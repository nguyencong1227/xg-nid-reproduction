"""Component 5 - Integrated Gradient Explainer (XG-NID Sec. 3.1.5, Eq. 10).

    IG_i(x) = (x_i - x'_i) * integral_{a=0..1} dF(x' + a(x - x')) / dx_i  da

with the zero baseline the paper specifies.  The integral is approximated with
a Riemann sum over ``steps`` interpolation points, which needs no change to the
trained network -- only repeated calls to ``autograd.grad``.

Attributions are produced for both node types and, additionally, for the edge
attributes, so a prediction can be traced to a flow statistic, to a specific
packet's payload, or to the timing between two packets.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .config import PAYLOAD_SPECIFIC_CLASSES
from .step3_graph import GraphSpec
from .step4_model import XGNIDHGNN, unpack


def integrated_gradients(
    model: XGNIDHGNN,
    batch,
    target: int,
    steps: int = 32,
    include_edges: bool = True,
) -> dict[str, torch.Tensor]:
    """Eq. 10 for one graph, against an all-zero baseline."""
    model.eval()
    x_dict, edge_index_dict, edge_attr_dict, batch_dict = unpack(batch)

    inputs = {("node", k): v.detach() for k, v in x_dict.items()}
    if include_edges:
        inputs.update({("edge", k): v.detach() for k, v in edge_attr_dict.items()})
    baselines = {k: torch.zeros_like(v) for k, v in inputs.items()}
    total = {k: torch.zeros_like(v) for k, v in inputs.items()}

    # Midpoint rule: alphas at the centre of each of the `steps` sub-intervals.
    for alpha in (np.arange(steps) + 0.5) / steps:
        scaled = {k: (baselines[k] + alpha * (inputs[k] - baselines[k])).requires_grad_(True)
                  for k in inputs}
        xs = {k[1]: v for k, v in scaled.items() if k[0] == "node"}
        eas = ({k[1]: v for k, v in scaled.items() if k[0] == "edge"}
               if include_edges else edge_attr_dict)
        out = model(xs, edge_index_dict, eas, batch_dict)
        score = out[0, target]
        grads = torch.autograd.grad(score, list(scaled.values()), allow_unused=True)
        for k, g in zip(scaled.keys(), grads):
            if g is not None:
                total[k] += g.detach()

    attributions = {}
    for k in inputs:
        avg_grad = total[k] / steps
        attr = (inputs[k] - baselines[k]) * avg_grad
        kind, name = k
        key = name if kind == "node" else f"edge:{'__'.join(name)}"
        attributions[key] = attr.detach().cpu()
    return attributions


def _payload_bytes(packet_x: torch.Tensor, spec: GraphSpec) -> np.ndarray:
    """Recover the 0-255 byte matrix from a packet node feature block."""
    offset = 8 if spec.include_packet_flags else 0
    payload = packet_x[:, offset: offset + spec.payload_dim].numpy()
    if spec.payload_scale == "unit":
        payload = payload * 255.0
    return np.clip(np.rint(payload), 0, 255).astype(np.uint8)


def _to_ascii(byte_row: np.ndarray, strip_trailing_zeros: bool = True) -> str:
    if strip_trailing_zeros:
        nz = np.nonzero(byte_row)[0]
        byte_row = byte_row[: nz[-1] + 1] if nz.size else byte_row[:0]
    return "".join(chr(b) if 32 <= b < 127 else "." for b in byte_row)


def explain_graph(
    model: XGNIDHGNN,
    data,
    spec: GraphSpec,
    class_names: list[str],
    device: torch.device,
    steps: int = 32,
    top_k: int = 10,
    top_packets: int = 3,
    label_mapping: dict[int, int] | None = None,
) -> dict:
    """Run the model plus IG on one flow and return a structured explanation."""
    batch = next(iter(DataLoader([data], batch_size=1))).to(device)
    model.eval()
    with torch.no_grad():
        log_probs = model(*unpack(batch))
    probs = log_probs.exp()[0].cpu().numpy()
    pred = int(probs.argmax())
    true = int(data.y) if getattr(data, "y", None) is not None else None
    if true is not None and label_mapping:
        # datasets keep the global 8-class id; the model uses the compacted one
        true = label_mapping.get(true, true)
    if true is not None and not 0 <= true < len(class_names):
        true = None

    attr = integrated_gradients(model, batch, pred, steps=steps)

    # --- flow-level importance (Algorithm 3 lines 9-14) ------------------
    flow_attr = attr["flow"][0].numpy()
    flow_x = batch["flow"].x[0].detach().cpu().numpy()
    if spec.standardize and spec.mean is not None:
        actual = flow_x * np.asarray(spec.std) + np.asarray(spec.mean)
    else:
        actual = flow_x
    # Constant columns have std forced to 1.0, so de-standardising them leaves
    # float noise around zero; snap it so the prompt does not read "-1.09e-08".
    actual = np.where(np.abs(actual) < 1e-6, 0.0, actual)
    order = np.argsort(-np.abs(flow_attr))[:top_k]
    top_flow = [
        {"feature": spec.flow_columns[i],
         "importance": float(flow_attr[i]),
         "value": float(actual[i])}
        for i in order
    ]

    # --- payload importance (Algorithm 3 lines 19-24) --------------------
    packet_attr = attr["packet"].numpy()
    norms = np.linalg.norm(packet_attr, axis=1, keepdims=True)
    normalised = packet_attr / np.maximum(norms, 1e-12)
    packet_score = np.abs(packet_attr).mean(axis=1)
    payload = _payload_bytes(batch["packet"].x.detach().cpu(), spec)
    pkt_order = np.argsort(-packet_score)[:top_packets]
    top_payloads = [
        {"packet_index": int(i),
         "importance": float(packet_score[i]),
         "n_nonzero_bytes": int(np.count_nonzero(payload[i])),
         "hex": payload[i][: np.max(np.nonzero(payload[i])[0]) + 1].tobytes().hex()
                if np.any(payload[i]) else "",
         "ascii": _to_ascii(payload[i]),
         "top_byte_offsets": np.argsort(-np.abs(normalised[i]))[:16].tolist()}
        for i in pkt_order
    ]

    edge_importance = {
        k: float(np.abs(v.numpy()).sum()) for k, v in attr.items() if k.startswith("edge:")
    }

    return {
        "predicted_class": class_names[pred],
        "predicted_index": pred,
        "confidence": float(probs[pred]),
        "probabilities": {n: float(p) for n, p in zip(class_names, probs)},
        "true_class": class_names[true] if true is not None else None,
        "num_packets": int(batch["packet"].x.shape[0]),
        "top_flow_features": top_flow,
        "top_payloads": top_payloads,
        "edge_importance": edge_importance,
        "payload_specific": class_names[pred] in PAYLOAD_SPECIFIC_CLASSES,
    }


def explain_run(
    run_dir: str,
    graphs_dir: str,
    split: str = "test",
    num: int = 5,
    steps: int = 32,
    top_k: int = 10,
    out_dir: str | None = None,
    llm: str | None = None,
    seed: int = 0,
) -> str:
    """Explain ``num`` flows from a split and write JSON + rendered prompts."""
    from .step3_graph import XGNIDGraphDataset
    from .step7_llm_explain import LLMExplainer, build_prompts

    out_dir = out_dir or os.path.join(run_dir, f"explanations_{split}")
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    label_mapping = {int(k): int(v) for k, v in ckpt.get("label_mapping", {}).items()}
    spec = GraphSpec.load(os.path.join(graphs_dir, "spec.json"))
    dataset = XGNIDGraphDataset(root=os.path.join(graphs_dir, split), name=split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ckpt["cfg"]
    model = XGNIDHGNN(ckpt["metadata"], len(class_names), hidden=cfg["hidden"],
                      heads=cfg["heads"], use_edge_attr=cfg["use_edge_attr"],
                      dropout=cfg["dropout"], head_dims=tuple(cfg["head_dims"]))
    warm = next(iter(DataLoader([dataset[0], dataset[0]], batch_size=2)))
    model.eval()
    with torch.no_grad():
        model(*unpack(warm))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    # Spread the sample across classes so payload-specific attacks show up.
    rng = np.random.default_rng(seed)
    labels = np.array([label_mapping.get(int(d.y), int(d.y)) for d in dataset])
    picks: list[int] = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        picks.extend(rng.choice(idx, size=min(len(idx), max(1, num // len(np.unique(labels)))),
                                replace=False).tolist())
    picks = picks[:num] if len(picks) >= num else picks

    generator = LLMExplainer(llm) if llm else None
    records = []
    for i in picks:
        exp = explain_graph(model, dataset[i], spec, class_names, device,
                            steps=steps, top_k=top_k, label_mapping=label_mapping)
        exp["graph_index"] = int(i)
        exp["prompts"] = build_prompts(exp, top_n=top_k)
        if generator is not None:
            exp["llm"] = generator.answer(exp["prompts"])
        records.append(exp)
        print(f"[step6] graph {i}: pred={exp['predicted_class']} "
              f"({exp['confidence']:.3f}) true={exp['true_class']}")

    json_path = os.path.join(out_dir, "explanations.json")
    with open(json_path, "w") as fh:
        json.dump(records, fh, indent=2)

    txt_path = os.path.join(out_dir, "explanations.txt")
    with open(txt_path, "w") as fh:
        for r in records:
            fh.write("=" * 78 + "\n")
            fh.write(f"graph {r['graph_index']}  pred={r['predicted_class']} "
                     f"conf={r['confidence']:.3f}  true={r['true_class']}\n\n")
            for key, prompt in r["prompts"].items():
                fh.write(f"--- prompt[{key}] ---\n{prompt}\n\n")
            for key, text in (r.get("llm") or {}).items():
                fh.write(f"--- llm[{key}] ---\n{text}\n\n")
    print(f"[step6] wrote {json_path} and {txt_path}")
    return json_path

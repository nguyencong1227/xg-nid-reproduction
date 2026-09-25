"""End-to-end inference: raw pcap -> per-flow prediction.

This is the path the paper's real-time claim rests on (Sec. 3.1.1): flows are
cut at 20 packets, so a verdict is available a handful of packets into a
connection rather than after a 30-minute flow timeout.

Components 1 -> 2 -> 3 -> 4 are chained with the ``spec.json`` and checkpoint
produced by training, so the feature space at inference is byte-identical to
the one the model was fitted on.

Caveat: the Component 2 temporal features are rolling statistics over the
flows *seen so far in this capture*.  Scoring a pcap in isolation therefore
gives a different context than scoring the same traffic inside a longer
capture, and the first few flows of any capture have a nearly empty window.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .step1_flow_extract import extract_pcap
from .step2_expl_features import add_explainable_features
from .step3_graph import GraphSpec, row_to_graph
from .step4_model import XGNIDHGNN, unpack

META_COLUMNS = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol",
                "bidirectional_first_seen_ms", "bidirectional_packets"]


def load_model(run_dir: str, device: torch.device | None = None):
    """Rebuild the trained HGNN from a run directory."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=False)
    cfg = ckpt["cfg"]
    model = XGNIDHGNN(ckpt["metadata"], len(ckpt["class_names"]), hidden=cfg["hidden"],
                      heads=cfg["heads"], use_edge_attr=cfg["use_edge_attr"],
                      dropout=cfg["dropout"], head_dims=tuple(cfg["head_dims"]))
    return model, ckpt, device


def _materialise(model, sample, device):
    """Run one forward pass so the lazy GATConv layers allocate their weights."""
    batch = next(iter(DataLoader([sample, sample], batch_size=2)))
    model.eval()
    with torch.no_grad():
        model(*unpack(batch))
    return model.to(device)


def predict_pcap(
    pcap_path: str,
    run_dir: str,
    spec_path: str | None = None,
    window=350,
    batch_size: int = 64,
    limit: int = 20,
    out_csv: str | None = None,
) -> pd.DataFrame:
    """Score every flow in ``pcap_path`` and return one row per flow."""
    spec = GraphSpec.load(spec_path or os.path.join(run_dir, "spec.json"))
    model, ckpt, device = load_model(run_dir)
    class_names = ckpt["class_names"]

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = extract_pcap(pcap_path, tmp, limit=limit, progress_every=0)
        df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(columns=META_COLUMNS + ["predicted_class", "confidence"])

    df = add_explainable_features(df, window=window)
    missing = [c for c in spec.flow_columns if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    if missing:
        print(f"[step9] {len(missing)} feature(s) absent from this capture, "
              f"zero-filled: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    graphs = [row_to_graph(row, spec) for _, row in df.iterrows()]
    model = _materialise(model, graphs[0], device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    probs = []
    for batch in DataLoader(graphs, batch_size=batch_size):
        batch = batch.to(device)
        with torch.no_grad():
            probs.append(model(*unpack(batch)).exp().cpu().numpy())
    probs = np.concatenate(probs)
    pred = probs.argmax(1)

    out = df[[c for c in META_COLUMNS if c in df.columns]].copy()
    out["predicted_class"] = [class_names[i] for i in pred]
    out["confidence"] = probs.max(1)
    for i, name in enumerate(class_names):
        out[f"p_{name}"] = probs[:, i]

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        out.to_csv(out_csv, index=False)
        print(f"[step9] wrote {len(out):,} predictions to {out_csv}")
    counts = out["predicted_class"].value_counts()
    print("[step9] " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return out

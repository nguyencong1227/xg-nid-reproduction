"""Training and evaluation for the XG-NID HGNN (paper Sec. 4.1).

Graph-level multi-class classification with NLL loss on the log-softmax output
of Eq. 9.  Reported metrics mirror Table 5/6: precision, recall and F1, both
macro and weighted, plus a per-class breakdown and a confusion matrix.

Labels are stored as the global 8-class ids from ``config.LABEL_DICT``.  When a
run only covers some of those classes, they are compacted to a contiguous range
and the mapping is written into the checkpoint, so a model trained on two
captures still says "BruteForce" rather than "class 1".
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch_geometric.loader import DataLoader

from .config import CLASS_NAMES
from .step4_model import XGNIDHGNN, unpack


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 64
    heads: int = 1
    dropout: float = 0.0
    use_edge_attr: bool = True
    class_weights: bool = True
    val_frac: float = 0.1
    patience: int = 8
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    head_dims: tuple[int, ...] = (64, 16)


def label_mapping(labels: np.ndarray) -> tuple[dict[int, int], list[str]]:
    present = sorted(set(int(v) for v in labels))
    mapping = {g: i for i, g in enumerate(present)}
    names = [CLASS_NAMES[g] if g < len(CLASS_NAMES) else str(g) for g in present]
    return mapping, names


def _remap(dataset, mapping: dict[int, int]) -> list:
    out = []
    for d in dataset:
        d = d.clone()
        d.y = torch.tensor([mapping[int(d.y)]], dtype=torch.long)
        out.append(d)
    return out


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    preds, trues, losses = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(*unpack(batch))
        losses.append(float(model.loss(out, batch.y)) * batch.num_graphs)
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(trues) if trues else np.array([])
    n = len(y_true)
    return {
        "loss": sum(losses) / max(n, 1),
        "accuracy": float((y_pred == y_true).mean()) if n else 0.0,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "_y_true": y_true,
        "_y_pred": y_pred,
    }


def train(train_set, test_set, cfg: TrainConfig, out_dir: str):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device)

    all_labels = np.array([int(d.y) for d in train_set] + [int(d.y) for d in test_set])
    mapping, class_names = label_mapping(all_labels)
    num_classes = len(class_names)
    print(f"[train] {num_classes} classes: {class_names}")

    train_all = _remap(train_set, mapping)
    test_all = _remap(test_set, mapping)

    # Stratified-ish validation split.
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(train_all))
    n_val = int(len(idx) * cfg.val_frac)
    val_data = [train_all[i] for i in idx[:n_val]]
    tr_data = [train_all[i] for i in idx[n_val:]]

    loaders = {
        "train": DataLoader(tr_data, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers),
        "val": DataLoader(val_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers),
        "test": DataLoader(test_all, batch_size=cfg.batch_size, num_workers=cfg.num_workers),
    }

    model = XGNIDHGNN(
        tr_data[0].metadata(), num_classes, hidden=cfg.hidden, heads=cfg.heads,
        use_edge_attr=cfg.use_edge_attr, dropout=cfg.dropout, head_dims=tuple(cfg.head_dims),
    )
    warm = next(iter(loaders["train"]))
    model.eval()
    with torch.no_grad():
        model(*unpack(warm))            # materialise the lazy GATConv shapes
    model = model.to(device)

    weight = None
    if cfg.class_weights:
        counts = np.bincount([int(d.y) for d in tr_data], minlength=num_classes)
        w = counts.sum() / (num_classes * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[train] class weights: {np.round(w, 3).tolist()}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    history, best_f1, best_epoch, bad = [], -1.0, -1, 0
    ckpt_path = os.path.join(out_dir, "model.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(*unpack(batch))
            loss = model.loss(out, batch.y, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss) * batch.num_graphs
            seen += batch.num_graphs

        val = evaluate(model, loaders["val"], device, num_classes)
        sched.step(val["f1_macro"])
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1),
                        **{k: v for k, v in val.items() if not k.startswith("_")}})
        print(f"[train] epoch {epoch:3d}  loss {total / max(seen, 1):.4f}  "
              f"val_acc {val['accuracy']:.4f}  val_f1 {val['f1_macro']:.4f}  "
              f"({time.time() - t0:.1f}s)")

        if val["f1_macro"] > best_f1:
            best_f1, best_epoch, bad = val["f1_macro"], epoch, 0
            torch.save({"state_dict": model.state_dict(),
                        "cfg": asdict(cfg),
                        "label_mapping": {str(k): v for k, v in mapping.items()},
                        "class_names": class_names,
                        "metadata": tr_data[0].metadata()}, ckpt_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[train] early stop at epoch {epoch} (best {best_epoch})")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test = evaluate(model, loaders["test"], device, num_classes)
    y_true, y_pred = test.pop("_y_true"), test.pop("_y_pred")
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    print("\n[test] " + "  ".join(f"{k}={v:.4f}" for k, v in test.items()))
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("confusion matrix (rows=true):")
    print(cm)

    results = {"config": asdict(cfg), "class_names": class_names,
               "best_epoch": best_epoch, "history": history,
               "test": test, "per_class": report, "confusion_matrix": cm.tolist()}
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[train] wrote {ckpt_path} and results.json")
    return model, results

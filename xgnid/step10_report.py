"""Comparison table in the shape of the paper's Tables 5 and 6.

Table 5 lists flow-only methods with a "Payload-Specific" column; Table 6 lists
packet-only methods with a "Flow-Specific" column.  Both exist to show that a
single-modality model is strong overall and weak exactly where its modality is
blind, which is the gap XG-NID claims to close -- so the proposed model is
appended to both tables from the same test split.
"""
from __future__ import annotations

import json
import os

from .config import PAYLOAD_SPECIFIC_CLASSES


def _proposed_row(results: dict) -> dict:
    """Pull the HGNN's overall and per-subset scores out of results.json."""
    per_class = results["per_class"]
    names = results["class_names"]
    payload_specific = [n for n in names if n in PAYLOAD_SPECIFIC_CLASSES]
    flow_specific = [n for n in names if n not in PAYLOAD_SPECIFIC_CLASSES]

    def mean_f1(subset):
        vals = [per_class[n]["f1-score"] for n in subset if n in per_class]
        return sum(vals) / len(vals) if vals else None

    return {
        "precision": results["test"]["precision_macro"],
        "recall": results["test"]["recall_macro"],
        "f1": results["test"]["f1_macro"],
        "payload_specific_f1": mean_f1(payload_specific),
        "flow_specific_f1": mean_f1(flow_specific),
    }


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def build_report(run_dir: str, baselines_dir: str | None = None,
                 out_path: str | None = None) -> str:
    with open(os.path.join(run_dir, "results.json")) as fh:
        results = json.load(fh)
    baselines = {}
    if baselines_dir:
        bpath = os.path.join(baselines_dir, "baselines.json")
        if os.path.exists(bpath):
            with open(bpath) as fh:
                baselines = json.load(fh)

    proposed = _proposed_row(results)
    lines: list[str] = []
    lines.append(f"# XG-NID reproduction — {', '.join(results['class_names'])}\n")
    lines.append(f"Test split: {sum(sum(r) for r in results['confusion_matrix'])} flows. "
                 f"Best epoch {results['best_epoch']}.\n")

    for modality, subset_key, subset_title in (
        ("flow", "payload_specific_f1", "Payload-Specific"),
        ("packet", "flow_specific_f1", "Flow-Specific"),
    ):
        rows = {k.split("/", 1)[1]: v for k, v in baselines.items()
                if k.startswith(modality + "/")}
        if not rows:
            continue
        title = ("Table 5 — comparison with flow-level methods" if modality == "flow"
                 else "Table 6 — comparison with packet-level methods")
        lines.append(f"\n## {title}\n")
        lines.append(f"| Method | Precision | Recall | F1 | {subset_title} F1 |")
        lines.append("|---|---|---|---|---|")
        for name, r in rows.items():
            lines.append(f"| {name} | {_fmt(r['precision'])} | {_fmt(r['recall'])} "
                         f"| {_fmt(r['f1'])} | {_fmt(r.get(subset_key))} |")
        lines.append(f"| **XG-NID (this work)** | **{_fmt(proposed['precision'])}** "
                     f"| **{_fmt(proposed['recall'])}** | **{_fmt(proposed['f1'])}** "
                     f"| **{_fmt(proposed[subset_key])}** |")

    lines.append("\n## Per-class results (XG-NID)\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for name in results["class_names"]:
        r = results["per_class"][name]
        lines.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} "
                     f"| {r['f1-score']:.3f} | {int(r['support'])} |")

    lines.append("\n## Confusion matrix (rows = true)\n")
    header = "| |" + "|".join(results["class_names"]) + "|"
    lines.append(header)
    lines.append("|---|" + "---|" * len(results["class_names"]))
    for name, row in zip(results["class_names"], results["confusion_matrix"]):
        lines.append(f"| **{name}** |" + "|".join(str(v) for v in row) + "|")

    text = "\n".join(lines) + "\n"
    out_path = out_path or os.path.join(run_dir, "REPORT.md")
    with open(out_path, "w") as fh:
        fh.write(text)
    print(f"[step10] wrote {out_path}")
    return text

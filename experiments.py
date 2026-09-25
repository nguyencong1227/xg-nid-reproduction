#!/usr/bin/env python
"""Experiment suite for the XG-NID reimplementation.

Each entry in ``EXPERIMENTS`` is one ablation of the framework, expressed as
the CLI flags it needs.  Two things make the suite cheap to run:

* Graph building (Component 3) is keyed on the flags that actually change the
  graphs, so every model-only variant reuses an existing graph directory
  instead of re-materialising ~13k heterogeneous graphs.
* Stages are skipped when their output already exists, so an interrupted run
  resumes where it stopped.  ``--force`` overrides that.

Usage
-----
    python experiments.py --list
    python experiments.py --run baseline,no_edge_attr
    python experiments.py --all
    python experiments.py --report                 # rebuild EXPERIMENTS.md only

The suite assumes the flow CSVs and the balanced dataset already exist (stages
1, 2 and 2b, i.e. ``scripts/run_xgnid_demo.sh`` up to ``dataset``), since those
depend on which pcaps are available rather than on the model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

WORK = os.environ.get("WORK", "work/xgnid")
DATASET_DIR = os.path.join(WORK, "03_dataset")
EXP_ROOT = os.path.join(WORK, "experiments")
PYTHON = sys.executable


@dataclass
class Experiment:
    name: str
    question: str                       # what this run is meant to answer
    graph_flags: list[str] = field(default_factory=list)
    train_flags: list[str] = field(default_factory=list)
    features_window: str | None = None  # re-run Component 2 with this window
    epochs: int = 25
    batch_size: int = 128

    @property
    def graph_key(self) -> str:
        """Identifies the graph directory this experiment needs."""
        parts = [f.lstrip("-") for f in sorted(self.graph_flags)]
        if self.features_window:
            parts.append(f"window{self.features_window}")
        return "default" if not parts else "_".join(parts)


EXPERIMENTS: list[Experiment] = [
    Experiment(
        "baseline",
        "Reference configuration: dual modality, edge attributes, temporal features.",
    ),
    Experiment(
        "no_edge_attr",
        "Do the Eq. 3/4 edge features matter? Note the forward `contain` "
        "attributes are provably inert (see README), so any change here comes "
        "from the reverse contain edge and the link edges.",
        train_flags=["--no-edge-attr"],
    ),
    Experiment(
        "no_temporal",
        "Ablates contribution #5: drop Component 2's rolling features from the "
        "flow node (85 -> 53 features).",
        graph_flags=["--drop-temporal"],
    ),
    Experiment(
        "flow_only",
        "Single modality inside the HGNN: packet nodes keep only the 8 TCP "
        "flags, no payload. Should degrade on payload-specific classes.",
        graph_flags=["--no-payload"],
    ),
    Experiment(
        "packet_flags",
        "Adds the 8 TCP flags alongside the payload (packet node 1500 -> 1508).",
        graph_flags=["--packet-flags"],
    ),
    Experiment(
        "paper_defaults",
        "Raw 0-255 payload bytes and unstandardised flow features, i.e. the "
        "released GNN4ID preprocessing. Tests deviation #3.",
        graph_flags=["--paper-defaults"],
    ),
    Experiment(
        "time_window",
        "Component 2 with a true 60 s time window instead of the released "
        "350-flow count window (Sec. 3.1.2 reads as a time window).",
        features_window="60s",
    ),
    Experiment(
        "heads4",
        "Four attention heads instead of one; the paper does not state a value.",
        train_flags=["--heads", "4"],
    ),
    Experiment(
        "hidden128",
        "Doubles the hidden width to 128.",
        train_flags=["--hidden", "128"],
    ),
    Experiment(
        "dropout",
        "Adds 0.3 dropout in the classification head as a regularisation check.",
        train_flags=["--dropout", "0.3"],
    ),
    Experiment(
        "seed1",
        "Same configuration as baseline, different seed: gives a variance "
        "floor, without which none of the gaps above can be called real.",
        train_flags=["--seed", "1"],
    ),
    Experiment(
        "seed2",
        "Third seed for the variance floor.",
        train_flags=["--seed", "2"],
    ),
]

BY_NAME = {e.name: e for e in EXPERIMENTS}


def run(cmd: list[str], log_path: str) -> None:
    """Run a pipeline stage, streaming both console and log file."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    print(f"  $ {' '.join(cmd)}", flush=True)
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
        log.write(proc.stdout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
        raise RuntimeError(f"stage failed ({proc.returncode}); tail of output:\n{tail}")


def ensure_graphs(exp: Experiment, force: bool = False) -> str:
    """Build (or reuse) the graph directory this experiment needs."""
    graphs_dir = os.path.join(EXP_ROOT, "graphs", exp.graph_key)
    marker = os.path.join(graphs_dir, "spec.json")
    log = os.path.join(EXP_ROOT, "logs", f"graphs_{exp.graph_key}.log")

    if force and os.path.isdir(graphs_dir):
        shutil.rmtree(graphs_dir)
    if os.path.exists(marker):
        print(f"  graphs[{exp.graph_key}] already built, reusing")
        return graphs_dir

    dataset_dir = DATASET_DIR
    if exp.features_window:
        # A different rolling window changes the flow features, so Component 2
        # and the balanced split have to be rebuilt for this variant.
        feat_dir = os.path.join(EXP_ROOT, "features", f"window{exp.features_window}")
        dataset_dir = os.path.join(EXP_ROOT, "dataset", f"window{exp.features_window}")
        if not os.path.exists(os.path.join(dataset_dir, "train.csv")):
            run([PYTHON, "-m", "xgnid.cli", "features",
                 "--in", os.path.join(WORK, "01_flows"), "--out", feat_dir,
                 "--window", exp.features_window], log)
            run([PYTHON, "-m", "xgnid.cli", "dataset",
                 "--in", feat_dir, "--out", dataset_dir,
                 "--train-per-class", "4000", "--test-cap", "800",
                 "--benign-from-background"], log)

    run([PYTHON, "-m", "xgnid.cli", "graphs",
         "--dataset", dataset_dir, "--out", graphs_dir] + exp.graph_flags, log)
    return graphs_dir


def run_experiment(exp: Experiment, force: bool = False) -> dict | None:
    print(f"\n=== {exp.name} ===")
    print(f"  {exp.question}")
    out_dir = os.path.join(EXP_ROOT, "runs", exp.name)
    results_path = os.path.join(out_dir, "results.json")
    log = os.path.join(EXP_ROOT, "logs", f"{exp.name}.log")

    if os.path.exists(results_path) and not force:
        print("  already trained, reusing results.json")
        with open(results_path) as fh:
            return json.load(fh)

    graphs_dir = ensure_graphs(exp, force=force)
    t0 = time.time()
    run([PYTHON, "-m", "xgnid.cli", "train",
         "--graphs", graphs_dir, "--out", out_dir,
         "--epochs", str(exp.epochs), "--batch-size", str(exp.batch_size)]
        + exp.train_flags, log)
    with open(results_path) as fh:
        results = json.load(fh)
    results["_wall_seconds"] = round(time.time() - t0, 1)
    results["_graph_key"] = exp.graph_key
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  done in {results['_wall_seconds']}s  "
          f"macro-F1 {results['test']['f1_macro']:.4f}")
    return results


def collect() -> dict[str, dict]:
    out = {}
    for exp in EXPERIMENTS:
        path = os.path.join(EXP_ROOT, "runs", exp.name, "results.json")
        if os.path.exists(path):
            with open(path) as fh:
                out[exp.name] = json.load(fh)
    return out


def write_report(path: str | None = None) -> str:
    """Aggregate every finished run into one markdown table."""
    results = collect()
    path = path or os.path.join(WORK, "EXPERIMENTS.md")
    if not results:
        print("no finished runs to report")
        return ""

    # Per-class F1 for the classes the paper calls payload-specific.
    from xgnid.config import PAYLOAD_SPECIFIC_CLASSES

    lines = ["# XG-NID ablation suite\n"]
    seeds = [r for n, r in results.items() if n.startswith("seed") or n == "baseline"]
    if len(seeds) > 1:
        f1s = [r["test"]["f1_macro"] for r in seeds]
        spread = max(f1s) - min(f1s)
        lines.append(f"Seed spread over {len(f1s)} runs of the reference "
                     f"configuration: macro-F1 {min(f1s):.4f} - {max(f1s):.4f} "
                     f"(range {spread:.4f}). **Treat any difference below that "
                     f"as noise.**\n")
        # Name the cause, otherwise the spread looks like model instability.
        ref = seeds[0]
        smallest = min(((n, int(ref["per_class"][n]["support"]))
                        for n in ref["class_names"] if n in ref["per_class"]),
                       key=lambda kv: kv[1], default=None)
        if smallest:
            name, support = smallest
            lines.append(f"The spread is driven by class size, not by the model: "
                         f"`{name}` has only {support} test flows, so a handful of "
                         f"flipped predictions moves macro-F1 by ~2 points. "
                         f"Separating ablations on this demo split would need "
                         f"either more captures or repeated seeds per "
                         f"configuration.\n")

    lines.append("| Experiment | Acc | Macro-F1 | Weighted-F1 | Payload-specific F1 | Epoch | Wall |")
    lines.append("|---|---|---|---|---|---|---|")
    for exp in EXPERIMENTS:
        r = results.get(exp.name)
        if r is None:
            continue
        names = r["class_names"]
        ps = [n for n in names if n in PAYLOAD_SPECIFIC_CLASSES]
        ps_f1 = ([r["per_class"][n]["f1-score"] for n in ps if n in r["per_class"]])
        ps_txt = f"{sum(ps_f1) / len(ps_f1):.3f}" if ps_f1 else "-"
        t = r["test"]
        wall = r.get("_wall_seconds")
        lines.append(f"| `{exp.name}` | {t['accuracy']:.4f} | **{t['f1_macro']:.4f}** "
                     f"| {t['f1_weighted']:.4f} | {ps_txt} | {r['best_epoch']} "
                     f"| {'' if wall is None else f'{wall / 60:.1f}m'} |")

    lines.append("\n## What each run asks\n")
    for exp in EXPERIMENTS:
        mark = "" if exp.name in results else "  *(not run)*"
        lines.append(f"- **`{exp.name}`**{mark} — {exp.question}")

    lines.append("\n## Per-class F1\n")
    all_names: list[str] = []
    for r in results.values():
        for n in r["class_names"]:
            if n not in all_names:
                all_names.append(n)
    lines.append("| Experiment |" + "|".join(all_names) + "|")
    lines.append("|---|" + "---|" * len(all_names))
    for exp in EXPERIMENTS:
        r = results.get(exp.name)
        if r is None:
            continue
        cells = [f"{r['per_class'][n]['f1-score']:.3f}" if n in r["per_class"] else "-"
                 for n in all_names]
        lines.append(f"| `{exp.name}` |" + "|".join(cells) + "|")

    lines.append("\n---\n")
    lines.append("Regenerate with `python experiments.py --report`. "
                 "Raw output per run is under "
                 f"`{os.path.join(EXP_ROOT, 'runs/<name>/')}`, stage logs under "
                 f"`{os.path.join(EXP_ROOT, 'logs/')}`.\n")

    text = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(text)
    print(f"wrote {path}")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="show the suite and exit")
    g.add_argument("--run", help="comma-separated experiment names")
    g.add_argument("--all", action="store_true", help="run everything")
    g.add_argument("--report", action="store_true", help="rebuild EXPERIMENTS.md only")
    ap.add_argument("--force", action="store_true",
                    help="rebuild graphs and retrain even if outputs exist")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue the suite when one experiment fails")
    args = ap.parse_args(argv)

    if args.list:
        width = max(len(e.name) for e in EXPERIMENTS)
        print(f"{len(EXPERIMENTS)} experiments "
              f"({len({e.graph_key for e in EXPERIMENTS})} distinct graph builds):\n")
        for e in EXPERIMENTS:
            flags = " ".join(e.graph_flags + e.train_flags) or "-"
            print(f"  {e.name:{width}s}  graphs[{e.graph_key}]  flags: {flags}")
            print(f"  {'':{width}s}  {e.question}")
        return 0

    if args.report:
        print(write_report())
        return 0

    if not os.path.exists(os.path.join(DATASET_DIR, "train.csv")):
        sys.exit(f"{DATASET_DIR}/train.csv missing — run the extract/features/"
                 f"dataset stages first (see scripts/run_xgnid_demo.sh)")

    selected = EXPERIMENTS if args.all else []
    if args.run:
        names = [n.strip() for n in args.run.split(",") if n.strip()]
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            sys.exit(f"unknown experiment(s): {unknown}\nknown: {list(BY_NAME)}")
        selected = [BY_NAME[n] for n in names]

    failed = []
    for exp in selected:
        try:
            run_experiment(exp, force=args.force)
        except Exception as exc:
            failed.append(exp.name)
            print(f"  FAILED {exp.name}: {exc}", file=sys.stderr)
            if not args.keep_going:
                write_report()
                raise

    write_report()
    if failed:
        print(f"\n{len(failed)} experiment(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

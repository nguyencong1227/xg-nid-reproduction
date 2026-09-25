"""Command-line driver for the XG-NID pipeline.

    python -m xgnid.cli extract  --pcaps 'data/cic_pcap/*.pcap' --out work/xgnid/01_flows
    python -m xgnid.cli features --in work/xgnid/01_flows       --out work/xgnid/02_features
    python -m xgnid.cli dataset  --in work/xgnid/02_features    --out work/xgnid/03_dataset
    python -m xgnid.cli graphs   --dataset work/xgnid/03_dataset --out work/xgnid/04_graphs
    python -m xgnid.cli train    --graphs  work/xgnid/04_graphs  --out work/xgnid/05_run
    python -m xgnid.cli baselines --dataset work/xgnid/03_dataset --out work/xgnid/06_baselines
    python -m xgnid.cli infer    --pcap data/cic_pcap/XSS.pcap --run work/xgnid/05_run
    python -m xgnid.cli explain  --run work/xgnid/05_run --graphs work/xgnid/04_graphs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _add_extract(sub):
    p = sub.add_parser("extract", help="pcap -> flow+packet CSV (Component 1)")
    p.add_argument("--pcaps", required=True, help="glob or directory of pcap files")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=20, help="max packets per flow")
    p.add_argument("--idle-timeout", type=int, default=120)


def _add_features(sub):
    p = sub.add_parser("features", help="add temporal features (Component 2)")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", default="350",
                   help="rolling window: an integer (flow count) or a pandas "
                        "offset such as 60s (true time window)")


def _add_dataset(sub):
    p = sub.add_parser("dataset", help="MAC filter + balance + split (Sec. 3.2.1)")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-per-class", type=int, default=20000)
    p.add_argument("--test-cap", type=int, default=4000)
    p.add_argument("--benign-from-background", action="store_true",
                   help="treat non-attacker flows inside attack captures as Benign")
    p.add_argument("--seed", type=int, default=42)


def _add_graphs(sub):
    p = sub.add_parser("graphs", help="CSV -> heterogeneous graphs (Component 3)")
    p.add_argument("--dataset", required=True, help="dir holding train.csv / test.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--packet-flags", action="store_true",
                   help="append the 8 TCP flags to each packet node (1500 -> 1508)")
    p.add_argument("--paper-defaults", action="store_true",
                   help="raw byte values and no flow standardisation")
    p.add_argument("--no-payload", action="store_true",
                   help="ablation: drop payload bytes from packet nodes "
                        "(implies --packet-flags so the nodes keep features)")
    p.add_argument("--drop-temporal", action="store_true",
                   help="ablation: drop the Component 2 temporal features "
                        "from the flow node")
    p.add_argument("--max-rows", type=int, default=None)


def _add_train(sub):
    p = sub.add_parser("train", help="train + evaluate the HGNN (Components 4)")
    p.add_argument("--graphs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-edge-attr", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)


def _add_baselines(sub):
    p = sub.add_parser("baselines", help="flow-only / packet-only baselines (Sec. 4.1.1)")
    p.add_argument("--dataset", required=True, help="dir holding train.csv / test.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--modalities", default="flow,packet")
    p.add_argument("--max-train", type=int, default=20000)
    p.add_argument("--n-packets", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)


def _add_report(sub):
    p = sub.add_parser("report", help="Table 5/6-style comparison markdown")
    p.add_argument("--run", required=True)
    p.add_argument("--baselines", default=None)
    p.add_argument("--out", default=None)


def _add_infer(sub):
    p = sub.add_parser("infer", help="pcap -> per-flow prediction (Components 1-4)")
    p.add_argument("--pcap", required=True)
    p.add_argument("--run", required=True, help="training output dir with model.pt")
    p.add_argument("--spec", default=None, help="defaults to <run>/spec.json")
    p.add_argument("--window", default="350")
    p.add_argument("--out", default=None, help="CSV to write predictions to")


def _add_explain(sub):
    p = sub.add_parser("explain", help="integrated gradients + LLM prompts (Components 5-6)")
    p.add_argument("--run", required=True, help="training output dir with model.pt")
    p.add_argument("--graphs", required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--num", type=int, default=5, help="how many flows to explain")
    p.add_argument("--steps", type=int, default=32, help="IG interpolation steps")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", default=None)
    p.add_argument("--llm", default=None,
                   help="HF model id to actually generate text, e.g. "
                        "meta-llama/Meta-Llama-3-8B-Instruct (default: prompts only)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xgnid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for add in (_add_extract, _add_features, _add_dataset, _add_graphs,
                _add_train, _add_baselines, _add_report, _add_infer, _add_explain):
        add(sub)
    args = ap.parse_args(argv)

    if args.cmd == "extract":
        from .step1_flow_extract import extract_many
        pattern = args.pcaps
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "*.pcap*")
        files = sorted(glob.glob(pattern))
        if not files:
            sys.exit(f"no pcap matched {pattern}")
        extract_many(files, args.out, limit=args.limit, idle_timeout=args.idle_timeout)

    elif args.cmd == "features":
        from .step2_expl_features import process_csv
        window = int(args.window) if args.window.isdigit() else args.window
        files = sorted(glob.glob(os.path.join(args.inp, "*.csv")))
        if not files:
            sys.exit(f"no CSV found in {args.inp}")
        for f in files:
            out = os.path.join(args.out, os.path.basename(f))
            process_csv(f, out, window=window)
            print(f"[step2] {os.path.basename(f)} -> {out}")

    elif args.cmd == "dataset":
        from .step2b_preprocess import build_dataset
        build_dataset(args.inp, args.out, train_per_class=args.train_per_class,
                      test_cap=args.test_cap, seed=args.seed,
                      benign_from_attack_captures=args.benign_from_background)

    elif args.cmd == "graphs":
        import pandas as pd
        from .step3_graph import TEMPORAL_PREFIXES, XGNIDGraphDataset, fit_spec
        os.makedirs(args.out, exist_ok=True)
        train_csv = os.path.join(args.dataset, "train.csv")
        test_csv = os.path.join(args.dataset, "test.csv")
        spec_path = os.path.join(args.out, "spec.json")
        head = pd.read_csv(train_csv, nrows=20000)
        spec = fit_spec(head,
                        include_packet_flags=args.packet_flags or args.no_payload,
                        include_payload=not args.no_payload,
                        paper_defaults=args.paper_defaults,
                        exclude_prefixes=TEMPORAL_PREFIXES if args.drop_temporal else ())
        spec.save(spec_path)
        print(f"[step3] flow dim {spec.flow_dim}, packet dim {spec.packet_dim} -> {spec_path}")
        for name, csv in (("train", train_csv), ("test", test_csv)):
            if not os.path.exists(csv):
                continue
            ds = XGNIDGraphDataset(root=os.path.join(args.out, name), csv_path=csv,
                                   spec=spec, name=name, max_rows=args.max_rows)
            print(f"[step3] {name}: {len(ds)} graphs")

    elif args.cmd == "train":
        from .step3_graph import GraphSpec, XGNIDGraphDataset
        from .step5_train import TrainConfig, train as run_train
        spec = GraphSpec.load(os.path.join(args.graphs, "spec.json"))
        train_set = XGNIDGraphDataset(root=os.path.join(args.graphs, "train"), name="train")
        test_set = XGNIDGraphDataset(root=os.path.join(args.graphs, "test"), name="test")
        cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                          hidden=args.hidden, heads=args.heads, dropout=args.dropout,
                          use_edge_attr=not args.no_edge_attr, seed=args.seed,
                          **({"device": args.device} if args.device else {}))
        os.makedirs(args.out, exist_ok=True)
        spec.save(os.path.join(args.out, "spec.json"))
        run_train(train_set, test_set, cfg, args.out)

    elif args.cmd == "baselines":
        from .step8_baselines import run_baselines
        run_baselines(args.dataset, args.out,
                      modalities=tuple(m.strip() for m in args.modalities.split(",") if m.strip()),
                      max_train=args.max_train, n_packets=args.n_packets, seed=args.seed)

    elif args.cmd == "report":
        from .step10_report import build_report
        print(build_report(args.run, args.baselines, args.out))

    elif args.cmd == "infer":
        from .step9_infer import predict_pcap
        window = int(args.window) if args.window.isdigit() else args.window
        predict_pcap(args.pcap, args.run, spec_path=args.spec, window=window,
                     out_csv=args.out)

    elif args.cmd == "explain":
        from .step6_ig_explain import explain_run
        explain_run(run_dir=args.run, graphs_dir=args.graphs, split=args.split,
                    num=args.num, steps=args.steps, top_k=args.top_k,
                    out_dir=args.out, llm=args.llm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

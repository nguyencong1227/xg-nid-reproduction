#!/usr/bin/env python3
"""Run the NI-Diff reproduction on CIC-IoT2023.

    python nidiff/scripts/run_nidiff.py --smoke            # ~minutes, sanity check
    python nidiff/scripts/run_nidiff.py                    # full paper recipe
    python nidiff/scripts/run_nidiff.py --clf-epochs 50 --vae-epochs 100 --diff-epochs 300
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from nidiff.config import Config, smoke          # noqa: E402
from nidiff.pipeline import run                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true', help='tiny budget end-to-end run')
    ap.add_argument('--force', action='store_true', help='retrain even if checkpoints exist')
    ap.add_argument('--no-ablation', action='store_true', help='skip the Fig. 2 sweep')
    ap.add_argument('--dataset', choices=['cic_raw', 'xgnid'], default=None,
                    help='cic_raw = official CIC-IoT2023 (39 feat, 34 classes, '
                         'paper protocol); xgnid = nfstream re-extraction (82 feat, '
                         '8 super classes)')
    ap.add_argument('--label-granularity', choices=['super', 'fine'], default=None,
                    help="cic_raw only: 'super' = 4-way head over known super classes, "
                         "'fine' = literal 20-way head over the remaining fine classes")
    ap.add_argument('--arch', choices=['base', 'large'], default=None)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--clf-epochs', type=int, default=None)
    ap.add_argument('--vae-epochs', type=int, default=None)
    ap.add_argument('--diff-epochs', type=int, default=None)
    ap.add_argument('--gan-epochs', type=int, default=None)
    ap.add_argument('--target-fpr', type=float, default=None)
    ap.add_argument('--lambda-perc', type=float, default=None,
                    help='lambda_2 of Eq. 2. Set 0 to drop the perceptual term, which '
                         'is what trains the VAE to preserve the classifier verdict')
    ap.add_argument('--lambda-kld', type=float, default=None)
    ap.add_argument('--n-pool', type=int, default=None,
                    help='VAE pooling stages; 2 = paper default (1x9 latent on 39 '
                         'features), 1 = 1x19, 0 = no bottleneck')
    ap.add_argument('--clip', type=float, default=None,
                    help='standardised-space clip; 0 disables it')
    ap.add_argument('--grad-eps', type=float, default=None,
                    help='PGD L-inf budget; schema dependent (10 for cic_raw, 1 for xgnid)')
    ap.add_argument('--grad-alpha', type=float, default=None)
    ap.add_argument('--max-attack-samples', type=int, default=None)
    ap.add_argument('--zero-day', nargs='*', default=None,
                    help='override the held-out classes '
                         '(default: BruteForce Spoofing Recon WebBased)')
    args = ap.parse_args()

    cfg = Config()
    if args.smoke:
        cfg = smoke(cfg)
    if args.dataset:
        cfg.data.dataset = args.dataset
    if args.label_granularity:
        cfg.data.label_granularity = args.label_granularity
    if args.arch:
        cfg.clf.arch = args.arch
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = cfg.data.seed = args.seed
    if args.clf_epochs is not None:
        cfg.clf.epochs = args.clf_epochs
    if args.vae_epochs is not None:
        cfg.vae.epochs = args.vae_epochs
    if args.diff_epochs is not None:
        cfg.diff.epochs = args.diff_epochs
    if args.gan_epochs is not None:
        cfg.attack.gan_epochs = args.gan_epochs
    if args.target_fpr is not None:
        cfg.detect.target_fpr = args.target_fpr
    if args.lambda_perc is not None:
        cfg.vae.lambda_perc = args.lambda_perc
    if args.lambda_kld is not None:
        cfg.vae.lambda_kld = args.lambda_kld
    if args.n_pool is not None:
        cfg.vae.n_pool = args.n_pool
    if args.clip is not None:
        cfg.data.clip = args.clip
    if args.grad_eps is not None:
        cfg.attack.grad_eps = args.grad_eps
    if args.grad_alpha is not None:
        cfg.attack.grad_alpha = args.grad_alpha
    if cfg.data.dataset == 'xgnid' and args.grad_eps is None:
        cfg.attack.grad_eps, cfg.attack.grad_alpha = 1.0, 0.1
        cfg.attack.gan_pert_scale = 1.0
    if args.max_attack_samples is not None:
        cfg.attack.max_attack_samples = args.max_attack_samples
    if args.zero_day is not None:
        cfg.data.zero_day_classes = list(args.zero_day)

    out = run(cfg, force=args.force, do_ablation=not args.no_ablation)
    print()
    print(out['results'].to_string(index=False))
    print()
    print(f"artifacts: {cfg.path(cfg.out_dir)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

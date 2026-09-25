#!/usr/bin/env python3
"""Side-by-side of a run's results against the paper's published CIC numbers.

    python nidiff/scripts/compare_to_paper.py --out-dir nidiff/work/nidiff_raw

Writes `comparison.md` into the run directory and prints it. Reads
`results_raw.csv` and `summary.json`, so it works on any finished run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# NI-Diff (Zhang et al., MILCOM 2025), CIC-IoT2023 columns of Tables III/V/VII/VIII.
PAPER_ACC = {'accuracy': 99.42, 'asr_grad': 89.29, 'asr_gan': 99.97}

PAPER = {
    # detector: {scenario: {metric: value in %}}
    'GEN':      {'zero-day': dict(TPR=38.38, Precision=88.70, F1=53.58, AUROC=82.05),
                 'grad':     dict(TPR=99.96, Precision=95.34, F1=97.59, AUROC=99.89),
                 'GAN':      dict(TPR=99.97, Precision=95.34, F1=97.60, AUROC=99.33),
                 'FPR': 4.89},
    'Energy':   {'zero-day': dict(TPR=39.64, Precision=90.19, F1=55.07, AUROC=84.01),
                 'grad':     dict(TPR=99.82, Precision=95.86, F1=97.80, AUROC=99.90),
                 'GAN':      dict(TPR=99.97, Precision=95.87, F1=97.88, AUROC=99.42),
                 'FPR': 4.31},
    'GradNorm': {'zero-day': dict(TPR=30.11, Precision=83.27, F1=44.23, AUROC=76.50),
                 'grad':     dict(TPR=99.86, Precision=94.29, F1=96.99, AUROC=99.71),
                 'GAN':      dict(TPR=99.94, Precision=94.29, F1=97.03, AUROC=99.39),
                 'FPR': 6.05},
    'MANDA':    {'zero-day': dict(TPR=11.77, Precision=45.90, F1=18.74, AUROC=26.17),
                 'grad':     dict(TPR=56.77, Precision=80.37, F1=66.54, AUROC=82.33),
                 'GAN':      dict(TPR=99.95, Precision=87.81, F1=93.49, AUROC=97.71),
                 'FPR': 13.87},
    'DeepRP':   {'zero-day': dict(TPR=16.96, Precision=89.36, F1=28.51, AUROC=58.41),
                 'grad':     dict(TPR=3.96,  Precision=97.92, F1=96.42, AUROC=62.97),
                 'GAN':      dict(TPR=94.97, Precision=66.22, F1=7.47,  AUROC=97.51),
                 'FPR': 2.02},
    'NIDS-DA':  {'zero-day': dict(TPR=13.38, Precision=78.34, F1=22.86, AUROC=83.69),
                 'grad':     dict(TPR=32.73, Precision=89.84, F1=47.98, AUROC=98.08),
                 'GAN':      dict(TPR=94.74, Precision=96.24, F1=95.48, AUROC=99.48),
                 'FPR': 3.70},
    'NI-Diff':  {'zero-day': dict(TPR=51.18, Precision=93.98, F1=66.27, AUROC=95.70),
                 'grad':     dict(TPR=94.31, Precision=96.64, F1=95.46, AUROC=99.46),
                 'GAN':      dict(TPR=99.98, Precision=96.82, F1=98.37, AUROC=99.94),
                 'FPR': 3.28},
}

ORDER = ['GEN', 'Energy', 'GradNorm', 'MANDA', 'DeepRP', 'NIDS-DA', 'NI-Diff']
SCEN = ['zero-day', 'grad', 'GAN']


def _fmt(v) -> str:
    return '--' if v is None or v != v else f'{v:.2f}'


def _recalibrated(run_dir: str):
    """Re-threshold every detector at the FPR the paper reports for it.

    Our runs calibrate all detectors to one common target, so their FPR column
    is flat while the paper's spans 2.02-13.87. Comparing precision or F1 across
    that mismatch is not apples-to-apples: a detector held at a higher FPR
    necessarily shows lower precision. This recomputes the threshold-dependent
    metrics at each detector's *published* FPR. AUROC is unaffected.
    """
    import numpy as np

    from nidiff.metrics import scenario_metrics
    p = os.path.join(run_dir, 'scores.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    out = {}
    for det in ORDER:
        if f'{det}/id' not in z:
            continue
        s_id = z[f'{det}/id']
        thr = float(np.quantile(s_id, 1.0 - PAPER[det]['FPR'] / 100.0))
        for s in SCEN:
            k = f'{det}/{s}'
            if k in z:
                out[(det, s)] = scenario_metrics(s_id, z[k], thr)
    return out


def build(run_dir: str, match_fpr: bool = False) -> str:
    import pandas as pd
    df = pd.read_csv(os.path.join(run_dir, 'results_raw.csv'))
    summary = {}
    sp = os.path.join(run_dir, 'summary.json')
    if os.path.exists(sp):
        summary = json.load(open(sp))
    recal = _recalibrated(run_dir) if match_fpr else None

    def ours(det, scen, metric):
        if recal is not None and (det, scen) in recal:
            return recal[(det, scen)][metric] * 100
        r = df[(df.detector == det) & (df.scenario == scen)]
        return float(r[metric].iloc[0]) * 100 if len(r) else float('nan')

    def ours_fpr(det):
        r = df[df.detector == det]
        return float(r['FPR'].mean()) * 100 if len(r) else float('nan')

    L = ['# NI-Diff reproduction vs the paper (CIC-IoT2023)', '',
         'Left column of each pair is this run, right is the value printed in the',
         'paper. All figures are percentages.', '']
    if recal is not None:
        L += ['Each detector is re-thresholded at **its own published FPR**, so precision',
              'and F1 are compared at equal false-positive cost. AUROC is threshold-free',
              'and therefore unchanged.', '']

    L += ['## Table III — accuracy and attack success rate', '',
          '| | ours | paper |', '|---|---|---|']
    for k, label in [('accuracy', 'Accuracy'), ('asr_grad', 'ASR (gradient)'),
                     ('asr_gan', 'ASR (GAN)')]:
        o = summary.get(k)
        L.append(f'| {label} | {_fmt(o * 100 if o is not None else None)} | {PAPER_ACC[k]:.2f} |')
    L.append('')

    for metric, title in [('TPR', 'Table V — true positive rate'),
                          ('Precision', 'Table VII — precision'),
                          ('F1', 'Table VII — F1'),
                          ('AUROC', 'Table VIII — AUROC')]:
        L += [f'## {title}', '',
              '| detector | ' + ' | '.join(f'{s} ours | {s} paper' for s in SCEN) + ' |',
              '|---' * (1 + 2 * len(SCEN)) + '|']
        for det in ORDER:
            cells = []
            for s in SCEN:
                cells.append(_fmt(ours(det, s, metric)))
                cells.append(_fmt(PAPER[det][s][metric]))
            L.append(f'| {det} | ' + ' | '.join(cells) + ' |')
        L.append('')

    L += ['## False positive rate', '', '| detector | ours | paper |', '|---|---|---|']
    for det in ORDER:
        L.append(f'| {det} | {_fmt(ours_fpr(det))} | {PAPER[det]["FPR"]:.2f} |')
    L += ['',
          'Our FPR column is flat by construction: every detector is calibrated to '
          'the same target on the ID validation split, so the spread the paper shows '
          '(2.02–13.87) reflects a threshold rule it does not state.', '']

    return '\n'.join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='nidiff/work/nidiff_raw')
    ap.add_argument('--match-fpr', action='store_true',
                    help="re-threshold each detector at the FPR the paper reports for "
                         "it, so precision and F1 are compared at equal false-positive "
                         "cost (AUROC is unaffected)")
    args = ap.parse_args()
    run_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    md = build(run_dir, match_fpr=args.match_fpr)
    path = os.path.join(run_dir, 'comparison_matchfpr.md' if args.match_fpr else 'comparison.md')
    with open(path, 'w') as fh:
        fh.write(md)
    print(md)
    print(f'\nwrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

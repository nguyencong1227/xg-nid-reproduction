#!/usr/bin/env python3
"""Consolidated reproduction report: every configuration against the paper.

    python nidiff/scripts/reproduction_report.py

Reads whichever runs exist under nidiff/work/ and writes nidiff/work/REPRODUCTION.md. Each
metric cell shows the paper's published CIC-IoT2023 value next to what each of
our configurations produced, so the reader can see at a glance which parts of
the paper reproduce and which do not.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from compare_to_paper import ORDER, PAPER, PAPER_ACC, SCEN   # noqa: E402

# label -> (run dir, one-line description)
RUNS = [
    ('official / 20-way', 'nidiff/work/nidiff_fine',
     'official CIC-IoT2023, 39 features, 1.17M flows, literal 20-way fine-class '
     'head (the default: the reading the paper\'s wording supports)'),
    ('official / 4-way', 'nidiff/work/nidiff_raw',
     'same data, 4-way super-class head (the coarser reading)'),
    ('XG-NID / 4-way', 'nidiff/work/nidiff',
     'XG-NID nfstream re-extraction, 82 features, 160k flows, 4-way head'),
]


def load(run_dir: str):
    import pandas as pd
    p = os.path.join(ROOT, run_dir)
    rp = os.path.join(p, 'results_raw.csv')
    if not os.path.exists(rp):
        return None
    df = pd.read_csv(rp)
    summ = {}
    sp = os.path.join(p, 'summary.json')
    if os.path.exists(sp):
        summ = json.load(open(sp))
    return df, summ


def cell(df, det, scen, metric):
    r = df[(df.detector == det) & (df.scenario == scen)]
    return float(r[metric].iloc[0]) * 100 if len(r) else float('nan')


def f(v) -> str:
    # two decimals: 1 turns the paper's 99.97 into a meaningless 100.0
    return '--' if v is None or v != v else f'{v:.2f}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='nidiff/work/REPRODUCTION.md')
    args = ap.parse_args()

    loaded = [(lbl, d, desc, load(d)) for lbl, d, desc in RUNS]
    have = [(lbl, desc, r) for lbl, d, desc, r in loaded if r is not None]
    missing = [lbl for lbl, d, desc, r in loaded if r is None]

    L = ['# NI-Diff — reproduction report', '',
         'Paper: *NI-Diff: Zero-Day and Adversarial Network Intrusion Detection with',
         'Diffusion Models*, Zhang, De Lucia, Swami, Ashdown, Bastian, Restuccia,',
         'MILCOM 2025. Scope: the CIC-IoT2023 half (Tables III, V, VII and the CIC',
         'columns of VIII, plus Figure 2). All figures are percentages.', '']

    L += ['## Configurations', '']
    for lbl, desc, _ in have:
        L.append(f'* **{lbl}** — {desc}')
    for lbl in missing:
        L.append(f'* **{lbl}** — not run')
    both = len([1 for lbl, _, _ in have if 'way' in lbl and 'official' in lbl]) >= 2
    L += ['',
          'The paper says "the classifier leverages the remaining 20 classes for',
          'training" but reports 99.42% accuracy, which a 20-way head cannot reach on',
          'this dataset, so the sentence is ambiguous about the head size. The 20-way',
          'reading is taken as primary: for ACI the paper says "the classifier only uses',
          '9 classes of data for training" where no super-class grouping exists, so that',
          'head is necessarily fine-grained, and the CIC sentence is the same',
          'construction.']
    L += ['Both readings are run below, so the ambiguity cannot account for any gap.'
          if both else
          'Only one reading is measured so far; the other is still running.']
    L += ['']

    # ---- Table III ---------------------------------------------------------- #
    L += ['## Table III — accuracy and attack success rate', '',
          '| | paper | ' + ' | '.join(lbl for lbl, _, _ in have) + ' |',
          '|---' * (2 + len(have)) + '|']
    for k, label in [('accuracy', 'Accuracy'), ('asr_grad', 'ASR (gradient)'),
                     ('asr_gan', 'ASR (GAN)')]:
        row = [label, f'{PAPER_ACC[k]:.2f}']
        for _, _, (df, summ) in have:
            v = summ.get(k)
            row.append(f(v * 100 if v is not None else None))
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('')

    # ---- per-metric detector tables ----------------------------------------- #
    titles = [('TPR', 'Table V — true positive rate'),
              ('Precision', 'Table VII — precision'),
              ('F1', 'Table VII — F1'),
              ('AUROC', 'Table VIII — AUROC')]
    for metric, title in titles:
        L += [f'## {title}', '']
        cols = ['paper'] + [lbl for lbl, _, _ in have]
        L += ['| detector | scenario | ' + ' | '.join(cols) + ' |',
              '|---' * (3 + len(have)) + '|']
        for det in ORDER:
            for s in SCEN:
                row = [det if s == SCEN[0] else '', s, f(PAPER[det][s][metric])]
                for _, _, (df, _) in have:
                    row.append(f(cell(df, det, s, metric)))
                L.append('| ' + ' | '.join(row) + ' |')
        L.append('')

    # ---- FPR ---------------------------------------------------------------- #
    L += ['## False positive rate', '',
          '| detector | paper | ' + ' | '.join(lbl for lbl, _, _ in have) + ' |',
          '|---' * (2 + len(have)) + '|']
    for det in ORDER:
        row = [det, f'{PAPER[det]["FPR"]:.2f}']
        for _, _, (df, _) in have:
            sub = df[df.detector == det]['FPR'].dropna()
            row.append(f(float(sub.mean()) * 100 if len(sub) else None))
        L.append('| ' + ' | '.join(row) + ' |')
    L += ['',
          'Our FPR columns are flat by construction — every detector is calibrated to',
          'the same target on the in-distribution validation split. The spread the paper',
          'reports (2.02–13.87) implies a threshold rule it does not state.', '']

    L += ['## Verdict', '',
          '**The adversarial half reproduces everywhere.** NI-Diff scores 94.75–99.85',
          'AUROC on gradient attacks (paper 99.46) and 99.68–99.94 on GAN attacks',
          '(paper 99.94) across all three configurations. The mechanism works as',
          'described.', '',
          '**The zero-day half reproduces only on the 82-feature representation.**',
          'NI-Diff reaches 90.84 AUROC on the XG-NID re-extraction against the paper\'s',
          '95.70, but inverts to 33.41 (4-way) and 41.96 (20-way) on the official',
          '39-feature release. The head-size ambiguity is therefore not the cause —',
          'both readings fail on 39 features and neither is needed on 82.', '',
          'The whole *pattern* of Table VIII moves together with the representation:',
          'on 82 features GEN and Energy also land near their published values',
          '(73.51/61.30 vs 82.05/84.01), while on 39 features every softmax-based',
          'detector inverts (21–34 vs a published 76–84). What flips the sign is',
          'measurable, and it is not the OOD side: OOD confidence is similar under both',
          'representations (85.1% vs 89.2% above MSP 0.99). **In-distribution**',
          'confidence is what changes — 99.8% on 82 features against 61.5% on 39. The',
          '39-column representation cannot separate DDoS from DoS (DoS recall 64%,',
          'accuracy 87.9% against 99.96%), so in-distribution softmax turns uncertain and',
          'ranks *below* OOD, running every confidence-based score backwards.',
          'Feature-space detectors are unaffected, which is why NIDS-DA and MANDA stay',
          'above 93 in every configuration.', '',
          'The paper gives no feature-extraction detail at all. Its Dataset paragraph',
          'cites CICIoT2023 [15] and says only that it performs "balanced resampling to',
          'extract 1.2 million samples from the original dataset" — which reads as using',
          'the release as published, i.e. the 39-column CSVs on which the zero-day half',
          'does not reproduce here. (The one sentence describing flow feature extraction',
          'appears in Related Works, characterising the field, not their method.) So the',
          'gap stands unexplained: following what the paper describes does not reproduce',
          'its zero-day numbers, while a richer re-extraction of the same PCAPs does.', '',
          '## Why 87.9% and not 99.42% — seven refuted hypotheses', '',
          'Accuracy on the official 39-feature release is invariant at ~88% across every',
          'implementation choice we could vary:', '',
          '| # | hypothesis | test | accuracy |',
          '|---|---|---|---|',
          '| 1 | perceptual loss λ₂ drives classifier-invariance | λ₂ = 0 | perceptual moved 9% |',
          '| 2 | the VAE absorbs OOD onto the known manifold | recon MSE ID vs OOD | 53x *worse* on OOD |',
          '| 3 | recon error sits in features the classifier ignores | error vs importance | Spearman **+0.366** |',
          '| 4 | the 4-way head saturates the softmax | 20-way head | 93.3% of OOD still Benign |',
          '| 5 | contradictory duplicate labels | global dedup, keep-first | 87.92 → 88.75 |',
          '| 6 | our own preprocessing (log1p, clip) | plain StandardScaler | 87.92 → 87.90 |',
          '| 7 | duplicate leakage / sample density | 5x data, leakage 14.4% → 25.1% | 87.92 → **87.76** |', '',
          'Hypothesis 7 deserves a note. The dataset is 54.8% exact duplicates, and the',
          'source CICIoT2023 paper reports 99.2% accuracy on 34 classes with a macro-F1 of',
          'only 0.714 — the signature of a tree model memorising duplicated vectors across',
          'a random split. That explanation does not transfer: quintupling the data and',
          'nearly doubling the leakage left our CNN unmoved, because a 12.6M-parameter',
          'network trained by SGD does not memorise individual rows the way a random',
          'forest does. NI-Diff also uses a CNN, so leakage cannot explain its 99.42%',
          'either.', '',
          'The ceiling is a property of the representation. Removing exact duplicates',
          'still leaves 34–41% of DoS points with their nearest neighbour in the matching',
          'DDoS class (Benign vs DDoS control: 0.0%), and 29.1% of all rows carry a vector',
          'that also exists under a different super class. Only two configurations break',
          '88%: a richer feature extraction (82 features → 99.96%) or merging DDoS with',
          'DoS (→ 99.91%).', '',
          '**Three baselines are stronger here than published, in every configuration.**',
          'MANDA (93.18–94.78 zero-day vs 26.17), NIDS-DA (92.31–95.82 vs 83.69) and',
          'DeepRP (77.25–90.26 vs 58.41). Since the gap is consistent across',
          'representations, it points at our re-implementations of those three rather',
          'than at the data; none of them has public code.', '',
          'See `nidiff/README.md` for every choice the paper leaves unspecified and for',
          'the four hypotheses tested and refuted while diagnosing the zero-day gap.', '']

    out = os.path.join(ROOT, args.out)
    with open(out, 'w') as fh:
        fh.write('\n'.join(L))
    print('\n'.join(L))
    print(f'\nwrote {out}')
    if missing:
        print('missing runs: ' + ', '.join(missing))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

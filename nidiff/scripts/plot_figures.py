#!/usr/bin/env python3
"""Figures for the NI-Diff reproduction.

    python nidiff/scripts/plot_figures.py [--out-dir nidiff/work/nidiff]

Writes into the run directory:
  fig2_timesteps.png  -- TPR/FPR as a function of denoising timesteps (paper Fig. 2)
  fig_scores.png      -- distribution of the NI-Diff KLD score per scenario
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Validated categorical slots: worst adjacent CVD dE 9.2, normal-vision 27.6.
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
SURFACE = '#fcfcfb'
INK, INK_2, INK_MUTED = '#0b0b0b', '#52514e', '#8a8a85'
GRID = '#e4e4e0'


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)


def fig_timesteps(run_dir: str) -> str:
    """Paper Fig. 2 — the threshold is held fixed at the one-step setting, so
    TPR and FPR are free to drift up together as the timestep grows."""
    import pandas as pd
    df = pd.read_csv(os.path.join(run_dir, 'ablation_timesteps.csv'))

    # grad and GAN both sit at ~100% for every timestep, so the pair is drawn as a
    # thick rim under a thin line -- the overlap treatment the marks spec prescribes --
    # and their markers are interleaved so neither series disappears.
    series = [
        ('TPR (grad)', ORANGE, '-', 's', 4.6, (0, 2), 3, (11, 1)),
        ('TPR (GAN)', AQUA, '-', '^', 2.0, (1, 2), 4, (11, 14)),
        ('TPR (zero-day)', BLUE, '-', 'o', 2.0, 1, 5, (11, -25)),
        ('FPR', VIOLET, '--', 'D', 2.0, 1, 5, (11, -12)),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.78)
    _style(ax)

    x = df['steps'].to_numpy()
    for col, color, ls, marker, lw, mevery, zo, lbl_off in series:
        y = 100 * df[col].to_numpy()
        ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, marker=marker,
                markevery=mevery, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=col, zorder=zo, clip_on=False)
        # direct label on the last point (also the relief for the low-contrast hue)
        ax.annotate(f'{y[-1]:.0f}%', (x[-1], y[-1]), textcoords='offset points',
                    xytext=lbl_off, color=color, fontsize=9, fontweight='medium',
                    annotation_clip=False)

    ax.axvline(1, color=INK_MUTED, linewidth=1.0, linestyle=':', zorder=1)
    ax.annotate('one-step\n(paper setting)', (1, 33), textcoords='offset points',
                xytext=(9, 0), color=INK_2, fontsize=8.5, va='center')

    ax.set_xscale('log')
    ax.set_xlim(0.9, 1150)
    ax.set_ylim(0, 103)
    ax.set_xticks([1, 5, 10, 25, 50, 100, 200, 400, 600, 1000])
    ax.set_xticklabels(['1', '5', '10', '25', '50', '100', '200', '400', '600', '1000'])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100%'])
    ax.set_xlabel('Denoising timesteps (log scale — samples were taken log-spaced)',
                  color=INK_2, fontsize=9.5)
    ax.set_ylabel('Rate', color=INK_2, fontsize=9.5)
    fig.text(0.055, 0.945, 'One-step denoising is the operating point',
             color=INK, fontsize=13, fontweight='semibold', va='top')
    fig.text(0.055, 0.885, 'Threshold fixed at the one-step setting. Past ~200 steps the '
             'reconstruction is semantically new,\nso the classifier stops separating normal '
             'from anomalous and both rates run to 100%.',
             color=INK_2, fontsize=9, va='top', linespacing=1.5)

    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.20, 0.62), frameon=False,
                    fontsize=9, handlelength=2.6)
    for t in leg.get_texts():
        t.set_color(INK_2)

    out = os.path.join(run_dir, 'fig2_timesteps.png')
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_scores(run_dir: str) -> str:
    """Where the detection threshold sits relative to each score distribution."""
    z = np.load(os.path.join(run_dir, 'scores.npz'))
    import pandas as pd
    res = pd.read_csv(os.path.join(run_dir, 'results_raw.csv'))
    T = float(res.loc[res['detector'] == 'NI-Diff', 'threshold'].iloc[0])

    floor = 1e-12
    ref = np.clip(z['NI-Diff/id'], floor, None)
    scen = [('zero-day', BLUE), ('grad', ORANGE), ('GAN', AQUA)]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.78)
    _style(ax)

    cols = [('id', 'in-distribution', INK_2)] + [(k, k, c) for k, c in scen]
    vals = {k: np.clip(z[f'NI-Diff/{k}'], floor, None) for k, _, _ in cols}
    lo = min(v.min() for v in vals.values())
    hi = max(v.max() for v in vals.values())
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)

    # share of flows per bin -- `density` on log-spaced bins would put a 1e11 spike
    # on the leftmost bin and flatten everything else to zero
    for key, label, color in cols:
        v = vals[key]
        w = np.ones(len(v)) / len(v)
        if key == 'id':
            ax.hist(v, bins=bins, weights=w, histtype='stepfilled',
                    color=INK_MUTED, alpha=0.20, zorder=2)
        ax.hist(v, bins=bins, weights=w, histtype='step', color=color,
                linewidth=2.0 if key != 'id' else 1.6, label=label, zorder=3)

    ax.axvline(T, color=INK, linewidth=1.4, linestyle='--', zorder=5)
    ax.annotate(f'T = {T:.1e}\n(5% FPR)', (T, ax.get_ylim()[1] * 0.97),
                textcoords='offset points', xytext=(8, 0), color=INK,
                fontsize=9, fontweight='medium', va='top')

    ax.set_xscale('log')
    ax.yaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f}%')
    ax.set_xlabel(r'$D_{KL}(y \parallel y^{\prime})$  —  softmax of the flow vs of its '
                  'VAE/diffusion round trip', color=INK_2, fontsize=9.5)
    ax.set_ylabel('Share of flows', color=INK_2, fontsize=9.5)
    fig.text(0.055, 0.945, 'The round trip barely moves in-distribution flows',
             color=INK, fontsize=13, fontweight='semibold', va='top')
    fig.text(0.055, 0.885, 'Adversarial flows snap back to their true class and zero-day '
             'flows decode poorly, so both land\norders of magnitude to the right of the '
             'threshold.', color=INK_2, fontsize=9, va='top', linespacing=1.5)

    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.97), frameon=False,
                    fontsize=9, handlelength=2.4)
    for t in leg.get_texts():
        t.set_color(INK_2)

    out = os.path.join(run_dir, 'fig_scores.png')
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='nidiff/work/nidiff')
    args = ap.parse_args()
    run_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    for fn in (fig_timesteps, fig_scores):
        print('wrote', fn(run_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

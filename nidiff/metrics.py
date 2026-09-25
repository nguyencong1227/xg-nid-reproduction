"""Evaluation metrics used in Tables IV-VIII: TPR, FPR, precision, F1, AUROC.

Positive class = "flagged" (zero-day or adversarial); negative = in-distribution.
A single threshold T is calibrated once on the held-out ID validation split and
then reused for every scenario, exactly as Algorithm 1 prescribes.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def pick_threshold(id_val_scores: np.ndarray, target_fpr: float) -> float:
    """T = the (1 - target_fpr) quantile of the in-distribution scores."""
    return float(np.quantile(np.asarray(id_val_scores, dtype=np.float64), 1.0 - target_fpr))


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Rank-based AUROC; ties get average ranks."""
    neg = np.asarray(neg, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    if len(neg) == 0 or len(pos) == 0:
        return float('nan')
    both = np.concatenate([pos, neg])
    order = both.argsort(kind='mergesort')
    ranks = np.empty(len(both), dtype=np.float64)
    ranks[order] = np.arange(1, len(both) + 1)
    srt = both[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def scenario_metrics(id_scores: np.ndarray, attack_scores: np.ndarray,
                     threshold: float) -> Dict[str, float]:
    """TPR/FPR/precision/F1 at `threshold` plus threshold-free AUROC."""
    id_scores = np.asarray(id_scores, dtype=np.float64)
    attack_scores = np.asarray(attack_scores, dtype=np.float64)
    tp = float((attack_scores > threshold).sum())
    fn = float(len(attack_scores) - tp)
    fp = float((id_scores > threshold).sum())
    tn = float(len(id_scores) - fp)
    tpr = tp / max(tp + fn, 1.0)
    fpr = fp / max(fp + tn, 1.0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0
    return {'TPR': tpr, 'FPR': fpr, 'Precision': prec, 'F1': f1,
            'AUROC': auroc(id_scores, attack_scores),
            'n_id': float(len(id_scores)), 'n_attack': float(len(attack_scores))}


def fpr_at_tpr(id_scores: np.ndarray, attack_scores: np.ndarray, tpr: float = 0.95) -> float:
    t = float(np.quantile(np.asarray(attack_scores, np.float64), 1.0 - tpr))
    return float((np.asarray(id_scores, np.float64) > t).mean())


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def table_tpr_fpr(df, scenarios: Sequence[str],
                  detector_order: Optional[Sequence[str]] = None):
    """Table IV / V: one row per detector, TPR per scenario plus a single FPR."""
    import pandas as pd
    rows = []
    dets = detector_order or list(dict.fromkeys(df['detector']))
    for d in dets:
        sub = df[df['detector'] == d]
        row = {'detector': d}
        for s in scenarios:
            v = sub.loc[sub['scenario'] == s, 'TPR']
            row[f'TPR ({s})'] = float(v.iloc[0]) if len(v) else np.nan
        fprs = sub['FPR'].dropna()
        row['FPR'] = float(fprs.mean()) if len(fprs) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('detector')


def table_metric(df, metrics: Sequence[str], scenarios: Sequence[str],
                 detector_order: Optional[Sequence[str]] = None):
    """Tables VI-VIII: a (scenario x metric) grid, one row per detector."""
    import pandas as pd
    rows = []
    dets = detector_order or list(dict.fromkeys(df['detector']))
    for d in dets:
        sub = df[df['detector'] == d]
        row = {'detector': d}
        for s in scenarios:
            for m in metrics:
                v = sub.loc[sub['scenario'] == s, m]
                row[f'{m} ({s})'] = float(v.iloc[0]) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('detector')


def to_markdown(table, pct: bool = True, digits: int = 2) -> str:
    def f(v):
        if v != v:
            return '--'
        return f'{100 * v:.{digits}f}%' if pct else f'{v:.{digits}f}'
    return table.map(f).to_markdown()

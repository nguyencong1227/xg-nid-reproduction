#!/usr/bin/env python3
"""Two experiments chained, for an overnight run.

A — preprocessing: drop the log1p and clip that we added ourselves and use a
    plain StandardScaler, which is what the CICIoT2023 source paper describes.
    Same 45k/class cache, so only the scaling changes.

B — scale and leakage: sample 250k per fine class instead of 45k. The corpus is
    54.8% exact duplicates, so a denser sample puts far more identical vectors on
    both sides of a random split. If the published 99.4% figures come from
    duplicate memorisation on the full data, accuracy should climb here.

Each stage reports accuracy, the 4x4 confusion matrix, and the measured
train/test duplicate leakage.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from nidiff import data as D                       # noqa: E402
from nidiff.config import Config                   # noqa: E402
from nidiff.models import build_classifier         # noqa: E402
from nidiff.pipeline import classifier_report      # noqa: E402
from nidiff.train import set_seed, train_classifier  # noqa: E402


def leakage(npz_path: str) -> float:
    z = np.load(npz_path, allow_pickle=True)
    h_tr = pd.util.hash_pandas_object(pd.DataFrame(z['Xtr']), index=False).to_numpy()
    h_te = pd.util.hash_pandas_object(pd.DataFrame(z['Xte']), index=False).to_numpy()
    return float(np.isin(h_te, np.unique(h_tr)).mean())


def confusion(cfg, clf, data) -> None:
    pred = []
    with torch.no_grad():
        for i in range(0, len(data.X_test_id), 4096):
            xb = torch.from_numpy(data.X_test_id[i:i + 4096]).to(cfg.device)
            pred.append(clf(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    y = data.y_test_id
    n = data.n_classes
    cm = np.zeros((n, n), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    print('%-8s' % '', ' '.join('%8s' % c[:8] for c in data.known_names), flush=True)
    for i, c in enumerate(data.known_names):
        print('%-8s' % c[:8],
              ' '.join('%8.2f' % (100 * v / max(cm[i].sum(), 1)) for v in cm[i]), flush=True)


def stage(tag: str, cfg: Config, npz: str) -> None:
    print('\n' + '=' * 74, flush=True)
    print(tag, flush=True)
    print('=' * 74, flush=True)
    print('  cache      : %s' % npz, flush=True)
    print('  log1p=%s  clip=%s  epochs=%d' % (cfg.data.log1p, cfg.data.clip, cfg.clf.epochs),
          flush=True)
    print('  leakage (test rows with an exact twin in train): %.2f%%' % (100 * leakage(npz)),
          flush=True)
    set_seed(cfg.seed)
    data = D.build_dataset(cfg)
    print('  ' + data.summary().replace('\n', '\n  '), flush=True)
    t0 = time.time()
    clf = train_classifier(cfg, data, force=True)
    acc = classifier_report(cfg, clf, data)['accuracy']
    print('  ACCURACY %.4f   (%.1f min)' % (acc, (time.time() - t0) / 60), flush=True)
    print(flush=True)
    confusion(cfg, clf, data)


def main() -> int:
    # ---- A: plain StandardScaler ------------------------------------------ #
    npz_a = '/home/congnc/data/ciciot2023_45k.npz'
    cfg = Config()
    cfg.out_dir = 'nidiff/work/nidiff_noscale'
    cfg.data.raw_cache_npz = npz_a
    cfg.data.log1p = False
    cfg.data.clip = 0.0
    stage('A — plain StandardScaler (no log1p, no clip), 45k/class', cfg, npz_a)

    # ---- B: dense sample, more duplicate leakage --------------------------- #
    npz_b = '/home/congnc/data/ciciot2023_250k.npz'
    if not os.path.exists(npz_b):
        print('\nbuilding dense cache (250k per fine class) ...', flush=True)
        from nidiff.data_cic_raw import build_cache
        build_cache('/home/congnc/data/CICIoT2023/CSV', npz_b,
                    per_class=250000, seed=42, test_frac=0.2, dedup=False)

    cfg = Config()                       # back to the default preprocessing
    cfg.out_dir = 'nidiff/work/nidiff_bigsample'
    cfg.data.raw_cache_npz = npz_b
    cfg.clf.epochs = 25                  # it plateaus by ~15 at every scale so far
    stage('B — 250k/class, default preprocessing, no dedup', cfg, npz_b)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

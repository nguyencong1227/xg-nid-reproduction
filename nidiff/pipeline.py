"""End-to-end reproduction of the CIC-IoT2023 half of NI-Diff.

Order of operations mirrors the paper:
  1. train the multi-class DNN classifier on the *known* classes
  2. train the VAE (Eq. 2) with the frozen classifier as perceptual network
  3. train the latent diffusion model (Eq. 5)
  4. craft gradient-based and GAN-based adversarial intrusions  -> Table III
  5. score every detector on {zero-day, gradient, GAN} vs in-distribution
       -> Tables V (TPR/FPR), VII (precision/F1), VIII (AUROC)
  6. sweep the number of denoising timesteps                    -> Figure 2
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import data as data_mod
from . import metrics as M
from .attacks import attack_pool, gan_attack, gradient_attack
from .detectors import (BASELINE_ORDER, NIDiffDetector, build_baselines)
from .train import (_log, encode_latents, evaluate_accuracy, set_seed,
                    train_classifier, train_diffusion, train_vae)

SCENARIOS = ('zero-day', 'grad', 'GAN')


def classifier_report(cfg, clf, data) -> Dict[str, float]:
    """Accuracy on in-distribution test flows -- the `Acc` column of Table III."""
    dl = data_mod.loaders(data.X_test_id, data.y_test_id, cfg.eval_batch_size, False, cfg.device)
    acc = evaluate_accuracy(clf, dl, cfg.device)
    _log(f'classifier in-distribution test accuracy: {acc:.4%}')
    return {'accuracy': acc}


def craft_attacks(cfg, clf, data) -> Dict[str, Dict]:
    """Table III: attack success rate of both adversarial attacks."""
    X_pool, y_pool = attack_pool(data, cfg.attack.max_attack_samples, cfg.seed)
    _log(f'attack pool: {len(X_pool):,} known-intrusion test flows '
         f'({int(data.mutable_mask().sum())} mutable features)')

    _log('crafting gradient-based adversarial flows (constrained PGD -> benign)')
    Xg, yg, okg, asr_g = gradient_attack(cfg, clf, data, X_pool, y_pool)
    _log(f'  ASR (grad) = {asr_g:.4%}')

    _log('crafting GAN-based adversarial flows')
    Xa, ya, oka, asr_gan = gan_attack(cfg, clf, data, X_pool, y_pool)
    _log(f'  ASR (GAN)  = {asr_gan:.4%}')

    return {
        'grad': {'X': Xg, 'y': yg, 'ok': okg, 'asr': asr_g},
        'GAN': {'X': Xa, 'y': ya, 'ok': oka, 'asr': asr_gan},
    }


def _positives(cfg, data, atk: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """The positive (should-be-flagged) set of each scenario."""
    pos = {'zero-day': data.X_test_ood}
    for k in ('grad', 'GAN'):
        X, ok = atk[k]['X'], atk[k]['ok']
        pos[k] = X[ok] if cfg.detect.adv_successful_only else X
    return pos


def evaluate_detectors(cfg, clf, vae, dm, data, atk: Dict[str, Dict]):
    """Score every detector on every scenario with one shared threshold."""
    import pandas as pd

    pos = _positives(cfg, data, atk)
    for k, v in pos.items():
        _log(f'scenario {k:9s}: {len(v):,} positives')

    detectors = build_baselines(cfg, clf, data)
    detectors.append(NIDiffDetector(cfg, clf, vae, dm))

    rows, raw_scores = [], {}
    for det in detectors:
        t0 = time.time()
        s_val = det.score(data.X_val)
        T = M.pick_threshold(s_val, cfg.detect.target_fpr)
        s_id = det.score(data.X_test_id)
        raw_scores[f'{det.name}/id'] = s_id
        for name in SCENARIOS:
            s_pos = det.score(pos[name])
            raw_scores[f'{det.name}/{name}'] = s_pos
            m = M.scenario_metrics(s_id, s_pos, T)
            m['FPR95'] = M.fpr_at_tpr(s_id, s_pos, 0.95)
            rows.append({'detector': det.name, 'scenario': name, 'threshold': T, **m})
        _log(f'{det.name:9s} done in {time.time() - t0:.1f}s  '
             f'(T={T:.4g}, FPR={rows[-1]["FPR"]:.4%})')

    df = pd.DataFrame(rows)
    np.savez_compressed(cfg.out('scores.npz'), **raw_scores)
    return df, detectors


def timestep_ablation(cfg, clf, vae, dm, data, atk: Dict[str, Dict],
                      steps: Optional[Sequence[int]] = None):
    """Figure 2: TPR/FPR of NI-Diff as a function of denoising timesteps.

    The threshold is calibrated *once*, at the paper's one-step setting, and then
    held fixed -- that is what makes TPR and FPR both drift upwards together as
    the timestep grows, the effect Fig. 2 reports.  Recalibrating per timestep
    would pin FPR at the target and hide it.
    """
    import pandas as pd

    pos = _positives(cfg, data, atk)
    ref = NIDiffDetector(cfg, clf, vae, dm, denoise_steps=cfg.diff.denoise_steps)
    T = M.pick_threshold(ref.score(data.X_val), cfg.detect.target_fpr)
    _log(f'ablation: fixed threshold T={T:.4g} from t={cfg.diff.denoise_steps}')

    rows = []
    for s in (steps or cfg.detect.ablation_steps):
        det = NIDiffDetector(cfg, clf, vae, dm, denoise_steps=s)
        s_id = det.score(data.X_test_id)
        row = {'steps': s, 'FPR': float((s_id > T).mean())}
        for name in SCENARIOS:
            row[f'TPR ({name})'] = float((det.score(pos[name]) > T).mean())
        rows.append(row)
        _log(f'ablation t={s:4d}  FPR {row["FPR"]:.4%}  '
             + '  '.join(f'{k} {v:.4%}' for k, v in row.items() if k.startswith('TPR')))
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out('ablation_timesteps.csv'), index=False)
    return df


def save_tables(cfg, df, clf_acc: float, atk: Dict[str, Dict]) -> str:
    """Write the paper-shaped tables as CSV and one markdown report."""
    order = BASELINE_ORDER
    df.to_csv(cfg.out('results_raw.csv'), index=False)

    t_tpr = M.table_tpr_fpr(df, SCENARIOS, order)
    t_pf1 = M.table_metric(df, ['Precision', 'F1'], SCENARIOS, order)
    t_auc = M.table_metric(df, ['AUROC'], SCENARIOS, order)
    for name, t in [('table_v_tpr_fpr', t_tpr), ('table_vii_precision_f1', t_pf1),
                    ('table_viii_auroc', t_auc)]:
        t.to_csv(cfg.out(f'{name}.csv'))

    lines = [
        '# NI-Diff on CIC-IoT2023 -- reproduction',
        '',
        f'known classes: {cfg.data.class_names}',
        f'zero-day (held out): {cfg.data.zero_day_classes}',
        f'classifier: NI-Diff-{cfg.clf.arch}   threshold: '
        f'{cfg.detect.target_fpr:.0%} nominal FPR on the ID validation split',
        f'adversarial flows scored: '
        f'{"successful only" if cfg.detect.adv_successful_only else "all"}',
        '',
        '## Table III -- accuracy and attack success rate',
        '',
        '| | Acc | ASR (Grad) | ASR (GAN) |',
        '|---|---|---|---|',
        f'| CICIoT | {clf_acc:.2%} | {atk["grad"]["asr"]:.2%} | {atk["GAN"]["asr"]:.2%} |',
        '',
        '## Table V -- true and false positive rate',
        '',
        M.to_markdown(t_tpr),
        '',
        '## Table VII -- precision and F1 score',
        '',
        M.to_markdown(t_pf1),
        '',
        '## Table VIII -- AUROC',
        '',
        M.to_markdown(t_auc),
        '',
    ]
    path = cfg.out('report.md')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    _log(f'report written to {path}')
    return path


# --------------------------------------------------------------------------- #
def run(cfg, force: bool = False, do_ablation: bool = True):
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        cfg.device = 'cpu'
        _log('CUDA unavailable, falling back to CPU')
    cfg.save()

    data = data_mod.build_dataset(cfg)
    _log(f'data [{cfg.data.dataset}]\n' + data.summary())

    clf = train_classifier(cfg, data, force)
    acc = classifier_report(cfg, clf, data)['accuracy']
    vae = train_vae(cfg, data, clf, force)
    dm = train_diffusion(cfg, data, vae, force)

    atk = craft_attacks(cfg, clf, data)
    df, _ = evaluate_detectors(cfg, clf, vae, dm, data, atk)
    save_tables(cfg, df, acc, atk)

    abl = timestep_ablation(cfg, clf, vae, dm, data, atk) if do_ablation else None

    with open(cfg.out('summary.json'), 'w') as fh:
        json.dump({'accuracy': acc,
                   'asr_grad': atk['grad']['asr'],
                   'asr_gan': atk['GAN']['asr'],
                   'latent': f'1x{vae.latent_len}'}, fh, indent=2)
    return {'data': data, 'classifier': clf, 'vae': vae, 'diffusion': dm,
            'attacks': atk, 'results': df, 'ablation': abl}

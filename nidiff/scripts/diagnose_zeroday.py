#!/usr/bin/env python3
"""Why does NI-Diff's zero-day AUROC invert on the official CIC release?

Three cheap experiments that reuse the trained checkpoints:

  1. Is the VAE absorbing OOD?  Reconstruction error on ID vs OOD, and the AUROC
     of reconstruction error used directly as a detector. If OOD reconstructs as
     well as ID, the latent path has erased the novelty and no softmax-based
     read-out can recover it.
  2. Is `clip` erasing it first?  Share of clipped cells and rows, ID vs OOD.
  3. Is it the read-out?  Re-score with each KLD direction x {sample z, use mu}.

    python nidiff/scripts/diagnose_zeroday.py [--out-dir nidiff/work/nidiff_raw]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from nidiff import data as D                          # noqa: E402
from nidiff.config import Config                      # noqa: E402
from nidiff.detectors import NIDiffDetector           # noqa: E402
from nidiff.diffusion import GaussianDiffusion        # noqa: E402
from nidiff.metrics import auroc                      # noqa: E402
from nidiff.models import DenoiseUNet1d, FlowVAE, build_classifier   # noqa: E402


def load_all(cfg, data):
    dev = cfg.device
    clf = build_classifier(cfg.clf.arch, data.n_classes, data.n_features).to(dev)
    clf.load_state_dict(torch.load(cfg.out('classifier.pt'), map_location=dev))
    vae = FlowVAE(data.n_features, cfg.vae.channels, cfg.vae.n_layers, cfg.vae.n_pool).to(dev)
    vae.load_state_dict(torch.load(cfg.out('vae.pt'), map_location=dev))
    unet = DenoiseUNet1d(in_channels=vae.latent_channels).to(dev)
    dm = GaussianDiffusion(unet, cfg.diff.timesteps, cfg.diff.beta_start, cfg.diff.beta_end).to(dev)
    dm.model.load_state_dict(torch.load(cfg.out('diffusion.pt'), map_location=dev))
    return clf.eval(), vae.eval(), dm.eval()


@torch.no_grad()
def _batched(fn, X, dev, bs=4096):
    out = []
    for i in range(0, len(X), bs):
        out.append(fn(torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(dev)))
    return np.concatenate(out)


def exp1_vae_absorption(cfg, data, vae, dm):
    print('=' * 74)
    print('1. Is the VAE absorbing OOD?   (latent = 1x%d for %d features)'
          % (vae.latent_len, data.n_features))
    print('=' * 74)
    dev = cfg.device

    def recon_mse(xb):
        r, _, _, _ = vae(xb, sample=False)
        return ((r - xb) ** 2).mean(1).cpu().numpy()

    def roundtrip_mse(xb):
        mu, logvar = vae.encode(xb)
        z2 = dm.denoise_once(mu, step=cfg.diff.denoise_steps)
        return ((vae.decode(z2) - xb) ** 2).mean(1).cpu().numpy()

    for label, fn in [('VAE only', recon_mse), ('VAE + 1-step diffusion', roundtrip_mse)]:
        m_id = _batched(fn, data.X_test_id, dev)
        m_ood = _batched(fn, data.X_test_ood, dev)
        print(f'\n  {label}')
        print('    ID  : median %.5f  mean %.5f  p90 %.5f' %
              (np.median(m_id), m_id.mean(), np.percentile(m_id, 90)))
        print('    OOD : median %.5f  mean %.5f  p90 %.5f' %
              (np.median(m_ood), m_ood.mean(), np.percentile(m_ood, 90)))
        print('    ratio of medians OOD/ID: %.2fx' % (np.median(m_ood) / max(np.median(m_id), 1e-12)))
        print('    AUROC using this error alone as the detector: %.2f%%'
              % (100 * auroc(m_id, m_ood)))


def exp2_clipping(cfg, data):
    print()
    print('=' * 74)
    print('2. Is `clip=%s` erasing the signal before the VAE sees it?' % cfg.data.clip)
    print('=' * 74)
    thr = cfg.data.clip - 1e-3 if cfg.data.clip else None
    if thr is None:
        print('  clip disabled')
        return
    for label, X in [('ID  test', data.X_test_id), ('OOD test', data.X_test_ood)]:
        at = np.abs(X) >= thr
        print('  %s : %.4f%% of cells clipped, %.2f%% of rows have >=1 clipped cell'
              % (label, 100 * at.mean(), 100 * at.any(1).mean()))
    # which features clip most on OOD
    at = np.abs(data.X_test_ood) >= thr
    rate = at.mean(0)
    order = np.argsort(-rate)[:6]
    print('  top clipped features on OOD:')
    for i in order:
        if rate[i] > 0:
            print('    %-16s %.2f%% (ID %.2f%%)'
                  % (data.feat_names[i], 100 * rate[i],
                     100 * (np.abs(data.X_test_id[:, i]) >= thr).mean()))


def exp3_readout(cfg, data, clf, vae, dm):
    print()
    print('=' * 74)
    print('3. Is it the read-out?  KLD direction x z-sampling')
    print('=' * 74)
    print('  %-11s %-8s %10s %10s %10s' % ('direction', 'z', 'AUROC-0day', 'AUROC-grad', 'AUROC-GAN'))
    scores = np.load(os.path.join(cfg.path(cfg.out_dir), 'scores.npz'))
    adv = {k: scores[f'NI-Diff/{k}'] for k in ('grad', 'GAN')}
    # the adversarial sets are not stored, so reuse the saved scores' sample count
    # only for reference; recompute 0-day and ID exactly.
    for direction in ('forward', 'reverse', 'symmetric'):
        for use_mu in (False, True):
            cfg.detect.kld_direction = direction
            cfg.detect.use_mu = use_mu
            det = NIDiffDetector(cfg, clf, vae, dm)
            s_id = det.score(data.X_test_id)
            s_ood = det.score(data.X_test_ood)
            print('  %-11s %-8s %9.2f%% %10s %10s'
                  % (direction, 'mu' if use_mu else 'sample',
                     100 * auroc(s_id, s_ood), '-', '-'))
    print('\n  (adversarial columns need the stored adversarial flows, which the'
          '\n   pipeline does not persist; re-run with --keep-adv to fill them in)')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='nidiff/work/nidiff_raw')
    args = ap.parse_args()

    cfg = Config()
    cfg.out_dir = args.out_dir
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        cfg.device = 'cpu'
    data = D.build_dataset(cfg)
    clf, vae, dm = load_all(cfg, data)

    exp1_vae_absorption(cfg, data, vae, dm)
    exp2_clipping(cfg, data)
    exp3_readout(cfg, data, clf, vae, dm)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

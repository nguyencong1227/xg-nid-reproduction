"""Detectors: NI-Diff itself plus the six baselines of Tables IV-VIII.

Every detector exposes `score(X) -> np.ndarray` where a *higher* score means
"more likely to be a zero-day / adversarial flow".  Thresholds are calibrated
once on the held-out in-distribution validation split
(`nidiff.metrics.pick_threshold`).

The three OOD baselines (GEN, Energy, GradNorm) are exact re-implementations of
their papers.  The three adversarial baselines (MANDA, DeepRP, NIDS-DA) are
re-implementations of the *mechanism* each paper describes, adapted to a 1D flow
classifier -- the original code is not public, so treat them as faithful in
spirit rather than line-for-line.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FlowData
from .diffusion import GaussianDiffusion
from .models import FlowVAE


class Detector:
    name = 'detector'

    def fit(self, data: FlowData) -> 'Detector':
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _batched(fn, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(device)
        out.append(np.asarray(fn(xb), dtype=np.float64).reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


# --------------------------------------------------------------------------- #
# NI-Diff (Algorithm 1)
# --------------------------------------------------------------------------- #
class NIDiffDetector(Detector):
    """KLD between the softmax score of x and of its VAE/diffusion round trip."""

    name = 'NI-Diff'

    def __init__(self, cfg, classifier: nn.Module, vae: FlowVAE, dm: GaussianDiffusion,
                 denoise_steps: Optional[int] = None):
        self.cfg = cfg
        self.f = classifier.eval()
        self.vae = vae.eval()
        self.dm = dm.eval()
        self.steps = cfg.diff.denoise_steps if denoise_steps is None else denoise_steps
        self.direction = cfg.detect.kld_direction
        self.use_mu = cfg.detect.use_mu

    @torch.no_grad()
    def _score_batch(self, xb: torch.Tensor) -> np.ndarray:
        y = F.softmax(self.f(xb), dim=1)                       # y = f(x)
        mu, logvar = self.vae.encode(xb)                       # E(x)
        z = self.vae.reparameterize(mu, logvar, sample=not self.use_mu)
        z2 = self.dm.denoise_once(z, step=self.steps)          # z' = DM(E(x))
        y2 = F.softmax(self.f(self.vae.decode(z2)), dim=1)      # y' = f(D(z'))
        eps = 1e-12
        ly, ly2 = (y + eps).log(), (y2 + eps).log()
        if self.direction == 'reverse':
            kld = (y2 * (ly2 - ly)).sum(1)
        elif self.direction == 'symmetric':
            kld = 0.5 * ((y * (ly - ly2)).sum(1) + (y2 * (ly2 - ly)).sum(1))
        else:
            kld = (y * (ly - ly2)).sum(1)
        return kld.cpu().numpy()

    def score(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        return _batched(self._score_batch, X, batch_size or self.cfg.eval_batch_size, self.cfg.device)


# --------------------------------------------------------------------------- #
# OOD baselines
# --------------------------------------------------------------------------- #
class GENDetector(Detector):
    """GEN, Liu et al. CVPR'23: generalised entropy of the top-M softmax mass."""

    name = 'GEN'

    def __init__(self, cfg, classifier: nn.Module, gamma: float = 0.1,
                 top_m: Optional[int] = None):
        self.cfg, self.f, self.gamma, self.top_m = cfg, classifier.eval(), gamma, top_m

    @torch.no_grad()
    def _score_batch(self, xb: torch.Tensor) -> np.ndarray:
        p = F.softmax(self.f(xb), dim=1)
        m = min(self.top_m or p.shape[1], p.shape[1])
        p = p.topk(m, dim=1).values
        return (p.pow(self.gamma) * (1.0 - p).pow(self.gamma)).sum(1).cpu().numpy()

    def score(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        return _batched(self._score_batch, X, batch_size or self.cfg.eval_batch_size, self.cfg.device)


class EnergyDetector(Detector):
    """Energy, Liu et al. NeurIPS'20: E(x) = -T * logsumexp(logits / T)."""

    name = 'Energy'

    def __init__(self, cfg, classifier: nn.Module, temperature: float = 1.0):
        self.cfg, self.f, self.T = cfg, classifier.eval(), temperature

    @torch.no_grad()
    def _score_batch(self, xb: torch.Tensor) -> np.ndarray:
        return (-self.T * torch.logsumexp(self.f(xb) / self.T, dim=1)).cpu().numpy()

    def score(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        return _batched(self._score_batch, X, batch_size or self.cfg.eval_batch_size, self.cfg.device)


class GradNormDetector(Detector):
    """GradNorm, Huang et al. NeurIPS'21: L1 norm of the gradient of
    KLD(uniform || softmax) w.r.t. the classifier head.

    For a linear head the per-sample gradient is (p - u) (x) f, so its L1 norm
    factorises exactly as ||p - u||_1 * ||f||_1 / T -- no per-sample backward
    pass needed.
    """

    name = 'GradNorm'

    def __init__(self, cfg, classifier: nn.Module, temperature: float = 1.0):
        self.cfg, self.f, self.T = cfg, classifier.eval(), temperature

    @torch.no_grad()
    def _score_batch(self, xb: torch.Tensor) -> np.ndarray:
        feats = self.f.features(xb)
        p = F.softmax(self.f.head(feats) / self.T, dim=1)
        u = 1.0 / p.shape[1]
        row = (p - u).abs().sum(1) / self.T
        col = feats.abs().sum(1)
        # GradNorm flags OOD by a *small* gradient norm -> negate
        return (-(row * col)).cpu().numpy()

    def score(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        return _batched(self._score_batch, X, batch_size or self.cfg.eval_batch_size, self.cfg.device)


# --------------------------------------------------------------------------- #
# Adversarial-detection baselines
# --------------------------------------------------------------------------- #
class MANDADetector(Detector):
    """MANDA, Wang et al. TDSC'22.

    Two signals, summed after standardisation:
      (i)  manifold inconsistency -- mean kNN distance, in the classifier's
           embedding space, to the training flows of the class the classifier
           predicted.  An adversarial flow is an outlier of the manifold it gets
           mapped onto.
      (ii) decision-boundary instability -- how often the prediction flips under
           small Gaussian perturbations of the input.
    """

    name = 'MANDA'

    def __init__(self, cfg, classifier: nn.Module, k: int = 20, n_probe: int = 8,
                 sigma: float = 0.1, fit_cap: int = 20000):
        self.cfg, self.f = cfg, classifier.eval()
        self.k, self.n_probe, self.sigma, self.fit_cap = k, n_probe, sigma, fit_cap

    @torch.no_grad()
    def _embed(self, X: np.ndarray, batch_size: int = 1024) -> torch.Tensor:
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(self.cfg.device)
            out.append(self.f.features(xb))
        return torch.cat(out)

    @torch.no_grad()
    def _predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(self.cfg.device)
            out.append(self.f(xb).argmax(1).cpu().numpy())
        return np.concatenate(out)

    def fit(self, data: FlowData) -> 'MANDADetector':
        rng = np.random.default_rng(self.cfg.seed)
        X, y = data.X_train, data.y_train
        if len(X) > self.fit_cap:
            idx = rng.choice(len(X), self.fit_cap, replace=False)
            X, y = X[idx], y[idx]
        E = self._embed(X)
        # per-class embedding banks, kept on the GPU for a chunked exact kNN
        self.bank_ = {int(c): E[torch.from_numpy(y == c).to(E.device)] for c in np.unique(y)}
        d = self._manifold_dist(E, self._predict(X))
        self._d_mu, self._d_sd = float(d.mean()), float(d.std() + 1e-8)
        return self

    @torch.no_grad()
    def _manifold_dist(self, E: torch.Tensor, pred: np.ndarray,
                       chunk: int = 512) -> np.ndarray:
        """Mean distance to the k nearest training flows of the predicted class."""
        d = torch.zeros(len(E), device=E.device)
        pred_t = torch.from_numpy(pred).to(E.device)
        for c, bank in self.bank_.items():
            sel = torch.nonzero(pred_t == c, as_tuple=True)[0]
            if not len(sel) or not len(bank):
                continue
            k = max(1, min(self.k, len(bank)))
            for i in range(0, len(sel), chunk):
                q = E[sel[i:i + chunk]]
                dist = torch.cdist(q, bank)
                d[sel[i:i + chunk]] = dist.topk(k, dim=1, largest=False).values.mean(1)
        return d.cpu().numpy().astype(np.float64)

    def _instability(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        dev = self.cfg.device
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(dev)
                base = self.f(xb).argmax(1)
                flips = torch.zeros(len(xb), device=dev)
                for _ in range(self.n_probe):
                    flips += (self.f(xb + self.sigma * torch.randn_like(xb)).argmax(1) != base).float()
                out.append((flips / self.n_probe).cpu().numpy())
        return np.concatenate(out)

    def score(self, X: np.ndarray) -> np.ndarray:
        pred = self._predict(X)
        d = (self._manifold_dist(self._embed(X), pred) - self._d_mu) / self._d_sd
        return d + self._instability(X)


class DeepRPDetector(Detector):
    """DeepRP, Drenkow et al. WACV'22 -- random subspace analysis.

    The classifier embedding is projected onto `n_proj` random low-dimensional
    subspaces.  Per subspace we measure the Mahalanobis deviation from the
    in-distribution statistics; the score is the mean deviation across
    subspaces, which is the magnification effect the paper relies on.
    """

    name = 'DeepRP'

    def __init__(self, cfg, classifier: nn.Module, n_proj: int = 32,
                 proj_dim: int = 16, fit_cap: int = 40000):
        self.cfg, self.f = cfg, classifier.eval()
        self.n_proj, self.proj_dim, self.fit_cap = n_proj, proj_dim, fit_cap

    @torch.no_grad()
    def _embed(self, X: np.ndarray, batch_size: int = 1024) -> torch.Tensor:
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(self.cfg.device)
            out.append(self.f.features(xb))
        return torch.cat(out)

    def fit(self, data: FlowData) -> 'DeepRPDetector':
        dev = self.cfg.device
        rng = np.random.default_rng(self.cfg.seed)
        X = data.X_train
        if len(X) > self.fit_cap:
            X = X[rng.choice(len(X), self.fit_cap, replace=False)]
        E = self._embed(X)
        g = torch.Generator().manual_seed(self.cfg.seed)
        self.P_ = (torch.randn(self.n_proj, E.shape[1], self.proj_dim, generator=g)
                   / np.sqrt(self.proj_dim)).to(dev)
        Z = torch.einsum('nd,pdk->pnk', E, self.P_)             # (n_proj, N, proj_dim)
        self.mu_ = Z.mean(dim=1, keepdim=True)
        prec = []
        eye = torch.eye(self.proj_dim, device=dev, dtype=torch.float64)
        for p in range(self.n_proj):
            Zc = (Z[p] - self.mu_[p]).double()
            cov = Zc.T @ Zc / max(len(Zc) - 1, 1) + 1e-4 * eye
            prec.append(torch.linalg.inv(cov))
        self.prec_ = torch.stack(prec).float()
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        batch_size = batch_size or self.cfg.eval_batch_size
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(self.cfg.device)
            Z = torch.einsum('nd,pdk->pnk', self.f.features(xb), self.P_) - self.mu_
            m = torch.einsum('pnk,pkl,pnl->pn', Z, self.prec_, Z)
            out.append(m.mean(0).cpu().numpy())
        return np.concatenate(out)


class NIDSDADetector(Detector):
    """NIDS-DA, Kumar et al. ESWA'25: a deep auto-encoder trained on known
    traffic; the reconstruction error is the detection score."""

    name = 'NIDS-DA'

    def __init__(self, cfg, hidden=(64, 32, 16), epochs: int = 50,
                 lr: float = 1e-3, batch_size: int = 1024):
        self.cfg, self.hidden = cfg, hidden
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size

    def _build(self, n: int) -> nn.Module:
        h1, h2, h3 = self.hidden
        return nn.Sequential(
            nn.Linear(n, h1), nn.ReLU(True), nn.Linear(h1, h2), nn.ReLU(True),
            nn.Linear(h2, h3), nn.ReLU(True),
            nn.Linear(h3, h2), nn.ReLU(True), nn.Linear(h2, h1), nn.ReLU(True),
            nn.Linear(h1, n))

    def fit(self, data: FlowData) -> 'NIDSDADetector':
        import os
        dev = self.cfg.device
        self.ae_ = self._build(data.n_features).to(dev)
        ckpt = self.cfg.out('nidsda_ae.pt')
        if os.path.exists(ckpt):
            self.ae_.load_state_dict(torch.load(ckpt, map_location=dev))
            return self
        from torch.utils.data import DataLoader, TensorDataset
        dl = DataLoader(TensorDataset(torch.from_numpy(data.X_train)),
                        batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.ae_.parameters(), lr=self.lr)
        for ep in range(1, self.epochs + 1):
            self.ae_.train()
            tot = n = 0
            for (xb,) in dl:
                xb = xb.to(dev)
                loss = F.mse_loss(self.ae_(xb), xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(xb)
                n += len(xb)
            if ep % 25 == 0 or ep == self.epochs:
                print(f'  nids-da epoch {ep:3d}/{self.epochs}  mse {tot / n:.5f}', flush=True)
        torch.save(self.ae_.state_dict(), ckpt)
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        batch_size = batch_size or self.cfg.eval_batch_size
        self.ae_.eval()
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(self.cfg.device)
            out.append(((self.ae_(xb) - xb) ** 2).mean(1).cpu().numpy())
        return np.concatenate(out)


BASELINE_ORDER = ['GEN', 'Energy', 'GradNorm', 'MANDA', 'DeepRP', 'NIDS-DA', 'NI-Diff']


def build_baselines(cfg, classifier: nn.Module, data: FlowData):
    """The six baselines of Tables IV-VIII, fitted on known training flows."""
    dets = [
        GENDetector(cfg, classifier),
        EnergyDetector(cfg, classifier),
        GradNormDetector(cfg, classifier),
        MANDADetector(cfg, classifier),
        DeepRPDetector(cfg, classifier),
        NIDSDADetector(cfg),
    ]
    for d in dets:
        print(f'  fitting {d.name} ...', flush=True)
        d.fit(data)
    return dets

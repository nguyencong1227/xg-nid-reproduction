"""Adversarial network intrusions used to *evaluate* the detector (Sec. IV).

Two targeted attacks that try to make a known intrusion be classified as
benign.  Both respect the paper's realism constraint: symbolic features (flag
counts, packet/byte counters, protocol one-hots, rolling counters) are frozen
and only statistical features (durations, packet-size and inter-arrival-time
statistics) may move.  On top of that every candidate is projected back onto
the feasible region in *raw* feature space -- non-negativity, and
min <= mean <= max inside each statistics group.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FlowData


class DomainProjector:
    """Projection onto the realistic-flow constraint set."""

    def __init__(self, data: FlowData, device: str):
        self.scaler = data.scaler.to_torch(device)
        self.mask = torch.from_numpy(data.mutable_mask()).to(device)
        self.triples = data.ordered_triples_idx()
        self.products = data.product_constraints_idx()
        self.nonneg = torch.from_numpy(np.asarray(data.scaler.log_mask)).to(device)

    def __call__(self, x_std: torch.Tensor) -> torch.Tensor:
        raw = self.scaler.t_inverse(x_std)
        raw = torch.where(self.nonneg, raw.clamp(min=0.0), raw)
        if self.triples or self.products:
            # rebuild column-by-column rather than assigning into `raw`: writing
            # into a tensor whose own columns are the operands bumps its autograd
            # version and breaks the GAN generator's backward pass
            cols = list(raw.unbind(dim=1))
            for lo, mid, hi in self.triples:
                tri, _ = torch.sort(torch.stack([cols[lo], cols[mid], cols[hi]], dim=1), dim=1)
                cols[lo], cols[mid], cols[hi] = tri[:, 0], tri[:, 1], tri[:, 2]
            # a summed quantity stays equal to count x mean, so the total follows
            # whatever the attacker did to the mean
            for tot, cnt, mean in self.products:
                cols[tot] = cols[cnt] * cols[mean]
            raw = torch.stack(cols, dim=1)
        return self.scaler.t_forward(raw)


def attack_pool(data: FlowData, cap: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Known-class *intrusion* test flows -- the samples an attacker perturbs."""
    mask = data.y_test_id != data.benign_idx
    X, y = data.X_test_id[mask], data.y_test_id[mask]
    if cap and len(X) > cap:
        idx = np.random.default_rng(seed).choice(len(X), cap, replace=False)
        X, y = X[idx], y[idx]
    return X, y


# --------------------------------------------------------------------------- #
def gradient_attack(cfg, classifier: nn.Module, data: FlowData,
                    X: Optional[np.ndarray] = None,
                    y: Optional[np.ndarray] = None,
                    batch_size: Optional[int] = None):
    """Constrained targeted PGD towards the benign class (FENCE-style).

    Returns (X_adv, y_true, success_mask, ASR).
    """
    dev = cfg.device
    ac = cfg.attack
    batch_size = batch_size or ac.batch_size
    if X is None:
        X, y = attack_pool(data, ac.max_attack_samples, cfg.seed)
    proj = DomainProjector(data, dev)
    mask = proj.mask.float().unsqueeze(0)
    target = data.benign_idx
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    out, flags = [], []
    for i in range(0, len(X), batch_size):
        x0 = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(dev)
        tgt = torch.full((len(x0),), target, device=dev, dtype=torch.long)
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(ac.grad_steps):
            loss = F.cross_entropy(classifier(x0 + delta * mask), tgt)
            grad, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                d = (delta - ac.grad_alpha * grad.sign()) * mask
                d = d.clamp(-ac.grad_eps, ac.grad_eps)
                xa = proj(x0 + d)
                d = ((xa - x0) * mask).clamp(-ac.grad_eps, ac.grad_eps)
            delta = d.detach().requires_grad_(True)
        with torch.no_grad():
            xa = proj(x0 + delta * mask)
            flags.append((classifier(xa).argmax(1) == target).cpu().numpy())
            out.append(xa.cpu().numpy())
    X_adv = np.concatenate(out)
    ok = np.concatenate(flags)
    return X_adv, y, ok, float(ok.mean())


# --------------------------------------------------------------------------- #
class PerturbGenerator(nn.Module):
    """G(x, z) -> bounded perturbation on the mutable features."""

    def __init__(self, n_features: int, noise_dim: int, hidden: int, scale: float):
        super().__init__()
        self.noise_dim = noise_dim
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(n_features + noise_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(True),
            nn.Linear(hidden, n_features), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.randn(x.shape[0], self.noise_dim, device=x.device)
        return self.net(torch.cat([x, z], dim=1)) * self.scale


class FlowDiscriminator(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def gan_attack(cfg, classifier: nn.Module, data: FlowData,
               X: Optional[np.ndarray] = None,
               y: Optional[np.ndarray] = None):
    """Generative-model-based evasion (Alhajjar et al.): the generator is
    rewarded both for fooling the classifier and for producing flows a
    discriminator cannot separate from real benign traffic.

    Returns (X_adv, y_true, success_mask, ASR).
    """
    dev = cfg.device
    ac = cfg.attack
    if X is None:
        X, y = attack_pool(data, ac.max_attack_samples, cfg.seed)
    proj = DomainProjector(data, dev)
    mask = proj.mask.float().unsqueeze(0)
    target = data.benign_idx
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    xb_real = torch.from_numpy(data.X_train[data.y_train == data.benign_idx]).to(dev)
    x_att = torch.from_numpy(np.ascontiguousarray(X)).to(dev)

    G = PerturbGenerator(data.n_features, ac.gan_noise_dim, ac.gan_hidden,
                         ac.gan_pert_scale).to(dev)
    D = FlowDiscriminator(data.n_features, ac.gan_hidden).to(dev)
    og = torch.optim.Adam(G.parameters(), lr=ac.gan_lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=ac.gan_lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(cfg.seed)

    n_batches = max(1, len(x_att) // ac.gan_batch_size)
    for ep in range(1, ac.gan_epochs + 1):
        perm = torch.randperm(len(x_att), generator=gen)
        d_tot = g_tot = 0.0
        for b in range(n_batches):
            idx = perm[b * ac.gan_batch_size:(b + 1) * ac.gan_batch_size].to(dev)
            xa0 = x_att[idx]
            xr = xb_real[torch.randint(0, len(xb_real), (len(xa0),), device=dev)]

            with torch.no_grad():
                fake = proj(xa0 + G(xa0) * mask)
            d_loss = bce(D(xr), torch.ones(len(xr), device=dev)) \
                + bce(D(fake), torch.zeros(len(fake), device=dev))
            od.zero_grad(set_to_none=True)
            d_loss.backward()
            od.step()

            fake = proj(xa0 + G(xa0) * mask)
            tgt = torch.full((len(fake),), target, device=dev, dtype=torch.long)
            g_loss = ac.gan_lambda_adv * F.cross_entropy(classifier(fake), tgt) \
                + ac.gan_lambda_real * bce(D(fake), torch.ones(len(fake), device=dev))
            og.zero_grad(set_to_none=True)
            g_loss.backward()
            og.step()
            d_tot += float(d_loss)
            g_tot += float(g_loss)
        if ep % 50 == 0 or ep == ac.gan_epochs:
            G.eval()
            with torch.no_grad():
                probe = x_att[:ac.batch_size]
                asr = float((classifier(proj(probe + G(probe) * mask)).argmax(1)
                             == target).float().mean())
            G.train()
            print(f'  gan epoch {ep:4d}/{ac.gan_epochs}  D {d_tot / n_batches:.3f}'
                  f'  G {g_tot / n_batches:.3f}  ASR {asr:.4f}', flush=True)

    G.eval()
    outs, flags = [], []
    with torch.no_grad():
        for i in range(0, len(x_att), ac.batch_size):
            xa0 = x_att[i:i + ac.batch_size]
            xa = proj(xa0 + G(xa0) * mask)
            flags.append((classifier(xa).argmax(1) == target).cpu().numpy())
            outs.append(xa.cpu().numpy())
    X_adv = np.concatenate(outs)
    ok = np.concatenate(flags)
    return X_adv, y, ok, float(ok.mean())

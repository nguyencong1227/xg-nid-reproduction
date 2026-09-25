"""Training recipes (Sec. IV).

classifier : Adam, lr 1e-4, batch 1024, 50 epochs
VAE        : Adam, lr 1e-4, batch 1024, 100 epochs, loss of Eq. 2
diffusion  : Adam, lr 1e-5, batch 1024, 1000 epochs, EMA
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FlowData, loaders
from .diffusion import GaussianDiffusion
from .models import EMA, DenoiseUNet1d, FlowVAE, build_classifier


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --------------------------------------------------------------------------- #
def train_classifier(cfg, data: FlowData, force: bool = False) -> nn.Module:
    dev = cfg.device
    ckpt = cfg.out('classifier.pt')
    model = build_classifier(cfg.clf.arch, data.n_classes, data.n_features).to(dev)
    if os.path.exists(ckpt) and not force:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        _log(f'classifier loaded from {ckpt}')
        return model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=cfg.clf.lr)
    tl = loaders(data.X_train, data.y_train, cfg.clf.batch_size, True, dev)
    vl = loaders(data.X_val, data.y_val, cfg.clf.batch_size, False, dev)
    best = -1.0
    for ep in range(1, cfg.clf.epochs + 1):
        model.train()
        tot = n = 0
        for xb, yb in tl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        acc = evaluate_accuracy(model, vl, dev)
        if acc > best:
            best, _ = acc, torch.save(model.state_dict(), ckpt)
        if ep % 5 == 0 or ep == 1 or ep == cfg.clf.epochs:
            _log(f'clf epoch {ep:3d}/{cfg.clf.epochs}  loss {tot / n:.4f}  val-acc {acc:.4f}')
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    _log(f'classifier best val-acc {best:.4f}')
    return model.eval()


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader, dev: str) -> float:
    model.eval()
    ok = n = 0
    for xb, yb in loader:
        pred = model(xb.to(dev)).argmax(1).cpu()
        ok += (pred == yb).sum().item()
        n += len(yb)
    return ok / max(n, 1)


# --------------------------------------------------------------------------- #
def vae_loss(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, classifier: nn.Module,
             lam_kld: float, lam_perc: float) -> Dict[str, torch.Tensor]:
    """Eq. 2: MSE reconstruction + lambda1 * KLD + lambda2 * perceptual loss.

    The perceptual network is the frozen DNN classifier -- its penultimate
    embedding is what "semantic properties of the network traffic" means here.
    """
    mse = F.mse_loss(recon, x)
    kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2))).mean()
    with torch.no_grad():
        f_real = classifier.features(x)
    perc = F.mse_loss(classifier.features(recon), f_real)
    return {'loss': mse + lam_kld * kld + lam_perc * perc,
            'mse': mse.detach(), 'kld': kld.detach(), 'perc': perc.detach()}


def train_vae(cfg, data: FlowData, classifier: nn.Module, force: bool = False) -> FlowVAE:
    dev = cfg.device
    ckpt = cfg.out('vae.pt')
    vae = FlowVAE(data.n_features, cfg.vae.channels, cfg.vae.n_layers, cfg.vae.n_pool).to(dev)
    if os.path.exists(ckpt) and not force:
        vae.load_state_dict(torch.load(ckpt, map_location=dev))
        _log(f'VAE loaded from {ckpt}  (latent 1x{vae.latent_len})')
        return vae.eval()

    for p in classifier.parameters():
        p.requires_grad_(False)
    classifier.eval()

    opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae.lr)
    tl = loaders(data.X_train, None, cfg.vae.batch_size, True, dev)
    best = float('inf')
    for ep in range(1, cfg.vae.epochs + 1):
        vae.train()
        agg = {'loss': 0.0, 'mse': 0.0, 'kld': 0.0, 'perc': 0.0}
        n = 0
        for (xb,) in tl:
            xb = xb.to(dev, non_blocking=True)
            recon, mu, logvar, _ = vae(xb)
            out = vae_loss(xb, recon, mu, logvar, classifier,
                           cfg.vae.lambda_kld, cfg.vae.lambda_perc)
            opt.zero_grad(set_to_none=True)
            out['loss'].backward()
            opt.step()
            for k in agg:
                agg[k] += float(out[k]) * len(xb)
            n += len(xb)
        for k in agg:
            agg[k] /= n
        if agg['loss'] < best:
            best, _ = agg['loss'], torch.save(vae.state_dict(), ckpt)
        if ep % 10 == 0 or ep == 1 or ep == cfg.vae.epochs:
            _log(f'vae epoch {ep:3d}/{cfg.vae.epochs}  loss {agg["loss"]:.5f} '
                 f'(mse {agg["mse"]:.5f}  kld {agg["kld"]:.2f}  perc {agg["perc"]:.5f})')
    vae.load_state_dict(torch.load(ckpt, map_location=dev))
    _log(f'VAE best loss {best:.5f}  latent 1x{vae.latent_len}')
    return vae.eval()


# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_latents(cfg, vae: FlowVAE, X: np.ndarray, batch_size: Optional[int] = None,
                   sample: bool = True) -> torch.Tensor:
    dev = cfg.device
    batch_size = batch_size or cfg.eval_batch_size
    vae.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(dev)
        mu, logvar = vae.encode(xb)
        out.append(vae.reparameterize(mu, logvar, sample).cpu())
    return torch.cat(out)


def train_diffusion(cfg, data: FlowData, vae: FlowVAE, force: bool = False) -> GaussianDiffusion:
    dev = cfg.device
    ckpt = cfg.out('diffusion.pt')
    unet = DenoiseUNet1d(in_channels=vae.latent_channels).to(dev)
    dm = GaussianDiffusion(unet, cfg.diff.timesteps, cfg.diff.beta_start, cfg.diff.beta_end).to(dev)
    if os.path.exists(ckpt) and not force:
        dm.model.load_state_dict(torch.load(ckpt, map_location=dev))
        _log(f'diffusion loaded from {ckpt}')
        return dm.eval()

    Z = encode_latents(cfg, vae, data.X_train)
    _log(f'diffusion training set: latents {tuple(Z.shape)}')
    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(TensorDataset(Z), batch_size=cfg.diff.batch_size, shuffle=True)
    opt = torch.optim.Adam(dm.model.parameters(), lr=cfg.diff.lr)
    ema = EMA(dm.model, cfg.diff.ema_decay)
    best = float('inf')
    for ep in range(1, cfg.diff.epochs + 1):
        dm.model.train()
        tot = n = 0
        for (zb,) in dl:
            zb = zb.to(dev, non_blocking=True)
            loss = dm.loss(zb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(dm.model)
            tot += loss.item() * len(zb)
            n += len(zb)
        cur = tot / n
        if cur < best:
            best = cur
            torch.save(ema.shadow, ckpt)
        if ep % 50 == 0 or ep == 1 or ep == cfg.diff.epochs:
            _log(f'diff epoch {ep:4d}/{cfg.diff.epochs}  eps-mse {cur:.5f}')
    dm.model.load_state_dict(torch.load(ckpt, map_location=dev))
    _log(f'diffusion best eps-mse {best:.5f} (EMA weights)')
    return dm.eval()

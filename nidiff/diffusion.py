"""DDPM over the VAE latent space (Eqs. 3-5).

Linear beta schedule from 1e-4 to 0.02 with a maximum timestep of 1000, and the
*one-step* diffusion-denoising used at detection time (Sec. III, following
Carlini et al.), which is what makes NI-Diff stable -- see Fig. 2 of the paper.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_bar', ab.float())
        self.register_buffer('sqrt_ab', ab.sqrt().float())
        self.register_buffer('sqrt_1mab', (1.0 - ab).sqrt().float())

    # ---- Eq. 3: forward (diffusion) process ------------------------------- #
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0) if noise is None else noise
        shape = (-1,) + (1,) * (x0.dim() - 1)
        a = self.sqrt_ab[t].view(shape)
        b = self.sqrt_1mab[t].view(shape)
        return a * x0 + b * noise, noise

    # ---- Eq. 5: training objective ---------------------------------------- #
    def loss(self, x0: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if t is None:
            t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        return F.mse_loss(self.model(xt, t), noise)

    # ---- detection-time reconstruction ------------------------------------ #
    @torch.no_grad()
    def denoise_once(self, x0: torch.Tensor, step: int = 1) -> torch.Tensor:
        """Noise x0 up to timestep `step` and recover it in a single shot.

        `step=1` is the paper's setting.  Larger values reproduce the sweep of
        Fig. 2 (detection degrades as the sample becomes semantically new).
        """
        t = torch.full((x0.shape[0],), step - 1, device=x0.device, dtype=torch.long)
        xt, _ = self.q_sample(x0, t)
        shape = (-1,) + (1,) * (x0.dim() - 1)
        eps = self.model(xt, t)
        return (xt - self.sqrt_1mab[t].view(shape) * eps) / self.sqrt_ab[t].view(shape)

    @torch.no_grad()
    def ddpm_sample_from(self, x0: torch.Tensor, step: int) -> torch.Tensor:
        """Full ancestral sampling back from timestep `step` (ablation only)."""
        shape = (-1,) + (1,) * (x0.dim() - 1)
        t0 = torch.full((x0.shape[0],), step - 1, device=x0.device, dtype=torch.long)
        x, _ = self.q_sample(x0, t0)
        for i in range(step - 1, -1, -1):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            eps = self.model(x, t)
            ab = self.alphas_bar[t].view(shape)
            ab_prev = self.alphas_bar[t - 1].view(shape) if i > 0 else torch.ones_like(ab)
            beta = self.betas[t].view(shape)
            alpha = 1.0 - beta
            x0_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            mean = (ab_prev.sqrt() * beta / (1 - ab)) * x0_hat \
                + (alpha.sqrt() * (1 - ab_prev) / (1 - ab)) * x
            if i > 0:
                var = beta * (1 - ab_prev) / (1 - ab)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x

"""Networks of NI-Diff: the DNN classifier (Table I), the VAE and the
denoising U-Net (Table II).

Conventions follow the paper: the classifier and the VAE use batch
normalisation + ReLU, the denoising U-Net uses group normalisation + SiLU.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c) if c % min(8, c) == 0 else 1, c)


class SelfAttention1d(nn.Module):
    """Single-head self-attention over the feature axis of a 1D signal."""

    def __init__(self, channels: int, norm: str = 'bn'):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels) if norm == 'bn' else _gn(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        attn = torch.softmax(torch.einsum('bcl,bcm->blm', q, k) * self.scale, dim=-1)
        out = torch.einsum('blm,bcm->bcl', attn, v)
        return x + self.proj(out)


class ResBlockBN(nn.Module):
    """Residual block used inside the DNN classifier (BN + ReLU)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.skip(x))


# --------------------------------------------------------------------------- #
# Table I -- DNN classifiers
# --------------------------------------------------------------------------- #
class NIDiffBase(nn.Module):
    """5 conv layers, each followed by BN, ReLU and maxpooling; Linear 64 x n."""

    CHANNELS = (16, 16, 32, 32, 64)

    def __init__(self, n_classes: int, in_len: int = 82):
        super().__init__()
        blocks: List[nn.Module] = []
        cin = 1
        for c in self.CHANNELS:
            blocks += [nn.Conv1d(cin, c, 3, padding=1), nn.BatchNorm1d(c),
                       nn.ReLU(inplace=True), nn.MaxPool1d(2)]
            cin = c
        self.body = nn.Sequential(*blocks)
        self.embed_dim = cin
        self.head = nn.Linear(cin, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.body(x).mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class NIDiffLarge(nn.Module):
    """Residual + self-attention stack (Res 64..1024); Linear 1024 x n."""

    CHANNELS = (64, 128, 256, 512, 1024)

    def __init__(self, n_classes: int, in_len: int = 82):
        super().__init__()
        stages: List[nn.Module] = []
        cin = 1
        for c in self.CHANNELS:
            stages += [ResBlockBN(cin, c), SelfAttention1d(c, norm='bn'), nn.MaxPool1d(2)]
            cin = c
        self.body = nn.Sequential(*stages)
        self.embed_dim = cin
        self.head = nn.Linear(cin, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.body(x).mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_classifier(arch: str, n_classes: int, in_len: int) -> nn.Module:
    if arch == 'base':
        return NIDiffBase(n_classes, in_len)
    if arch == 'large':
        return NIDiffLarge(n_classes, in_len)
    raise ValueError(f'unknown classifier arch: {arch}')


# --------------------------------------------------------------------------- #
# Variational auto-encoder (Sec. III)
#   6 x 1x3 conv per side, 64 channels, pool/upsample after every two layers
#   except the last pair; encoder ends in 2 channels (reparameterisation),
#   decoder ends in 1 channel.
# --------------------------------------------------------------------------- #
class FlowVAE(nn.Module):
    """`n_pool` controls how tight the bottleneck is.

    The paper pools "after every two convolutional layers ... except for the last
    layer pairs", i.e. `n_pool = n_layers // 2 - 1` (the default), which on 39
    features gives a 1x9 latent. Lowering it widens the latent without changing
    the layer count, which is how the bottleneck's contribution is tested.
    """

    def __init__(self, in_len: int = 82, channels: int = 64, n_layers: int = 6,
                 n_pool: Optional[int] = None):
        super().__init__()
        assert n_layers % 2 == 0, 'layers come in pairs'
        self.in_len = in_len
        self.n_pairs = n_layers // 2
        self.n_pool = self.n_pairs - 1 if n_pool is None else int(n_pool)
        assert 0 <= self.n_pool <= self.n_pairs - 1, 'n_pool out of range'

        # length trace, so the decoder can undo the pooling exactly
        self.lengths: List[int] = [in_len]
        l = in_len
        for _ in range(self.n_pool):
            l = l // 2
            self.lengths.append(l)
        self.latent_len = l
        self.latent_channels = 1

        enc, cin = [], 1
        for p in range(self.n_pairs):
            last_pair = (p == self.n_pairs - 1)
            for j in range(2):
                cout = 2 if (last_pair and j == 1) else channels
                enc.append(nn.Conv1d(cin, cout, 3, padding=1))
                if not (last_pair and j == 1):
                    enc += [nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
                cin = cout
            if p < self.n_pool:
                enc.append(nn.MaxPool1d(2))
        self.encoder = nn.Sequential(*enc)

        dec, cin = [], 1
        for p in range(self.n_pairs):
            last_pair = (p == self.n_pairs - 1)
            for j in range(2):
                cout = 1 if (last_pair and j == 1) else channels
                dec.append(nn.Conv1d(cin, cout, 3, padding=1))
                if not (last_pair and j == 1):
                    dec += [nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
                cin = cout
            # mirror the encoder: the first n_pool pairs upsample back up
            dec.append(_Upsample(self.lengths[self.n_pool - 1 - p])
                       if p < self.n_pool else nn.Identity())
        self.decoder = nn.Sequential(*dec)

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar.clamp(-20.0, 10.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        if out.shape[-1] != self.in_len:
            out = F.interpolate(out, size=self.in_len, mode='linear', align_corners=False)
        return out.squeeze(1)

    def forward(self, x: torch.Tensor, sample: bool = True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sample)
        return self.decode(z), mu, logvar, z


class _Upsample(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.size, mode='nearest')


# --------------------------------------------------------------------------- #
# Table II -- denoising U-Net (GroupNorm + SiLU)
# --------------------------------------------------------------------------- #
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


class ResBlockGN(nn.Module):
    """Residual block of the U-Net: GroupNorm + SiLU, timestep conditioning."""

    def __init__(self, cin: int, cout: int, t_dim: int):
        super().__init__()
        self.n1, self.c1 = _gn(cin), nn.Conv1d(cin, cout, 3, padding=1)
        self.t = nn.Linear(t_dim, cout)
        self.n2, self.c2 = _gn(cout), nn.Conv1d(cout, cout, 3, padding=1)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb)).unsqueeze(-1)
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class DenoiseUNet1d(nn.Module):
    """Down: Conv 1x3 32 / Res 32 / Res 64 -- Mid: Res 128 / SelfAttn 128 /
    Res 128 -- Up: Res 64 / Res 32 / Conv 1x3 1."""

    def __init__(self, in_channels: int = 1, base: int = 32, t_dim: int = 128):
        super().__init__()
        self.temb = TimeEmbedding(t_dim)
        c1, c2, c3 = base, base * 2, base * 4          # 32, 64, 128
        self.conv_in = nn.Conv1d(in_channels, c1, 3, padding=1)
        self.d1 = ResBlockGN(c1, c1, t_dim)
        self.d2 = ResBlockGN(c1, c2, t_dim)
        self.m1 = ResBlockGN(c2, c3, t_dim)
        self.attn = SelfAttention1d(c3, norm='gn')
        self.m2 = ResBlockGN(c3, c3, t_dim)
        self.u1 = ResBlockGN(c3 + c2, c2, t_dim)
        self.u2 = ResBlockGN(c2 + c1, c1, t_dim)
        self.out = nn.Sequential(_gn(c1), nn.SiLU(), nn.Conv1d(c1, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb(t)
        h0 = self.conv_in(x)
        h1 = self.d1(h0, temb)                                  # skip @ L
        h = F.avg_pool1d(h1, 2, ceil_mode=True)
        h2 = self.d2(h, temb)                                   # skip @ L/2
        h = F.avg_pool1d(h2, 2, ceil_mode=True)
        h = self.m2(self.attn(self.m1(h, temb)), temb)
        h = F.interpolate(h, size=h2.shape[-1], mode='nearest')
        h = self.u1(torch.cat([h, h2], dim=1), temb)
        h = F.interpolate(h, size=h1.shape[-1], mode='nearest')
        h = self.u2(torch.cat([h, h1], dim=1), temb)
        return self.out(h)


class EMA:
    """Exponential moving weight average over the diffusion model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

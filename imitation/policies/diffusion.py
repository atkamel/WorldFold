"""Diffusion Policy (Chi et al. 2023), state-input variant.

A 1D temporal conv U-Net denoises the K x A action chunk, conditioned (FiLM) on
the encoded obs history and the diffusion timestep. Trained as epsilon-prediction
with a squared-cosine DDPM schedule; sampled with deterministic DDIM in far fewer
steps, which is what makes batched GPU inference cheap enough for closed loop.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from imitation.policies.common import ChunkPolicy
from imitation.spec import ACTION_DIM, OBS_DIM


def _cosine_alphas_cumprod(T, s=0.008):
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


class _SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        x = t.float()[:, None] * freq[None]
        return torch.cat([x.sin(), x.cos()], -1)


class _ResBlock(nn.Module):
    """Conv -> GroupNorm -> Mish, twice, with a FiLM (scale, shift) from the condition."""

    def __init__(self, cin, cout, cond_dim, groups=8):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2), nn.GroupNorm(groups, cout), nn.Mish())
        self.c2 = nn.Sequential(nn.Conv1d(cout, cout, 5, padding=2), nn.GroupNorm(groups, cout), nn.Mish())
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(x)
        scale, shift = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h = self.c2(h * (1 + scale) + shift)
        return h + self.skip(x)


class _UNet1D(nn.Module):
    def __init__(self, action_dim, cond_dim, channels=(128, 256)):
        super().__init__()
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        cin = action_dim
        for c in channels:
            self.down.append(nn.ModuleList([_ResBlock(cin, c, cond_dim), _ResBlock(c, c, cond_dim),
                                            nn.Conv1d(c, c, 3, stride=2, padding=1)]))
            cin = c
        self.mid = nn.ModuleList([_ResBlock(cin, cin, cond_dim), _ResBlock(cin, cin, cond_dim)])
        for c in reversed(channels):
            self.up.append(nn.ModuleList([nn.ConvTranspose1d(cin, cin, 4, stride=2, padding=1),
                                          _ResBlock(cin + c, c, cond_dim), _ResBlock(c, c, cond_dim)]))
            cin = c
        self.final = nn.Sequential(nn.Conv1d(cin, cin, 5, padding=2), nn.Mish(), nn.Conv1d(cin, action_dim, 1))

    def forward(self, x, cond):           # x [B, A, K]
        skips = []
        for r1, r2, down in self.down:
            x = r2(r1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for r in self.mid:
            x = r(x, cond)
        for up, r1, r2 in self.up:
            x = up(x)
            x = r2(r1(torch.cat([x, skips.pop()], 1), cond), cond)
        return self.final(x)


class DiffusionPolicy(ChunkPolicy):
    kind = "diffusion"

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, obs_horizon=2, chunk=16, cond_dim=256,
                 channels=(128, 256), train_steps=100, infer_steps=10):
        super().__init__(obs_dim, action_dim, obs_horizon, chunk)
        assert chunk % (2 ** len(channels)) == 0, "chunk must divide by the U-Net's downsampling"
        self.cond_dim, self.channels = cond_dim, tuple(channels)
        self.train_steps, self.infer_steps = train_steps, infer_steps
        self.obs_enc = nn.Sequential(nn.Linear(obs_horizon * obs_dim, 512), nn.Mish(), nn.Linear(512, cond_dim))
        self.t_enc = nn.Sequential(_SinEmb(128), nn.Linear(128, cond_dim), nn.Mish(), nn.Linear(cond_dim, cond_dim))
        self.net = _UNet1D(action_dim, cond_dim, self.channels)
        self.register_buffer("alphas_cumprod", _cosine_alphas_cumprod(train_steps))

    def config(self):
        return super().config() | {"cond_dim": self.cond_dim, "channels": list(self.channels),
                                   "train_steps": self.train_steps, "infer_steps": self.infer_steps}

    def _eps(self, noisy, t, obs_cond):
        cond = obs_cond + self.t_enc(t)
        return self.net(noisy.transpose(1, 2), cond).transpose(1, 2)

    def _encode(self, obs):
        return self.obs_enc(self.normalize_obs(obs).flatten(1))

    def compute_loss(self, obs, actions, mask):
        B = actions.shape[0]
        t = torch.randint(0, self.train_steps, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        a = self.alphas_cumprod[t].view(B, 1, 1)
        noisy = a.sqrt() * actions + (1 - a).sqrt() * noise
        err = F.mse_loss(self._eps(noisy, t, self._encode(obs)), noise, reduction="none").mean(-1)
        return (err * mask).sum() / mask.sum().clamp(min=1.0)

    def sample(self, obs, steps=None):
        steps = steps or self.infer_steps
        B = obs.shape[0]
        cond = self._encode(obs)
        # One fixed-seed initial noise shared by every row: DDIM is then a deterministic
        # function of the observation. Fresh global-RNG noise made a row's action depend on
        # its batch position and call order, i.e. on worker timing (id_hard 160 -> 157/200).
        g = torch.Generator(device=obs.device).manual_seed(0)
        x = torch.randn((1, self.chunk, self.action_dim), generator=g, device=obs.device).expand(B, -1, -1).clone()
        ts = torch.linspace(self.train_steps - 1, 0, steps, device=obs.device).round().long()
        for i, t in enumerate(ts):
            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < steps else torch.ones((), device=obs.device)
            eps = self._eps(x, t.expand(B), cond)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1.0, 1.0)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()      # re-derive eps from the clipped x0
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x

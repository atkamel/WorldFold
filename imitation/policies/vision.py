"""Sensor-only chunk policy (Phase 4): camera images + sensor-available proprio -> chunk.

Inputs are exactly what a real robot has: the `main` camera and both wrist cameras
(current frame, uint8 [3, H, W]) and the 48 sensor-available proprio dims (history of
`obs_horizon`). Privileged dims of the 139-D observation are zeroed by `obs_mask`, so
the policy consumes the same obs vector as every other policy and the rollout plumbing
is unchanged. Each camera gets its own small CNN; features are fused with the proprio
embedding by the chunk-MLP trunk and trained with the same masked L1 loss.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from imitation.policies.chunk_mlp import _Block
from imitation.policies.common import ChunkPolicy
from imitation.spec import ACTION_DIM, OBS_DIM, OBS_SUBSETS

DEFAULT_CAMERAS = (("main", 128), ("left_wrist_cam", 64), ("right_wrist_cam", 64))   # M4.1: >= 128^2


class _Encoder(nn.Module):
    def __init__(self, out=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(3))
        self.proj = nn.Linear(128 * 9, out)

    def forward(self, x):
        return self.proj(self.conv(x).flatten(1))


def random_shift(x, pad=4):
    """DrQ-style augmentation: replicate-pad then random crop back, per sample."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad,) * 4, mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None])            # [n, h]
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None])            # [n, w]
    idx = torch.arange(n, device=x.device)[:, None, None]
    return x.permute(0, 2, 3, 1)[idx, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)


class VisionChunkPolicy(ChunkPolicy):
    kind = "vision"
    needs_images = True

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, obs_horizon=2, chunk=16, width=512, depth=3,
                 dropout=0.1, cameras=DEFAULT_CAMERAS, feat=256):
        super().__init__(obs_dim, action_dim, obs_horizon, chunk)
        self.width, self.depth, self.dropout, self.feat = width, depth, dropout, feat
        self.cameras = [tuple(c) for c in cameras]
        self.encoders = nn.ModuleDict({name: _Encoder(feat) for name, _ in self.cameras})
        self.proprio = nn.Linear(obs_horizon * obs_dim, feat)
        self.inp = nn.Linear(feat * (1 + len(self.cameras)), width)
        self.blocks = nn.Sequential(*[_Block(width, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, chunk * action_dim))
        self.set_obs_subset(OBS_SUBSETS["proprio"])
        # per-modality normalization (M4.1): per camera, per channel; the defaults (0.5, 1)
        # reproduce the fixed /255 - 0.5 of checkpoints trained before these buffers existed
        for name, _ in self.cameras:
            self.register_buffer(f"img_mean_{name}", torch.full((3,), 0.5))
            self.register_buffer(f"img_std_{name}", torch.ones(3))

    def set_image_normalizer(self, stats):
        """stats: {camera: (mean[3], std[3])} over [0, 1] pixel values of the training frames."""
        for name, (mean, std) in stats.items():
            getattr(self, f"img_mean_{name}").copy_(torch.as_tensor(mean))
            getattr(self, f"img_std_{name}").copy_(torch.as_tensor(std))

    def normalize_image(self, name, img):
        mean, std = getattr(self, f"img_mean_{name}"), getattr(self, f"img_std_{name}")
        return (img.float() / 255.0 - mean[:, None, None]) / std[:, None, None]

    def config(self):
        return super().config() | {"width": self.width, "depth": self.depth, "dropout": self.dropout,
                                   "cameras": self.cameras, "feat": self.feat}

    def forward(self, obs, images, augment=False):
        feats = [self.proprio(self.normalize_obs(obs).flatten(1))]
        for name, _ in self.cameras:
            x = self.normalize_image(name, images[name])
            if augment:
                x = random_shift(x)
            feats.append(self.encoders[name](x))
        y = self.out(self.blocks(self.inp(torch.cat(feats, -1))))
        return torch.tanh(y).view(-1, self.chunk, self.action_dim)

    def compute_loss(self, obs, actions, mask, images=None):
        err = (self(obs, images, augment=self.training) - actions).abs().mean(-1)
        return (err * mask).sum() / mask.sum().clamp(min=1.0)

    def sample(self, obs, images=None):
        return self(obs, images)

    @torch.no_grad()
    def predict(self, obs_hist: np.ndarray, images=None) -> np.ndarray:
        """images: list (one per batch row) of {camera: uint8 [3, H, W]}."""
        was_training = self.training
        self.eval()
        obs = torch.as_tensor(obs_hist, dtype=torch.float32, device=self.device)
        imgs = {name: torch.as_tensor(np.stack([im[name] for im in images]), device=self.device)
                for name, _ in self.cameras}
        out = self.sample(obs, imgs).clamp(-1.0, 1.0).float().cpu().numpy()
        self.train(was_training)
        return out

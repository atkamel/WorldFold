"""Regression action-chunking baseline (ACT-lite for state inputs).

Flattened normalized obs history -> residual MLP -> K x A actions in [-1, 1],
trained with masked L1 (ACT's loss choice: robust to the bimodal gripper dims).
"""

from __future__ import annotations

import torch
from torch import nn

from imitation.policies.common import ChunkPolicy
from imitation.spec import ACTION_DIM, OBS_DIM


class _Block(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * width, width))

    def forward(self, x):
        return x + self.net(x)


class ChunkMLP(ChunkPolicy):
    kind = "chunk_mlp"

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, obs_horizon=2, chunk=16, width=512, depth=4, dropout=0.1):
        super().__init__(obs_dim, action_dim, obs_horizon, chunk)
        self.width, self.depth, self.dropout = width, depth, dropout
        self.inp = nn.Linear(obs_horizon * obs_dim, width)
        self.blocks = nn.Sequential(*[_Block(width, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, chunk * action_dim))

    def config(self):
        return super().config() | {"width": self.width, "depth": self.depth, "dropout": self.dropout}

    def forward(self, obs):
        x = self.normalize_obs(obs).flatten(1)
        y = self.out(self.blocks(self.inp(x)))
        return torch.tanh(y).view(-1, self.chunk, self.action_dim)

    def compute_loss(self, obs, actions, mask):
        err = (self(obs) - actions).abs().mean(-1)            # [B, K]
        return (err * mask).sum() / mask.sum().clamp(min=1.0)

    def sample(self, obs):
        return self(obs)

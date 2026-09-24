"""Shared policy interface: obs history [B, H, D] -> action chunk [B, K, A].

Every policy owns its observation normalizer (as buffers, so it travels with the
checkpoint) and exposes:
  compute_loss(obs, actions, mask)   training, on normalized-inside tensors
  sample(obs)                        a chunk, torch in / torch out
  predict(obs_np)                    numpy in / numpy out, no grad -- the rollout API
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChunkPolicy(nn.Module):
    kind = "base"
    needs_images = False

    def __init__(self, obs_dim, action_dim, obs_horizon, chunk):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.obs_horizon, self.chunk = obs_horizon, chunk
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("obs_mask", torch.ones(obs_dim))    # 0 = dim hidden (M2.3 ablation)

    def set_normalizer(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean))
        self.obs_std.copy_(torch.as_tensor(std))

    def set_obs_subset(self, dims):
        self.obs_mask.zero_()
        self.obs_mask[list(dims)] = 1.0

    def normalize_obs(self, obs):
        return ((obs - self.obs_mean) / self.obs_std).clamp(-10.0, 10.0) * self.obs_mask

    def compute_loss(self, obs, actions, mask):
        raise NotImplementedError

    def sample(self, obs):
        raise NotImplementedError

    def config(self) -> dict:
        return {"obs_dim": self.obs_dim, "action_dim": self.action_dim,
                "obs_horizon": self.obs_horizon, "chunk": self.chunk}

    @property
    def device(self):
        return self.obs_mean.device

    @torch.no_grad()
    def predict(self, obs_hist: np.ndarray) -> np.ndarray:
        was_training = self.training
        self.eval()
        obs = torch.as_tensor(obs_hist, dtype=torch.float32, device=self.device)
        out = self.sample(obs).clamp(-1.0, 1.0).float().cpu().numpy()
        self.train(was_training)
        return out


def _registry():
    from imitation.policies.chunk_mlp import ChunkMLP
    from imitation.policies.diffusion import DiffusionPolicy
    from imitation.policies.vision import VisionChunkPolicy
    return {c.kind: c for c in (ChunkMLP, DiffusionPolicy, VisionChunkPolicy)}


def policy_class(kind) -> type[ChunkPolicy]:
    return _registry()[kind]


def build_policy(kind, **config) -> ChunkPolicy:
    return _registry()[kind](**config)


def save_policy(policy: ChunkPolicy, path, extra=None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": policy.kind, "config": policy.config(), "state_dict": policy.state_dict(),
                "extra": extra or {}}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(path, device=None) -> ChunkPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = build_policy(ckpt["kind"], **ckpt["config"])
    state = ckpt["state_dict"]
    state.setdefault("obs_mask", torch.ones(policy.obs_dim))   # checkpoints from before M2.3
    policy.load_state_dict(state)
    return policy.to(device or default_device()).eval()


def n_params(module):
    return sum(p.numel() for p in module.parameters())

"""Learned grasp flags: each arm's grasp flag at t+1 from (state, previous state, action).

The residual state predictor treats grasp and release, about one transition per arm per carry, as outliers of its L1
loss, and the friction grasp has no closed-form rule (it depends on the jaw settling with cloth in it). This head is a
classifier over the predictor's own normalized inputs, trained alongside it on the same transitions. Changes of the
flag are rare, so a transition where an arm's flag changes is weighted so that all changes together weigh as much as
all steady transitions, per arm (change_weights). Its events and metrics are in cloth_angles.model.grasp_events.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GraspHead(nn.Module):
    def __init__(self, input_dim: int, n_arms: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, n_arms))
        self.register_buffer("change_weight", torch.ones(n_arms))     # change_weights() of its training data

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: the predictor's _features(state, prev_state, action) -> a logit per arm."""
        return self.net(features)


def change_weights(now, nxt):
    """float[arms]: the weight of a transition whose flag changes, per arm: steady count / change count (1 if the
    flag never changes). now / nxt: [N, arms] flags."""
    changed = (now > 0.5) != (nxt > 0.5)
    n_change = changed.sum(0)
    return np.where(n_change > 0, (len(changed) - n_change) / np.maximum(n_change, 1), 1.0).astype(np.float32)


def weighted_loss(logits, now, nxt, weights):
    """Binary cross-entropy on the next flags, each changing transition weighted by its arm's weight."""
    changed = (now > 0.5) != (nxt > 0.5)
    w = torch.where(changed, weights.to(logits), torch.ones_like(logits))
    return nn.functional.binary_cross_entropy_with_logits(logits, (nxt > 0.5).to(logits), weight=w)

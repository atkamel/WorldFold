"""Latent -> [N, N] angle field. MLP-only, per spec, for small N."""

from __future__ import annotations

import math

import torch
from torch import nn


DELTA_MAX = 0.5   # rad; bound on per-step predicted change in delta mode


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, grid_size: int, hidden_dim: int = 128,
                 angle_convention: str = "signed", decode_mode: str = "absolute"):
        super().__init__()
        if angle_convention not in ("signed", "unsigned"):
            raise ValueError(f"unknown angle_convention {angle_convention!r}")
        if decode_mode not in ("absolute", "delta"):
            raise ValueError(f"unknown decode_mode {decode_mode!r}")
        self.grid_size = grid_size
        self.angle_convention = angle_convention
        self.decode_mode = decode_mode
        out_dim = grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [..., latent_dim] -> [..., N, N].

        absolute mode: the predicted angle field itself, range-bounded via
        pi * tanh(raw) (signed, [-pi, pi]) or pi/2 * (tanh(raw)+1) (unsigned).

        delta mode: a bounded per-step CHANGE, DELTA_MAX * tanh(raw). The
        caller adds it to the current field (and wraps/clamps). Residual
        decoding makes zero output equal the persistence baseline exactly, so
        everything learned is improvement over "predict no change" -- an
        absolute decoder must first re-earn its whole reconstruction floor
        before it can even tie persistence on quasi-static cells.
        """
        raw = self.net(latent)
        if self.decode_mode == "delta":
            angle = DELTA_MAX * torch.tanh(raw)
        elif self.angle_convention == "signed":
            angle = math.pi * torch.tanh(raw)
        else:
            angle = (math.pi / 2.0) * (torch.tanh(raw) + 1.0)
        shape = angle.shape[:-1] + (self.grid_size, self.grid_size)
        return angle.reshape(shape)

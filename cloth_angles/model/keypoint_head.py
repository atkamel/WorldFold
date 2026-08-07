"""Keypoint head: RSSM latent -> 4 cloth-corner DISPLACEMENTS from episode start.

The angle field is translation-invariant, so absolute corner positions are
not decodable from the latent; displacements caused by folding are. The
planner reconstructs absolute corners as corners0 + predicted delta and
scores imagined rollouts by distance to the goal corners.

Trained on frozen world-model posterior states; used at planning time on
prior (imagined) states -- the standard Dreamer head regime.
"""

from __future__ import annotations

import torch
from torch import nn


class KeypointHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 12),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """feature: [..., latent_dim] ([h, z]) -> [..., 12] corner displacement (m)."""
        return self.net(feature)


def save_keypoint_checkpoint(path, head: KeypointHead, step: int, latent_dim: int,
                              hidden_dim: int, wm_checkpoint: str) -> None:
    torch.save({
        "state_dict": head.state_dict(),
        "step": step,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "wm_checkpoint": wm_checkpoint,   # provenance: which frozen WM produced the states
    }, path)


def load_keypoint_checkpoint(path, expected_latent_dim: int, device=None) -> KeypointHead:
    checkpoint = torch.load(path, map_location=device or "cpu")
    if checkpoint["latent_dim"] != expected_latent_dim:
        raise ValueError(f"keypoint head latent_dim {checkpoint['latent_dim']} != "
                          f"expected {expected_latent_dim} (wrong world model?)")
    head = KeypointHead(checkpoint["latent_dim"], checkpoint["hidden_dim"])
    head.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        head = head.to(device)
    head.eval()
    return head

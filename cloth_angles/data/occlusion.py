"""Synthetic self-occlusion for angle fields: contiguous flap-like masks.

The DPM experiments (EXPERIMENTS.md section 1) need a partial observation:
a binary mask of hidden cells plus the masked field. Real self-occlusion
comes from a fold flap covering part of the cloth, so masks are contiguous
half-plane regions (a random cut direction, hiding the cells on one side),
not salt-and-pepper noise. Severity is the exact fraction of hidden cells.

Strata (per the experiment design):
    light: 10-20% of cells hidden
    heavy: 40-60% of cells hidden
"""

from __future__ import annotations

import numpy as np

STRATA = {"light": (0.10, 0.20), "heavy": (0.40, 0.60)}


def half_plane_mask(grid_size: int, hidden_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """bool[N, N], True = hidden. Cells are ranked by their projection onto a
    random direction; the top `hidden_fraction` quantile is hidden, giving an
    exactly-sized, contiguous, flap-like half-plane region."""
    if not 0.0 < hidden_fraction < 1.0:
        raise ValueError(f"hidden_fraction must be in (0, 1), got {hidden_fraction}")
    theta = rng.uniform(0.0, 2.0 * np.pi)
    direction = np.array([np.cos(theta), np.sin(theta)])
    ii, jj = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    projection = ii * direction[0] + jj * direction[1]
    n_hidden = max(1, int(round(hidden_fraction * grid_size * grid_size)))
    # random tie-break so axis-aligned cuts don't always hide the same cells
    order = np.argsort(projection.ravel() + rng.uniform(0, 1e-6, projection.size))
    mask = np.zeros(grid_size * grid_size, dtype=bool)
    mask[order[-n_hidden:]] = True
    return mask.reshape(grid_size, grid_size)


def sample_mask(grid_size: int, stratum: str, rng: np.random.Generator) -> np.ndarray:
    """Draw a mask with severity uniform in the stratum's range."""
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum {stratum!r}, expected one of {sorted(STRATA)}")
    lo, hi = STRATA[stratum]
    return half_plane_mask(grid_size, float(rng.uniform(lo, hi)), rng)


def apply_mask(field: np.ndarray, mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """field: [N, N] angles -> copy with hidden cells replaced by `fill`.
    (0.0 = flat-cloth angle, the least-informative physical value.)"""
    if field.shape != mask.shape:
        raise ValueError(f"field shape {field.shape} != mask shape {mask.shape}")
    out = field.copy()
    out[mask] = fill
    return out


def nearest_fill(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Deterministic inpainting floor baseline: each hidden cell takes the
    value of its nearest visible cell (Euclidean grid distance, ties by scan
    order). The 'raw' front-end reconstructs occluded cells this way."""
    if mask.all():
        return np.zeros_like(field)
    out = field.copy()
    vis_idx = np.argwhere(~mask)
    for i, j in np.argwhere(mask):
        d2 = (vis_idx[:, 0] - i) ** 2 + (vis_idx[:, 1] - j) ** 2
        vi, vj = vis_idx[int(np.argmin(d2))]
        out[i, j] = field[vi, vj]
    return out

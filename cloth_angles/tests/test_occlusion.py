"""Occlusion module invariants: contiguity, exact severity, reproducibility,
and nearest-fill correctness. These masks are the experimental treatment in
EXPERIMENTS.md section 1 -- a broken mask generator silently corrupts the
whole comparison, so the invariants are pinned here."""

import numpy as np
import pytest

from cloth_angles.data.occlusion import (
    STRATA,
    apply_mask,
    half_plane_mask,
    nearest_fill,
    sample_mask,
)

GRID = 11


def _contiguous(mask: np.ndarray) -> bool:
    """True if the hidden region is one 4-connected component."""
    hidden = set(map(tuple, np.argwhere(mask)))
    if not hidden:
        return False
    start = next(iter(hidden))
    frontier, seen = [start], {start}
    while frontier:
        i, j = frontier.pop()
        for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if (ni, nj) in hidden and (ni, nj) not in seen:
                seen.add((ni, nj))
                frontier.append((ni, nj))
    return seen == hidden


def test_mask_fraction_is_exact():
    rng = np.random.default_rng(0)
    for frac in (0.1, 0.25, 0.5):
        mask = half_plane_mask(GRID, frac, rng)
        expected = round(frac * GRID * GRID)
        assert mask.sum() == expected


def test_mask_is_contiguous():
    rng = np.random.default_rng(1)
    for _ in range(50):
        mask = half_plane_mask(GRID, float(rng.uniform(0.1, 0.6)), rng)
        assert _contiguous(mask)


def test_mask_seed_reproducible():
    m1 = half_plane_mask(GRID, 0.4, np.random.default_rng(42))
    m2 = half_plane_mask(GRID, 0.4, np.random.default_rng(42))
    assert np.array_equal(m1, m2)


def test_sample_mask_respects_stratum_bounds():
    rng = np.random.default_rng(2)
    for stratum, (lo, hi) in STRATA.items():
        for _ in range(20):
            mask = sample_mask(GRID, stratum, rng)
            frac = mask.sum() / mask.size
            # rounding to whole cells can nudge past the bound by < 1 cell
            slack = 1.0 / mask.size
            assert lo - slack <= frac <= hi + slack, (stratum, frac)


def test_sample_mask_rejects_unknown_stratum():
    with pytest.raises(ValueError):
        sample_mask(GRID, "medium", np.random.default_rng(0))


def test_apply_mask_only_touches_hidden_cells():
    rng = np.random.default_rng(3)
    field = rng.uniform(-np.pi, np.pi, (GRID, GRID)).astype(np.float32)
    mask = half_plane_mask(GRID, 0.3, rng)
    masked = apply_mask(field, mask)
    assert np.array_equal(masked[~mask], field[~mask])
    assert np.all(masked[mask] == 0.0)


def test_nearest_fill_copies_nearest_visible_value():
    field = np.zeros((3, 3), dtype=np.float32)
    field[:, 0] = 1.0   # visible column
    mask = np.zeros((3, 3), dtype=bool)
    mask[:, 1:] = True  # hide the rest
    filled = nearest_fill(field, mask)
    assert np.all(filled == 1.0)  # everything copies from the visible column


def test_nearest_fill_leaves_visible_cells_untouched():
    rng = np.random.default_rng(4)
    field = rng.uniform(-np.pi, np.pi, (GRID, GRID)).astype(np.float32)
    mask = half_plane_mask(GRID, 0.5, rng)
    filled = nearest_fill(field, mask)
    assert np.array_equal(filled[~mask], field[~mask])

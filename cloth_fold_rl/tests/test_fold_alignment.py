"""Layer-alignment score (cloth_fold_rl.fold_alignment)."""

import numpy as np

from cloth_fold_rl.fold_alignment import QUARTER_FOLDS, STRIP_FOLDS, folded_coords, groups, score

N, SPACING = 11, 0.03


def _folded(folds, layer_dz=0.002):
    """An ideal fold of the flat 30 cm grid: each vertex at its folded coordinate, layers stacked in z."""
    c = folded_coords(N, folds) * SPACING
    z = np.zeros(N * N)
    members, _ = groups(N, folds)
    for idx in members:
        z[idx] = np.arange(len(idx)) * layer_dz
    return np.column_stack([c, z])


def test_quarter_groups():
    members, edge = groups(N, QUARTER_FOLDS)
    sizes = sorted(len(m) for m in members)
    assert len(members) == 36 and sum(sizes) == N * N
    assert sizes.count(4) == 25 and sizes.count(2) == 10 and sizes.count(1) == 1
    corners = {0, N - 1, (N - 1) * N, N * N - 1}
    assert any(set(m) == corners for m in members)          # the four cloth corners stack
    assert sum(edge) == 6 * 4 - 4                             # the 6 x 6 folded grid's outline


def test_strip_groups():
    members, _ = groups(N, STRIP_FOLDS)
    assert sum(len(m) for m in members) == N * N
    assert max(len(m) for m in members) == 4
    iy = {i % N for m in members if 0 in m for i in m}        # cloth_0's stack: rows 0, 5, 10 (and 5's mirror)
    assert iy == {0, 5, 10}


def test_ideal_fold_passes_and_slide_is_free():
    pos = _folded(QUARTER_FOLDS)
    s = score(pos)
    assert s["passed"] and s["mean"] < 1e-12 and s["edge_max"] < 1e-12
    slid = pos + [0.05, -0.03, 0.0]
    assert score(slid)["passed"]


def test_misaligned_layer_fails():
    pos = _folded(QUARTER_FOLDS)
    members, _ = groups(N, QUARTER_FOLDS)
    top = [m[-1] for m in members if len(m) == 4]            # one layer of every four-stack, shifted 2 cm
    pos[top, 0] += 0.02
    s = score(pos)
    assert not s["passed"] and abs(s["edge_max"] - 0.02) < 1e-9


def test_flat_cloth_is_far_off():
    ix, iy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    flat = np.column_stack([ix.ravel() * SPACING, iy.ravel() * SPACING, np.zeros(N * N)])
    s = score(flat)
    assert not s["passed"] and s["mean"] > 0.1

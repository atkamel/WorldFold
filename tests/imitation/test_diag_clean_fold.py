"""R0 diagnostic start (isaac.diag_clean_fold): the perfect half fold's geometry and the stage-1 start, without Isaac."""

import numpy as np
import pytest

from isaac.diag_clean_fold import CREASE_RADIUS, begin_stage, fold_half, maybe_install

H, SPACING = 0.15, 0.003
N = int(round(2 * H / SPACING)) + 1


def _flat_grid(z=0.424):
    xs = np.linspace(-H, H, N)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")          # row-major ix * N + iy, like the cloth
    return np.stack([gx.ravel(), gy.ravel(), np.full(N * N, z)], axis=1)


def _corner(ix, iy):
    return ix * N + iy


def test_flap_lands_on_the_fixed_half():
    p = _flat_grid()
    v0, v10, vn = _corner(0, 0), _corner(0, N - 1), _corner(N - 1, N - 1)
    out = fold_half(p, p[v10], p[v0])
    # the flap's far edge lies on the fixed edge, 2r above it
    assert np.linalg.norm(out[v10, :2] - p[v0, :2]) < 1e-3
    assert np.linalg.norm(out[vn, :2] - p[_corner(N - 1, 0), :2]) < 1e-3
    assert out[v10, 2] == pytest.approx(p[v0, 2] + 2 * CREASE_RADIUS)
    # every particle ends inside the fixed half's footprint (the flap's mirror), within one particle spacing
    assert out[:, 1].max() <= SPACING and out[:, 1].min() >= -H - 1e-9
    flap = p[:, 1] > SPACING
    top = out[flap]
    assert np.all(top[:, 2] > p[0, 2])
    # each flap particle sits over the bottom particle it mirrors, within one spacing
    mirror_y = -p[flap, 1]
    assert np.abs(top[:, 1] - mirror_y).max() < SPACING
    assert np.allclose(out[:, 0], p[:, 0])          # the fold moves nothing across it


def test_fold_preserves_length_along_the_fold():
    p = _flat_grid()
    out = fold_half(p, p[_corner(0, N - 1)], p[_corner(0, 0)])
    col = out[_corner(5, 0):_corner(5, N - 1) + 1]  # one column across the crease
    seg = np.linalg.norm(np.diff(col, axis=0), axis=1)
    # the crease's chords are a hair shorter than their arcs; the total stays within 0.5 mm of 2H
    assert sum(seg) == pytest.approx(2 * H, abs=5e-4)
    assert seg.max() <= SPACING + 1e-9


def test_strip_width_meets_the_r0_check():
    # fold1_summary's width |y5 - y0| (vertex 5 = the west crease end) must be >= 14 cm
    p = _flat_grid()
    out = fold_half(p, p[_corner(0, N - 1)], p[_corner(0, 0)])
    assert abs(out[_corner(0, N // 2), 1] - out[_corner(0, 0), 1]) >= 0.14


def test_begin_stage_starts_the_second_fold():
    from cloth_fold_rl.quarter_fold_env import CLOTH_110, CLOTH_115, STAGES_FOLD_OVER, QuarterFoldEnv
    env = QuarterFoldEnv(seed=0, stages=STAGES_FOLD_OVER)
    env.start_hook = lambda e: begin_stage(e, 1)
    obs, info = env.reset(seed=0)
    assert env.stage == 1 and info["stage"] == 1
    goals = {m.prefix: env.goal(m) for m in STAGES_FOLD_OVER[1].moves}
    now = env.unwrapped.cloth_positions()
    assert np.allclose(goals["left_"], now[CLOTH_110]) and np.allclose(goals["right_"], now[CLOTH_115])
    assert info["fold_score"] == pytest.approx(0.5, abs=0.1)     # stage 0 counts as done
    env.close()


def test_install_is_opt_in(monkeypatch):
    class E:
        start_hook = None
    monkeypatch.delenv("WORLDFOLD_QF_START", raising=False)
    assert maybe_install(E(), "quarter_fold", "isaac_friction").start_hook is None
    monkeypatch.setenv("WORLDFOLD_QF_START", "clean")
    assert maybe_install(E(), "half_fold", "isaac_friction").start_hook is None
    assert maybe_install(E(), "quarter_fold", "isaac_weld").start_hook is None
    assert maybe_install(E(), "quarter_fold", "isaac_friction").start_hook is not None

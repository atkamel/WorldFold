"""Q2: the friction expert's place point for the second fold (mirror along the carry axis, surface height)."""
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def expert_cls(monkeypatch):
    # isaac.fold_expert imports isaac.half_fold_demo (LeHome) at module level; stub just what it reads
    demo = types.ModuleType("isaac.half_fold_demo")
    for k, v in dict(ARC_HEIGHT=0.08, ARC_WAYPOINTS=6, PINCH_HEIGHT=0.01, PLACE_HEIGHT=0.011, POSE_SETTLE_STEPS=3,
                     SEED_Q={}, TRACK_TOL=0.02, WAYPOINT_TOL=0.05).items():
        setattr(demo, k, v)
    monkeypatch.setitem(sys.modules, "isaac.half_fold_demo", demo)
    monkeypatch.delitem(sys.modules, "isaac.fold_expert", raising=False)
    from isaac.fold_expert import IsaacArmExpert
    return IsaacArmExpert


def _expert(cls, stage, cloth):
    e = cls.__new__(cls)
    e.env = types.SimpleNamespace(stage=stage)
    e.base = types.SimpleNamespace(cloth_positions=lambda: cloth)
    e.corners = (0,)
    return e


def test_stage0_mirror_is_unchanged_and_stage1_mirrors_x(expert_cls):
    from mujuco.cloth_params import TABLE_TOP_Z
    cloth = np.zeros((121, 3))
    cloth[:, 2] = TABLE_TOP_Z + 0.004
    off = np.array([0.003, -0.004])
    e0 = _expert(expert_cls, 0, cloth)
    goal0 = np.array([-0.15, -0.285, TABLE_TOP_Z])
    np.testing.assert_allclose(e0._place_point(goal0, off, np.array([-0.15, 0.0, 0.0])),
                               [goal0[0] + off[0], goal0[1] - off[1], TABLE_TOP_Z + e0.PLACE_HEIGHT])
    # stage 1 carries along x (west edge -> east edge): flip x; the goal lies on a 2-layer stack 8 mm up
    goal1 = np.array([0.15, -0.285, TABLE_TOP_Z])
    cloth[110] = [0.15, -0.285, TABLE_TOP_Z + 0.012]
    e1 = _expert(expert_cls, 1, cloth)
    p = e1._place_point(goal1, off, np.array([-0.15, -0.285, 0.0]))
    np.testing.assert_allclose(p[:2], [goal1[0] - off[0], goal1[1] + off[1]])
    assert np.isclose(p[2], TABLE_TOP_Z + 0.012 - 0.004 + e1.PLACE_HEIGHT)

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


def test_lift_gate_holds_the_first_closer_until_both_closed():
    from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
    phases = ("approach", "descend", "close", "lift")

    class Arm:
        PHASES = phases
        RELEASE_PHASES = ()

        def __init__(self, phase):
            self.phase, self.lift_allowed = phases.index(phase), True

    q = QuarterFoldExpert.__new__(QuarterFoldExpert)
    q.env = types.SimpleNamespace(stage=1)
    q.experts = {(1, "left_"): Arm("close"), (1, "right_"): Arm("descend"), (0, "left_"): Arm("lift")}
    q._update_lift_gate()
    assert not q.experts[(1, "left_")].lift_allowed and not q.experts[(1, "right_")].lift_allowed
    q.experts[(1, "right_")].phase = phases.index("close")
    q._update_lift_gate()
    assert q.experts[(1, "left_")].lift_allowed and q.experts[(1, "right_")].lift_allowed
    q.env.stage = 0                     # stage 0: the gate never touches the arms
    q.experts[(0, "left_")].lift_allowed = "untouched"
    q._update_lift_gate()
    assert q.experts[(0, "left_")].lift_allowed == "untouched"


def test_sequenced_stage_parks_the_left_arm_until_the_right_arm_closed():
    from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
    phases = ("approach", "descend", "close", "lift", "hold", "release", "done")

    class Arm:
        PHASES = phases
        RELEASE_PHASES = ("release", "done")

        def __init__(self, phase):
            self.phase, self.waiting, self.release_allowed, self.lift_allowed = phases.index(phase), False, False, True

    q = QuarterFoldExpert.__new__(QuarterFoldExpert)
    q.env = types.SimpleNamespace(stage=1)
    left, right = Arm("approach"), Arm("descend")
    q.experts = {(1, "left_"): left, (1, "right_"): right}
    q.turn_done = set()
    q._update_sequence()
    q._update_lift_gate()
    assert left.waiting and not right.waiting and not right.lift_allowed
    right.phase = phases.index("close")
    q._update_sequence()
    q._update_lift_gate()
    assert not left.waiting and not right.lift_allowed          # right holds its pinch until the left closes too
    right.phase = phases.index("approach")                      # a regrasp keeps the left arm's turn
    q._update_sequence()
    assert not left.waiting

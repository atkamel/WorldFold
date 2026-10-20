"""IsaacPlaceServoExpert without Isaac: the residual and the live goal reach the IK target, the release vote and the
hold cap, and a slip ends the carry."""

import numpy as np

from isaac.fold_expert import IsaacArmExpert
from isaac.place_servo import IsaacPlaceServoExpert
from mujuco.cloth_params import TABLE_TOP_Z


class _IK:
    def __init__(self):
        self.tips = []

    def solve(self, prefix, tip, jaw, q, orientation_weight=0.0, iters=None, jaw_weight=None):
        self.tips.append(np.asarray(tip, float).copy())
        return np.r_[tip, 0.0, 0.0], 0.0


class _Base:
    grasp_knobs = {}

    def __init__(self):
        ix, iy = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
        self.cloth = np.column_stack([ix.ravel() * 0.03 - 0.15, iy.ravel() * 0.03 - 0.15, np.full(121, TABLE_TOP_Z)])
        self.site = self.cloth[10] + [0.0, 0.0, 0.03]

    def cloth_positions(self):
        return self.cloth

    def gripper_position(self, prefix):
        return self.site

    def joint_positions(self, prefix):
        return np.zeros(6)


def _expert():
    e = object.__new__(IsaacPlaceServoExpert)
    e.base, e._ik, e.prefix, e.corners = _Base(), _IK(), "left_", (10,)
    e.PHASES = IsaacArmExpert.PHASES + IsaacArmExpert.RELEASE_PHASES
    e.goal = lambda: e.base.cloth[0].copy()
    e.live_goal = lambda: e.base.cloth[0].copy()
    e.reset()
    e.release_allowed = False
    e.plan = {"jaw": np.array([0.0, -1.0, 0.0])}
    return e


def test_residual_shifts_the_carry_target():
    e = _expert()
    e._start_carry()
    e.act()
    plain = e._ik.tips[-1]
    e.residual = np.array([0.01, -0.02, 0.0])
    e.act()
    assert np.allclose(e._ik.tips[-1] - plain, e.residual, atol=1e-9)


def test_place_follows_the_partner_where_it_is_now():
    e = _expert()
    e._go("place")
    e.base.cloth[0] += [0.04, 0.0, 0.0]                    # the bottom layer slid 4 cm
    e.base.site = e.base.cloth[10] + [0.0, 0.0, 0.03]
    e.act()
    corner, goal = e.base.cloth[10], e.base.cloth[0]
    want = e.base.site + np.clip(goal + [0, 0, e.PRESS_Z] - corner, -e.STEER_CLIP, e.STEER_CLIP)
    assert np.allclose(e._ik.tips[-1], want)


def test_release_waits_for_the_gate_and_the_vote_or_the_cap():
    e = _expert()
    e._start_hold()
    e.release_vote = False
    e.act()
    assert e.PHASES[e.phase] == "hold"                     # gate closed
    e.release_allowed = True
    for _ in range(e.HOLD_CAP - 2):
        e.act()
    assert e.PHASES[e.phase] == "hold"                     # gate open, no vote, under the cap
    e.act()
    assert e.PHASES[e.phase] == "release"                  # the cap


def test_vote_releases_at_once_once_the_gate_opens():
    e = _expert()
    e._start_hold()
    e.release_allowed = True
    e.act()
    assert e.PHASES[e.phase] == "release"


def test_slip_in_the_carry_marks_a_drop_and_holds():
    e = _expert()
    e._start_carry()
    e.base.site = e.base.cloth[10] + [0.0, 0.0, 0.2]       # the corner is far from the jaw
    for _ in range(e.SLIP_STEPS):
        e.act()
    assert e.dropped and e.PHASES[e.phase] == "hold"


def test_live_goal_is_capped_around_the_start(monkeypatch):
    e = _expert()
    monkeypatch.setattr(IsaacPlaceServoExpert, "LIVE_MAX", 0.03)
    start = e.base.cloth[0].copy()
    e.goal = lambda: start
    e.base.cloth[0] += [0.08, -0.01, 0.0]
    assert np.allclose(e._goal_now(), start + [0.03, -0.01, 0.0])

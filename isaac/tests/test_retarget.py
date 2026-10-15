"""Placing onto the partner corner where it is now: QuarterFoldEnv.goal_follows and IsaacArmExpert.RETARGET."""

import numpy as np

from cloth_fold_rl.quarter_fold_env import Move, QuarterFoldEnv
from isaac.fold_expert import IsaacArmExpert


def test_goal_follows_the_vertex():
    env = object.__new__(QuarterFoldEnv)
    env._start = np.zeros((121, 3))
    now = np.zeros((121, 3))
    now[0] = [0.0, 0.08, 0.0]
    env._vertex = lambda v: now[v]
    mv = Move("left_", (5,), 0)
    assert np.allclose(env.goal(mv), 0.0)
    env.goal_follows = True
    assert np.allclose(env.goal(mv), [0.0, 0.08, 0.0])


class _IK:
    def solve(self, prefix, tip, jaw, q, orientation_weight=0.0):
        return np.r_[tip, 0.0, 0.0], 0.0          # "joints" = the tip, so the plan shows where it aims


def _expert(goal_now, wp):
    e = object.__new__(IsaacArmExpert)
    e._ik, e.prefix, e.RETARGET, e.wp = _IK(), "left_", 0.01, wp
    e._q = lambda: np.zeros(5)
    e.goal = lambda: np.asarray(goal_now, float)
    tips = [np.array([0.0, 0.1 - 0.02 * i, 0.1]) for i in range(6)]
    e.plan = {"arc_tips": tips, "arc": [np.r_[t, 0, 0] for t in tips], "jaw": np.array([1.0, 0, 0]),
              "goal": np.zeros(3), "retreat": None}
    return e


def test_retarget_bends_the_rest_of_the_arc():
    e = _expert([0.0, 0.05, 0.0], wp=3)
    before = [t.copy() for t in e.plan["arc_tips"]]
    e._retarget()
    tips = e.plan["arc_tips"]
    assert all(np.allclose(tips[i], before[i]) for i in range(3))        # the part already flown stays
    assert np.allclose(tips[-1] - before[-1], [0.0, 0.05, 0.0])          # the end lands on the moved goal
    assert np.allclose(e.plan["goal"], [0.0, 0.05, 0.0])


def test_retarget_waits_for_the_second_half_and_a_real_move():
    e = _expert([0.0, 0.05, 0.0], wp=1)
    e._retarget()
    assert np.allclose(e.plan["goal"], 0.0)
    e = _expert([0.0, 0.005, 0.0], wp=4)
    e._retarget()
    assert np.allclose(e.plan["goal"], 0.0)

"""Scripted (open-loop) half-fold baseline."""

import numpy as np
import pytest

from cloth_fold_rl.quarter_fold_env import HALF_FOLD_MAX_STEPS, SETTLE_STEPS
from cloth_fold_rl.scripted_half_fold import CLOSED, OPEN, SCHEDULE, WAYPOINTS, ScriptedHalfFold


class _ArmsOnly:
    """Just the proprio accessor: the scripted policy must not need anything else (no cloth state)."""

    def __init__(self):
        self.unwrapped = self
        self.q = {"left_": np.zeros(5), "right_": np.zeros(5)}

    def joint_positions(self, prefix):
        return self.q[prefix]


def test_tables_are_consistent():
    names = [name for name, _, _ in SCHEDULE]
    for prefix in ("left_", "right_"):
        assert list(WAYPOINTS[prefix]) == names
        assert all(len(q) == 5 for q in WAYPOINTS[prefix].values())
    # the script plus the settle window fits in the episode
    assert sum(steps for _, _, steps in SCHEDULE) + SETTLE_STEPS <= HALF_FOLD_MAX_STEPS


def test_open_loop_follows_schedule():
    env = _ArmsOnly()
    policy = ScriptedHalfFold(env)
    t = 0
    for name, grip, steps in SCHEDULE:
        for _ in range(steps):
            action = policy.act()
            assert action.shape == (12,)
            assert action[5] == grip and action[11] == grip
            assert np.all(np.abs(action) <= 1.0)
            t += 1
    assert policy.segment() == (SCHEDULE[-1][0], SCHEDULE[-1][1])   # holds the last pose after the script
    # deterministic in (step, joints)
    policy.reset()
    a = policy.act()
    policy.reset()
    assert np.array_equal(a, policy.act())
    assert {OPEN, CLOSED} == {grip for _, grip, _ in SCHEDULE}


def test_servo_direction():
    env = _ArmsOnly()
    policy = ScriptedHalfFold(env)
    target = np.array(WAYPOINTS["left_"][SCHEDULE[0][0]])
    env.q["left_"] = target - 0.01
    np.testing.assert_allclose(policy.act()[:5], 0.01 / 0.05, rtol=1e-5)


@pytest.mark.slow
def test_folds_on_nominal_seed():
    from cloth_fold_rl.quarter_fold_env import HalfFoldEnv
    from cloth_fold_rl.scripted_half_fold import run_episode

    env = HalfFoldEnv()
    row = run_episode(env, ScriptedHalfFold(env), seed=0)
    assert row["success"], row

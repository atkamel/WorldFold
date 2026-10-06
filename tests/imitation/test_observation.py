"""Observation spec (docs/imitation.md §2): goal slots, stage/settle dims, recorded offset."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow  # builds a sim

from cloth_fold_rl.quarter_fold_env import CLOTH_0, CLOTH_10, CLOTH_110, CLOTH_120, SETTLE_STEPS, QuarterFoldEnv
from imitation.spec import OBS_DIM
from imitation.tasks import HalfFoldEnv
from imitation.teachers import ScriptedTeacher

TASK = 50 + 69                  # task block start; one-hot removed
GOALS = slice(TASK + 1, TASK + 13)


def _start(env, v):
    return env._start[v]


def test_shape_and_goal_slots():
    env = HalfFoldEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) == env.observation_space.shape
    goals = obs[GOALS].reshape(4, 3)
    # slot order [v0, v10, v110, v120]: carried corners point at their fold targets,
    # anchors at where they are
    np.testing.assert_allclose(goals[1], _start(env, CLOTH_0), atol=1e-6)
    np.testing.assert_allclose(goals[3], _start(env, CLOTH_110), atol=1e-6)
    np.testing.assert_allclose(goals[0], _start(env, CLOTH_0), atol=1e-6)
    np.testing.assert_allclose(goals[2], _start(env, CLOTH_110), atol=1e-6)
    assert obs[-2] == 0.0 and obs[-1] == 0.0


def test_cloth_offset_recorded():
    env = HalfFoldEnv()
    _, info = env.reset(seed=5)
    off = env.unwrapped._domain_params["cloth_offset_xy"]
    assert len(off) == 2 and max(abs(v) for v in off) > 0
    assert info["domain_parameters"]["cloth_offset_xy"] == off


def test_settle_counter_observed_until_success():
    env = HalfFoldEnv()
    teacher = ScriptedTeacher(env)
    env.reset(seed=0)
    teacher.reset()
    settles = []
    for _ in range(env.unwrapped.max_episode_steps):
        obs, _, terminated, truncated, info = env.step(teacher.act())
        settles.append(obs[-1])
        assert obs[-1] == pytest.approx(info["settle_steps"] / SETTLE_STEPS)
        if terminated or truncated:
            break
    assert info["success"]
    assert max(settles) > 0.5


def test_quarter_fold_goals_follow_stage():
    env = QuarterFoldEnv()
    env.reset(seed=0)
    env.stage = 1
    env._stage_start = env._start.copy()
    env._set_goals()
    obs = env._observe()
    goals = obs[GOALS].reshape(4, 3)
    np.testing.assert_allclose(goals[0], _start(env, CLOTH_110), atol=1e-6)   # stack carried east
    np.testing.assert_allclose(goals[1], _start(env, CLOTH_110), atol=1e-6)
    np.testing.assert_allclose(goals[3], _start(env, CLOTH_120), atol=1e-6)   # anchor stays
    assert obs[-2] == pytest.approx(0.5)

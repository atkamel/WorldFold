"""Invariants that must survive the imitation/isaac merge: stage counts and the 139-D observation."""

import numpy as np
import pytest

import imitation.tasks
from cloth_fold_rl import quarter_fold_env as qfe
from imitation.spec import OBS_DIM
from isaac.tests.test_half_fold import ScriptedBase


def test_imitation_half_fold_has_one_stage():
    env = imitation.tasks.HalfFoldEnv(base_env=ScriptedBase(), cloth_jitter=0.0)
    assert len(env.stages) == 1


def test_quarter_fold_has_two_stages():
    env = qfe.QuarterFoldEnv(base_env=ScriptedBase(), cloth_jitter=0.0)
    assert len(env.stages) == 2


def test_rl_half_fold_has_one_stage():
    env = qfe.HalfFoldEnv(base_env=ScriptedBase(), cloth_jitter=0.0)
    assert len(env.stages) == 1


@pytest.mark.parametrize("make", [
    lambda: imitation.tasks.HalfFoldEnv(base_env=ScriptedBase(), cloth_jitter=0.0),
    lambda: qfe.QuarterFoldEnv(base_env=ScriptedBase(), cloth_jitter=0.0),
    lambda: qfe.HalfFoldEnv(base_env=ScriptedBase(), cloth_jitter=0.0),
])
def test_observation_length_is_obs_dim(make):
    env = make()
    obs, _ = env.reset(seed=0)
    assert len(obs) == OBS_DIM == env.observation_space.shape[0]


@pytest.mark.slow
def test_mujoco_half_fold_reset():
    env = imitation.tasks.HalfFoldEnv()
    obs, _ = env.reset(seed=0)
    assert np.array_equal(env._start, env.unwrapped.cloth_positions())
    assert len(env.stages) == 1
    assert obs.shape == (OBS_DIM,)

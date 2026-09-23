"""Snapshot/restore determinism and side-effect-free teacher labelling."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow  # builds a sim and runs 40 expert IK steps

from imitation.sim_state import restore, snapshot
from imitation.tasks import HalfFoldEnv
from imitation.teachers import ScriptedTeacher


def _obs(env):
    return env._observe()


@pytest.fixture(scope="module")
def mid_episode():
    """An env 40 expert steps into an episode (the arms have grasped by then)."""
    env = HalfFoldEnv()
    teacher = ScriptedTeacher(env)
    env.reset(seed=3)
    teacher.reset()
    for _ in range(40):
        env.step(teacher.act())
    return env, teacher


def test_restore_replays_bit_for_bit(mid_episode):
    env, _ = mid_episode
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(8, 12)).astype(np.float32)
    snap = snapshot(env)
    first = [env.step(a) for a in actions]
    restore(env, snap)
    second = [env.step(a) for a in actions]
    restore(env, snap)
    for (o1, r1, *_), (o2, r2, *_) in zip(first, second):
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2


def test_label_chunk_leaves_env_and_expert_untouched(mid_episode):
    env, teacher = mid_episode
    obs = _obs(env)
    stage, step = env.stage, env.unwrapped._step_count
    phases = teacher.phases()
    chunk = teacher.label_chunk(env, horizon=6)
    assert chunk.shape == (6, 12)
    assert np.all(np.abs(chunk) <= 1.0)
    np.testing.assert_array_equal(obs, _obs(env))
    assert (env.stage, env.unwrapped._step_count) == (stage, step)
    assert teacher.phases() == phases



def test_teacher_reset_restarts_the_ik_rngs():
    """A pooled worker reuses one teacher across episodes; its IK restart rngs must
    restart per episode, or which worker a seed lands on changes the demo."""
    env = HalfFoldEnv()
    teacher = ScriptedTeacher(env)
    state = lambda: [e.rng.bit_generator.state for e in teacher.expert.experts.values()]
    fresh = state()
    for e in teacher.expert.experts.values():
        e.rng.random()                                      # an earlier episode drew restarts
    teacher.reset()
    assert state() == fresh

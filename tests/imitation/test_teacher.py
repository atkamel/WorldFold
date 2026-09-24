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


def test_labels_agree_with_the_expert_on_its_own_trajectory():
    """DAgger labels must say what the expert would do. Along the expert's own
    episode, the chunk labelled at t should match the actions it then took (M3.2:
    resync-from-scratch labels disagreed by up to 0.57 per joint during carry)."""
    env = HalfFoldEnv()
    teacher = ScriptedTeacher(env)
    joints = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    errs = []
    for seed in (0, 1):
        env.reset(seed=seed)
        teacher.reset()
        acts, labels = [], {}
        for t in range(250):
            if t % 8 == 0:
                labels[t] = teacher.label_chunk(env, 16)
            acts.append(teacher.act())
            _, _, term, trunc, _ = env.step(acts[-1])
            if term or trunc:
                break
        acts = np.array(acts)
        for t, lab in labels.items():
            n = min(16, len(acts) - t)
            if n >= 8:
                errs.append(np.abs(lab[:n, joints] - acts[t:t + n, joints]).mean())
    assert np.median(errs) < 0.02 and np.max(errs) < 0.1, np.round(errs, 3)


def test_a_shadowing_teacher_labels_like_the_one_driving():
    """Someone else drives (here: a second expert); the labelling teacher only observes.
    Its labels must match what the driver then does."""
    env = HalfFoldEnv()
    driver, labeller = ScriptedTeacher(env), ScriptedTeacher(env)
    joints = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    env.reset(seed=3)
    driver.reset()
    labeller.reset()
    acts, labels = [], {}
    for t in range(250):
        if t % 8 == 0:
            labels[t] = labeller.label_chunk(env, 16)
        labeller.observe()
        acts.append(driver.act())
        _, _, term, trunc, _ = env.step(acts[-1])
        if term or trunc:
            break
    acts = np.array(acts)
    errs = [np.abs(lab[:16, joints] - acts[t:t + 16, joints]).mean() for t, lab in labels.items() if t + 16 <= len(acts)]
    assert np.max(errs) < 0.02, np.round(errs, 3)

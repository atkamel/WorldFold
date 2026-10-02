"""HalfFoldEnv's success state, checked on a scripted stand-in for the simulator. Runs without Isaac Sim or MuJoCo."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cloth_fold_rl.quarter_fold_env import (  # noqa: E402
    CLOTH_0, CLOTH_10, CLOTH_110, CLOTH_120, SETTLE_STEPS, HalfFoldEnv,
)
from isaac.isaac_env import cloth_grid_mesh  # noqa: E402


class ScriptedBase(gym.Env):
    """Exposes only what the fold wrappers read; the test moves vertices and grasps by hand."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            k: gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
            for k, n in (("proprio", 4), ("cloth_state", 4), ("task", 4))})
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.prefixes = ["left_", "right_"]
        self.weld_mask = {}
        self.max_episode_steps = 1000
        self._step_count = 0
        self.positions = None
        self.grasped = {p: False for p in self.prefixes}

    def _obs(self):
        return {k: np.zeros(4, dtype=np.float32) for k in ("proprio", "cloth_state", "task")}

    def reset(self, seed=None, options=None):
        points, _ = cloth_grid_mesh()
        self.positions = np.array(points) + np.array([0.0, 0.0, 0.43])
        self.grasped = {p: False for p in self.prefixes}
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        self._step_count += 1
        return self._obs(), 0.0, False, False, {}

    def cloth_positions(self):
        return self.positions.copy()

    def gripper_position(self, prefix):
        return np.zeros(3)

    def grasp_active(self, prefix):
        return self.grasped[prefix]

    def _failed(self):
        return False


def make_env():
    env = HalfFoldEnv(base_env=ScriptedBase(), cloth_jitter=0.0)
    env.reset(seed=0)
    return env, env.unwrapped


def fold(base, left=True, right=True, offset=0.0):
    start = base.positions.copy()
    if left:
        base.positions[CLOTH_10] = start[CLOTH_0] + [offset, 0.0, 0.02]
    if right:
        base.positions[CLOTH_120] = start[CLOTH_110] + [offset, 0.0, 0.02]


def run(env, n):
    results = []
    for _ in range(n):
        results.append(env.step(np.zeros(12, dtype=np.float32)))
    return results


def test_folded_and_released_succeeds_after_settle():
    env, base = make_env()
    fold(base, offset=0.04)                       # inside the 5 cm tolerance
    results = run(env, SETTLE_STEPS)
    assert not any(r[2] for r in results[:-1])
    _, reward, terminated, _, info = results[-1]
    assert terminated and info["success"] and info["termination_reason"] == "success"
    assert info["fold_score"] > 0.8
    assert reward > 10.0


def test_held_in_the_gripper_is_not_success():
    env, base = make_env()
    fold(base)
    base.grasped["left_"] = True
    assert not any(r[2] for r in run(env, 3 * SETTLE_STEPS))


def test_one_corner_is_not_success():
    env, base = make_env()
    fold(base, right=False)
    assert not any(r[2] for r in run(env, 3 * SETTLE_STEPS))


def test_corner_outside_tolerance_is_not_success():
    env, base = make_env()
    fold(base, offset=0.06)
    assert not any(r[2] for r in run(env, 3 * SETTLE_STEPS))


def test_releasing_early_restarts_the_settle_count():
    env, base = make_env()
    fold(base)
    run(env, SETTLE_STEPS - 1)
    base.grasped["right_"] = True
    run(env, 1)
    base.grasped["right_"] = False
    results = run(env, SETTLE_STEPS)
    assert not any(r[2] for r in results[:-1]) and results[-1][4]["success"]


def test_dragging_an_anchor_fails():
    env, base = make_env()
    base.positions[CLOTH_0] += [0.0, -0.25, 0.0]
    _, _, terminated, _, info = env.step(np.zeros(12, dtype=np.float32))
    assert terminated and info["termination_reason"] == "cloth_dragged" and not info["success"]

"""A pure-numpy stand-in for isaac.isaac_env.IsaacClothFoldBatch (milestone V tests): B copies whose state moves
only in `advance()`, which steps EVERY copy -- idle ones too, as the real shared physics does. A copy stepped while
its episode waits for the driver would drift, so an episode's trajectory depends only on its seed iff the lockstep
barrier is right."""

from __future__ import annotations

import threading

import numpy as np

from imitation.lockstep import BatchBase

D = 12
SETTLE = 3


class FakeBatch(BatchBase):
    def __init__(self, n):
        self.x = np.zeros((n, D))
        self.u = np.zeros((n, D))
        self.steps = 0
        self.main_calls = []
        self.envs = [FakeSubEnv(self, i) for i in range(n)]

    def advance(self):
        self.steps += 1
        self.x += 0.2 * np.tanh(self.u - self.x) + 0.01 * np.sin(3.0 * self.x)


class FakeSubEnv:
    """Looks like HalfFoldEnv over a sub-env, as far as the rollout worker reads it."""

    def __init__(self, batch, i):
        self.batch, self.i = batch, i
        self.unwrapped = self
        self.last_render_s = 0.0
        self._domain_params = {}
        self._start = np.zeros((121, 3))
        self.max_episode_steps = 0
        self.t = 0
        self.parked = 0

    @property
    def state(self):
        return self.batch.x[self.i]

    def _obs(self):
        return np.concatenate([self.state, [self.t / max(self.max_episode_steps, 1)]]).astype(np.float32)

    def _info(self, reason=None):
        return {"stage": 0, "fold_score": float(-np.linalg.norm(self.state)), "grasped": {"left_": False,
                "right_": False}, "success": reason == "success", "termination_reason": reason,
                "anchor_drift": 0.0, "move_distance": [0.0, 0.0]}

    def _write(self, x0):
        self.batch.main_calls.append((self.i, threading.current_thread().name))
        self.batch.x[self.i] = x0
        self.batch.u[self.i] = 0.0

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed)
        self.batch.on_main(self._write, rng.normal(size=D))
        self._domain_params = {"mass": float(rng.uniform(0.7, 1.3))}
        self.max_episode_steps = 6 + int(seed) % 9
        self._start = np.full((121, 3), float(seed))
        for _ in range(SETTLE):
            self.batch.sync(self.i)
        self.t = 0
        return self._obs(), self._info()

    def step(self, action):
        self.batch.u[self.i] = np.asarray(action, dtype=float)[:D]
        self.batch.sync(self.i)
        self.t += 1
        term = bool(np.linalg.norm(self.state) < 0.3)
        trunc = (not term) and self.t >= self.max_episode_steps
        return self._obs(), float(-np.linalg.norm(self.state)), term, trunc, self._info("success" if term else None)

    def _failed(self):
        return False

    def park(self):
        self.parked += 1


class FakeExpert:
    def resync(self):
        return {"resynced": True}


class FakeTeacher:
    """Acts from the copy's state; label_chunk looks ahead through real (shared) physics steps and restores."""

    def __init__(self, env):
        self.env, self.expert, self.k = env, FakeExpert(), 0

    def reset(self):
        self.k = 0

    def act(self):
        self.k += 1
        return (-0.5 * self.env.state + 0.1 * np.sin(self.k)).astype(np.float32)

    def observe(self):
        self.act()

    def see(self, obs):
        pass

    def label_chunk(self, env, horizon):
        saved_x, saved_u, saved_t, saved_k = env.state.copy(), env.batch.u[env.i].copy(), env.t, self.k
        chunk = []
        for _ in range(horizon):
            a = self.act()
            chunk.append(a)
            env.step(a)
        env.batch.x[env.i], env.batch.u[env.i], env.t, self.k = saved_x, saved_u, saved_t, saved_k
        return np.stack(chunk)


class FakePolicy:
    obs_horizon = 1
    chunk = 4
    needs_images = False

    def predict(self, obs, images=None):
        last = obs[:, -1, :D]
        steps = np.linspace(0.5, 1.0, self.chunk)[None, :, None]
        return (np.tanh(last)[:, None, :] * steps).astype(np.float32)

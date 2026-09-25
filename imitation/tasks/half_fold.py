"""Half fold: stage 0 of the quarter fold, ending in success when it settles.

Both arms carry the north corners onto the south corners (a fold about y = 0),
release, and the corners must stay placed for SETTLE_STEPS. Everything else --
observation (139-D state), action (12-D joint deltas + grippers), reward shaping,
drag/instability termination -- is inherited from QuarterFoldEnv unchanged, so
QuarterFoldExpert drives it as-is.
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from cloth_fold_rl.quarter_fold_env import STAGES, QuarterFoldEnv
from imitation.spec import ACTION_DIM, OBS_DIM  # noqa: F401 (re-exported)

HALF_FOLD_MAX_STEPS = 250


class HalfFoldEnv(QuarterFoldEnv):
    stages = STAGES[:1]

    def __init__(self, max_episode_steps=HALF_FOLD_MAX_STEPS, seed=None, domain_randomization=True,
                 obs_mode="state", cameras=None, **kwargs):
        """obs_mode="state": the 139-D vector (imitation.md section 2).
        obs_mode="dict":  {"state": 139-D, <camera>: uint8 [3, H, W], ...} -- the sensor-only
        student's view (roadmap M4.1). `cameras` maps camera name -> square size (default
        `imitation.vision.render.CAMERAS`); per-episode visual randomization is drawn from
        the reset seed and only touches rendering, so `state` is identical in both modes."""
        super().__init__(max_episode_steps=max_episode_steps, seed=seed, **kwargs)
        self.unwrapped.domain_randomization = domain_randomization
        self.obs_mode, self.rig = obs_mode, None
        self.last_render_s = 0.0          # profiling (M5c.1): time of the last camera render
        if obs_mode == "dict":
            from imitation.vision.render import CAMERAS, CameraRig
            self.rig = CameraRig(self, cameras or CAMERAS)
            self.observation_space = gym.spaces.Dict(
                {"state": self.observation_space,
                 **{c: gym.spaces.Box(0, 255, (3, n, n), np.uint8) for c, n in self.rig.cameras.items()}})
        elif obs_mode != "state":
            raise ValueError(obs_mode)

    def _wrap(self, obs):
        if not self.rig:
            return obs
        t0 = time.perf_counter()
        images = self.rig.render()
        self.last_render_s = time.perf_counter() - t0
        return {"state": obs, **images}

    def reset(self, seed=None, options=None):
        # the base env samples a random task one-hot into the observation; for
        # a single-task policy that is pure noise, so pin it
        opts = dict(options or {})
        opts.setdefault("task", 0)
        obs, info = super().reset(seed=seed, options=opts)
        if self.rig:
            self.rig.reset(0 if seed is None else seed)
        return self._wrap(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._wrap(obs), r, term, trunc, info

    def step_state(self, action):
        """step() without rendering: for teacher look-ahead that is rolled back anyway."""
        return super().step(action)


def make_env(**kwargs):
    return HalfFoldEnv(**kwargs)

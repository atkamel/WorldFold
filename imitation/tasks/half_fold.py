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

from cloth_fold_rl.quarter_fold_env import HALF_FOLD_MAX_STEPS, STAGES, QuarterFoldEnv  # noqa: F401 (re-exported)
from imitation.spec import ACTION_DIM, OBS_DIM  # noqa: F401 (re-exported)


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
            if hasattr(self.unwrapped, "render_rig"):     # Isaac: the cameras are part of the scene (I1.3)
                from imitation.vision.isaac_render import IsaacCameraRig
                self.rig = IsaacCameraRig(self, cameras or CAMERAS)
            else:
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


BACKENDS = ("mujoco", "isaac", "isaac_weld")


def is_isaac(backend) -> bool:
    """True for every Isaac Sim backend ("isaac", "isaac_weld")."""
    return str(backend).startswith("isaac")


def make_env(backend="mujoco", **kwargs):
    """The half-fold env on either simulator; the only place an env should be built (Phase I).

    backend="mujoco": the MuJoCo ClothFoldEnv the pipeline was built on (default, unchanged).
    backend="isaac":  isaac.isaac_env.IsaacClothFoldEnv (Isaac Sim 5.1, friction grasp) with the Isaac
                      episode cap and jitter; needs the .venv-isaac environment and one env per process.
    backend="isaac_weld": the same Isaac env built with profile="mujoco" (weld grasp, MuJoCo arm drives); episode
                      cap (HALF_FOLD_MAX_STEPS) and cloth jitter (MuJoCo CLOTH_JITTER) are MuJoCo's.
    """
    if backend == "mujoco":
        return HalfFoldEnv(**kwargs)
    if backend == "isaac":
        from imitation.isaac_runtime import ISAAC_CLOTH_JITTER, ISAAC_MAX_STEPS, make_isaac_base
        kwargs.setdefault("max_episode_steps", ISAAC_MAX_STEPS)
        kwargs.setdefault("cloth_jitter", ISAAC_CLOTH_JITTER)
        cameras = None
        if kwargs.get("obs_mode", "state") == "dict":
            from imitation.vision.render import CAMERAS
            cameras = kwargs.get("cameras") or CAMERAS
        kwargs["base_env"] = make_isaac_base(kwargs["max_episode_steps"], cameras=cameras)
        return HalfFoldEnv(**kwargs)
    if backend == "isaac_weld":
        from cloth_fold_rl.fold_env import CLOTH_JITTER
        from imitation.isaac_runtime import ISAAC_PROFILES, make_isaac_base
        kwargs.setdefault("max_episode_steps", HALF_FOLD_MAX_STEPS)
        kwargs.setdefault("cloth_jitter", CLOTH_JITTER)
        cameras = None
        if kwargs.get("obs_mode", "state") == "dict":
            from imitation.vision.render import CAMERAS
            cameras = kwargs.get("cameras") or CAMERAS
        kwargs["base_env"] = make_isaac_base(kwargs["max_episode_steps"], cameras=cameras,
                                             profile=ISAAC_PROFILES[backend])
        return HalfFoldEnv(**kwargs)
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

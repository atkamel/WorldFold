"""Half fold: stage 0 of the quarter fold, ending in success when it settles.

Both arms carry the north corners onto the south corners (a fold about y = 0),
release, and the corners must stay placed for SETTLE_STEPS. Everything else --
observation (139-D state), action (12-D joint deltas + grippers), reward shaping,
drag/instability termination -- is inherited from QuarterFoldEnv unchanged, so
QuarterFoldExpert drives it as-is.
"""

from __future__ import annotations

import os
import time

import gymnasium as gym
import numpy as np

from cloth_fold_rl.quarter_fold_env import HALF_FOLD_MAX_STEPS, MAX_STEPS, STAGES, QuarterFoldEnv  # noqa: F401 (re-exported)
from imitation.spec import ACTION_DIM, OBS_DIM  # noqa: F401 (re-exported)


def _image_shape(n):
    """[3, H, W] of a rig camera: a square size, or an Isaac hero-camera spec {"size": (W, H), ...} (isaac/lab_scene)."""
    if isinstance(n, dict):
        return (3, int(n["size"][1]), int(n["size"][0]))
    return (3, n, n)


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
                 **{c: gym.spaces.Box(0, 255, _image_shape(n), np.uint8) for c, n in self.rig.cameras.items()}})
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


BACKENDS = ("mujoco", "isaac", "isaac_weld", "isaac_friction")


# the GPU-pipeline Isaac profiles: MuJoCo's eval sets, jitter and DR; vectorised (milestone V)
GPU_BACKENDS = ("isaac_weld", "isaac_friction")


TASKS = ("half_fold", "quarter_fold")


def _fold_over(backend):
    """The quarter fold's second fold is the bimanual fold-over (Q2) on LeHome's task layout: there the arms reach
    the west edge only with STAGE1_FOLD_OVER's roles. MuJoCo and the weld profile keep STAGES[1]."""
    from imitation.seeds import LEHOME_TASK_BACKENDS
    return backend in LEHOME_TASK_BACKENDS


def task_stages(task="half_fold", backend="mujoco"):
    """The stages a task runs on a backend. One place to swap a stage definition."""
    if task == "half_fold":
        return STAGES[:1]
    if task == "quarter_fold":
        from cloth_fold_rl.quarter_fold_env import STAGES_FOLD_OVER, STAGES_FOLD_OVER_STRAIGHTEN
        if not _fold_over(backend):
            return STAGES
        if _second() == "parallel":
            from cloth_fold_rl.quarter_fold_env import STAGES_PARALLEL
            return STAGES_PARALLEL
        if _second() == "left_fold":
            from cloth_fold_rl.quarter_fold_env import STAGES_LEFT_FOLD
            return STAGES_LEFT_FOLD
        if _second() == "left_fold_square":
            from cloth_fold_rl.quarter_fold_env import STAGES_LEFT_FOLD_SQUARE
            return STAGES_LEFT_FOLD_SQUARE
        if _second() == "left_fold_push":
            from cloth_fold_rl.quarter_fold_env import STAGES_LEFT_FOLD_PUSH
            return STAGES_LEFT_FOLD_PUSH
        if _second() == "reversed_square":
            from cloth_fold_rl.quarter_fold_env import STAGES_REVERSED_SQUARE
            return STAGES_REVERSED_SQUARE
        return STAGES_FOLD_OVER_STRAIGHTEN if _straighten() else STAGES_FOLD_OVER
    raise ValueError(f"task must be one of {TASKS}, got {task!r}")


def _second():
    """The quarter fold's second fold on LeHome backends. "parallel" (default since 2026-10-14, user: the strip folded
    in half again along the same axis, 7/16 untuned against 0/16 for the fold-over; results.md); "fold_over" (the
    bimanual west-onto-east fold, kept for follow-on work) via WORLDFOLD_QF_SECOND=fold_over; "left_fold" (R2, user
    2026-10-15: the square by the left arm alone, the right arm anchoring the east half); "left_fold_square" (that, then
    the right arm squares the folded corners up, two more stages); "left_fold_push" (the square-up as drag-only pushes
    with the left arm anchoring at the fold line); "reversed_square" (fold 1 reversed -- near corners onto far ones, the
    strip away from the arm bases -- then the rough left fold and the right-arm square-up)."""
    import os
    return os.environ.get("WORLDFOLD_QF_SECOND", "parallel")


def _straighten():
    """WORLDFOLD_QF_STRAIGHTEN=1: the fold-over quarter fold with a straighten stage between the folds (Q3 trial)."""
    import os
    return os.environ.get("WORLDFOLD_QF_STRAIGHTEN", "0") == "1"


def task_grasp_corners(task="half_fold", backend="mujoco"):
    """Per-arm corners the base env may grasp (the wrapper's per-stage mask narrows them)."""
    from cloth_fold_rl.quarter_fold_env import (GRASP_CORNERS, GRASP_CORNERS_FOLD_OVER,
                                                GRASP_CORNERS_FOLD_OVER_STRAIGHTEN)
    if task == "quarter_fold" and _fold_over(backend):
        if _second() == "parallel":
            from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS_PARALLEL
            return GRASP_CORNERS_PARALLEL
        if _second() == "left_fold":
            from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS_LEFT_FOLD
            return GRASP_CORNERS_LEFT_FOLD
        if _second() in ("left_fold_square", "left_fold_push"):
            from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS_LEFT_FOLD_SQUARE
            return GRASP_CORNERS_LEFT_FOLD_SQUARE
        if _second() == "reversed_square":
            from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS_REVERSED_SQUARE
            return GRASP_CORNERS_REVERSED_SQUARE
        return GRASP_CORNERS_FOLD_OVER_STRAIGHTEN if _straighten() else GRASP_CORNERS_FOLD_OVER
    return GRASP_CORNERS


def _gpu_cap(backend, task="half_fold"):
    if backend == "isaac_friction":
        if task == "quarter_fold":
            import os
            from imitation.isaac_runtime import FRICTION_QF_MAX_STEPS
            # WORLDFOLD_QF_MAX_STEPS: a longer cap for the 4-stage square-up variant (R2); unset = the default
            return int(os.environ.get("WORLDFOLD_QF_MAX_STEPS") or FRICTION_QF_MAX_STEPS)
        from imitation.isaac_runtime import FRICTION_MAX_STEPS
        return FRICTION_MAX_STEPS
    return MAX_STEPS if task == "quarter_fold" else HALF_FOLD_MAX_STEPS


def _jitter(backend):
    """Reset cloth jitter: LeHome's 1 cm for the backends on LeHome's task distribution (imitation.seeds), else MuJoCo's."""
    from imitation.seeds import LEHOME_TASK_BACKENDS
    if backend in LEHOME_TASK_BACKENDS:
        from imitation.isaac_runtime import ISAAC_CLOTH_JITTER
        return ISAAC_CLOTH_JITTER
    from cloth_fold_rl.fold_env import CLOTH_JITTER
    return CLOTH_JITTER


def is_isaac(backend) -> bool:
    """True for every Isaac Sim backend ("isaac", "isaac_weld", "isaac_friction")."""
    return str(backend).startswith("isaac")


# QD-2 (user 2026-10-16: the releases are violent; the second fold's friction grip is the problem). The quarter fold on
# isaac_friction lets go gently and picks up the second fold low and slow (results.md QD-2; tune_id_easy n = 16:
# 16/16, 15 real second folds, against 9/16 and 5). Jaw: after a close it opens only to 0.6 rad, at 0.04 rad/step,
# until the next grasp. Expert: the closed jaw holds 6 steps at the place, the open dwell covers the slow opening, the
# arm rises at 30% speed; stage 1 lifts at most half its (short) move, at 50% speed. The half fold is unchanged (not
# re-measured). WORLDFOLD_QF_GENTLE=0 turns it off; WORLDFOLD_EXPERT_PARAMS / WORLDFOLD_GRASP_KNOBS still override.
QF_GENTLE_GRASP = {"release_open": 0.6, "release_open_steps": 80, "jaw_open_rate": 0.04}
QF_GENTLE_EXPERT = {"OPEN_DWELL": 18, "SETTLE_DWELL": 6, "RETREAT_SPEED": 0.3, "STAGE1_ARC_HEIGHT": 0.06,
                    "STAGE1_CARRY_SPEED": 0.5}
# R1 (user 2026-10-17: adopt the fold-1 friction change). Fold 1 left a 7-9 cm strip because the laid flap slid across
# the bottom layer (cloth on cloth at 0.5 x 0.6 = 0.3) and pushed the crease. With the cloth at 1.0 against the rigids
# and 2.0 against itself (adhesion stays 0.1, no weld) and the stage-0 flat arc / chord cap / 11 cm lay-down, the strip
# is 13.6 / 13.0 cm on n = 8 (results.md R1). Quarter fold on isaac_friction only, on top of the gentle preset;
# WORLDFOLD_QF_FOLD1=0 turns it off (the QD-2 preset, which the n = 100 QD-2 evals and isaac_qf_v1 were made with).
QF_FOLD1_GRASP = {"particle_friction": 1.0, "particle_friction_scale": 2.0}
QF_FOLD1_EXPERT = {"QF0_ARC_FLAT": 2, "QF0_ARC_HEIGHT": 0.10, "QF0_CHORD_CAP": 0.04, "QF0_LAYDOWN": 0.11,
                   "QF0_OVERSHOOT_SCALE": 0}


def qf_gentle(task, backend):
    return task == "quarter_fold" and backend == "isaac_friction" and os.environ.get("WORLDFOLD_QF_GENTLE", "1") != "0"


def qf_fold1(task, backend):
    return qf_gentle(task, backend) and os.environ.get("WORLDFOLD_QF_FOLD1", "1") != "0"


def qf_preset(task, backend):
    """(grasp knobs, expert defaults) for this task on this backend: the gentle preset plus the fold-1 change, or
    (None, None) when neither applies (the half fold, other backends, or WORLDFOLD_QF_GENTLE=0)."""
    if not qf_gentle(task, backend):
        return None, None
    grasp, expert = dict(QF_GENTLE_GRASP), dict(QF_GENTLE_EXPERT)
    if qf_fold1(task, backend):
        grasp.update(QF_FOLD1_GRASP)
        expert.update(QF_FOLD1_EXPERT)
    return grasp, expert


def make_env(backend="mujoco", task="half_fold", **kwargs):
    """The half-fold env on either simulator; the only place an env should be built (Phase I).

    backend="mujoco": the MuJoCo ClothFoldEnv the pipeline was built on (default, unchanged).
    backend="isaac":  isaac.isaac_env.IsaacClothFoldEnv (Isaac Sim 5.1, friction grasp) with the Isaac
                      episode cap and jitter; needs the .venv-isaac environment and one env per process.
    backend="isaac_weld": the same Isaac env built with profile="weld" (weld grasp, MuJoCo arm drives); episode
                      cap (HALF_FOLD_MAX_STEPS) and cloth jitter (MuJoCo CLOTH_JITTER) are MuJoCo's.
    backend="isaac_friction": as isaac_weld, with profile="friction" (the jaws hold the cloth by contact, no weld) and
                      the FRICTION_MAX_STEPS cap (Phase F).
    """
    kwargs["stages"] = task_stages(task, backend)
    kwargs["carry_obs"] = task == "quarter_fold"         # obs v2 (imitation.spec.obs_dim)
    if backend == "mujoco":
        if task == "quarter_fold":
            kwargs.setdefault("max_episode_steps", MAX_STEPS)
        return HalfFoldEnv(**kwargs)
    if backend == "isaac":
        from imitation.isaac_runtime import ISAAC_CLOTH_JITTER, ISAAC_MAX_STEPS, make_isaac_base
        kwargs.setdefault("max_episode_steps", ISAAC_MAX_STEPS * (2 if task == "quarter_fold" else 1))
        kwargs.setdefault("cloth_jitter", ISAAC_CLOTH_JITTER)
        cameras = None
        if kwargs.get("obs_mode", "state") == "dict":
            from imitation.vision.render import CAMERAS
            cameras = kwargs.get("cameras") or CAMERAS
        kwargs["base_env"] = make_isaac_base(kwargs["max_episode_steps"], cameras=cameras,
                                             grasp_corners=task_grasp_corners(task, backend))
        return HalfFoldEnv(**kwargs)
    if backend in GPU_BACKENDS:
        from imitation.isaac_runtime import ISAAC_PROFILES, make_isaac_base
        kwargs.setdefault("max_episode_steps", _gpu_cap(backend, task))
        kwargs.setdefault("cloth_jitter", _jitter(backend))
        cameras = None
        if kwargs.get("obs_mode", "state") == "dict":
            from imitation.vision.render import CAMERAS
            cameras = kwargs.get("cameras") or CAMERAS
        grasp, expert = qf_preset(task, backend)
        kwargs["base_env"] = make_isaac_base(kwargs["max_episode_steps"], cameras=cameras,
                                             profile=ISAAC_PROFILES[backend],
                                             grasp_corners=task_grasp_corners(task, backend), grasp_knobs=grasp)
        if expert is not None:
            kwargs["base_env"].expert_defaults = expert
        return _diag_start(HalfFoldEnv(**kwargs), task, backend)
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


def make_env_batch(backend="isaac_weld", n=1, task="half_fold", **kwargs):
    """n half-fold envs sharing one Isaac scene (milestone V): -> (batch, [HalfFoldEnv over sub-env i]).

    Each env is built exactly as make_env(backend, **kwargs) builds one, except that its base env is
    sub-env i of an isaac.isaac_env.IsaacClothFoldBatch; physics advances only through the batch (see
    imitation/lockstep.py). Only the GPU weld profile (isaac_weld) is vectorised."""
    if backend not in GPU_BACKENDS:
        raise ValueError(f"only {GPU_BACKENDS} run several envs per process, got {backend!r}")
    from imitation.isaac_runtime import ISAAC_PROFILES, make_isaac_batch
    kwargs["stages"] = task_stages(task, backend)
    kwargs["carry_obs"] = task == "quarter_fold"
    kwargs.setdefault("max_episode_steps", _gpu_cap(backend, task))
    kwargs.setdefault("cloth_jitter", _jitter(backend))
    cameras = None
    if kwargs.get("obs_mode", "state") == "dict":
        from imitation.vision.render import CAMERAS
        cameras = kwargs.get("cameras") or CAMERAS
    # QD-2: the gentle preset applies on this path too (collect and multi-env evals build their envs here). Until
    # 2026-10-16 22:00 only make_env had it, so the first Q4 collect ran the old release (results.md QD-3).
    grasp, expert = qf_preset(task, backend)
    batch = make_isaac_batch(n, kwargs["max_episode_steps"], cameras=cameras, profile=ISAAC_PROFILES[backend],
                             grasp_corners=task_grasp_corners(task, backend), grasp_knobs=grasp)
    if expert is not None:
        for sub in batch.envs:
            sub.expert_defaults = dict(expert)
    return batch, [_diag_start(HalfFoldEnv(**dict(kwargs, base_env=sub)), task, backend) for sub in batch.envs]


def _diag_start(env, task, backend):
    """R0 diagnostic: WORLDFOLD_QF_START=clean starts the isaac_friction quarter fold at stage 1 from a perfect half
    fold (isaac.diag_clean_fold). Off by default; the env is returned unchanged."""
    import os
    if os.environ.get("WORLDFOLD_QF_START", "") != "clean":
        return env
    from isaac.diag_clean_fold import maybe_install
    return maybe_install(env, task, backend)

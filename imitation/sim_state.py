"""Snapshot / restore of a running fold env, so a teacher can roll forward from a
student-visited state and the env can be put back exactly where it was.

MuJoCo's own integration state (qpos, qvel, act, warmstart, ctrl, eq_active, ...)
is covered by mj_getState(mjSTATE_INTEGRATION). On top of that the fold envs keep
Python-side state that also decides what happens next:
  model.eq_data      weld offsets, rewritten each time a grasp engages
  _gripper_closed    gripper hysteresis (commands in (-0.3, 0.3) hold the state)
  counters           _step_count / _success_steps / _action_clipped
  weld_mask          which vertices each arm may weld this stage
  goal keypoints     _goal_corners / _goal_scale, reset per stage by the wrapper
  wrapper fields     stage, settle counter, stage-start cloth pose, previous potential
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
_BASE_FIELDS = ("_gripper_closed", "_step_count", "_success_steps", "_action_clipped", "weld_mask",
                "_goal_corners", "_goal_scale")
_WRAPPER_FIELDS = ("stage", "_settle_steps", "_stage_start", "_start", "_prev_potential")


@dataclass
class SimSnapshot:
    physics: np.ndarray
    eq_data: np.ndarray
    base: dict
    wrapper: dict


def _wrapper_chain(env):
    while hasattr(env, "env"):
        yield env
        env = env.env


def snapshot(env) -> SimSnapshot:
    base = env.unwrapped
    physics = np.empty(mujoco.mj_stateSize(base.model, _SPEC))
    mujoco.mj_getState(base.model, base.data, physics, _SPEC)
    wrapper = {}
    for i, w in enumerate(_wrapper_chain(env)):
        for f in _WRAPPER_FIELDS:
            if f in vars(w):
                wrapper[(i, f)] = copy.deepcopy(vars(w)[f])
    return SimSnapshot(physics=physics, eq_data=base.model.eq_data.copy(),
                       base={f: copy.deepcopy(getattr(base, f)) for f in _BASE_FIELDS}, wrapper=wrapper)


def restore(env, snap: SimSnapshot) -> None:
    base = env.unwrapped
    mujoco.mj_setState(base.model, base.data, snap.physics, _SPEC)
    base.model.eq_data[:] = snap.eq_data
    for f, v in snap.base.items():
        setattr(base, f, copy.deepcopy(v))
    for i, w in enumerate(_wrapper_chain(env)):
        for f in _WRAPPER_FIELDS:
            if (i, f) in snap.wrapper:
                setattr(w, f, copy.deepcopy(snap.wrapper[(i, f)]))
    # derived quantities (xpos, site_xpos, ...) must match the restored qpos
    mujoco.mj_forward(base.model, base.data)

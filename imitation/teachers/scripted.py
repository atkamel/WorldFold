"""Scripted teacher: QuarterFoldExpert with phase re-inference and sim snapshots.

act()          the expert driving from the start of an episode (demo collection).
label_chunk()  from ANY state, e.g. one the student reached: snapshot the env,
               re-infer each arm's phase from the sim, run the expert forward
               `horizon` steps recording its actions, restore the env. If the
               episode ends inside the chunk, the rest is padded with the last
               action (grippers keep their command, joints are zeroed).
"""

from __future__ import annotations

import copy

import numpy as np

from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
from imitation.sim_state import restore, snapshot
from imitation.teachers.base import Teacher

# FoldExpert fields that change while it acts (the rest is configuration)
_PHASE_FIELDS = ("phase", "q_target", "phase_steps", "retreat_target", "release_allowed", "rng")


class ScriptedTeacher(Teacher):
    def __init__(self, env, seed=0):
        self.env = env
        self.expert = QuarterFoldExpert(env, seed=seed)

    def reset(self, env=None) -> None:
        self.expert.reset()

    def act(self, env=None) -> np.ndarray:
        return self.expert.act()

    def phases(self) -> dict:
        return self.expert.phases()

    def label_chunk(self, env=None, horizon: int = 16) -> np.ndarray:
        env = self.env
        snap = snapshot(env)
        saved = self._save_expert()
        self.expert.resync()
        actions = []
        try:
            for _ in range(horizon):
                a = self.expert.act()
                actions.append(a)
                _, _, terminated, truncated, _ = env.step(a)
                if terminated or truncated:
                    break
        finally:
            restore(env, snap)
            self._restore_expert(saved)
        chunk = np.zeros((horizon, len(actions[0])), dtype=np.float32)
        chunk[:len(actions)] = actions
        if len(actions) < horizon:        # hold still, keep the gripper commands
            pad = np.zeros_like(chunk[0])
            pad[[5, 11]] = actions[-1][[5, 11]]
            chunk[len(actions):] = pad
        return chunk

    def label(self, env=None) -> np.ndarray:
        return self.label_chunk(env, horizon=1)[0]

    # the expert's phase machine, IK caches and rngs, so labelling is side-effect free
    def _save_expert(self):
        e = self.expert
        return {"experts": {k: {f: copy.deepcopy(getattr(x, f)) for f in _PHASE_FIELDS} for k, x in e.experts.items()},
                "correction": copy.deepcopy(e.correction), "retries": dict(e.retries),
                "done_steps": dict(e.done_steps)}

    def _restore_expert(self, saved):
        e = self.expert
        for k, state in saved["experts"].items():
            for f, v in state.items():
                setattr(e.experts[k], f, v)
        e.correction.clear()
        e.correction.update(saved["correction"])
        e.retries, e.done_steps = saved["retries"], saved["done_steps"]

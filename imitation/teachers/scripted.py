"""Scripted teacher: QuarterFoldExpert with phase re-inference and sim snapshots.

act()          the expert driving from the start of an episode (demo collection).
label_chunk()  from ANY state, e.g. one the student reached: snapshot the env,
               work out each arm's phase, run the expert forward `horizon` steps
               recording its actions, restore the env. If the episode ends inside
               the chunk, the rest is padded with the last action (grippers keep
               their command, joints are zeroed).

Phase tracking (M3.2). Re-inferring every arm's phase from geometry at each label
("resync from scratch") disagreed with the live expert on its *own* trajectory -- up to
0.57 per joint during carry -- because the live machine advances on conditions the
geometry does not show (carry -> place fires when the wrist reaches the carry target,
while the hanging corner lags) and a fresh 5-DOF position-only IK solve can pick another
joint configuration. So while someone else drives, the teacher *shadows* the episode:
`observe()` runs its phase machine on every step and discards the action, so its phases,
IK targets and timers evolve exactly as if it were acting. At a label, an arm's shadow
state is kept unless geometry puts it in a different phase *group* (pre-grasp / holding /
released) -- the student dropped a corner, grasped, or let go -- and only then is that
arm re-inferred from the sim.
"""

from __future__ import annotations

import copy

import numpy as np

from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
from imitation.sim_state import restore, snapshot
from imitation.teachers.base import Teacher

# FoldExpert fields that change while it acts (the rest is configuration)
_PHASE_FIELDS = ("phase", "q_target", "phase_steps", "retreat_target", "release_allowed", "rng")
_GROUPS = ({"approach", "descend"}, {"lift", "carry", "place", "hold"}, {"release", "retreat", "done"})


def _group(name):
    return next(i for i, g in enumerate(_GROUPS) if name in g)


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
        self._sync_phases()
        actions = []
        try:
            for _ in range(horizon):
                a = self.expert.act()
                actions.append(a)
                _, _, terminated, truncated, _ = getattr(env, "step_state", env.step)(a)
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

    def observe(self, env=None) -> None:
        """Shadow one step someone else is about to take (call before env.step)."""
        self.expert.act()

    def see(self, obs) -> None:
        """A new observation after reset/step. The scripted teacher reads the sim instead."""

    def _sync_phases(self):
        """Keep each arm's shadow state unless geometry says its phase group changed."""
        e = self.expert
        for key, x in e.experts.items():
            if key[0] != e.env.stage:
                continue
            kept = {f: copy.deepcopy(getattr(x, f)) for f in _PHASE_FIELDS}
            x.infer_phase(placed=e.env._placed(e.moves[key]))
            if _group(x.PHASES[x.phase]) == _group(x.PHASES[kept["phase"]]):
                for f, v in kept.items():
                    setattr(x, f, v)
            else:
                e.done_steps[key] = 0
        e._update_release_gate()

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

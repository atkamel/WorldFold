"""Scripted teacher on Isaac Sim: isaac.half_fold_expert.IsaacHalfFoldExpert behind the Teacher interface.

The expert decides every step from the live state, so there is nothing to shadow while someone else drives and
nothing to resync after; its labels come from a kinematic look-ahead instead of a sim snapshot (see that module).
"""

from __future__ import annotations

import numpy as np

from imitation.teachers.base import Teacher


class IsaacScriptedTeacher(Teacher):
    def __init__(self, env, seed=0):
        from isaac.half_fold_expert import IsaacHalfFoldExpert
        self.env = env
        self.expert = IsaacHalfFoldExpert(env)

    def reset(self, env=None) -> None:
        self.expert.reset()

    def act(self, env=None) -> np.ndarray:
        return self.expert.act()

    def phases(self) -> dict:
        return self.expert.phases()

    def label_chunk(self, env=None, horizon: int = 16) -> np.ndarray:
        return self.expert.label_chunk(horizon)

    def label(self, env=None) -> np.ndarray:
        return self.label_chunk(env, horizon=1)[0]

    def observe(self, env=None) -> None:
        """Markov expert: nothing to shadow."""

    def see(self, obs) -> None:
        """The scripted teacher reads the sim instead."""

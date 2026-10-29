"""Teacher interface. A teacher labels states of a live env with expert actions.

The scripted teacher reads privileged sim state; a VLA or teleop teacher would
implement the same two calls from observations instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Teacher(ABC):
    @abstractmethod
    def reset(self, env) -> None:
        """Start of an episode that the teacher drives from the beginning."""

    @abstractmethod
    def act(self, env) -> np.ndarray:
        """Next action when the teacher is in control (stateful, advances internal phase)."""

    @abstractmethod
    def label_chunk(self, env, horizon: int) -> np.ndarray:
        """Expert actions for the next `horizon` steps from the env's CURRENT state,
        which the teacher may not have produced (e.g. a student-visited state).
        Must leave the env exactly as it found it. Returns [horizon, action_dim]."""

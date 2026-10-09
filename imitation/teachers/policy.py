"""A trained policy as the teacher (roadmap M4.2: `PrivilegedPolicyTeacher`).

Implements the same `Teacher` interface as the scripted expert, so the Phase 3 DAgger
loop runs unchanged with either: the worker hosts the teacher, feeds it every new
observation (`see`), and asks it for `label_chunk` at replan points. The teacher is a
*privileged* state policy (full 139-D observation) teaching a sensor-only student --
distillation is DAgger where the teacher is a network. Inference runs on the worker
CPU; with a 2-step history an 8.6M-parameter chunk MLP takes a few ms per label.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from imitation.teachers.base import Teacher


class PolicyTeacher(Teacher):
    def __init__(self, env, ckpt, replan_every=8):
        import torch
        from imitation.policies.common import load_policy
        torch.set_num_threads(1)
        self.env, self.ckpt, self.replan_every = env, str(ckpt), replan_every
        self.policy = load_policy(ckpt, "cpu")
        if self.policy.needs_images:
            raise ValueError("a teacher must be a state policy (it labels from the privileged obs)")
        self.hist = None
        self.queue = deque()

    def reset(self, env=None) -> None:
        self.hist, self.queue = None, deque()

    def see(self, obs) -> None:
        """Every new observation (after reset and after each step)."""
        obs = np.asarray(obs["state"] if isinstance(obs, dict) else obs, dtype=np.float32)
        if self.hist is None:
            self.hist = deque([obs] * self.policy.obs_horizon, maxlen=self.policy.obs_horizon)
        else:
            self.hist.append(obs)

    def observe(self, env=None) -> None:
        """Someone else is about to act; a policy teacher's state is its obs history."""

    def _chunk(self):
        return self.policy.predict(np.stack(self.hist)[None])[0]

    def act(self, env=None) -> np.ndarray:
        if not self.queue:
            self.queue.extend(self._chunk()[:self.replan_every])
        return self.queue.popleft()

    def label_chunk(self, env=None, horizon: int = 16) -> np.ndarray:
        chunk = self._chunk()
        if len(chunk) < horizon:      # a shorter-chunk teacher: hold still, keep the grippers
            pad = np.zeros((horizon - len(chunk), chunk.shape[1]), np.float32)
            pad[:, [5, 11]] = chunk[-1, [5, 11]]
            chunk = np.concatenate([chunk, pad])
        return chunk[:horizon].astype(np.float32)

    def phases(self) -> dict:
        return {}

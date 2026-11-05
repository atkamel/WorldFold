"""The Isaac half fold's scripted base, one env: isaac.rl.precision_env.PrecisionBatch's driver without the copies.

QuarterFoldEnv's stage 0 (cloth_10 onto cloth_0, cloth_120 onto cloth_110, release, settle) over an
IsaacClothFoldEnv, the cloth jittered by the half fold's +-1 cm. Each arm is an IsaacPlaceServoExpert: IsaacArmExpert's
pinch, lift and carry start, corner steering onto the live partner vertex from the carry on, and the gentle release
(place_servo.RELEASE_JAW). precision_env.coordinate gates both: no arm lifts before both have closed, both release
once both hold.
"""

from __future__ import annotations

import numpy as np

from cloth_fold_rl.quarter_fold_env import STAGES, QuarterFoldEnv
from isaac.experts.place_servo import IsaacPlaceServoExpert
from isaac.rl.precision_env import ARMS, coordinate
from isaac.sim.isaac_env import CLOTH_JITTER


class ScriptedHalfFold:
    def __init__(self, base):
        self.task = QuarterFoldEnv(base_env=base, cloth_jitter=CLOTH_JITTER, stages=STAGES[:1])
        self.experts = {m.prefix: IsaacPlaceServoExpert(self.task, prefix=m.prefix, corner=list(m.corners),
                                                        release=True, goal=lambda m=m: self.task.goal(m),
                                                        live_goal=lambda m=m: base.cloth_positions()[m.goal])
                        for m in STAGES[0].moves}

    def reset(self, seed):
        """Reset the task (its seed draws the cloth offset and the dynamics DR) and both experts -> reset info."""
        _, info = self.task.reset(seed=seed)
        for k, e in enumerate(self.experts.values()):
            e.reset()
            e.rng = np.random.default_rng(seed + k)
            e.release_allowed = False
        return info

    def act(self):
        """The 12-dim action: both arms' experts after the gates."""
        coordinate(self.experts)
        return np.concatenate([self.experts[p].act() for p in ARMS]).astype(np.float32)

    def cycles(self):
        """Each arm's carried corner from the first state on, in play's cycle format (for the manipulated vertex)."""
        return [{"arm": m.prefix, "start": 0, "vertex": m.corners[0]} for m in STAGES[0].moves]

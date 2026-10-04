"""The FoldExpert (written for MuJoCo) on the Isaac env's weld profile (Phase W, W3).

cloth_fold_rl.expert.FoldExpert's phase machine unchanged -- approach (corner + 6 cm), descend (+ 0.5 cm, wait for the
weld), lift to table + 12 cm, carry, place (goal + 2 cm), hold, release, retreat; a phase advances within 2 cm or
after 45 steps -- with its sim reads taken from IsaacClothFoldEnv's accessors and its position-only IK solved by
isaac.pinch.PinchIK.solve_position (the same damped least squares, 0.1 rad step cap, 12 restarts and 6 mm tolerance,
on LeHome's SO101 kinematics, which match the MJCF's to 0.3 mm). QuarterFoldExpert(expert_cls=IsaacFoldExpert)
supplies the overshoot, retries and release gate, as on MuJoCo. Needs grasp_mode="weld" (profile "weld").
"""

from __future__ import annotations

import numpy as np

from cloth_fold_rl.expert import FoldExpert

# per (stage, arm) placement offset past the goal: minus the mean miss of a corner placed exactly on its goal, measured
# on the Isaac weld profile with the MuJoCo-calibrated weld as MuJoCo's OVERSHOOT was on the stock cloth: 20
# tune-block seeds (600000-600019), sd ~1 cm (isaac/overshoot_measure.py, docs/results.md W3). Under 5 mm: the Isaac
# cloth barely springs back, unlike MuJoCo's flexcomp ((-0.04, -0.03) / (0.02, -0.03)). Stage 1 (quarter fold, not run
# on Isaac) keeps MuJoCo's.
from cloth_fold_rl.quarter_fold_expert import OVERSHOOT   # noqa: E402

OVERSHOOT_ISAAC = {key: np.array(v, dtype=float) for key, v in OVERSHOOT.items()}
OVERSHOOT_ISAAC[(0, "left_")] = np.array([0.000, 0.004, 0.0])
OVERSHOOT_ISAAC[(0, "right_")] = np.array([0.002, 0.002, 0.0])


class IsaacFoldExpert(FoldExpert):

    _pinch = None            # one PinchIK per process (it parses LeHome's URDF)

    def _init_sim(self, raw_vertex):
        if not raw_vertex:
            raise ValueError("IsaacFoldExpert indexes cloth_positions() by grid vertex (raw_vertex=True)")
        if getattr(self.base, "grasp_mode", None) != "weld":
            raise ValueError("IsaacFoldExpert needs the weld grasp (IsaacClothFoldEnv profile 'weld')")
        if IsaacFoldExpert._pinch is None:
            from isaac.pinch import PinchIK
            IsaacFoldExpert._pinch = PinchIK()

    def _site(self):
        return np.asarray(self.base.gripper_position(self.prefix), dtype=float)

    def _q_now(self):
        return np.asarray(self.base.joint_positions(self.prefix)[:5], dtype=float)

    def _corner(self):
        cloth = self.base.cloth_positions()
        return np.mean([cloth[c] for c in self.corners], axis=0)

    # W3b knobs (None = FoldExpert's behaviour). Tuned on recovery replays of tune seeds 610000+ (docs/results.md W3b).
    # REGRASP_OFFSET: above this sideways offset of a held corner, aim the gripper so the corner reaches the target.
    #   Attempt A (0.02, lift/carry/place): 30/40 vs 35/40 baseline, rejected.
    # RETRY_LIFT / RETRY_APPROACH: a retry (QuarterFoldExpert resets the expert mid-episode to nudge a placed corner)
    #   repeats FoldExpert's full motion -- approach 6 cm above, lift to table + 12 cm -- for a corner that is already
    #   near its goal; these lower the retry's lift (above the table) and approach (above the corner), in metres.
    #   Attempt B (0.05 / 0.03): 34/40, more but cheaper retries, rejected.
    REGRASP_OFFSET = None
    REGRASP_PHASES = ("lift", "carry", "place")   # where REGRASP_OFFSET applies; attempt C: lift only
    RETRY_LIFT = None
    RETRY_APPROACH = None

    def reset(self):
        super().reset()
        # a reset after the episode's first step is QuarterFoldExpert's retry, not a new episode
        self.retrying = int(getattr(self.base, "_step_count", 0)) > 0

    def _plan(self):
        from mujuco.cloth_params import TABLE_TOP_Z
        target = super()._plan()
        if target is None:
            return None
        name = self.PHASES[self.phase]
        if getattr(self, "retrying", False):
            if name == "approach" and self.RETRY_APPROACH is not None:
                target = self._corner() + np.array([0.0, 0.0, self.RETRY_APPROACH])
            elif name in ("lift", "carry") and self.RETRY_LIFT is not None:
                target = np.array(target, dtype=float)
                target[2] = TABLE_TOP_Z + self.RETRY_LIFT
        if (self.REGRASP_OFFSET is not None and name in self.REGRASP_PHASES
                and self.base.grasp_active(self.prefix)):
            off = (self._corner() - self._site())[:2]
            if float(np.linalg.norm(off)) > self.REGRASP_OFFSET:
                target = np.array(target, dtype=float)
                target[:2] -= off
        return target

    def _ik(self, target):
        # solve_ik seeds its first try from the live arm pose, then restarts at random
        return self._pinch.solve_position(self.prefix, target, self._q_now(), rng=self.rng)

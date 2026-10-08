"""Privileged closed-loop half-fold expert (Phase F3b): IsaacArmExpert's phases, but every motion servoes on the live
cloth state instead of tracking joint waypoints planned once per phase.

Why (results.md, F3b drop traces): after a disturbance the open-loop expert's descends ended 1-4 cm from the corner
and its joint-space arrival test (TRACK_TOL) never converged with the arm folded near its base, so it timed out,
closed off the corner or ping-ponged between approach and descend until the 400-step cap. It has the exact state of
every particle; it should use it every step.

Each step in approach / descend / close / lift / carry / place:
  target   recomputed from the corner as it lies now (approach: 4 cm above the pinch point; descend / close: the pinch
           point; lift: straight up from where the jaw closed; carry / place: the GRIPPER target that moves the CORNER
           toward its next waypoint -- the corner is steered, so cloth spring-back and lag need no overshoot table)
  motion   one warm-started IK solve (isaac.pinch.PinchIK) from the current joints, joints moved toward it at the speed
           limit; if the upright pinch is out of reach the jaw is re-aimed along the arm and then allowed to tilt
  arrival  Cartesian (gripper site within ARRIVE_XY / ARRIVE_Z of the pinch point), not joint-space
  grasp    verified, not assumed: after the close, the corner must rise with the jaw; otherwise open and servo again
hold / release / retreat / done are IsaacArmExpert's. Selected with WORLDFOLD_EXPERT_PARAMS="SERVO=1" (or SERVO = 1
on IsaacArmExpert), read by imitation.teachers.scripted.ScriptedTeacher.
"""

from __future__ import annotations

import math

import numpy as np

from isaac.fold_expert import (ABOVE, ARC_HEIGHT, CORNER_REST_DZ, JAW_SHUT_TOL, ORIENTATION_WEIGHT, RETREAT,
                               TILT_WEIGHT, IsaacArmExpert)
from mujuco.cloth_params import ARM_BASE_LEFT, ARM_BASE_RIGHT, JOINT_DELTA_SCALE, TABLE_TOP_Z

SERVO_PHASES = ("approach", "descend", "close", "lift", "carry", "place")


class IsaacServoExpert(IsaacArmExpert):
    OVERSHOOT_SCALE = 0.0    # the corner is steered onto its goal: no spring-back table
    # descend -> close: the site is within ARRIVE_XY (horizontally) of the pinch point and has stopped closing in
    # (moved < STALL_MOVE over STALL_STEPS steps). The fingertip meets the table and cloth 1-1.5 cm short of the
    # nominal pinch point (first servo run: descends parked at 1.4-1.5 cm, never met a 6 mm test, and dragged the
    # cloth); the 4.3 cm pads grip from there, which is what the open-loop expert's patience close relied on
    ARRIVE_XY = 0.02
    STALL_MOVE = 0.002
    STALL_STEPS = 4
    APPROACH_TOL = 0.012     # approach -> descend
    IK_ITERS = 20            # per step, warm-started from the current joints
    ESCALATE_STEPS = 25      # a descend this long without arriving re-aims the jaw (then lets it tilt)
    GIVE_UP_STEPS = 40       # ... and after this many it closes where it is
    LIFT_RISE = 0.04
    PINCH_CHECK = 6          # lift steps before judging "did the corner rise with the jaw"
    MAX_REGRASPS = 3
    CORNER_STEP = 0.02       # corner waypoint spacing along the carry arc (m)
    CORNER_WP_TOL = 0.015
    WP_PATIENCE = 15         # steps per corner waypoint before moving on anyway
    STEER_CLIP = 0.03        # max gripper-target offset per step (m)
    PLACE_TOL = 0.012
    PLACE_PATIENCE = 40
    SLIP_STEPS = 3
    HOLD_DIST = 0.05         # = isaac.isaac_env.HOLD_DIST

    PHASE_FIELDS = IsaacArmExpert.PHASE_FIELDS + ("sv",)

    def reset(self):
        super().reset()
        self.sv = {"aim": 0, "regrasps": 0, "lift0": None, "path": None, "k": 0, "k_steps": 0, "slip": 0, "hist": []}

    # ---- geometry ---------------------------------------------------------------------------------------------------
    def _site(self):
        return np.asarray(self.base.gripper_position(self.prefix), dtype=float)

    def _jaw_dir(self, corner):
        """Jaw aim for the current attempt: cloth centre -> corner (aim 0), arm base -> corner (aim >= 1)."""
        if self.sv["aim"] == 0:
            center = self.base.cloth_positions().mean(axis=0)
            j = np.r_[center[:2] - corner[:2], 0.0]
        else:
            b = np.asarray(ARM_BASE_LEFT if self.prefix == "left_" else ARM_BASE_RIGHT, dtype=float)
            j = np.r_[b[:2] - corner[:2], 0.0]
        return j / max(np.linalg.norm(j), 1e-9)

    def _pinch_point(self, corner, jaw):
        z = max(TABLE_TOP_Z + self.PINCH_HEIGHT, float(corner[2]) - CORNER_REST_DZ + self.PINCH_HEIGHT)
        xy = corner[:2] - self.PINCH_INSET * jaw[:2]
        return np.array([xy[0], xy[1], z])

    def _toward(self, tip, jaw, weight=None):
        """Joint-delta action (5,) toward the IK solution for `tip` from the current joints, at the speed limit."""
        w = weight if weight is not None else (TILT_WEIGHT if self.sv["aim"] >= 2 else ORIENTATION_WEIGHT)
        q = self._q()
        q_des, err = self._ik.solve(self.prefix, tip, jaw, q, orientation_weight=w, iters=self.IK_ITERS)
        delta = (q_des - q) / JOINT_DELTA_SCALE
        return delta / max(1.0, float(np.max(np.abs(delta)))), err

    def _corner_path(self, corner, goal):
        """Corner waypoints from where it is to the goal: the parent's arc shape, applied to the corner."""
        n = max(3, int(np.ceil(np.linalg.norm(goal[:2] - corner[:2]) / self.CORNER_STEP)))
        start, end = corner.copy(), np.array([goal[0], goal[1], goal[2]])
        path = []
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            p = start + (end - start) * (1.0 - math.cos(math.pi * s)) / 2.0
            p[2] = start[2] + (end[2] - start[2]) * s + ARC_HEIGHT * 0.6 * math.sin(math.pi * s)
            path.append(p)
        return path

    def _steer(self, corner_target):
        """Gripper target that moves the corner toward corner_target (the corner hangs from the jaw)."""
        corner, site = self._corner(), self._site()
        off = np.clip(corner_target - corner, -self.STEER_CLIP, self.STEER_CLIP)
        return site + off

    # ---- phase control ------------------------------------------------------------------------------------------------
    def _go(self, name):
        self.sv["hist"] = []
        self.phase = self.PHASES.index(name)
        self.phase_steps = 0
        self.q_target = None

    def _regrasp(self):
        self.sv.update(regrasps=self.sv["regrasps"] + 1, aim=0, lift0=None, path=None, slip=0)
        self._go("approach")

    def _start_carry(self):
        goal = np.asarray(self.goal(), dtype=float)
        self.sv.update(path=self._corner_path(self._corner(), goal), k=0, k_steps=0, slip=0)
        self._go("carry")

    def _start_hold(self):
        q = self._q()
        site = self._site()
        jaw = self._jaw_dir(self._corner())
        retreat, _ = self._ik.solve(self.prefix, site + [0, 0, RETREAT], jaw, q, orientation_weight=0.02)
        self.plan = dict(self.plan or {}, jaw=jaw, arc=[q], arc_tips=[site], retreat=retreat)
        self.wp = 0
        self._go("hold")

    def act(self):
        name = self.PHASES[self.phase]
        if name not in SERVO_PHASES:
            return super().act()          # hold / release / retreat / done: the parent's joint targets
        action = np.zeros(6, dtype=np.float32)
        action[5] = 1.0 if name in self.OPEN_PHASES else -1.0
        corner = self._corner()
        jaw = self._jaw_dir(corner)
        pinch = self._pinch_point(corner, jaw)
        site = self._site()
        self.phase_steps += 1

        if name == "approach":
            tip = pinch + [0.0, 0.0, ABOVE]
            action[0:5], _ = self._toward(tip, jaw)
            if np.linalg.norm(site - tip) < self.APPROACH_TOL:
                self._go("descend")
        elif name == "descend":
            action[0:5], err = self._toward(pinch, jaw)
            hist = self.sv["hist"] = (self.sv["hist"] + [site.copy()])[-(self.STALL_STEPS + 1):]
            stalled = len(hist) > self.STALL_STEPS and np.linalg.norm(hist[-1] - hist[0]) < self.STALL_MOVE
            low = site[2] - pinch[2] < 0.02
            arrived = np.linalg.norm(site[:2] - pinch[:2]) < self.ARRIVE_XY and low and stalled
            if arrived:
                self._go("close")
            elif self.phase_steps >= self.ESCALATE_STEPS and self.sv["aim"] < 2:
                self.sv["aim"] += 1               # re-aim the jaw along the arm, then let it tilt
                self.phase_steps = 0
            elif self.phase_steps >= self.GIVE_UP_STEPS:
                self._go("close")                 # as close as it gets: close and let the lift check decide
        elif name == "close":
            action[0:5], _ = self._toward(pinch, jaw)
            jaw_q = float(self.base.joint_positions(self.prefix)[5])
            shut = jaw_q <= getattr(self.base, "_closed_q", jaw_q) + JAW_SHUT_TOL
            if (self.phase_steps >= self.CLOSE_DWELL and shut) or self.phase_steps >= 3 * self.CLOSE_DWELL:
                self.sv["lift0"] = (site.copy(), corner.copy())
                self._go("lift")
        elif name == "lift":
            s0, c0 = self.sv["lift0"]
            tip = s0 + [0.0, 0.0, self.LIFT_RISE]
            action[0:5], _ = self._toward(tip, jaw, weight=0.05)
            if self.phase_steps >= self.PINCH_CHECK:
                rise_j, rise_c = site[2] - s0[2], corner[2] - c0[2]
                if rise_j > 0.01 and rise_c < 0.5 * rise_j:          # the jaw rose, the corner didn't: not pinched
                    if self.sv["regrasps"] < self.MAX_REGRASPS:
                        self._regrasp()
                        action[5] = 1.0
                    else:
                        self._start_carry()
                elif site[2] - s0[2] > 0.8 * self.LIFT_RISE:
                    self._start_carry()
        elif name in ("carry", "place"):
            if np.linalg.norm(corner - site) > self.HOLD_DIST:           # the corner left the jaw: slip
                self.sv["slip"] += 1
                if self.sv["slip"] >= self.SLIP_STEPS and self.sv["regrasps"] < self.MAX_REGRASPS:
                    self._regrasp()
                    action[5] = 1.0
                    return action
            else:
                self.sv["slip"] = 0
            goal = np.asarray(self.goal(), dtype=float)
            if name == "carry":
                path, k = self.sv["path"], self.sv["k"]
                target = path[min(k, len(path) - 1)]
                action[0:5], _ = self._toward(self._steer(target), jaw, weight=0.05)
                self.sv["k_steps"] += 1
                if np.linalg.norm(corner - target) < self.CORNER_WP_TOL or self.sv["k_steps"] > self.WP_PATIENCE:
                    self.sv["k"], self.sv["k_steps"] = k + 1, 0
                if self.sv["k"] >= len(path):
                    self._go("place")
            else:
                target = goal + [0.0, 0.0, 0.005]
                action[0:5], _ = self._toward(self._steer(target), jaw, weight=0.05)
                if (np.linalg.norm(corner[:2] - goal[:2]) < self.PLACE_TOL and corner[2] - goal[2] < 0.015) \
                        or self.phase_steps >= self.PLACE_PATIENCE:
                    self._start_hold()
        if name in ("lift", "carry"):
            action[0:5] *= self.CARRY_SPEED
        return action

    # ---- resync -----------------------------------------------------------------------------------------------------
    def infer_phase(self, placed=None):
        corner = self._corner()
        goal = np.asarray(self.goal(), dtype=float)
        if placed is None:
            placed = float(np.linalg.norm(corner - goal)) < 0.05
        self.sv.update(aim=0, lift0=None, slip=0)
        if self.base.grasp_active(self.prefix):
            if float(np.linalg.norm(corner[:2] - goal[:2])) < self.PLACE_TOL:
                self._start_hold()
                return "hold"
            self._start_carry()
            return "carry"
        if placed and "done" in self.PHASES:
            return super().infer_phase(placed=placed)
        self._go("approach")
        return "approach"

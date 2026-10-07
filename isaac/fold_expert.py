"""Scripted half-fold expert for the Isaac env, with the friction grasp as built (Phase I, I2.1).

A drop-in for cloth_fold_rl.expert.FoldExpert inside QuarterFoldExpert (same constructor, phase machine fields,
act / reset / infer_phase), so the imitation pipeline's ScriptedTeacher, release gate, retry-on-measured-miss and
resync work unchanged. The motion is isaac/half_fold_demo.py's plan, unchanged (no grasp tuning in this pass): a
top-down pinch solved by PinchIK on the corner's live position, the jaw across the corner; close and dwell; an IK
waypoint arc over the fold line onto the goal; open and dwell; back off. Every pose is held until the joints get
there (the env moves each joint at most JOINT_DELTA_SCALE per step and the joints lag), with a step budget as the
per-phase patience. It reads the sim only through env accessors (cloth_positions, gripper_position,
joint_positions, grasp_active).
"""

from __future__ import annotations

import math

import numpy as np

from isaac.half_fold_demo import (ARC_HEIGHT, ARC_WAYPOINTS, PINCH_HEIGHT, PLACE_HEIGHT, POSE_SETTLE_STEPS, SEED_Q,
                                  TRACK_TOL, WAYPOINT_TOL)
from mujuco.cloth_params import JOINT_DELTA_SCALE, TABLE_TOP_Z

SUCCESS_DIST = 0.05          # = cloth_fold_rl.fold_env.SUCCESS_DIST (placed)
CLOSE_DWELL = 8              # half_fold_demo SCHEDULE: ("pinch", -1.0, 8)
OPEN_DWELL = 6               # ("arc", 1.0, 6)
LIFT_WAYPOINTS = 3           # the first arc waypoints count as "lift" (phase groups for the teacher)
ABOVE = 0.04                 # pre-pinch height above the pinch point
RETREAT = 0.05
# per (stage, arm): metres past the goal to place at, the friction grasp's measured spring-back (cf.
# cloth_fold_rl.quarter_fold_expert.OVERSHOOT on MuJoCo, isaac.weld_expert.OVERSHOOT_ISAAC on the weld profile).
# Stage 0 is calibrated (track G, IG.2 attempt 8: minus the settled miss of attempt 3); stage 1 stays zero.
OVERSHOOT_FRICTION = {key: np.zeros(3) for key in ((0, "left_"), (0, "right_"), (1, "left_"), (1, "right_"))}
OVERSHOOT_FRICTION[(0, "left_")][:] = (-0.001, 0.052, 0.0)
OVERSHOOT_FRICTION[(0, "right_")][:] = (0.013, 0.050, 0.0)
# The "friction" profile (Phase F) keeps #9's offsets. Subtracting F2-A's mean settled miss (left (+1.9, +0.8) cm,
# right (-1.3, +1.5) cm) made it worse: F2 attempts F and G, which both carried it, held 11/20 and placed 10/20 on the
# right arm against A's 16 / 15 (results.md). The landing point is not independent of the target.
OVERSHOOT_FRICTION_PROFILE = {key: v.copy() for key, v in OVERSHOOT_FRICTION.items()}
OVERSHOOTS = {"lehome": OVERSHOOT_FRICTION, "friction": OVERSHOOT_FRICTION_PROFILE, "anchor": OVERSHOOT_FRICTION_PROFILE}


JAW_SHUT_TOL = 0.05          # = isaac_env.JAW_SETTLED_TOL
JAW_STILL = 0.01             # rad per control step


class IsaacArmExpert:
    PHASES = ("approach", "descend", "close", "lift", "carry", "place", "hold")
    RELEASE_PHASES = ("release", "retreat", "done")
    OPEN_PHASES = ("approach", "descend", "release", "retreat", "done")
    # everything that changes while it acts: ScriptedTeacher saves / restores these around a label
    PHASE_FIELDS = ("phase", "q_target", "phase_steps", "retreat_target", "release_allowed", "rng", "plan", "wp",
                    "budget", "regrasps", "miss_steps", "last_jaw")

    # the pinch / place geometry and dwells, as class attributes so the grasp bench can try candidates (track G)
    PINCH_HEIGHT = 0.005             # fingertip above the table at the pinch (IG.2 #9; half_fold_demo / weld: 0.010)
    PINCH_INSET = 0.005              # pinch point this far in from the corner, toward the cloth centre
    PLACE_HEIGHT = PLACE_HEIGHT
    ARC_HEIGHT = ARC_HEIGHT
    CLOSE_DWELL = CLOSE_DWELL
    OPEN_DWELL = OPEN_DWELL
    # Phase F (F2/F3), ideas from origin/feat/isaac-half-fold's Markov expert; 0 = off (track G's behaviour), so the
    # grasp bench can A/B each one (--expert SETTLE_LIFT=1,...)
    SETTLE_LIFT = 0          # 1: close ends only once the jaw is shut and still (lifting mid-squeeze slips)
    REGRASP = 0              # n > 0: corner not held for MISS_STEPS steps in lift/carry -> open and re-approach, n times
    MISS_STEPS = 3
    REPLAN_MOVE = 0.0        # m > 0: re-plan the approach whenever the corner has moved this far from the planned pinch
    CARRY_SPEED = 1.0        # joint-speed scale in lift/carry (< 1: a slower carry)

    _ik = None               # one PinchIK for all arms in the process (it parses LeHome's URDF)

    def __init__(self, env, seed=0, prefix="left_", corner=(10,), goal=None, release=False, raw_vertex=True):
        self.env = env
        self.base = env.unwrapped
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.prefix = prefix
        self.corners = tuple(np.atleast_1d(corner))
        self.goal = goal
        if release:
            self.PHASES = self.PHASES + self.RELEASE_PHASES
        self.release_allowed = True
        if IsaacArmExpert._ik is None:
            from isaac.pinch import PinchIK
            IsaacArmExpert._ik = PinchIK()
        self.reset()

    # ---- state ------------------------------------------------------------
    def reset(self):
        self.phase = 0
        self.q_target = None
        self.phase_steps = 0
        self.retreat_target = None
        self.plan = None          # {"jaw", "above", "pinch", "arc": [q...], "arc_tips": [...], "retreat"}
        self.wp = 0               # index into the arc while lifting / carrying / placing
        self.budget = 0           # steps allowed on the current target (half_fold_demo's per-pose budget)
        self.regrasps = 0
        self.miss_steps = 0
        self.last_jaw = None

    def _corner(self):
        cloth = self.base.cloth_positions()
        return np.mean([cloth[c] for c in self.corners], axis=0)

    def _q(self):
        return np.asarray(self.base.joint_positions(self.prefix)[:5], dtype=float)

    # ---- planning (half_fold_demo.plan, per arm) ---------------------------
    def _make_plan(self, from_q=None):
        ik, p = self._ik, self.prefix
        cloth = self.base.cloth_positions()
        center = cloth.mean(axis=0)
        corner = self._corner()
        goal = np.asarray(self.goal(), dtype=float)
        jaw = np.r_[center[:2] - corner[:2], 0.0]
        jaw /= np.linalg.norm(jaw)
        offset = -self.PINCH_INSET * jaw[:2]
        pinch = np.array([corner[0] + offset[0], corner[1] + offset[1], TABLE_TOP_Z + self.PINCH_HEIGHT])
        # the held point's mirror image across the fold line, so the corner lands on the goal
        place = np.array([goal[0] + offset[0], goal[1] - offset[1], TABLE_TOP_Z + self.PLACE_HEIGHT])
        q = list(SEED_Q[p]) if from_q is None else list(from_q)
        plan = {"jaw": jaw, "corner": corner.copy()}
        q, _ = ik.solve(p, pinch + [0, 0, ABOVE], jaw, q)
        plan["above"] = q
        q, _ = ik.solve(p, pinch, jaw, q)
        plan["pinch"] = q
        plan["arc"], plan["arc_tips"] = [], []
        start = pinch
        for s in np.linspace(0.0, 1.0, ARC_WAYPOINTS + 1)[1:]:
            tip = start + (place - start) * (1.0 - math.cos(math.pi * s)) / 2.0
            tip[2] = start[2] + (place[2] - start[2]) * s + self.ARC_HEIGHT * math.sin(math.pi * s)
            q, _ = ik.solve(p, tip, jaw, q, orientation_weight=0.05)
            plan["arc"].append(q)
            plan["arc_tips"].append(tip)
        q, _ = ik.solve(p, place + [0, 0, RETREAT], jaw, q, orientation_weight=0.02)
        plan["retreat"] = q
        return plan

    def _arc_from_here(self):
        """After a resync while holding: a fresh arc from where the gripper is to the place pose."""
        ik, p = self._ik, self.prefix
        goal = np.asarray(self.goal(), dtype=float)
        jaw = self.plan["jaw"] if self.plan else np.array([0.0, -1.0, 0.0])
        offset = -self.PINCH_INSET * jaw[:2]
        place = np.array([goal[0] + offset[0], goal[1] - offset[1], TABLE_TOP_Z + self.PLACE_HEIGHT])
        start = np.asarray(self.base.gripper_position(p), dtype=float)
        q, arc, tips = self._q(), [], []
        n = max(2, int(np.ceil(np.linalg.norm(place - start) / 0.02)))
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            tip = start + (place - start) * s
            tip[2] = max(tip[2], place[2]) + 0.5 * self.ARC_HEIGHT * math.sin(math.pi * s) * (start[2] > place[2] + 0.02)
            q, _ = ik.solve(p, tip, jaw, q, orientation_weight=0.05)
            arc.append(q)
            tips.append(tip)
        plan = dict(self.plan or {"jaw": jaw})
        plan["arc"], plan["arc_tips"] = arc, tips
        q, _ = ik.solve(p, place + [0, 0, RETREAT], jaw, q, orientation_weight=0.02)
        plan["retreat"] = q
        return plan

    # ---- acting -------------------------------------------------------------
    def _target(self, name):
        if name in ("approach",):
            return self.plan["above"]
        if name in ("descend", "close"):
            return self.plan["pinch"]
        if name in ("lift", "carry", "place", "hold", "release"):
            return self.plan["arc"][min(self.wp, len(self.plan["arc"]) - 1)]
        if name == "retreat":
            return self.plan["retreat"]
        return None          # done: stop moving

    def act(self):
        action = np.zeros(6, dtype=np.float32)
        name = self.PHASES[self.phase]
        action[5] = 1.0 if name in self.OPEN_PHASES else -1.0
        if self.plan is None:
            self.plan = self._make_plan()
        target = self._target(name)
        if target is not None:
            self.q_target = np.asarray(target, dtype=float)
            if self.phase_steps == 0:     # a new target: what its largest joint move needs at the speed limit
                last = name not in ("lift", "carry") or self.wp >= len(self.plan["arc"]) - 1
                self.budget = (int(np.ceil(np.max(np.abs(self.q_target - self._q())) / JOINT_DELTA_SCALE))
                               + (POSE_SETTLE_STEPS if last else 0))
            delta = (self.q_target - self._q()) / JOINT_DELTA_SCALE
            action[0:5] = delta / max(1.0, float(np.max(np.abs(delta))))     # all joints arrive together
            if name in ("lift", "carry"):
                action[0:5] *= self.CARRY_SPEED
        self.phase_steps += 1
        self._maybe_advance(name)
        return action

    def _arrived(self, tol):
        return self.q_target is not None and float(np.max(np.abs(self.q_target - self._q()))) < tol

    def _next(self):
        self.phase = min(self.phase + 1, len(self.PHASES) - 1)
        self.phase_steps = 0

    def _jaw_shut(self):
        """The jaw has reached its closed target and stopped (SETTLE_LIFT)."""
        jaw = float(self.base.joint_positions(self.prefix)[5])
        last, self.last_jaw = self.last_jaw, jaw
        closed_q = getattr(self.base, "_closed_q", jaw)
        return jaw <= closed_q + JAW_SHUT_TOL and last is not None and abs(jaw - last) < JAW_STILL

    def _restart_grasp(self):
        """Missed or slipped: open, re-plan on the corner where it lies now, and approach again (REGRASP)."""
        self.regrasps += 1
        self.miss_steps = 0
        self.plan = self._make_plan(from_q=self._q())
        self.phase = self.PHASES.index("approach")
        self.phase_steps = 0
        self.q_target = None
        self.wp = 0

    def _maybe_advance(self, name):
        if name == "done" or (name == "hold" and not (self.release_allowed and "release" in self.PHASES)):
            return
        if self.REGRASP and name in ("lift", "carry"):
            held = self.base.grasp_active(self.prefix)
            self.miss_steps = 0 if held else self.miss_steps + 1
            if self.miss_steps >= self.MISS_STEPS and self.regrasps < self.REGRASP:
                self._restart_grasp()
                return
        if self.REPLAN_MOVE and name == "approach" and self.plan is not None and "corner" in self.plan:
            if float(np.linalg.norm(self._corner() - self.plan["corner"])) > self.REPLAN_MOVE:
                self.plan = self._make_plan(from_q=self._q())
                self.phase_steps = 0
        patience = self.phase_steps > self.budget    # a pose pressed into the table or cloth may never get closer
        if name in ("approach", "descend"):
            if self._arrived(TRACK_TOL) or patience:
                if name == "approach":       # re-solve the pinch on the corner as it lies now (closed loop)
                    self.plan = self._make_plan(from_q=self._q())
                self._next()
        elif name == "close":
            shut = self._jaw_shut() if self.SETTLE_LIFT else True
            if (self.phase_steps >= self.CLOSE_DWELL and shut) or self.phase_steps >= 3 * self.CLOSE_DWELL:
                self.wp = 0
                self.miss_steps = 0
                self._next()
        elif name in ("lift", "carry"):
            last = len(self.plan["arc"]) - 1
            if self._arrived(WAYPOINT_TOL) or patience:
                self.wp = min(self.wp + 1, last)
                self.phase_steps = 0
            if name == "lift" and self.wp >= LIFT_WAYPOINTS:
                self._next()
            elif name == "carry" and self.wp >= last:
                self._next()
        elif name == "place":
            if self._arrived(TRACK_TOL) or patience:
                self._next()
        elif name == "hold":
            self._next()
        elif name == "release":
            if self.phase_steps >= self.OPEN_DWELL:
                self._next()
        elif name == "retreat":
            if self._arrived(TRACK_TOL) or patience:
                self._next()

    # ---- resync --------------------------------------------------------------
    def _jaw(self):
        if self.plan is not None and "jaw" in self.plan:
            return self.plan["jaw"]
        center, corner = self.base.cloth_positions().mean(axis=0), self._corner()
        jaw = np.r_[center[:2] - corner[:2], 0.0]
        return jaw / np.linalg.norm(jaw)

    def infer_phase(self, placed=None):
        """Phase from the live sim (a learner or a perturbation has been driving), as FoldExpert.infer_phase."""
        tip = np.asarray(self.base.gripper_position(self.prefix), dtype=float)
        corner = self._corner()
        goal = np.asarray(self.goal(), dtype=float)
        if placed is None:
            placed = float(np.linalg.norm(corner - goal)) < SUCCESS_DIST
        if self.base.grasp_active(self.prefix):
            self.plan = dict(self.plan or {}, jaw=self._jaw())
            self.plan = self._arc_from_here()
            self.wp = 0
            if float(np.linalg.norm(corner - goal)) < SUCCESS_DIST:
                name, self.wp = "hold", len(self.plan["arc"]) - 1
            elif float(np.linalg.norm(corner[:2] - goal[:2])) < 0.02:
                name, self.wp = "place", len(self.plan["arc"]) - 1
            elif corner[2] >= TABLE_TOP_Z + 0.04:
                name = "carry"
            else:
                name = "lift"
        elif placed and "done" in self.PHASES:
            name = "done" if tip[2] >= corner[2] + RETREAT else "retreat"
            self.plan = dict(self.plan or {}, jaw=self._jaw())
            self.plan["retreat"], _ = self._ik.solve(self.prefix, tip + [0, 0, RETREAT], self.plan["jaw"], self._q(),
                                                     orientation_weight=0.02)
        else:
            self.plan = self._make_plan(from_q=self._q())
            near = float(np.linalg.norm(tip[:2] - corner[:2])) < 0.02 and tip[2] - corner[2] < 0.07
            name = "descend" if near else "approach"
        self.phase = self.PHASES.index(name)
        self.phase_steps = 0
        self.q_target = None
        self.retreat_target = None
        return name
"""Closed-loop placement base for the residual RL (docs/superpowers/plans/2026-10-20-closed-loop-rl.md).

IsaacArmExpert's open-loop pinch does the approach, descend, close and lift (friction profile: 40/40 held). From the
carry on, IsaacServoExpert's corner steering takes over: each step the gripper target moves the CORNER toward its next
waypoint, so spring-back and lag need no overshoot table. The place target is the partner vertex's current position
(`live_goal`), not where it started: the bottom layer slides 6-14 cm during a carry (strip runs, results.md).

Two hooks for a learned policy, both neutral by default (the scripted base):
  residual      (3,) m, added to the steered gripper target in carry, place and hold
  release_vote  the arm releases once its gate is open (every arm of the fold in hold) and its vote is True, or after
                HOLD_CAP steps in hold
No retries: a corner that leaves the jaw in the carry sets `dropped` and the arm goes on to release.

WORLDFOLD_PLACE_PARAMS="PRESS_Z=0.012,LIVE_MAX=0.03" overrides this class's attributes for a run (as
WORLDFOLD_EXPERT_PARAMS does IsaacArmExpert's).
"""

from __future__ import annotations

import json
import os

import numpy as np

from isaac.fold_expert import ABOVE, RETREAT, IsaacArmExpert
from isaac.servo_expert import IsaacServoExpert
from mujuco.cloth_params import TABLE_TOP_Z


class IsaacPlaceServoExpert(IsaacServoExpert):
    PLACE_TOL = 0.005        # the servo's 1.2 cm is as large as the bar
    PLACE_PATIENCE = 40
    PRESS_Z = 0.003          # place target this far above the partner vertex: the corner rests on the bottom layer
    HOLD_CAP = 20            # steps in hold before the arm releases without its vote
    # m: the live goal may stray at most this far (per axis) from the partner's start position; None: unbounded.
    # R0 (results.md): unbounded chasing dragged the bottom layer 7-12 cm
    LIVE_MAX = None
    # m above the table: the steered tip (residual included) never goes lower, so a residual can't drive the jaws
    # into the table (p1, 2026-10-20: the noisy prior's episodes blew the cloth up); None: no floor
    Z_FLOOR = 0.005
    # Base fixes from the 2026-10-21 brainstorm (results.md), all off by default:
    # LATCH_GOAL 1: the place target is the partner where it was when the place began (the fingers push the partner,
    # and an unlatched goal chases it). BACKOFF m: the release slides the open jaws this far out along -jaw at hold
    # height before lifting, so a corner on the pads isn't carried up. GFOLD_SLOPE: the carried corner's path height is
    # capped at goal z + GFOLD_Z0 + GFOLD_SLOPE * (xy distance to the goal), so the top half lays down from the crease
    # out instead of swinging down taut (the fold line formed ~1.4 cm short)
    LATCH_GOAL = 0
    BACKOFF = 0.0
    GFOLD_SLOPE = None
    GFOLD_Z0 = 0.01
    # Touch-up after lay-down (demo hunt 2026-10-22: the closest fold had a 0.75 cm mean but one corner 2.3 cm off):
    # start_touchup(offset) comes down TOUCH_BEHIND behind the top corner with the jaw closed, presses TOUCH_PRESS above
    # it and slides by offset, then lifts. Whether the top layer slides over the bottom one is what it tests
    TOUCH_BEHIND = 0.015
    TOUCH_PRESS = 0.002
    TOUCH_STEP = 0.004       # m per slide waypoint

    PHASE_FIELDS = IsaacServoExpert.PHASE_FIELDS + ("residual", "release_vote", "hold_site", "dropped", "latched", "touch")

    def __init__(self, *args, live_goal=None, **kwargs):
        self.live_goal = live_goal       # callable -> the partner vertex's current position; None: self.goal
        super().__init__(*args, **kwargs)
        self.PHASES = self.PHASES + ("touch",)

    def reset(self):
        super().reset()
        self.residual = np.zeros(3)
        self.release_vote = True
        self.hold_site = None
        self.dropped = False
        self.latched = None
        self.touch = None

    def _goal_now(self):
        if self.latched is not None:
            return self.latched
        goal = np.asarray((self.live_goal or self.goal)(), dtype=float)
        if self.LIVE_MAX is not None and self.live_goal is not None:
            start = np.asarray(self.goal(), dtype=float)
            goal = start + np.clip(goal - start, -self.LIVE_MAX, self.LIVE_MAX)
        return goal

    def _tip(self, tip):
        if self.Z_FLOOR is not None:
            tip = np.r_[tip[:2], max(float(tip[2]), TABLE_TOP_Z + self.Z_FLOOR)]
        return tip

    def _corner_path(self, corner, goal):
        path = super()._corner_path(corner, goal)
        if self.GFOLD_SLOPE is None:
            return path
        return [np.r_[p[:2], min(float(p[2]), float(goal[2]) + self.GFOLD_Z0
                                  + self.GFOLD_SLOPE * float(np.linalg.norm(goal[:2] - p[:2])))] for p in path]

    def _start_carry(self):
        self.sv.update(path=self._corner_path(self._corner(), self._goal_now()), k=0, k_steps=0, slip=0)
        self._go("carry")

    def _start_hold(self):
        super()._start_hold()
        self.hold_site = self._site() - self.residual      # hold where the arm is now
        if self.BACKOFF:
            jaw = self.plan["jaw"]
            back = self._site() - self.BACKOFF * np.r_[jaw[:2], 0.0]
            q, self.plan["retreat_path"] = self._q(), []
            # two moves: out, then up (the slide grasp's 3 mm lift steps made 17 waypoints and timed out: sweep 1)
            for tip in (back, back + [0.0, 0.0, RETREAT]):
                q, _ = self._ik.solve(self.prefix, tip, jaw, q, orientation_weight=0.02)
                self.plan["retreat_path"].append(q)

    def start_touchup(self, offset):
        """Slide the settled top corner by offset (xy, m) with the closed jaw pressed on the top layer."""
        corner = self._corner()
        off = np.r_[np.asarray(offset, float)[:2], 0.0]
        d = off / max(np.linalg.norm(off), 1e-9)
        z = float(corner[2]) + self.TOUCH_PRESS
        start = np.r_[corner[:2], z] - self.TOUCH_BEHIND * d
        n = max(2, int(np.ceil(np.linalg.norm(off) / self.TOUCH_STEP)))
        path = [start + [0.0, 0.0, ABOVE], start] + [start + off * s for s in np.linspace(0.0, 1.0, n + 1)[1:]]
        path.append(path[-1] + [0.0, 0.0, RETREAT])
        self.touch = {"path": path, "k": 0, "k_steps": 0, "offset": off[:2].tolist()}
        self._go("touch")

    def act(self):
        name = self.PHASES[self.phase]
        if name == "touch":
            action = np.zeros(6, dtype=np.float32)
            action[5] = self._close_command()
            t = self.touch
            tip = t["path"][t["k"]]
            action[0:5], _ = self._toward(self._tip(tip + self.residual), self.plan["jaw"], weight=0.05)
            t["k_steps"] += 1
            if np.linalg.norm(self._site() - tip) < 0.006 or t["k_steps"] > 12:
                t["k"], t["k_steps"] = t["k"] + 1, 0
                if t["k"] >= len(t["path"]):
                    self._go("done")
            return action
        if name in ("approach", "descend", "close", "lift"):
            action = IsaacArmExpert.act(self)
            if self.PHASES[self.phase] == "carry":        # the parent's lift just ended: steer from here
                self._start_carry()
            return action
        if name not in ("carry", "place", "hold"):
            return IsaacArmExpert.act(self)               # release, retreat, done
        action = np.zeros(6, dtype=np.float32)
        action[5] = self._close_command()
        corner, site = self._corner(), self._site()
        jaw = self.plan["jaw"]
        self.phase_steps += 1
        if name in ("carry", "place") and np.linalg.norm(corner - site) > self.HOLD_DIST:
            self.sv["slip"] += 1
            if self.sv["slip"] >= self.SLIP_STEPS:
                self.dropped = True
                self._start_hold()
                return action
        else:
            self.sv["slip"] = 0
        if name == "carry":
            path, k = self.sv["path"], self.sv["k"]
            target = path[min(k, len(path) - 1)]
            action[0:5], _ = self._toward(self._tip(self._steer(target) + self.residual), jaw, weight=0.05)
            self.sv["k_steps"] += 1
            if np.linalg.norm(corner - target) < self.CORNER_WP_TOL or self.sv["k_steps"] > self.WP_PATIENCE:
                self.sv["k"], self.sv["k_steps"] = k + 1, 0
            if self.sv["k"] >= len(path):
                if self.LATCH_GOAL:
                    self.latched = self._goal_now()
                self._go("place")
            action[0:5] *= self.CARRY_SPEED
        elif name == "place":
            goal = self._goal_now()
            action[0:5], _ = self._toward(self._tip(self._steer(goal + [0.0, 0.0, self.PRESS_Z]) + self.residual),
                                          jaw, weight=0.05)
            if (np.linalg.norm(corner[:2] - goal[:2]) < self.PLACE_TOL and corner[2] - goal[2] < 0.015) \
                    or self.phase_steps >= self.PLACE_PATIENCE:
                self._start_hold()
        else:   # hold: stay, plus the residual, until the gate and the vote let go
            action[0:5], _ = self._toward(self._tip(self.hold_site + self.residual), jaw, weight=0.05)
            if self.release_allowed and (self.release_vote or self.phase_steps >= self.HOLD_CAP):
                self.plan["arc"], self.wp = [self._q()], 0      # release where the arm is, not at the hold start
                self._go("release")
        return action


def _apply_env_params():
    for item in filter(None, (s.strip() for s in os.environ.get("WORLDFOLD_PLACE_PARAMS", "").split(","))):
        key, val = item.split("=", 1)
        if not hasattr(IsaacPlaceServoExpert, key):
            raise ValueError(f"WORLDFOLD_PLACE_PARAMS: IsaacPlaceServoExpert has no {key}")
        setattr(IsaacPlaceServoExpert, key, json.loads(val))


_apply_env_params()

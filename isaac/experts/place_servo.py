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

from isaac.experts.fold_expert import ABOVE, RETREAT, IsaacArmExpert
from isaac.experts.servo_expert import IsaacServoExpert
from mujoco_sim.cloth_params import GRIPPER_OPEN, TABLE_TOP_Z


class IsaacPlaceServoExpert(IsaacServoExpert):
    PLACE_TOL = 0.005        # the servo's 1.2 cm is as large as the bar
    PLACE_PATIENCE = 40
    PRESS_Z = 0.003          # place target this far above the partner vertex: the corner rests on the bottom layer
    HOLD_CAP = 20            # steps in hold before the arm releases without its vote
    # m: the live goal may stray at most this far (per axis) from the partner's start position; None: unbounded.
    # R0 (results.md): unbounded chasing dragged the bottom layer 7-12 cm
    LIVE_MAX = None
    # m above the table: the steered tip (residual included) never goes lower, so a residual can't drive the jaws
    # into the table (p1, 2026-10-20: the noisy prior's episodes blew the cloth up); None: no floor. 17 mm (pads clear
    # of the bottom layer) stopped the plough but cost 1.4 cm of alignment (audit round 2, results.md)
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
    # Gentle release (user, 2026-10-22: the release distorted the cloth; the 980008 video shows the jaw snapping open
    # to GRIPPER_OPEN, sweeping the top layer, then a full-speed lift): the arm holds still while the jaw opens
    # RELEASE_STEP rad per step to RELEASE_JAW only, lifts straight up RETREAT_STEP m per step to RETREAT, and opens
    # the rest of the way up there (BACKOFF: after sliding the jaws out that far, at the same speed). RELEASE_JAW None:
    # the old release (jaw straight to GRIPPER_OPEN, a move at the speed limit)
    RELEASE_JAW = 0.3        # rad (closed -0.1, open 1.0)
    RELEASE_STEP = 0.02      # rad per step: 20 steps (1 s) from closed to RELEASE_JAW
    RETREAT_STEP = 0.002     # m per step: 4 cm/s
    # Placement offset (overnight plan 2026-11-08): the place target moves PLACE_ALONG m from the partner back toward
    # where the carried corner started (toward the crease) and PLACE_INWARD m toward the cloth's centre line. Audit
    # round 2's base ends with the interior top layer 1.8 cm past the bottom and the top corners 1.5-2.3 cm outside
    # their partners (median)
    PLACE_ALONG = 0.0
    PLACE_INWARD = 0.0
    # Place high, then put down (overnight plan, batch B): CARRY_FLOOR m above the table is the steered tip's floor in
    # the carry and the place (None: Z_FLOOR), so the pads stay off the bottom layer while the corner is steered over
    # its partner; hold then lowers the site straight down to Z_FLOOR, and the arm may release once it has stopped
    # coming down (< DESCEND_STILL over 3 steps) or after DESCEND_STEPS. 17 mm everywhere cut the bottom shift 2.0 ->
    # 0.5 cm but cost 1.4 cm on the mean, released from higher (audit round 2)
    CARRY_FLOOR = None
    DESCEND_STILL = 0.001
    DESCEND_STEPS = 20
    # Layer-shift correction (audit 2026-11-07: the top layer lies a rigid ~2.2 cm off the bottom one before release).
    # Once every arm of the fold holds, each step both arms move their held point by -SHIFT_GAIN times the top-minus-
    # bottom offset the batch measures (layer_offset), at most SHIFT_STEP per step and SHIFT_MAX in all, until it is
    # under SHIFT_TOL or SHIFT_STEPS steps have passed; only then can they release. SHIFT_GAIN 0: off
    SHIFT_GAIN = 0.0
    SHIFT_STEP = 0.002
    SHIFT_MAX = 0.03
    SHIFT_TOL = 0.003
    SHIFT_STEPS = 40
    # Slip recovery (batch I, 2026-11-09). On the stiff cloth the right corner leaves the shut jaw mid-carry in about
    # half the seeds, 2.3x as often when it sits under 1.5 cm above the site at carry start (batches G and H). Both off
    # by default. SEAT_MIN: after the lift, a SEAT_ARMS corner sitting lower than this above the site is re-grasped, at
    # most SEAT_REGRASPS times (right arm only: the left seats under 2 cm in 91/96 control seeds and slips less), and no
    # carry starts until every arm has passed its check (`coordinate`). SLIP_REGRASPS: once every other arm holds, a
    # dropped corner at rest (moved under SETTLED_MM over 5 steps) more than REGRASP_FAR from its goal is re-grasped
    # where it lies, at most that many times; one that fell near its goal stays (18 of the control's 21 passes had the
    # right corner dropped and carried over by the stiff sheet; real losses end 9-21 cm off).
    # The carry's IK orientation weight (batch J, 2026-11-09). None: _carry_w (0.05). On the friction layout the right
    # arm starts the carry with its wrist near the joint limit, and at 0.05 its first waypoints miss by over 5 mm in
    # 73/128 control rows; those corners drop earlier, before the crease (offline: 0.01 brings every s3 seed under 3 mm
    # at the same tilt and carry length). Carry only: place, hold and release keep _carry_w.
    CARRY_WEIGHT = None
    SEAT_MIN = None
    SEAT_ARMS = ("right_",)
    SEAT_REGRASPS = 2
    SLIP_REGRASPS = 0
    REGRASP_FAR = 0.05
    SETTLED_MM = 3.0
    SLIP_DECIDE = 40         # hold steps with the other arms holding before a corner that never settles stays

    PHASE_FIELDS = IsaacServoExpert.PHASE_FIELDS + ("residual", "release_vote", "hold_site", "dropped", "latched", "touch",
                                                    "release_site", "hold_offset", "shift", "drop_at", "seat_regrasps",
                                                    "slip_regrasps", "slipped", "corner_hist", "seats")

    def __init__(self, *args, live_goal=None, layer_offset=None, **kwargs):
        self.live_goal = live_goal       # callable -> the partner vertex's current position; None: self.goal
        self.layer_offset = layer_offset     # callable -> the fold's top-minus-bottom offset (m); None: no correction
        super().__init__(*args, **kwargs)
        self.PHASES = self.PHASES + ("touch",)

    def reset(self):
        super().reset()
        self.residual = np.zeros(3)
        self.release_vote = True
        self.hold_site = None
        self.dropped = False
        self.drop_at = None
        self.seat_regrasps = 0
        self.slip_regrasps = 0
        self.slipped = False             # the corner left the jaw since the last grasp: a re-grasp decision is pending
        self.corner_hist = []
        self.seats = []                  # cm, the corner above the site at each lift's end
        self.latched = None
        self.touch = None
        self.release_site = None
        self.hold_offset = np.zeros(3)
        self.shift = {"steps": 0, "first": None, "last": None}

    def _hold_jaw(self):
        # the pinch's jaw axis through hold, release and retreat: servo_expert re-aimed it from the folded cloth's
        # centre, ~70 deg off, so the wrist rolled while gripping and letting go (audit sweep: edge -2.8 cm)
        return self.plan["jaw"]

    def _shifting(self):
        """SHIFT_GAIN: move the held point against the layer offset; True while the correction still runs."""
        if not (self.SHIFT_GAIN and self.layer_offset is not None and self.release_allowed) \
                or self.shift["steps"] >= self.SHIFT_STEPS:
            return False
        off = np.asarray(self.layer_offset(), dtype=float)[:2]
        if self.shift["first"] is None:
            self.shift["first"] = off.round(4).tolist()
        self.shift["last"] = off.round(4).tolist()
        if np.linalg.norm(off) <= self.SHIFT_TOL:
            self.shift["steps"] = self.SHIFT_STEPS
            return False
        step = np.clip(-self.SHIFT_GAIN * off, -self.SHIFT_STEP, self.SHIFT_STEP)
        self.hold_offset[:2] = np.clip(self.hold_offset[:2] + step, -self.SHIFT_MAX, self.SHIFT_MAX)
        self.shift["steps"] += 1
        return True

    def _goal_now(self):
        if self.latched is not None:
            return self.latched
        goal = np.asarray((self.live_goal or self.goal)(), dtype=float)
        if self.LIVE_MAX is not None and self.live_goal is not None:
            start = np.asarray(self.goal(), dtype=float)
            goal = start + np.clip(goal - start, -self.LIVE_MAX, self.LIVE_MAX)
        return goal + self._place_offset(goal)

    def _place_offset(self, goal):
        """PLACE_ALONG / PLACE_INWARD as a (3,) offset from the partner; zero when both are 0 or before a plan."""
        if not (self.PLACE_ALONG or self.PLACE_INWARD) or not self.plan or "corner" not in self.plan:
            return np.zeros(3)
        back = np.r_[np.asarray(self.plan["corner"], float)[:2] - goal[:2], 0.0]
        back /= max(np.linalg.norm(back), 1e-9)
        inward = np.array([-back[1], back[0], 0.0])
        if inward[:2] @ (self.base.cloth_positions().mean(axis=0)[:2] - goal[:2]) < 0:
            inward = -inward
        return self.PLACE_ALONG * back + self.PLACE_INWARD * inward

    def _tip(self, tip, floor=None):
        floor = self.Z_FLOOR if floor is None else floor
        if floor is not None:
            tip = np.r_[tip[:2], max(float(tip[2]), TABLE_TOP_Z + floor)]
        return tip

    def _descending(self, site):
        """CARRY_FLOOR: True while hold is still lowering the site onto the stack."""
        if self.CARRY_FLOOR is None or self.phase_steps > self.DESCEND_STEPS:
            return False
        hist = self.sv["hist"] = (self.sv["hist"] + [float(site[2])])[-4:]
        return len(hist) < 4 or hist[0] - hist[-1] >= self.DESCEND_STILL

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
        if self.CARRY_FLOOR is not None and not self.dropped:     # ... but put it down (an empty jaw stays up)
            self.hold_site[2] = TABLE_TOP_Z + (self.Z_FLOOR or 0.0)
        if self.BACKOFF:
            jaw = self.plan["jaw"]
            back = self._site() - self.BACKOFF * np.r_[jaw[:2], 0.0]
            q, self.plan["retreat_path"] = self._q(), []
            # two moves: out, then up (the slide grasp's 3 mm lift steps made 17 waypoints and timed out: sweep 1)
            for tip in (back, back + [0.0, 0.0, RETREAT]):
                q, _ = self._ik.solve(self.prefix, tip, jaw, q, orientation_weight=0.02)
                self.plan["retreat_path"].append(q)

    def start_touchup(self, offset, press=None):
        """Slide the settled top corner by offset (xy, m) with the closed jaw pressed on the top layer, press m above the
        corner (default TOUCH_PRESS)."""
        corner = self._corner()
        off = np.r_[np.asarray(offset, float)[:2], 0.0]
        d = off / max(np.linalg.norm(off), 1e-9)
        z = float(corner[2]) + (self.TOUCH_PRESS if press is None else float(press))
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
            action[0:5], _ = self._toward(self._tip(tip + self.residual), self.plan["jaw"], weight=self._carry_w)
            t["k_steps"] += 1
            if np.linalg.norm(self._site() - tip) < 0.006 or t["k_steps"] > 12:
                t["k"], t["k_steps"] = t["k"] + 1, 0
                if t["k"] >= len(t["path"]):
                    self._go("done")
            return action
        if name in ("approach", "descend", "close", "lift"):
            action = IsaacArmExpert.act(self)
            if self.PHASES[self.phase] == "carry":        # the parent's lift just ended: steer from here
                self.seats.append(round(100 * float(self._corner()[2] - self._site()[2]), 2))
                if self._unseated():
                    self.seat_regrasps += 1
                    self._regrasp_here(action)
                else:
                    self._start_carry()
                    self.sv["go"] = False        # coordinate's carry gate decides when it moves
            return action
        if name in ("release", "retreat", "done") and self.RELEASE_JAW is not None:
            return self._release_gently(name)
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
                # when and how far (batch G review: on stiff cloth the corner rides ~2.8 cm up the pads, so the 5 cm
                # rule may fire with it still pinched)
                self.drop_at = {"phase": name, "wp": [int(self.sv.get("k", 0)), len(self.sv.get("path", ()))],
                                "dist_cm": round(100 * float(np.linalg.norm(corner - site)), 2),
                                "dz_cm": round(100 * float(corner[2] - site[2]), 2)}
                self.slipped = True
                self._start_hold()
                return action
        else:
            self.sv["slip"] = 0
        if name == "carry":
            path, k = self.sv["path"], self.sv["k"]
            if not self.sv.get("go", True):
                if not getattr(self, "carry_allowed", True):    # an arm is still re-grasping before its carry: wait
                    action[0:5], _ = self._toward(self._tip(self._steer(path[0]) + self.residual, self.CARRY_FLOOR),
                                                  jaw, weight=self._carry_ik_w)
                    return action
                self.sv["go"] = True
            target = path[min(k, len(path) - 1)]
            action[0:5], _ = self._toward(self._tip(self._steer(target) + self.residual, self.CARRY_FLOOR), jaw,
                                          weight=self._carry_ik_w)
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
            action[0:5], _ = self._toward(self._tip(self._steer(goal + [0.0, 0.0, self.PRESS_Z]) + self.residual,
                                                    self.CARRY_FLOOR), jaw, weight=self._carry_w)
            if (np.linalg.norm(corner[:2] - goal[:2]) < self.PLACE_TOL and corner[2] - goal[2] < 0.015) \
                    or self.phase_steps >= self.PLACE_PATIENCE:
                self._start_hold()
        else:   # hold: stay, plus the residual, until the gate and the vote let go
            if self._regrasp_dropped(corner):
                self.slip_regrasps += 1
                self._regrasp_here(action)
                return action
            shifting = self._shifting()
            action[0:5], _ = self._toward(self._tip(self.hold_site + self.hold_offset + self.residual), jaw,
                                          weight=self._carry_w)
            if self._descending(site):
                return action
            if not shifting and self.release_allowed and (self.release_vote or self.phase_steps >= self.HOLD_CAP):
                self.plan["arc"], self.wp = [self._q()], 0      # release where the arm is, not at the hold start
                self.release_site = self._site()
                self._go("release")
        return action

    @property
    def _carry_ik_w(self):
        return self._carry_w if self.CARRY_WEIGHT is None else float(self.CARRY_WEIGHT)

    def _unseated(self):
        """SEAT_MIN: the lift just ended with a SEAT_ARMS corner sitting too low in the jaw, and re-grasps are left."""
        return (self.SEAT_MIN is not None and self.prefix in self.SEAT_ARMS
                and self.seat_regrasps < self.SEAT_REGRASPS and self.seats[-1] < 100 * self.SEAT_MIN)

    def slip_pending(self):
        """A dropped corner's re-grasp decision is still open (coordinate keeps the release shut meanwhile)."""
        return self.slipped and self.slip_regrasps < self.SLIP_REGRASPS

    def _regrasp_dropped(self, corner):
        """SLIP_REGRASPS: once every other arm holds, re-grasp the dropped corner if it rests more than REGRASP_FAR
        from its goal; if it rests nearer, the decision is made and it stays."""
        if not self.slip_pending():
            return False
        self.corner_hist = (self.corner_hist + [np.asarray(corner, float).copy()])[-5:]
        hist = np.asarray(self.corner_hist)
        settled = len(hist) == 5 and 1000 * float(np.linalg.norm(hist - hist[-1], axis=1).max()) < self.SETTLED_MM
        if not getattr(self, "partners_hold", True):
            return False
        if not settled:
            if self.phase_steps >= self.SLIP_DECIDE:     # still creeping: decide that it stays
                self.slipped = False
            return False
        if float(np.linalg.norm(corner[:2] - self._goal_now()[:2])) > self.REGRASP_FAR:
            return True
        self.slipped = False
        return False

    def _regrasp_here(self, action):
        """Open, re-plan on the corner where it lies now and approach again (IsaacArmExpert's REGRASP restart), with
        the last grasp's goal latch and retry counts cleared."""
        self.slipped, self.corner_hist, self.latched = False, [], None
        self.align_tries, self.descend_retries, self.site_hist = 0, 0, []
        self._restart_grasp()
        action[5] = 1.0

    def _set_jaw(self, action, q):
        """Open target q (rad) for this arm's jaw, None: fully open."""
        if self.base.grasp_knobs.get("gripper_mode") == "position":
            action[5] = self.base.jaw_command(GRIPPER_OPEN if q is None else q)
        else:
            self.base.jaw_open[self.prefix] = q
            action[5] = 1.0

    def _release_gently(self, name):
        """RELEASE_JAW: release opens the jaw slowly part way with the arm still, retreat lifts straight up slowly,
        done opens the jaw fully."""
        action = np.zeros(6, dtype=np.float32)
        self.phase_steps += 1
        if name == "done":
            self._set_jaw(action, None)
            return action
        jaw = self.plan["jaw"]
        if self.release_site is None:            # put into release / retreat by infer_phase, not by hold
            self.release_site = self._site()
        if name == "release":
            q = min(self.RELEASE_JAW, float(self.base._joint_targets[self.prefix][5]) + self.RELEASE_STEP)
            self._set_jaw(action, q)
            action[0:5], _ = self._toward(self.release_site, jaw, weight=self._carry_w)
            opened = float(self.base.joint_positions(self.prefix)[5]) >= self.RELEASE_JAW - 0.05
            if q >= self.RELEASE_JAW and (opened or self.phase_steps >= 60):     # 3 s patience
                self._go("retreat")      # the jaw itself, not just its target, is open: the pads have let go
            return action
        self._set_jaw(action, self.RELEASE_JAW)
        back = min(self.BACKOFF, self.phase_steps * self.RETREAT_STEP)        # BACKOFF: slide out first, then lift
        lift = min(RETREAT, self.phase_steps * self.RETREAT_STEP - back)
        top = self.release_site - back * np.r_[jaw[:2], 0.0]
        action[0:5], _ = self._toward(top + [0.0, 0.0, lift], jaw, weight=self._carry_w)
        if lift >= RETREAT and (self._site()[2] >= self.release_site[2] + RETREAT - 0.005
                                or self.phase_steps >= (self.BACKOFF + RETREAT) / self.RETREAT_STEP + 20):
            self._go("done")
        return action


def _apply_env_params():
    for item in filter(None, (s.strip() for s in os.environ.get("WORLDFOLD_PLACE_PARAMS", "").split(","))):
        key, val = item.split("=", 1)
        if not hasattr(IsaacPlaceServoExpert, key):
            raise ValueError(f"WORLDFOLD_PLACE_PARAMS: IsaacPlaceServoExpert has no {key}")
        setattr(IsaacPlaceServoExpert, key, json.loads(val))


_apply_env_params()

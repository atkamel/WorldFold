"""Markov scripted expert for the Isaac half fold: the half_fold_demo motion, decided from the current state.

half_fold_demo.py plans each arm's joint targets once at reset and plays them back on a schedule, which is fine for
recording demos but cannot say what to do from a state it did not produce (a student's, or after a perturbation).
This expert keeps the same plan (half_fold_demo.plan_arm: pinch from above, carry along an arc over the fold line,
release, back off) but re-decides every step, from what the sim shows, where each arm is in it:

    pre-grasp   jaws open, corner not on its goal     above the pinch -> down to it -> close
    closing     jaws commanded shut, still moving     hold the pinch pose
    holding     jaws shut, corner at the fingertip    next arc waypoint past the nearest one; at the end, open
                                                      (both arms in step: see decide())
    missed      jaws shut, corner not at the tip      open and go back above the pinch
    released    jaws open, corner on its goal         stay until the jaws are open, then retreat and hold

So the same call labels any state, and a dropped or sprung-back corner is simply grasped again: its pinch is
re-solved where the corner now lies. Nothing is timed or remembered between steps except the plan cache, which
depends only on where each corner was when its plan was solved.

label_chunk() looks ahead without the simulator: Isaac's CPU cloth has no velocity state to snapshot and restore,
so the expert runs on a kinematic copy of the state in which each joint reaches its commanded target, the jaws
move at JAW_RATE, and a held corner rides along with the fingertip. That is the motion the expert asks for, so a
label is the chunk it would play from here if the arm tracked it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from isaac import half_fold_demo
from isaac.half_fold_demo import TRACK_TOL, plan_arm
from mujuco.cloth_params import GRIPPER_CLOSED, GRIPPER_OPEN, JOINT_DELTA_SCALE

HOLD_RADIUS = 0.04        # m: corner within this of the fingertip with the jaws shut -> held
PLACED_DIST = 0.05        # m: corner within this of its goal -> placed (HalfFoldEnv's SUCCESS_DIST)
RELEASE_DIST = 0.035      # m: a held corner this close (horizontally) to its goal at the end of the arc -> let go
ALIGN_DIST = 0.015        # m: fingertip within this (horizontally) of the pinch -> descend
WAYPOINT_DIST = 0.01      # m: fingertip this far from its pinch -> it has started up the arc
REPLAN_DIST = 0.01        # m: re-solve an arm's plan once its (unheld) corner is this far from where it was planned
JAW_SHUT = 0.25           # rad: jaws below this and no longer closing -> shut (they meet at about -0.015)
JAW_OPENED = 0.8          # rad: jaws above this -> open enough to back away
JAW_SPEED_STILL = 0.05    # rad/s: jaws this slow have finished squeezing (closing runs ~3 rad/s)
JAW_FULLY_SHUT = 0.05     # rad: jaws this far closed count as shut whatever their speed
STALL_TOL = 0.1           # rad: a pose pressed into the table or the cloth may stop short of TRACK_TOL
STALL_SPEED = 0.05        # rad/s: ...then the arm counts as arrived once every joint is this slow
JAW_RATE = 0.15           # rad per control step, the jaws' speed in the look-ahead model
CONTROL_DT = 0.05
GRIPPER_OPEN_CMD, GRIPPER_CLOSE_CMD = 1.0, -1.0
CARRY_SPEED = 1.0         # fraction of the per-step joint limit used while holding (a slower lift may slip less)

# ISAAC_EXPERT_PARAMS='{"PINCH_OFFSET": -0.01, "CARRY_SPEED": 0.5}' overrides constants here or in half_fold_demo
# (the plan geometry), for sweeping the expert on Isaac without editing code
for _name, _value in json.loads(os.environ.get("ISAAC_EXPERT_PARAMS", "{}")).items():
    if _name in globals():
        globals()[_name] = _value
    elif hasattr(half_fold_demo, _name):
        setattr(half_fold_demo, _name, _value)
    else:
        raise KeyError(f"ISAAC_EXPERT_PARAMS: unknown constant {_name}")


@dataclass
class ArmState:
    """What the expert reads about one arm: the 5 arm joints, the jaw, its command, its fingertip and its corner."""
    q: np.ndarray
    qd: np.ndarray
    jaw: float
    jaw_speed: float
    closed: bool
    tip: np.ndarray
    corner: np.ndarray


class IsaacHalfFoldExpert:
    """Acts on a HalfFoldEnv (cloth_fold_rl.quarter_fold_env or imitation.tasks) whose base is IsaacClothFoldEnv."""

    def __init__(self, env, ik=None):
        self.env = env
        if ik is None:
            from isaac.pinch import PinchIK
            ik = PinchIK()
        self.ik = ik
        self.moves = {m.prefix: m for m in env.stages[0].moves}
        self.plans = {}

    # ---- plan cache ----
    def reset(self):
        self.plans = {}
        self._center = self.env.unwrapped.cloth_positions().mean(axis=0)
        for p in self.moves:
            self._plan(p, self._corner(p))

    def _plan(self, prefix, corner):
        poses = plan_arm(self.ik, prefix, corner, self.env.goal(self.moves[prefix]), self._center)
        poses["corner"] = np.array(corner, dtype=float)
        self.plans[prefix] = poses
        return poses

    # ---- reading the sim ----
    def _corner(self, prefix):
        return np.mean([self.env._vertex(c) for c in self.moves[prefix].corners], axis=0)

    def read(self, prefix) -> ArmState:
        base = self.env.unwrapped
        q, qd = base.joint_positions(prefix), base.joint_velocities(prefix)
        return ArmState(q=q[:5].copy(), qd=qd[:5].copy(), jaw=float(q[5]), jaw_speed=abs(float(qd[5])),
                        closed=bool(base._gripper_closed[prefix]), tip=self.ik.site_pose(prefix, q[:5])[0],
                        corner=self._corner(prefix))

    # ---- the decision ----
    def phase(self, prefix, s: ArmState) -> str:
        # lifting before the jaws finish squeezing lets the corner slip out (seen on Isaac, seeds 16-17)
        shut = s.jaw < JAW_SHUT and (s.jaw_speed < JAW_SPEED_STILL or s.jaw < JAW_FULLY_SHUT)
        if s.closed:
            if not shut:
                return "closing"
            return "holding" if np.linalg.norm(s.corner - s.tip) < HOLD_RADIUS else "missed"
        if np.linalg.norm(s.corner - self.env.goal(self.moves[prefix])) < PLACED_DIST:
            return "released"
        return "pre_grasp"

    @staticmethod
    def _arrived(s, target):
        err = float(np.max(np.abs(target - s.q)))
        return err < TRACK_TOL or (err < STALL_TOL and float(np.max(np.abs(s.qd))) < STALL_SPEED)

    def _arc_index(self, prefix, s, advance=True):
        """The arc waypoint to aim at: one past the waypoint nearest the arm in joint space (a pure-pursuit carrot,
        so the arm keeps moving forward along the arc and never turns back between two waypoints)."""
        arc = self.plans[prefix]["arc"]
        k = int(np.argmin([np.max(np.abs(s.q - q)) for q in arc]))
        return min(k + 1, len(arc) - 1) if advance else k

    def status(self, prefix, s: ArmState, replan=True):
        """(phase, at the end of the arc) for one arm; re-solves a pre-grasp arm's plan if its corner has moved."""
        phase = self.phase(prefix, s)
        if phase == "pre_grasp" and replan and np.linalg.norm(s.corner - self.plans[prefix]["corner"]) > REPLAN_DIST:
            self._plan(prefix, s.corner)
        at_end = False
        if phase == "holding":
            arc = self.plans[prefix]["arc"]
            k = self._arc_index(prefix, s)
            # let go where the corner is, not where the arm is: the last pose presses into the cloth already folded
            # under it and may never be reached, and an arm can stop at it with its corner still short of the goal
            # (released there, the cloth springs back)
            to_target = s.corner - self.plans[prefix]["corner_target"]
            placed = np.linalg.norm(to_target[:2]) < RELEASE_DIST or (
                self._arrived(s, arc[k]) and np.linalg.norm(to_target) < PLACED_DIST)
            at_end = k == len(arc) - 1 and placed
        return phase, at_end

    def decide(self, prefix, s: ArmState, own, partner):
        """5 joint targets and a gripper command for one arm, given its own and the other arm's status().

        The arms move together, as half_fold_demo's schedule has them: lifting one corner drags the whole far edge,
        the other corner with it, so a holding arm waits at its pinch (or, if it lost its partner mid-carry, where
        it is) until the other arm holds too, and an arm at the end of the arc opens only once the other one is
        there or has let go."""
        phase, at_end = own
        plan = self.plans[prefix]
        above, pinch = plan["above"][0], plan["pinch"][0]
        if phase == "pre_grasp":
            if s.jaw < JAW_OPENED or np.linalg.norm((s.tip - plan["pinch_tip"])[:2]) > ALIGN_DIST:
                return above, GRIPPER_OPEN_CMD
            return pinch, (GRIPPER_CLOSE_CMD if self._arrived(s, pinch) else GRIPPER_OPEN_CMD)
        if phase == "closing":
            return pinch, GRIPPER_CLOSE_CMD
        if phase == "missed":
            return above, GRIPPER_OPEN_CMD
        arc = plan["arc"]
        if phase == "holding":
            if partner[0] not in ("holding", "released"):
                lifted = np.linalg.norm(s.tip - plan["pinch_tip"]) > WAYPOINT_DIST
                return (arc[self._arc_index(prefix, s, advance=False)] if lifted else pinch), GRIPPER_CLOSE_CMD
            if at_end:
                together = partner[0] == "released" or partner[1]
                return arc[-1], (GRIPPER_OPEN_CMD if together else GRIPPER_CLOSE_CMD)
            return arc[self._arc_index(prefix, s)], GRIPPER_CLOSE_CMD
        # released: let the jaws open before backing away, then hold the retreat pose
        return (arc[-1] if s.jaw < JAW_OPENED else plan["retreat"][0]), GRIPPER_OPEN_CMD

    def action_from(self, states: dict, replan=True):
        status = {p: self.status(p, states[p], replan) for p in self.moves}
        action = np.zeros(12, dtype=np.float32)
        for p, other, off in (("left_", "right_", 0), ("right_", "left_", 6)):
            target, grip = self.decide(p, states[p], status[p], status[other])
            # each arm's joints scaled together, so they arrive at once (as half_fold_demo servoes)
            delta = (target - states[p].q) / JOINT_DELTA_SCALE
            limit = CARRY_SPEED if status[p][0] == "holding" else 1.0
            action[off:off + 5] = delta / max(1.0, float(np.max(np.abs(delta))) / limit)
            action[off + 5] = grip
        return action, {p: st[0] for p, st in status.items()}

    # ---- teacher interface (imitation.teachers.base.Teacher) ----
    def act(self):
        return self.action_from({p: self.read(p) for p in self.moves})[0]

    def phases(self):
        return {p: self.phase(p, self.read(p)) for p in self.moves}

    def resync(self):
        """After someone else drove: nothing to re-infer, every step already starts from the live state."""
        return self.phases()

    def label_chunk(self, horizon=16):
        """The next `horizon` actions on the kinematic look-ahead model (module docstring). Leaves the env alone."""
        states = {p: self.read(p) for p in self.moves}
        held = {p: (states[p].corner - states[p].tip) if self.phase(p, states[p]) == "holding" else None
                for p in self.moves}
        chunk = np.zeros((horizon, 12), dtype=np.float32)
        for t in range(horizon):
            # no re-planning inside the look-ahead: an unheld corner does not move in this model
            chunk[t], _ = self.action_from(states, replan=False)
            for p, off in (("left_", 0), ("right_", 6)):
                states[p], held[p] = self._predict(p, states[p], chunk[t, off:off + 6], held[p])
        return chunk

    def _predict(self, prefix, s, a, held):
        q = s.q + a[:5] * JOINT_DELTA_SCALE
        closed = True if a[5] < -0.3 else False if a[5] > 0.3 else s.closed
        goal = GRIPPER_CLOSED if closed else GRIPPER_OPEN
        jaw = s.jaw + float(np.clip(goal - s.jaw, -JAW_RATE, JAW_RATE))
        tip = self.ik.site_pose(prefix, q)[0]
        nxt = ArmState(q=q, qd=(q - s.q) / CONTROL_DT, jaw=jaw, jaw_speed=abs(jaw - s.jaw) / CONTROL_DT,
                       closed=closed, tip=tip, corner=s.corner)
        if not closed:
            held = None
        elif held is None and self.phase(prefix, nxt) == "holding":
            held = s.corner - s.tip
        if held is not None:
            nxt.corner = tip + held
        return nxt, held


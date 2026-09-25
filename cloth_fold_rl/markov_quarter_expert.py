"""Quarter-fold expert whose action is a function of the current state only.

QuarterFoldExpert is a phase machine: which phase it is in depends on what
happened earlier in the episode (patience timeouts, hold gating, the stage-1
retry corrections), and its joint targets come from IK solved when a phase
began. That is fine for driving the arms, but an imitation learner only sees
the state, so labels from the phase machine contradict each other on states
another policy reached (DAgger). This expert reads the phase off the state
every step instead:

    per move, grasped:  corner within RELEASE_DIST of the overshot goal
                          -> hold, or open once every move of the stage is there
                        gripper at the carry point      -> place
                        gripper below lift height       -> lift
                        otherwise                       -> carry
    per move, free:     corners placed (the env's rule), or still
                        settling where they were let go       -> retreat upward
                        gripper just above / within reach     -> descend, closed
                        otherwise                             -> approach

Stage 1 parks the left arm as QuarterFoldExpert does, and the right arm waits
until the left gripper is clear of the stack. A stack that springs off target
after release is simply grasped again (there is no hidden correction).

IK is solved every step from the current joints (part of the state) with a
fixed seed for any restarts, so the joint targets are a function of the state
and stay near the arm's current configuration. (Pulling the redundant joints
toward the home pose tilts the wrist until the jaw meets the table while placing.) The carry point above a far goal can be out of
reach, so "carried" allows for IK's error there.

    python -m cloth_fold_rl.markov_quarter_expert --episodes 5
"""

from __future__ import annotations

import argparse

import numpy as np

from cloth_fold_rl.expert import JOINT_DELTA_SCALE, FoldExpert, solve_ik
from cloth_fold_rl.fold_env import LIFT_TARGET_Z, SUCCESS_DIST
from cloth_fold_rl.quarter_fold_env import CLOTH_10, STAGES, QuarterFoldEnv
from cloth_fold_rl.quarter_fold_expert import CLEAR_DISTANCE, OVERSHOOT, PARK_HEIGHT, corner_index

APPROACH_HEIGHT = 0.06   # FoldExpert's approach point above the corner
DESCEND_RADIUS = 0.02    # a free gripper this close (across) and ...
DESCEND_HEIGHT = 0.07    # ... this low above its corner descends onto it
REACH_RADIUS = 0.03      # or one this close in any direction (inside the 4 cm weld radius)
CARRY_REACHED = 0.02     # a carrying gripper this close across to the goal (plus IK's reach error) lowers the corner
LIFT_MARGIN = 0.03       # a carrying gripper this far below lift height goes straight up
RETREAT_HEIGHT = 0.07    # a released gripper backs off to this height above its corner
# The stacked top layer springs back toward -x, +y after release (QuarterFoldExpert
# instead corrects a missed stack with a hidden retry offset). Over 40 seeds this
# aim folded 29 where QuarterFoldExpert's (0.03, 0, 0) folded 25.
STACK_OVERSHOOT = np.array([0.05, -0.015, 0.0])
RELEASE_DIST = SUCCESS_DIST   # a carried corner this close to its overshot goal is let go
SETTLING_DIST = 0.03     # a released corner still this close to its overshot goal is left to settle


class MarkovQuarterExpert:
    def __init__(self, env, seed=0):
        self.env = env
        self.base = env.unwrapped
        # FoldExpert provides the per-arm IK plumbing; its phase machine is unused
        self.overshoot = {**OVERSHOOT, (1, "right_"): STACK_OVERSHOOT}
        self.arms = {}
        for prefix in ("left_", "right_"):
            self.arms[prefix] = FoldExpert(env, seed=seed, prefix=prefix, corner=corner_index(env, CLOTH_10))
        self.reset()

    # interface shared with QuarterFoldExpert
    def reset(self):
        self._ik_cache = {}
        self._phases = {}
        self.retries = 0    # reported by run_episode; this expert re-grasps without counting

    def resync(self):
        pass

    def use_rest_posture(self):
        pass

    def phases(self):
        return dict(self._phases)

    # ---- helpers ----------------------------------------------------------

    def _site(self, prefix):
        return self.base.data.site_xpos[self.base._site_id[prefix]].copy()

    def _corner(self, move):
        return np.mean([self.env._vertex(c) for c in move.corners], axis=0)

    def _overshot_goal(self, stage, move):
        return self.env.goal(move) + self.overshoot[(stage, move.prefix)]

    def _at_release(self, stage, move):
        corner = self._corner(move)
        return float(np.linalg.norm(corner - self._overshot_goal(stage, move))) < RELEASE_DIST

    def _placed(self, move):
        """The env's own test: every carried corner within SUCCESS_DIST of the goal."""
        return self.env._placed(move)

    def _solve(self, prefix, target):
        """(q, err) from the current joints; cached within a step."""
        key = (prefix, *np.asarray(target, dtype=float).round(6))
        if key not in self._ik_cache:
            arm = self.arms[prefix]
            self._ik_cache[key] = solve_ik(self.base.model, self.base.data, arm.site_id, arm.qpos_adr,
                                           arm.dof_adr, arm.joint_range, target, rng=np.random.default_rng(0))
        return self._ik_cache[key]

    def _joint_action(self, prefix, target):
        arm = self.arms[prefix]
        q_now = np.array([self.base.data.qpos[a] for a in arm.qpos_adr])
        return np.clip((self._solve(prefix, target)[0] - q_now) / JOINT_DELTA_SCALE, -1.0, 1.0)

    def _action(self, prefix, target, gripper_open):
        action = np.zeros(6, dtype=np.float32)
        if target is not None:
            action[0:5] = self._joint_action(prefix, target)
        action[5] = 1.0 if gripper_open else -1.0
        return action

    def _move_action(self, stage, move):
        """(phase, action) for one arm carrying out one move of the stage."""
        prefix = move.prefix
        site = self._site(prefix)
        corner = self._corner(move)
        goal = self._overshot_goal(stage, move)
        if self.base.grasp_active(prefix):
            if self._at_release(stage, move):
                everyone = all(self._at_release(stage, m) if self.base.grasp_active(m.prefix)
                               else self._placed(m) for m in STAGES[stage].moves)
                return ("release", self._action(prefix, None, True)) if everyone \
                    else ("hold", self._action(prefix, None, False))
            carry = np.array([goal[0], goal[1], LIFT_TARGET_Z])
            reach_err = self._solve(prefix, carry)[1]
            if float(np.linalg.norm((site - goal)[:2])) < reach_err + CARRY_REACHED:
                return "place", self._action(prefix, goal + np.array([0.0, 0.0, 0.02]), False)
            if site[2] < LIFT_TARGET_Z - LIFT_MARGIN:
                return "lift", self._action(prefix, np.array([corner[0], corner[1], LIFT_TARGET_Z]), False)
            return "carry", self._action(prefix, carry, False)
        if self._placed(move) or float(np.linalg.norm(corner - goal)) < SETTLING_DIST:
            return "retreat", self._action(prefix, corner + np.array([0.0, 0.0, RETREAT_HEIGHT]), True)
        offset = site - corner
        across = float(np.linalg.norm(offset[:2]))
        if (across < DESCEND_RADIUS and offset[2] < DESCEND_HEIGHT) or float(np.linalg.norm(offset)) < REACH_RADIUS:
            return "descend", self._action(prefix, corner + np.array([0.0, 0.0, 0.005]), False)
        return "approach", self._action(prefix, corner + np.array([0.0, 0.0, APPROACH_HEIGHT]), True)

    def _left_clear(self):
        stack = self.env._vertex(STAGES[1].moves[0].corners[0])
        return float(np.linalg.norm(self._site("left_") - stack)) > CLEAR_DISTANCE

    # ---- policy -----------------------------------------------------------

    def act(self):
        self._ik_cache = {}
        stage = self.env.stage
        if stage == 0:
            actions = {}
            for move in STAGES[0].moves:
                self._phases[move.prefix], actions[move.prefix] = self._move_action(0, move)
            return np.concatenate([actions["left_"], actions["right_"]])
        park = self.env._start[CLOTH_10] + np.array([0.0, 0.0, PARK_HEIGHT])
        left = self._action("left_", park, True)
        self._phases["left_"] = "park"
        if self._left_clear():
            self._phases["right_"], right = self._move_action(1, STAGES[1].moves[0])
        else:
            self._phases["right_"], right = "wait", self._action("right_", None, True)
        return np.concatenate([left, right])


def main():
    from cloth_fold_rl.quarter_fold_expert import run_episode

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    env = QuarterFoldEnv()
    env.unwrapped.domain_randomization = True
    expert = MarkovQuarterExpert(env)
    rows = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        if not args.quiet:
            print(f"episode seed={seed}")
        row = run_episode(env, expert, seed, verbose=not args.quiet)
        rows.append(row)
        print(f"  -> {row}")
    print(f"success {sum(r['success'] for r in rows)}/{len(rows)}, reached stage 1 in "
          f"{sum(r['stage'] >= 1 for r in rows)}, mean steps {np.mean([r['steps'] for r in rows]):.0f}, "
          f"mean score {np.mean([r['fold_score'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()

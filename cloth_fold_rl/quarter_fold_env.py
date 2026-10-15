"""Quarter fold: two arms fold the cloth in half and release, then the same two
arms fold the halved cloth in half again and release. The fold has to hold on
its own.

A wrapper around ClothFoldEnv in the style of fold_env.SingleCornerFoldEnv,
with two stages. Corner positions at reset: cloth_0 (-h, -h), cloth_10
(-h, +h), cloth_110 (+h, -h), cloth_120 (+h, +h), h = 0.15 m. Both arms sit
side-by-side south of the cloth (see mujuco/sim_main.py's ARM_BASE_LEFT/RIGHT),
each reaching its own near south corner -- unlike the old diagonal placement,
where the two arms sat at opposite corners.

  stage 0  the left arm carries cloth_10 onto cloth_0 and the right arm carries
           cloth_120 onto cloth_110 (a fold about y = 0); both release and the
           corners must stay placed for SETTLE_STEPS. cloth_0 and cloth_110 must
           not be dragged. This leaves a half-height sheet along the south edge
           with a two-layer stack at each end: cloth_0+cloth_10 at the west
           corner, cloth_110+cloth_120 at the east corner.
  stage 1  a fold about x = 0: the whole WEST short edge flips onto the EAST
           short edge, both its corners carried at once by different arms
           (a real two-gripper edge fold, not each arm handling its own side).
           The right arm carries the two-layer south stack (cloth_0+cloth_10)
           onto cloth_110's start; the left arm carries the single-layer
           crease-end vertex (cloth_5) onto cloth_115's start. Both release and
           the two carried points must stay placed; the east edge's two
           corners (cloth_110, cloth_120) are the anchors that must not drag.

The base env welds one hard-wired vertex per gripper within GRASP_RADIUS. Here
each gripper's weld list holds its stage-0 corner and its stage-1 corner (see
GRASP_CORNERS). In stage 0 an arm's two weld vertices are 0.30 m apart so only
the near one welds; in stage 1 the arm grasps a stacked south corner, where
both its weld vertices sit within GRASP_RADIUS, so it lifts the whole two-layer
stack. GRASP_RADIUS is 6 cm (> SUCCESS_DIST 5 cm) so both layers still weld
when the stage-0 fold leaves them a few cm apart.

Reward: potential-based shaping summed over the stage's moves. Each move has
three regimes, free (reach the corner) / grasped (carry it) / released and
placed, joined so that grasping and letting go of a placed corner are steps up
and dropping an unplaced corner is a step down. STAGE_BONUS when stage 0
settles, SUCCESS_BONUS when stage 1 does. Drag and instability terminate.

Action: 12 values, left joint deltas(5) + left gripper, right joint deltas(5)
+ right gripper. Observation: the base env's flat state vector without its task
one-hot, goal keypoints set per stage, plus stage and settle progress (139-D).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from cloth_fold_rl.fold_env import (
    CLOTH_JITTER, CTRL_COST, DRAG_LIMIT, GRASP_BONUS, SUCCESS_BONUS, SUCCESS_DIST, W_CARRY, W_REACH,
)
from cloth_params import CLOTH_COUNT, StateOnlyWrapper  # noqa: E402 (path set by fold_env; numpy only, so Isaac Sim can import it)

N = CLOTH_COUNT
CLOTH_0, CLOTH_10, CLOTH_110, CLOTH_120 = 0, N - 1, (N - 1) * N, N * N - 1
# Stage-1 goal vertices, the midpoints of the west/east SHORT edges after the
# stage-0 fold (row-major index = ix*N + iy). CLOTH_5 (-h, 0) is the crease-end
# of the west short edge; CLOTH_115 (+h, 0) is its mirror on the east edge.
CLOTH_5 = (N // 2)               # (-h, 0): west short edge, crease end
CLOTH_115 = (N - 1) * N + N // 2  # (+h, 0): east short edge, crease end
# Each arm welds its stage-0 corner and its stage-1 corner. In stage 0 the two
# weld vertices of an arm are 0.30 m apart (only the near one is in range), so
# one weld each. Stage 1 folds the whole WEST short edge onto the east short
# edge (both its corners: the two-layer south stack and the single-layer
# crease-end vertex) -- a real edge-to-edge fold, not each arm handling its own
# side. Reach is asymmetric though: both west-edge points sit at x=-h, so
# they're near the LEFT arm and a genuine cross-table reach for the RIGHT arm
# (measured via solve_ik: right_ -> west stack 0.5cm, right_ -> west midpoint
# 3.3cm, both well under GRASP_RADIUS). The right arm keeps doing the stack
# (its stage-0 role, and the easier of the two reaches); the left arm -- which
# is already home at the west edge -- carries the lone midpoint vertex across
# to CLOTH_115 (measured reach 4.3cm, also within tolerance).
GRASP_CORNERS = {"left_": (CLOTH_10, CLOTH_5), "right_": (CLOTH_120, CLOTH_0, CLOTH_10)}
GRASP_RADIUS = 0.06      # both layers of a stacked corner weld even if the
                         # stage-0 fold left them a few cm apart
RELEASE_BONUS = 3.0      # potential step for letting go of a placed corner
STAGE_BONUS = 10.0
SETTLE_STEPS = 20        # 1.0 s released and placed before a stage completes
MAX_STEPS = 400
HALF_FOLD_MAX_STEPS = 250        # stage 0 alone; the expert takes 77-214 (retries included)
ACTION_DIM = 12
CORNERS = (CLOTH_0, CLOTH_10, CLOTH_110, CLOTH_120)   # the base env's corner slot order


@dataclass(frozen=True)
class Move:
    prefix: str        # the arm
    corners: tuple     # cloth vertices it carries (reached for at their mean position)
    goal: int          # cloth vertex whose start position they go to
    goal_from: str = "stage"   # "stage": the goal vertex where it lies when the stage starts; "reset": at reset (Q3)
    kind: str = "carry"        # "carry"; "anchor" (R2): press the cloth down at `corners` (goal = itself), jaw open
    tol: float = None          # placed within this of the goal (m); None = SUCCESS_DIST


@dataclass(frozen=True)
class Stage:
    moves: tuple
    anchors: tuple     # (vertex, reference vertex): vertex must stay within DRAG_LIMIT of the reference's start
    checks: tuple = ()  # (vertices, goal vertex): their mean must also lie within SUCCESS_DIST of the goal's stage start


STAGES = (
    Stage(moves=(Move("left_", (CLOTH_10,), CLOTH_0), Move("right_", (CLOTH_120,), CLOTH_110)),
          anchors=((CLOTH_0, CLOTH_0), (CLOTH_110, CLOTH_110))),
    Stage(moves=(Move("right_", (CLOTH_0, CLOTH_10), CLOTH_110), Move("left_", (CLOTH_5,), CLOTH_115)),
          anchors=((CLOTH_110, CLOTH_110), (CLOTH_120, CLOTH_120))),
)

# Phase Q2 (2026-10-13, user): the second fold on Isaac as a bimanual fold-over. Both arms take the west edge, one end
# each, lift it over the crease and lower it onto the east edge. The roles are swapped from STAGES[1]: on LeHome's
# cloth position the left arm carries the near-row stack (0+10) and the right arm the far-row crease end (5); IK reach
# 50/50 for both against 31/50 for the left arm in STAGES[1]'s roles (isaac/reach_check_qf.py, multi-seed IK).
STAGE1_FOLD_OVER = Stage(moves=(Move("left_", (CLOTH_0, CLOTH_10), CLOTH_110), Move("right_", (CLOTH_5,), CLOTH_115)),
                         anchors=((CLOTH_110, CLOTH_110), (CLOTH_120, CLOTH_120)))
STAGES_FOLD_OVER = (STAGES[0], STAGE1_FOLD_OVER)
# Q3 (2026-10-14): the first fold leaves the strip wrinkled into ~7 cm of its 15 cm width (crease 4-5 cm toward the
# arms, bottom corners ~3 cm away from them; results.md), too narrow for both grippers on its west edge. A straighten
# stage between the folds: each arm pulls its own crease end back to where it lay at reset (each on its own side,
# 4-9 cm), flattening the strip; the bottom corners must stay (anchors).
STAGE_STRAIGHTEN = Stage(moves=(Move("left_", (CLOTH_5,), CLOTH_5, "reset"), Move("right_", (CLOTH_115,), CLOTH_115, "reset")),
                         anchors=((CLOTH_0, CLOTH_0), (CLOTH_110, CLOTH_110)))
STAGES_FOLD_OVER_STRAIGHTEN = (STAGES[0], STAGE_STRAIGHTEN, STAGE1_FOLD_OVER)
# Q3 comparison: the parallel second fold. The strip is folded in half again along the same axis; each arm carries its
# crease end onto its own near (bottom) corner. That is short moves in each arm's own lane, and the result is a
# 4-layer strip, not a square.
STAGE1_PARALLEL = Stage(moves=(Move("left_", (CLOTH_5,), CLOTH_0), Move("right_", (CLOTH_115,), CLOTH_110)),
                        anchors=((CLOTH_0, CLOTH_0), (CLOTH_110, CLOTH_110)))
STAGES_PARALLEL = (STAGES[0], STAGE1_PARALLEL)
GRASP_CORNERS_PARALLEL = {"left_": (CLOTH_10, CLOTH_5), "right_": (CLOTH_120, CLOTH_115)}
GRASP_CORNERS_FOLD_OVER_STRAIGHTEN = {"left_": (CLOTH_10, CLOTH_5, CLOTH_0), "right_": (CLOTH_120, CLOTH_115, CLOTH_5)}
GRASP_CORNERS_FOLD_OVER = {"left_": (CLOTH_10, CLOTH_0), "right_": (CLOTH_120, CLOTH_5)}
# R2 (2026-10-15, user: option 1): the square's second fold by the left arm alone. Only the left arm pinches the west
# short edge top-down (the right arm's minimum tilt there is >= 35 deg and every such grip slipped at the lift;
# results.md R2). The left arm carries the west edge's midpoint, both layers (bottom vertex 3, top vertex 7 lies on
# it), onto the east edge's (113); the right arm presses the east half down at vertex 90 (x +0.09) with its jaw open
# and lifts off before the flap lands. Success also needs the west corners on the east ones (a square, not a sag).
CLOTH_W_MID, CLOTH_W_MID_TOP, CLOTH_E_MID = 3, N - 1 - 3, (N - 1) * N + 3
CLOTH_ANCHOR = (N - 3) * N + 2
STAGE1_LEFT_FOLD = Stage(moves=(Move("left_", (CLOTH_W_MID, CLOTH_W_MID_TOP), CLOTH_E_MID),
                                Move("right_", (CLOTH_ANCHOR,), CLOTH_ANCHOR, kind="anchor")),
                         anchors=((CLOTH_110, CLOTH_110), (CLOTH_120, CLOTH_120)),
                         checks=(((CLOTH_0, CLOTH_10), CLOTH_110), ((CLOTH_5,), CLOTH_115)))
STAGES_LEFT_FOLD = (STAGES[0], STAGE1_LEFT_FOLD)
GRASP_CORNERS_LEFT_FOLD = {"left_": (CLOTH_10, CLOTH_W_MID, CLOTH_W_MID_TOP), "right_": (CLOTH_120,)}
# R2 square-up (user 2026-10-15): one mid-edge pinch lets the folded edge collapse, its corners landing 5-10 cm in
# toward the grip (results.md R2 option 1). After the left fold settles (its own move and anchor, no square checks),
# the right arm -- which reaches the east half straight down -- pulls the folded corner stack (0+10, now the top layers
# over 110) onto 110, then the crease end (5) onto 115. The last stage carries the square checks for both.
# the rough fold only has to bring the flap over: its midpoint within ROUGH_TOL (the short place alone is 3.5 cm, and
# 3 traces landed 5.8-11.7 cm off without advancing; trace_sq1)
ROUGH_TOL = 0.10
STAGE1_LEFT_FOLD_ROUGH = Stage(moves=(Move("left_", (CLOTH_W_MID, CLOTH_W_MID_TOP), CLOTH_E_MID, tol=ROUGH_TOL),
                                      STAGE1_LEFT_FOLD.moves[1]),
                               anchors=STAGE1_LEFT_FOLD.anchors)
STAGE_SQUARE_STACK = Stage(moves=(Move("right_", (CLOTH_0, CLOTH_10), CLOTH_110),),
                           anchors=((CLOTH_110, CLOTH_110), (CLOTH_120, CLOTH_120)))
STAGE_SQUARE_CREASE = Stage(moves=(Move("right_", (CLOTH_5,), CLOTH_115),),
                            anchors=((CLOTH_110, CLOTH_110), (CLOTH_120, CLOTH_120)),
                            checks=STAGE1_LEFT_FOLD.checks)
STAGES_LEFT_FOLD_SQUARE = (STAGES[0], STAGE1_LEFT_FOLD_ROUGH, STAGE_SQUARE_STACK, STAGE_SQUARE_CREASE)
GRASP_CORNERS_LEFT_FOLD_SQUARE = {"left_": GRASP_CORNERS_LEFT_FOLD["left_"], "right_": (CLOTH_120, CLOTH_0, CLOTH_10, CLOTH_5)}


class QuarterFoldEnv(gym.Wrapper):
    """Fold in half, release, fold in half again, release. See module docstring."""

    stages = STAGES     # subclasses may run a prefix of the stages (see imitation.tasks.HalfFoldEnv)
    start_hook = None   # fn(env) run at the end of reset, e.g. isaac.diag_clean_fold.clean_start (R0 diagnostic)

    def __init__(self, max_episode_steps=MAX_STEPS, seed=None, cloth_jitter=CLOTH_JITTER, base_env=None,
                 stages=None, carry_obs=False):
        # base_env: any ClothFoldEnv drop-in (e.g. isaac.isaac_env.IsaacClothFoldEnv) built in joint_delta
        # mode with grasp_corners=GRASP_CORNERS and grasp_radius=GRASP_RADIUS; the wrapper reads it through
        # cloth_positions / gripper_position / grasp_active, plus _get_obs, the goal fields and
        # _domain_params for the 139-D observation. max_episode_steps only reaches the default env; a
        # passed-in base_env must already be built with its own (truncation reads base_env.max_episode_steps).
        if base_env is None:
            from sim_main import ClothFoldEnv   # mujoco only where the default env is used
            base_env = ClothFoldEnv(observation_mode="state", action_mode="joint_delta",
                                    max_episode_steps=max_episode_steps, grasp_corners=GRASP_CORNERS,
                                    grasp_radius=GRASP_RADIUS)
        env = base_env
        super().__init__(env)
        # an explicit stages argument wins; otherwise the class attribute stands, so a
        # subclass that sets `stages` keeps it (an instance attribute would shadow it)
        if stages is not None:
            self.stages = tuple(stages)
        self.cloth_jitter = cloth_jitter
        self._rng = np.random.default_rng(seed)
        self._flat = StateOnlyWrapper(env)
        # observation = base flat state minus the task one-hot (constant per env,
        # so dead) plus stage and settle progress, which drive termination and
        # would otherwise be hidden wrapper state (reward non-Markovian). The
        # one-hot opens the task block, which follows proprio and cloth_state in
        # StateOnlyWrapper's key order.
        spaces = env.observation_space
        task_start = spaces["proprio"].shape[0] + spaces["cloth_state"].shape[0]
        self._onehot = slice(task_start, task_start + env.n_tasks)
        dim = self._flat.observation_space.shape[0] - env.n_tasks + 2
        # Q2 obs v2 (imitation.spec.CARRY_OBS_DIM): per arm, the carried point, its goal and goal - point
        self.carry_obs = carry_obs
        dim += 18 if carry_obs else 0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.stage = 0
        self._start = None          # cloth vertex positions at reset, [N*N, 3]
        self._stage_start = None    # cloth vertex positions at the current stage's start
        self._goals = {}            # move -> its goal position, frozen when the move's stage begins
        self._settle_steps = 0
        self._prev_potential = 0.0

    # ---- geometry helpers -------------------------------------------------

    def _vertex(self, index):
        return self.env.cloth_positions()[index]

    def goal(self, move):
        # Q2: a stage's goal is its goal vertex where it lies when the stage starts. Stage 0 is unchanged (the stage
        # start is the reset). For the second fold the first one has moved the cloth by a few cm, so the reset
        # position of 110/115 no longer marks the east edge (traces 2026-10-14: the crease end lay 9 cm from its
        # reset pose).
        # Frozen per move when its stage begins (_begin_stage); a later stage not begun yet uses the reset pose.
        return self._goals[move] if move in self._goals else self._start[move.goal]

    def _begin_stage(self):
        for move in self.stages[self.stage].moves:
            ref = self._start if getattr(move, "goal_from", "stage") == "reset" else self._stage_start
            self._goals[move] = ref[move.goal].copy()

    def _apply_weld_mask(self):
        # Each arm may weld only the corners it is meant to carry this stage.
        # CLOTH_10 lives in both arms' grasp_corners (left carries it in stage 0,
        # the right-arm stack includes it in stage 1); without this mask the left
        # arm welds CLOTH_10 along with CLOTH_5 at the stage-1 grasp and drags it
        # to the wrong goal, so the right arm's stack never places.
        mask = {p: set() for p in self.env.prefixes}
        for move in self.stages[self.stage].moves:
            mask[move.prefix].update(move.corners)
        self.env.weld_mask = mask

    def _set_goals(self):
        # The base env's goal keypoints default to a diagonal fold that is none
        # of our stages. Point each corner slot at its own goal this stage: a
        # carried corner at its move's goal, every other corner where it is now.
        corners = self._stage_start[list(CORNERS)].copy()
        goals = corners.copy()
        for move in self.stages[self.stage].moves:
            for c in move.corners:
                if c in CORNERS:
                    goals[CORNERS.index(c)] = self.goal(move)
        self.env._goal_corners = goals
        self.env._goal_scale = np.maximum(np.linalg.norm(goals - corners, axis=1), 1e-6)

    def _observe(self):
        flat = np.delete(self._flat.observation(self.env._get_obs()), self._onehot)
        extra = [self.stage / len(self.stages), self._settle_steps / SETTLE_STEPS]
        parts = [flat, extra]
        if self.carry_obs:
            parts.append(self._carry_features())
        return np.concatenate(parts).astype(np.float32)

    def _carry_features(self):
        out = np.zeros((2, 3, 3))
        for move in self.stages[self.stage].moves:
            arm = 0 if move.prefix == "left_" else 1
            point = np.mean([self._vertex(c) for c in move.corners], axis=0)
            goal = self.goal(move)
            out[arm] = (point, goal, goal - point)
        return out.ravel()

    def _grasped(self, prefix):
        return self.env.grasp_active(prefix)

    def _move_distance(self, move):
        """Mean distance of the move's corners to its goal."""
        return float(np.mean([np.linalg.norm(self._vertex(c) - self.goal(move)) for c in move.corners]))

    def _spread(self, move):
        pts = [self._vertex(c) for c in move.corners]
        return float(max((np.linalg.norm(a - b) for i, a in enumerate(pts) for b in pts[i + 1:]), default=0.0))

    def _placed(self, move):
        tol = SUCCESS_DIST if getattr(move, "tol", None) is None else move.tol
        return all(np.linalg.norm(self._vertex(c) - self.goal(move)) < tol for c in move.corners)

    def _start_distance(self, move):
        return max(float(np.mean([np.linalg.norm(self._start[c] - self.goal(move)) for c in move.corners])), 1e-6)

    def _anchor_drift(self):
        # reference is the vertex's position at the START of the current stage,
        # not at reset: an anchor corner may have been legitimately relocated by
        # an earlier stage (e.g. CLOTH_120 is folded onto CLOTH_110 in stage 0,
        # ~0.30 m from its reset pose). What must not move is the already-placed
        # cloth as THIS stage runs, so we measure drift from the stage's start.
        return max(float(np.linalg.norm(self._vertex(v) - self._stage_start[ref])) for v, ref in self.stages[self.stage].anchors)

    def fold_score(self):
        """Fraction of the two stages' total carry distance that has been covered."""
        done = sum(self._start_distance(m) for s in self.stages[:self.stage] for m in s.moves)
        current = sum(self._start_distance(m) - min(self._move_distance(m), self._start_distance(m))
                      for m in self.stages[self.stage].moves)
        total = sum(self._start_distance(m) for s in self.stages for m in s.moves)
        return float((done + current) / total)

    # ---- reward -----------------------------------------------------------

    def _move_potential(self, move):
        d = self._move_distance(move)
        if self._grasped(move.prefix):
            return GRASP_BONUS - W_CARRY * d
        if self._placed(move):
            return GRASP_BONUS + RELEASE_BONUS - W_CARRY * d
        gripper = self.env.gripper_position(move.prefix)
        reach = float(np.linalg.norm(gripper - np.mean([self._vertex(c) for c in move.corners], axis=0)))
        return -W_REACH * reach - W_CARRY * self._start_distance(move)

    def _potential(self):
        return sum(self._move_potential(m) for m in self.stages[self.stage].moves)

    def _check_met(self, vertices, goal):
        return float(np.linalg.norm(np.mean([self._vertex(v) for v in vertices], axis=0)
                                    - self._stage_start[goal])) < SUCCESS_DIST

    def _stage_settled(self):
        stage = self.stages[self.stage]
        return (all(not self._grasped(m.prefix) and self._placed(m) for m in stage.moves)
                and all(self._check_met(v, g) for v, g in stage.checks))

    # ---- gym API ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.cloth_jitter > 0 and "cloth_pose" not in opts:
            opts["cloth_pose"] = self._rng.uniform(-self.cloth_jitter, self.cloth_jitter, size=2)
        _, info = self.env.reset(seed=seed, options=opts)
        if "cloth_pose" in opts:
            # the base env only records offsets it draws itself
            self.env._domain_params["cloth_offset_xy"] = [float(v) for v in np.ravel(opts["cloth_pose"])[:2]]
        self._start = self.env.cloth_positions()
        self._stage_start = self._start.copy()
        self.stage = 0
        self._goals = {}
        self._begin_stage()
        self._apply_weld_mask()
        self._set_goals()
        self._settle_steps = 0
        self._prev_potential = self._potential()
        if self.start_hook is not None:     # opt-in diagnostic start (isaac.diag_clean_fold), None by default
            self.start_hook(self)
        info = dict(info)
        info["goal_keypoints"] = self.env._goal_corners.copy()
        info["domain_parameters"] = dict(self.env._domain_params)
        info.update(self._info(None))
        return self._observe(), info

    def _expand(self, action):
        a = np.zeros(14, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).ravel()
        a[0:5], a[6] = action[0:5], action[5]       # left joint deltas, left gripper
        a[7:12], a[13] = action[6:11], action[11]   # right joint deltas, right gripper
        return a

    def _info(self, reason):
        return {"stage": self.stage, "fold_score": self.fold_score(), "settle_steps": self._settle_steps,
                "grasped": {p: self._grasped(p) for p in self.env.prefixes},
                "move_distance": [self._move_distance(m) for m in self.stages[self.stage].moves],
                # Q2: how far apart the vertices a move carries together are (a stack's two layers); 0 for one vertex
                "carried_spread": [self._spread(m) for m in self.stages[self.stage].moves],
                "anchor_drift": self._anchor_drift(), "success": reason == "success",
                "termination_reason": reason}

    def step(self, action):
        _, _, _, _, info = self.env.step(self._expand(action))

        potential = self._potential()
        reward = potential - self._prev_potential - CTRL_COST * float(np.square(action).sum())
        self._settle_steps = self._settle_steps + 1 if self._stage_settled() else 0

        terminated, reason = False, None
        if self._settle_steps >= SETTLE_STEPS:
            if self.stage == len(self.stages) - 1:
                reward += SUCCESS_BONUS
                terminated, reason = True, "success"
            else:
                reward += STAGE_BONUS
                self.stage += 1
                self._stage_start = self.env.cloth_positions()
                self._begin_stage()
                self._apply_weld_mask()
                self._set_goals()
                self._settle_steps = 0
                potential = self._potential()
        elif self._anchor_drift() > DRAG_LIMIT:
            reward -= 5.0
            terminated, reason = True, "cloth_dragged"
        elif self.env._failed():
            reward -= 5.0
            terminated, reason = True, "unstable"
        self._prev_potential = potential
        truncated = (not terminated) and self.env._step_count >= self.env.max_episode_steps

        info = dict(info)
        info.update(self._info(reason))
        return self._observe(), float(reward), terminated, truncated, info


class HalfFoldEnv(QuarterFoldEnv):
    """Stage 0 of the quarter fold on its own: fold the cloth in half about y = 0.

    Success state: the left arm has carried cloth_10 to within SUCCESS_DIST (5 cm)
    of cloth_0's start and the right arm cloth_120 to within 5 cm of cloth_110's
    start, BOTH grippers are open (the fold holds on its own, not in the jaws),
    and that has held for SETTLE_STEPS (20 control steps, 1.0 s) in a row. The
    episode fails first if cloth_0 or cloth_110 is dragged more than DRAG_LIMIT
    (0.20 m) from its start, or the cloth goes unstable. Success pays
    SUCCESS_BONUS and info["success"] is True; fold_score is the fraction of the
    two corners' carry distance covered.
    """

    def __init__(self, max_episode_steps=HALF_FOLD_MAX_STEPS, seed=None, cloth_jitter=CLOTH_JITTER,
                 base_env=None):
        super().__init__(max_episode_steps=max_episode_steps, seed=seed, cloth_jitter=cloth_jitter,
                         base_env=base_env, stages=STAGES[:1])

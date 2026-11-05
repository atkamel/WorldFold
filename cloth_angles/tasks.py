"""Fold tasks as seen by the world model: state layout, arms, staged moves, success rule.

The state vector is 363 cloth vertex coordinates followed by 15 robot values
per arm (joint pos 5, joint vel 5, gripper command, end-effector position 3,
grasp flag). An episode's goal vector holds the start positions of the goal
vertices, 3 values per goal slot; it comes from the env at reset.

    single   the left arm carries cloth_10 to cloth_120's start and holds it (MuJoCo)
    half     cloth_fold_rl.quarter_fold_env.STAGES[0]: both arms fold the y > 0 edge over and release
    quarter  all of STAGES: the half fold, then the west short edge onto the east one

half and quarter are built from STAGES, so they cannot drift from the env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cloth_fold_rl import quarter_fold_env

VERTEX_DIM = 363
ROBOT_DIM = 15
CORNER_LIST = (0, 10, 110, 120)


def corner_slot(vertex: int) -> int:
    return CORNER_LIST.index(vertex)


@dataclass(frozen=True)
class Arm:
    prefix: str
    robot: int          # offset of its robot block in the state vector
    gripper: int        # index of its gripper command in the action
    joints: slice       # its joint deltas in the action
    weld_corners: tuple # cloth vertices its gripper welds to when closed within grasp_radius (MuJoCo's weld)

    @property
    def ctrl_index(self) -> int:
        return self.robot + 10

    @property
    def ee_slice(self) -> slice:
        return slice(self.robot + 11, self.robot + 14)

    @property
    def grasp_index(self) -> int:
        return self.robot + 14


@dataclass(frozen=True)
class Move:
    arm: int            # index into Task.arms
    corners: tuple      # cloth vertices carried, reached for at their mean
    goal: int           # goal slot in the goal vector


@dataclass(frozen=True)
class Stage:
    moves: tuple
    anchors: tuple      # (vertex, reference vertex): vertex stays within DRAG_LIMIT of where the reference was when
                        # the stage began


@dataclass(frozen=True)
class Task:
    name: str
    arms: tuple
    stages: tuple
    goal_corners: tuple   # corner per goal slot
    action_dim: int
    grasp_radius: float
    release: bool         # placed corners must be let go (else held) to settle
    settle_steps: int     # consecutive settled steps that complete a stage
    max_steps: int        # the env's episode length

    @property
    def state_dim(self) -> int:
        return VERTEX_DIM + ROBOT_DIM * len(self.arms)

    @property
    def goal_dim(self) -> int:
        return 3 * len(self.goal_corners)

    def make_env(self, max_steps=None, cloth_jitter=None):
        kwargs = {} if cloth_jitter is None else {"cloth_jitter": cloth_jitter}
        kwargs["max_episode_steps"] = self.max_steps if max_steps is None else max_steps
        if self.name == "single":
            from cloth_fold_rl.fold_env import SingleCornerFoldEnv
            return SingleCornerFoldEnv(**kwargs)
        if self.name == "half":
            return quarter_fold_env.HalfFoldEnv(**kwargs)
        return quarter_fold_env.QuarterFoldEnv(**kwargs)

    def make_expert(self, env, seed):
        if self.name == "single":
            from cloth_fold_rl.expert import FoldExpert
            return FoldExpert(env, seed=seed)
        from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
        return QuarterFoldExpert(env, seed=seed)

    def corner_starts(self, env) -> np.ndarray:
        """float32[4, 3]: start positions of CORNER_LIST, from a reset env."""
        if self.name == "single":
            return np.asarray(env._anchors0, dtype=np.float32)
        return np.asarray(env._start[list(CORNER_LIST)], dtype=np.float32)

    def goals(self, env) -> np.ndarray:
        """float32[goal_dim]: start positions of the goal vertices."""
        if self.name == "single":
            starts = self.corner_starts(env)
            return np.concatenate([starts[corner_slot(c)] for c in self.goal_corners])
        return np.asarray(env._start[list(self.goal_corners)], dtype=np.float32).reshape(-1)

    def stage(self, info) -> int:
        return int(info.get("stage", 0))

    def grasped(self, info) -> bool:
        g = info["grasped"]
        return any(g.values()) if isinstance(g, dict) else bool(g)


LEFT = Arm("left_", VERTEX_DIM, 5, slice(0, 5), weld_corners=(10,))
SINGLE = Task("single", arms=(LEFT,),
              stages=(Stage(moves=(Move(0, (10,), 0),), anchors=((0, 0), (110, 110))),),
              goal_corners=(120,), action_dim=6, grasp_radius=0.03, release=False, settle_steps=10, max_steps=200)

PREFIXES = ("left_", "right_")     # the two-arm state and action order


def from_stages(name, stages, max_steps) -> Task:
    """A two-arm Task from quarter_fold_env stages: one goal slot per distinct goal vertex in order of appearance,
    each arm welding every vertex it carries in any stage."""
    goals = tuple(dict.fromkeys(m.goal for st in stages for m in st.moves))
    arms = tuple(Arm(p, VERTEX_DIM + ROBOT_DIM * i, 6 * i + 5, slice(6 * i, 6 * i + 5),
                     weld_corners=tuple(dict.fromkeys(c for st in stages for m in st.moves if m.prefix == p
                                                      for c in m.corners)))
                 for i, p in enumerate(PREFIXES))
    return Task(name, arms=arms,
                stages=tuple(Stage(moves=tuple(Move(PREFIXES.index(m.prefix), m.corners, goals.index(m.goal))
                                               for m in st.moves), anchors=st.anchors) for st in stages),
                goal_corners=goals, action_dim=quarter_fold_env.ACTION_DIM, grasp_radius=quarter_fold_env.GRASP_RADIUS,
                release=True, settle_steps=quarter_fold_env.SETTLE_STEPS, max_steps=max_steps)


HALF = from_stages("half", quarter_fold_env.STAGES[:1], quarter_fold_env.HALF_FOLD_MAX_STEPS)
QUARTER = from_stages("quarter", quarter_fold_env.STAGES, quarter_fold_env.MAX_STEPS)
TASKS = {"single": SINGLE, "half": HALF, "quarter": QUARTER}

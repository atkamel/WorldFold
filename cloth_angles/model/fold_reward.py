"""Fold-env reward and termination, computed from state vectors for any Task.

The world model predicts physical state (cloth vertices, arm states, grasp
flags), so the fold reward can be evaluated with the environments' own
formulas instead of a learned head. This mirrors cloth_fold_rl/fold_env.py
and cloth_fold_rl/quarter_fold_env.py: potential shaping per move (free /
grasped / released-and-placed regimes), control cost, settle counter, stage
bonus, success bonus, drag penalty. The one rule it cannot reproduce is the
"unstable" termination, which reads what the state omits (MuJoCo's joint
accelerations, Isaac's particle speeds and every particle's position).

The reward machine carries a stage, a settle counter and the cloth as it was
when the stage began per batch row (`init`, `step`), because a task's stage
and its anchors depend on the trajectory so far.
"""

from __future__ import annotations

import torch

from cloth_angles.tasks import SINGLE, VERTEX_DIM, Arm, Task
from cloth_fold_rl.fold_env import CTRL_COST, DRAG_LIMIT, GRASP_BONUS, SUCCESS_BONUS, SUCCESS_DIST, W_CARRY, W_REACH
from cloth_fold_rl.quarter_fold_env import RELEASE_BONUS, STAGE_BONUS

DRAG_PENALTY = 5.0


def _vertex(state: torch.Tensor, index: int) -> torch.Tensor:
    return state[..., 3 * index:3 * index + 3]


class FoldReward:
    """Per-episode constants (goals, start positions) plus the batched reward rules.

    goal: [goal_dim] or [B, goal_dim]; starts: [N, 3] or [B, N, 3], every cloth
    vertex's position at reset (an episode's first state). A batched spec pairs
    each batch row of the states with its own episode.

    soft=True blends the grasped and free potentials by the clamped grasp value
    instead of thresholding it. On real states (flag exactly 0 or 1) this is
    identical; on predicted states it lets reward gradients reach the gripper
    action through the grasp prediction. Placement, success and drag rules
    always use hard thresholds.
    """

    def __init__(self, goal, starts, task: Task = SINGLE, soft: bool = False):
        self.task, self.soft = task, soft
        goal = torch.as_tensor(goal, dtype=torch.float32)
        self.goal = goal.reshape(*goal.shape[:-1], len(task.goal_corners), 3)
        self.starts = torch.as_tensor(starts, dtype=torch.float32)

    # ---- geometry ------------------------------------------------------------

    def grasped(self, state, arm: Arm):
        return state[..., arm.grasp_index] > 0.5

    def _goal(self, move):
        return self.goal[..., move.goal, :]

    def _corner_distances(self, state, move):
        return torch.stack([(_vertex(state, c) - self._goal(move)).norm(dim=-1) for c in move.corners], -1)

    def move_distance(self, state, move):
        return self._corner_distances(state, move).mean(-1)

    def placed(self, state, move):
        return (self._corner_distances(state, move) < SUCCESS_DIST).all(-1)

    def start_distance(self, move):
        return torch.stack([(self.starts[..., c, :] - self._goal(move)).norm(dim=-1)
                            for c in move.corners], -1).mean(-1).clamp(min=1e-6)

    def _per_stage(self, fn, state, stage):
        """Evaluate fn(state, stage_index) for every stage and pick each row's own."""
        values = torch.stack([fn(state, s) for s in range(len(self.task.stages))], -1)
        index = stage.expand(values.shape[:-1]) if stage.dim() < values.dim() - 1 else stage
        return values.gather(-1, index.unsqueeze(-1)).squeeze(-1)

    # ---- potential -----------------------------------------------------------

    def move_potential(self, state, move):
        arm = self.task.arms[move.arm]
        d = self.move_distance(state, move)
        carried = GRASP_BONUS - W_CARRY * d
        reach = (state[..., arm.ee_slice] - torch.stack([_vertex(state, c) for c in move.corners]).mean(0)).norm(dim=-1)
        free = -W_REACH * reach - W_CARRY * self.start_distance(move)
        if self.task.release:
            free = torch.where(self.placed(state, move), GRASP_BONUS + RELEASE_BONUS - W_CARRY * d, free)
        weight = state[..., arm.grasp_index].clamp(0.0, 1.0) if self.soft else self.grasped(state, arm).float()
        return weight * carried + (1.0 - weight) * free

    def stage_potential(self, state, s: int):
        return sum(self.move_potential(state, m) for m in self.task.stages[s].moves)

    def potential(self, state, stage):
        return self._per_stage(self.stage_potential, state, stage)

    def settled(self, state, stage):
        def one(state, s):
            moves = self.task.stages[s].moves
            placed = torch.stack([self.placed(state, m) for m in moves], -1).all(-1)
            held = torch.stack([self.grasped(state, self.task.arms[m.arm]) for m in moves], -1)
            return placed & (~held if self.task.release else held).all(-1)
        return self._per_stage(one, state, stage)

    def anchor_drift(self, state, stage, stage_start):
        """stage_start: [B, N, 3], the cloth when the current stage began."""
        def one(state, s):
            return torch.stack([(_vertex(state, v) - stage_start[..., ref, :]).norm(dim=-1)
                                for v, ref in self.task.stages[s].anchors], -1).amax(-1)
        return self._per_stage(one, state, stage)

    # ---- reward machine --------------------------------------------------------

    def init(self, stage, stage_start=None):
        """stage: long [B], the task stage at the first state; stage_start: [B, N, 3], the cloth when that stage
        began (default the reset starts, exact for stage 0)."""
        stage = torch.as_tensor(stage, dtype=torch.long)
        if stage_start is None:
            stage_start = self.starts.expand(*stage.shape, *self.starts.shape[-2:])
        return {"stage": stage, "hold": torch.zeros_like(stage), "stage_start": stage_start}

    def step(self, state, next_state, action, rs):
        """One transition -> reward [B], terminated [B], next machine state."""
        stage, hold = rs["stage"], rs["hold"]
        reward = self.potential(next_state, stage) - self.potential(state, stage) - CTRL_COST * action.pow(2).sum(-1)
        hold = torch.where(self.settled(next_state, stage), hold + 1, torch.zeros_like(hold))
        complete = hold >= self.task.settle_steps
        last = stage == len(self.task.stages) - 1
        success, advance = complete & last, complete & ~last
        dragged = ~complete & (self.anchor_drift(next_state, stage, rs["stage_start"]) > DRAG_LIMIT)
        reward = reward + SUCCESS_BONUS * success.float() + STAGE_BONUS * advance.float() - DRAG_PENALTY * dragged.float()
        cloth = next_state[..., :VERTEX_DIM].reshape(*next_state.shape[:-1], -1, 3)
        new = {"stage": torch.where(advance, stage + 1, stage),
               "hold": torch.where(advance, torch.zeros_like(hold), hold),
               "stage_start": torch.where(advance[..., None, None], cloth, rs["stage_start"])}
        return reward, success | dragged, new

    def rollout(self, states, actions, stage=None):
        """states: [B, T+1, D], actions: [B, T, A] -> rewards [B, T], terminated [B, T], stages [B, T+1].

        Rewards after a termination are still computed; callers mask them.
        """
        batch, time = actions.shape[:2]
        rs = self.init(torch.zeros(batch, dtype=torch.long) if stage is None else stage)
        rewards, terminated, stages = [], [], [rs["stage"]]
        for t in range(time):
            reward, term, rs = self.step(states[:, t], states[:, t + 1], actions[:, t], rs)
            rewards.append(reward)
            terminated.append(term)
            stages.append(rs["stage"])
        return torch.stack(rewards, 1), torch.stack(terminated, 1), torch.stack(stages, 1)


"""Analytic fold reward mirrors the fold envs' rules, for the single and quarter tasks."""

import torch

from cloth_angles.model.fold_reward import FoldReward
from cloth_angles.tasks import LEFT, QUARTER, SINGLE
from cloth_fold_rl.fold_env import CTRL_COST, DRAG_LIMIT, GRASP_BONUS, SUCCESS_BONUS, W_CARRY, W_REACH
from cloth_fold_rl.quarter_fold_env import RELEASE_BONUS, STAGE_BONUS


def flat_state(corner, ee, grasp, anchor0=(0.0, 0.0, 0.0)):
    state = torch.zeros(SINGLE.state_dim)
    state[0:3] = torch.tensor(anchor0)
    state[30:33] = torch.tensor(corner)          # cloth_10
    state[LEFT.ee_slice] = torch.tensor(ee)
    state[LEFT.grasp_index] = grasp
    return state


def spec():
    # cloth_0 and cloth_110 both sit at the origin in these synthetic states
    starts = torch.zeros(121, 3)
    starts[10], starts[120] = torch.tensor([0.0, 0.3, 0.0]), torch.tensor([0.3, 0.3, 0.0])
    return FoldReward(goal=[0.3, 0.3, 0.0], starts=starts)


def test_potential_switches_regime_on_grasp():
    r = spec()
    zero = torch.zeros(1, dtype=torch.long)
    free = flat_state((0.0, 0.3, 0.0), (0.1, 0.3, 0.0), 0.0)
    held = flat_state((0.0, 0.3, 0.0), (0.0, 0.3, 0.0), 1.0)
    torch.testing.assert_close(r.potential(free[None], zero), torch.tensor([-W_REACH * 0.1 - W_CARRY * 0.3]))
    torch.testing.assert_close(r.potential(held[None], zero), torch.tensor([GRASP_BONUS - W_CARRY * 0.3]))


def test_step_reward_is_potential_difference_minus_control_cost():
    r = spec()
    s0 = flat_state((0.0, 0.3, 0.0), (0.0, 0.3, 0.0), 1.0)
    s1 = flat_state((0.1, 0.3, 0.0), (0.1, 0.3, 0.0), 1.0)
    reward, terminated, _ = r.step(s0[None], s1[None], torch.ones(1, 6), r.init([0]))
    torch.testing.assert_close(reward, torch.tensor([W_CARRY * 0.1 - CTRL_COST * 6]))
    assert not terminated.item()


def test_success_bonus_after_hold_and_drag_penalty():
    r = spec()
    placed = flat_state((0.3, 0.3, 0.0), (0.3, 0.3, 0.0), 1.0)
    states = placed.expand(12, -1)[None]
    rewards, terminated, stages = r.rollout(states, torch.zeros(1, 11, 6))
    assert terminated[0, 9] and not terminated[0, 8]
    assert rewards[0, 9].item() == SUCCESS_BONUS and rewards[0, 8].item() == 0.0
    assert (stages == 0).all()
    dragged = flat_state((0.3, 0.3, 0.0), (0.3, 0.3, 0.0), 1.0, anchor0=(DRAG_LIMIT + 0.01, 0.0, 0.0))
    rewards, terminated, _ = r.rollout(torch.stack([placed, dragged])[None], torch.zeros(1, 1, 6))
    assert terminated[0, 0] and rewards[0, 0].item() == -5.0


def test_soft_potential_matches_hard_on_real_flags_and_blends_between():
    hard, soft = spec(), spec()
    soft.soft = True
    zero = torch.zeros(1, dtype=torch.long)
    for flag in (0.0, 1.0):
        state = flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), flag)[None]
        torch.testing.assert_close(soft.potential(state, zero), hard.potential(state, zero))
    half = flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), 0.5)[None]
    lo = hard.potential(flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), 0.0)[None], zero)
    hi = hard.potential(flat_state((0.1, 0.3, 0.0), (0.2, 0.3, 0.0), 1.0)[None], zero)
    torch.testing.assert_close(soft.potential(half, zero), 0.5 * (lo + hi))


H = 0.15
GRID = torch.stack(torch.meshgrid(torch.linspace(-H, H, 11), torch.linspace(-H, H, 11), indexing="ij"), -1)
STARTS = torch.cat([GRID.reshape(121, 2), torch.zeros(121, 1)], -1)        # index = ix * 11 + iy, as the env's grid
GOAL = torch.cat([STARTS[0], STARTS[110], STARTS[115]])                    # QUARTER's goal slots: 0, 110, 115


def quarter_state(c0, c10, c110, c120, left_grasp=0.0, right_grasp=0.0, right_ee=(0.0, 0.0, 0.0), c5=STARTS[5]):
    state = torch.zeros(QUARTER.state_dim)
    state[:363] = STARTS.reshape(-1)
    for vertex, pos in ((0, c0), (10, c10), (110, c110), (120, c120), (5, c5)):
        state[3 * vertex:3 * vertex + 3] = torch.as_tensor(pos)
    left, right = QUARTER.arms
    state[left.grasp_index], state[right.grasp_index] = left_grasp, right_grasp
    state[right.ee_slice] = torch.as_tensor(right_ee)
    return state


def test_quarter_stage_advances_with_bonus_then_succeeds():
    r = FoldReward(goal=GOAL, starts=STARTS, task=QUARTER)
    s0, s110 = STARTS[0], STARTS[110]
    half = quarter_state(s0, s0, s110, s110)                              # both corners placed, released
    rewards, terminated, stages = r.rollout(half.expand(22, -1)[None], torch.zeros(1, 21, 12))
    assert stages[0, 19] == 0 and stages[0, 20] == 1                      # advance after 20 settled steps
    assert rewards[0, 19].item() == STAGE_BONUS and not terminated.any()
    # stage 1: stack at (h, -h) and cloth_5 on cloth_115 released -> success after another 20 steps; cloth_120, folded
    # onto cloth_110 in stage 0, is anchored where the stage began, not at its reset start
    quarter = quarter_state(s110, s110, s110, s110, c5=STARTS[115])
    states = torch.cat([half.expand(21, -1), quarter.expand(21, -1)])[None]
    rewards, terminated, stages = r.rollout(states, torch.zeros(1, 41, 12))
    assert terminated[0, 39] and rewards[0, 39].item() == SUCCESS_BONUS      # 20 settled steps in stage 1
    assert not terminated[0, :39].any() and stages[0, 20] == 1


def test_quarter_release_regime_is_a_step_up_from_carrying():
    r = FoldReward(goal=GOAL, starts=STARTS, task=QUARTER)
    one = torch.ones(1, dtype=torch.long)
    s110 = STARTS[110]
    held = quarter_state(s110, s110, s110, s110, right_grasp=1.0, right_ee=s110)
    released = quarter_state(s110, s110, s110, s110, right_grasp=0.0, right_ee=s110)
    torch.testing.assert_close(r.potential(released[None], one) - r.potential(held[None], one), torch.tensor([RELEASE_BONUS]))

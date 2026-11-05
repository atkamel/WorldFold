"""Lambda returns, squashed actor, policy features, and imagination updates on tiny models."""

import torch

from cloth_angles.model.actor_critic import (
    Actor, Critic, ImaginationTrainer, feature_dim, lambda_returns, policy_features,
)
from cloth_angles.model.fold_reward import FoldReward
from cloth_angles.model.grasp_head import GraspHead
from cloth_angles.model.state_predictor import ResidualStatePredictor
from cloth_angles.tasks import QUARTER, SINGLE
from cloth_fold_rl.quarter_fold_env import STAGE_BONUS


def test_lambda_returns_reduce_to_discounted_sum_when_lambda_is_one():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[9.0, 9.0, 9.0, 4.0]])
    continues = torch.ones(1, 3)
    returns = lambda_returns(rewards, values, continues, gamma=0.5, lam=1.0)
    expected = torch.tensor([[1 + 0.5 * (2 + 0.5 * (3 + 0.5 * 4)), 2 + 0.5 * (3 + 0.5 * 4), 3 + 0.5 * 4]])
    torch.testing.assert_close(returns, expected)


def test_lambda_returns_stop_at_termination():
    rewards = torch.tensor([[1.0, 5.0]])
    values = torch.tensor([[0.0, 7.0, 7.0]])
    continues = torch.tensor([[0.0, 1.0]])
    returns = lambda_returns(rewards, values, continues, gamma=0.9, lam=0.8)
    assert returns[0, 0].item() == 1.0


def test_actor_outputs_stay_in_action_range():
    actor = Actor(input_dim=4, action_dim=2)
    features = torch.randn(16, 4) * 50
    assert actor(features).abs().max() <= 1.0
    sample, entropy = actor.sample(features)
    assert sample.abs().max() <= 1.0 and torch.isfinite(entropy)


def test_single_task_features_keep_the_original_layout():
    state = torch.randn(2, SINGLE.state_dim)
    goal = torch.tensor([[0.15, 0.1, 0.43]]).expand(2, 3)
    features = policy_features(state, goal, torch.zeros(SINGLE.state_dim), torch.ones(SINGLE.state_dim))
    assert features.shape == (2, feature_dim(SINGLE)) and feature_dim(SINGLE) == SINGLE.state_dim + 6
    torch.testing.assert_close(features[:, :SINGLE.state_dim], state)
    torch.testing.assert_close(features[:, -3:], (state[:, 30:33] - goal) * 10)


def test_quarter_features_follow_the_stage():
    state = torch.randn(3, QUARTER.state_dim)
    goal = torch.randn(3, QUARTER.goal_dim)
    stage = torch.tensor([0, 1, 0])
    features = policy_features(state, goal, torch.zeros(QUARTER.state_dim), torch.ones(QUARTER.state_dim), QUARTER, stage)
    assert features.shape == (3, feature_dim(QUARTER))
    offsets = features[:, -8:-2].reshape(3, 2, 3)
    # stage 0: left carries cloth_10 to slot 0, right carries cloth_120 to slot 1
    torch.testing.assert_close(offsets[0, 0], (state[0, 30:33] - goal[0, :3]) * 10)
    torch.testing.assert_close(offsets[0, 1], (state[0, 360:363] - goal[0, 3:6]) * 10)
    # stage 1: left carries cloth_5 to slot 2, right carries the mean of cloth_0 and cloth_10 to slot 1
    torch.testing.assert_close(offsets[1, 0], (state[1, 15:18] - goal[1, 6:9]) * 10)
    torch.testing.assert_close(offsets[1, 1], ((state[1, 0:3] + state[1, 30:33]) / 2 - goal[1, 3:6]) * 10)
    torch.testing.assert_close(features[:, -2:], torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))


def make_trainer(task, horizon=3, **kwargs):
    torch.manual_seed(0)
    world_model = ResidualStatePredictor(task.state_dim, task.action_dim, hidden_dim=16)
    world_model.fit_normalizer(torch.randn(20, task.state_dim) * 0.1, torch.randn(20, task.state_dim) * 0.1)
    head = GraspHead(world_model.net[0].in_features, len(task.arms), hidden_dim=16)
    actor, critic = Actor(feature_dim(task), task.action_dim, hidden_dim=16), Critic(feature_dim(task), hidden_dim=16)
    return ImaginationTrainer(world_model, head, actor, critic, task, horizon=horizon, **kwargs), actor, world_model


def test_imagination_update_changes_actor_and_returns_metrics():
    trainer, actor, world_model = make_trainer(SINGLE)
    before = [p.clone() for p in actor.parameters()]
    state = torch.randn(4, SINGLE.state_dim) * 0.1
    goal = torch.tensor([[0.15, 0.1, 0.43]]).expand(4, 3)
    starts = torch.zeros(4, 121, 3)
    metrics = trainer.update(state, state, goal, starts)
    assert set(metrics) >= {"actor_loss", "critic_loss", "mean_return", "imagined_grasp_rate", "imagined_stage_advance"}
    assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))
    assert all(not p.requires_grad for p in world_model.parameters())


def test_quarter_imagination_update_runs_from_either_stage():
    trainer, actor, _ = make_trainer(QUARTER)
    state = torch.randn(4, QUARTER.state_dim) * 0.1
    goal = torch.randn(4, QUARTER.goal_dim)
    starts = torch.zeros(4, 121, 3)
    metrics = trainer.update(state, state, goal, starts, stage=torch.tensor([0, 1, 0, 1]),
                             anchor=(state, goal, torch.tensor([0, 1, 0, 1]), torch.zeros(4, QUARTER.action_dim)))
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_ensemble_disagreement_penalizes_reward():
    torch.manual_seed(0)
    members = []
    for k in range(3):
        m = ResidualStatePredictor(SINGLE.state_dim, 6, hidden_dim=16)
        m.fit_normalizer(torch.randn(20, SINGLE.state_dim) * 0.1, torch.randn(20, SINGLE.state_dim) * 0.1)
        with torch.no_grad():
            m.net[-1].weight.normal_(std=0.1 * (k + 1))
        members.append(m)
    actor, critic = Actor(feature_dim(SINGLE), 6, hidden_dim=16), Critic(feature_dim(SINGLE), hidden_dim=16)
    head = GraspHead(members[0].net[0].in_features, 1, hidden_dim=16)
    state = torch.randn(4, SINGLE.state_dim) * 0.1
    goal = torch.tensor([[0.15, 0.1, 0.43]]).expand(4, 3)
    starts = torch.zeros(4, 121, 3)
    torch.manual_seed(1)
    plain = ImaginationTrainer(members[0], head, actor, critic, horizon=3, ensemble=members, disagreement_coef=0.0)
    plain_metrics = plain.update(state, state, goal, starts)
    torch.manual_seed(1)
    penalized = ImaginationTrainer(members[0], head, actor, critic, horizon=3, ensemble=members,
                                   disagreement_coef=5.0)
    penalized_metrics = penalized.update(state, state, goal, starts)
    assert plain_metrics["disagreement"] > 0
    assert penalized_metrics["mean_reward"] < plain_metrics["mean_reward"]


def test_grasp_flags_come_from_the_head_and_carry_the_gradient():
    trainer, _, world_model = make_trainer(QUARTER)
    state, prev = torch.randn(3, QUARTER.state_dim) * 0.1, torch.randn(3, QUARTER.state_dim) * 0.1
    action = (torch.rand(3, 12) * 2 - 1).requires_grad_()
    nxt = trainer.step(state, prev, action)
    flags = [a.grasp_index for a in QUARTER.arms]
    expected = torch.sigmoid(trainer.grasp_head(world_model._features(state, prev, action)))
    torch.testing.assert_close(nxt[:, flags], expected)
    rest = [i for i in range(QUARTER.state_dim) if i not in flags]
    torch.testing.assert_close(nxt[:, rest], world_model(state, prev, action)[:, rest])
    grad, = torch.autograd.grad(nxt[:, flags].sum(), action)
    assert grad.abs().sum() > 0                                   # the soft flag passes the actor's gradient


def test_imagined_stage_advances_once_the_stage_settles():
    trainer, _, _ = make_trainer(QUARTER, horizon=2)
    trainer.world_model = ResidualStatePredictor(QUARTER.state_dim, 12, hidden_dim=8)     # exact persistence
    trainer.ensemble = [trainer.world_model]
    with torch.no_grad():
        trainer.grasp_head.net[-1].weight.zero_()
        trainer.grasp_head.net[-1].bias.fill_(-10.0)                                       # both jaws released
    grid = torch.stack(torch.meshgrid(torch.linspace(-0.15, 0.15, 11), torch.linspace(-0.15, 0.15, 11),
                                      indexing="ij"), -1).reshape(121, 2)
    starts = torch.cat([grid, torch.zeros(121, 1)], -1)
    state = torch.zeros(1, QUARTER.state_dim)
    state[0, :363] = starts.reshape(-1)
    state[0, 30:33], state[0, 360:363] = starts[0], starts[110]          # the half fold placed, released
    goal = torch.cat([starts[0], starts[110], starts[115]])
    spec = FoldReward(goal, starts, QUARTER)
    spec.flat_goal = goal[None]                                          # as update() sets it
    rs = spec.init(torch.zeros(1, dtype=torch.long))
    rs["hold"] = torch.full((1,), QUARTER.settle_steps - 1)
    _, _, rewards, terminated, stages, _ = trainer.imagine(state, state, spec, rs)
    assert stages[0].tolist() == [0, 1, 1] and not terminated[0, 0]
    assert rewards[0, 0] > STAGE_BONUS - 1.0

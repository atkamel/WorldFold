"""The grasp head: shapes, change weighting, event metrics, training in the predictor's run, checkpoint round trip."""

import numpy as np
import pytest
import torch

from cloth_angles.data.state_episode import StateEpisode
from cloth_angles.model.grasp_events import grasp_counts, grasp_events, grasp_rates
from cloth_angles.model.grasp_head import GraspHead, change_weights, weighted_loss
from cloth_angles.tasks import HALF
from cloth_angles.train_state_predictor import load_checkpoint, save_checkpoint, train


def test_head_gives_a_logit_per_arm_and_keeps_its_weights():
    head = GraspHead(input_dim=10, n_arms=2, hidden_dim=8)
    assert head(torch.randn(5, 10)).shape == (5, 2)
    head.change_weight.copy_(torch.tensor([3.0, 4.0]))
    assert torch.equal(head.state_dict()["change_weight"], torch.tensor([3.0, 4.0]))


def test_changes_weigh_as_much_as_steady_transitions():
    now = np.array([[0, 0], [0, 1], [1, 1], [1, 1], [0, 1]], dtype=np.float32)
    nxt = np.array([[0, 0], [1, 1], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    np.testing.assert_allclose(change_weights(now, nxt), [4.0, 4.0])          # 1 change, 4 steady per arm
    np.testing.assert_allclose(change_weights(now, now), [1.0, 1.0])          # no change: no reweighting
    logits = torch.tensor(np.random.default_rng(0).normal(size=(5, 2)), dtype=torch.float32)
    w = torch.tensor([4.0, 4.0])
    manual = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.as_tensor(nxt), weight=torch.where(torch.as_tensor(now != nxt), w, torch.ones(5, 2)))
    torch.testing.assert_close(weighted_loss(logits, torch.as_tensor(now), torch.as_tensor(nxt), w), manual)


def test_events_split_losses_by_the_gripper_command():
    now = np.array([0, 1, 1, 1, 0])
    nxt = np.array([1, 0, 0, 1, 0])
    command = np.array([-1.0, 1.0, -1.0, -1.0, 0.2])                           # 0.2: hysteresis, not open
    e = grasp_events(now, nxt, command)
    assert e["onset"].tolist() == [True, False, False, False, False]
    assert e["release"].tolist() == [False, True, False, False, False]
    assert e["slip"].tolist() == [False, False, True, False, False]


def test_counts_and_rates():
    now = np.array([0, 0, 1, 1, 1, 0])
    true = np.array([1, 1, 0, 0, 1, 0])
    pred = np.array([1, 0, 0, 1, 0, 1])                # onsets: 1 of 2 found, 1 false; release found; 1 false slip
    command = np.array([-1, -1, 1, -1, -1, -1.0])
    rates = grasp_rates(grasp_counts(now, true, pred, command))
    assert rates["accuracy"] == pytest.approx(2 / 6)
    assert rates["onset"] == {"precision": 0.5, "recall": 0.5, "n": 2}
    assert rates["release"] == {"precision": 1.0, "recall": 1.0, "n": 1}
    assert rates["slip"] == {"precision": 0.0, "recall": 0.0, "n": 1}
    persistence = grasp_rates(grasp_counts(now, true, now, command))
    assert persistence["onset"]["precision"] is None and persistence["onset"]["recall"] == 0.0


def _episodes(n=6, T=40, seed=0):
    """The left flag follows the left gripper command one step later, the right one stays shut: a learnable rule."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        actions = rng.uniform(-1, 1, (T, 12)).astype(np.float32)
        robot = np.zeros((T + 1, 30), dtype=np.float32)
        robot[1:, 14] = actions[:, 5] < 0
        robot[:, 29] = 1.0
        vertices = np.zeros((T + 1, 121, 3), dtype=np.float32)
        out.append(StateEpisode(vertices, robot, actions, {"split": "train"}))
    return out


def test_head_trains_with_the_predictor_and_round_trips(tmp_path):
    episodes = _episodes()
    model, head, _ = train(episodes, HALF, steps=300, hidden_dim=32, seed=0, log_every=0)
    changed = np.concatenate([ep.robot[1:, 14] != ep.robot[:-1, 14] for ep in episodes])
    np.testing.assert_allclose(head.change_weight, [(~changed).sum() / changed.sum(), 1.0], rtol=1e-6)
    ep = _episodes(n=1, seed=1)[0]
    states = torch.as_tensor(ep.states())
    prev = torch.cat([states[:1], states[:-2]])
    with torch.no_grad():
        logits = head(model._features(states[:-1], prev, torch.as_tensor(ep.actions)))
    accuracy = ((logits > 0).numpy() == (states[1:, [HALF.arms[0].grasp_index, HALF.arms[1].grasp_index]] > 0.5)
                .numpy()).mean()
    assert accuracy > 0.9
    path = tmp_path / "wm.pt"
    save_checkpoint(model, head, path, "state_l1", "half", 300, 0, 32)
    model2, head2, saved = load_checkpoint(path)
    assert saved["task"] == "half"
    with torch.no_grad():
        torch.testing.assert_close(head2(model2._features(states[:-1], prev, torch.as_tensor(ep.actions))), logits)
    _, no_head, _ = train(_episodes(n=2), HALF, variant="vertex_l1", steps=2, hidden_dim=8, log_every=0)
    assert no_head is None

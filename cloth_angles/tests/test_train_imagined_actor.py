"""The imagination script's data, its bias log and one tiny end-to-end run on a synthetic half-fold store."""

import json
import sys

import numpy as np
import torch

from cloth_angles import train_imagined_actor as tia
from cloth_angles.data.state_episode import StateEpisode, StateEpisodeStore
from cloth_angles.model.state_predictor import ResidualStatePredictor
from cloth_angles.tasks import HALF, QUARTER
from cloth_angles.train_state_predictor import save_checkpoint, train


def episode(kind="expert", split="train", T=30, stage_from=None, seed=0):
    rng = np.random.default_rng(seed)
    vertices = rng.normal(0, 0.05, (T + 1, 121, 3)).astype(np.float32)
    robot = rng.normal(0, 0.1, (T + 1, 30)).astype(np.float32)
    robot[:, [14, 29]] = (rng.random((T + 1, 2)) < 0.3).astype(np.float32)
    stage = np.zeros(T + 1, dtype=np.int64)
    if stage_from is not None:
        stage[stage_from:] = 1
    meta = {"kind": kind, "split": split, "goal": vertices[0][[0, 110]].ravel().tolist()}
    return StateEpisode(vertices, robot, rng.uniform(-1, 1, (T, 12)).astype(np.float32), meta,
                        rewards=np.zeros(T, np.float32), terminated=np.zeros(T, bool), stage=stage)


def test_transitions_keep_the_task_stages_with_their_starts():
    ep = episode(T=30, stage_from=20)
    half = tia.task_transitions([ep], HALF)
    assert len(half["cur"]) == 20 and (half["stage"] == 0).all()             # HALF has stage 0 only
    np.testing.assert_array_equal(half["starts"][7].numpy(), ep.vertices[0])
    np.testing.assert_array_equal(half["stage_start"][7].numpy(), ep.vertices[0])
    np.testing.assert_array_equal(half["prev"][5].numpy(), ep.states()[4])
    ep.metadata["goal"] += [0.0, 0.0, 0.0]                                   # QUARTER's third goal slot
    quarter = tia.task_transitions([ep], QUARTER)
    assert len(quarter["cur"]) == 30
    np.testing.assert_array_equal(quarter["stage_start"][25].numpy(), ep.vertices[20])   # where stage 1 began


def test_bias_is_the_signed_mean_per_action_dimension():
    data = tia.task_transitions([episode()], HALF)
    model = ResidualStatePredictor(HALF.state_dim, 12, hidden_dim=8)
    offset = torch.linspace(-0.2, 0.2, 12)
    bias = tia.action_bias(lambda features: data["act"] + offset, model, HALF, data)
    np.testing.assert_allclose(bias["per_dim"], offset.numpy(), atol=1e-4)
    assert abs(bias["mean_abs"] - float(offset.abs().mean())) < 1e-4


def test_clone_then_imagination_runs_and_logs_the_bias(tmp_path, monkeypatch):
    store = StateEpisodeStore(tmp_path / "data")
    eps = [episode("expert", "train", seed=1), episode("expert_noisy", "train", seed=2),
           episode("expert", "test", seed=3), episode("play", "train", seed=4)]
    for ep in eps:
        store.append(ep)
    model, head, _ = train(eps[:2], HALF, steps=5, hidden_dim=16, log_every=0)
    save_checkpoint(model, head, tmp_path / "wm.pt", "state_l1", "half", 5, 0, 16)
    out = tmp_path / "actor"
    monkeypatch.setattr(sys, "argv", ["x", "--model", str(tmp_path / "wm.pt"), "--data", str(tmp_path / "data"),
                                      "--output", str(out), "--bc-steps", "5", "--imagine-steps", "4",
                                      "--horizon", "2", "--batch", "8", "--log-every", "2"])
    tia.main()
    log = json.loads((out / "log.json").read_text())
    assert [row["stage"] for row in log] == ["init", "clone", "imagination", "imagination"]
    assert all(len(row["bias"]["per_dim"]) == 12 for row in log)
    assert {"actor_loss", "imagined_grasp_rate", "mean_return"} <= set(log[-1])
    actor, mean, scale, task = tia.load_actor(out / "imagined.pt")
    assert task is HALF and torch.equal(mean, model.state_mean) and actor(torch.zeros(1, 405)).shape == (1, 12)

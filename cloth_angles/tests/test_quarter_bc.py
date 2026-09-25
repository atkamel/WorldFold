"""Dataset loading, event balancing, BC losses and evaluation summaries."""

import json

import numpy as np
import torch

from cloth_angles.data.state_episode import StateEpisode, StateEpisodeStore
from cloth_angles.tasks import QUARTER
from scripts.audit_fold_dataset import audit_dataset
from scripts.evaluate_quarter_policy import summarize
from scripts.train_quarter_bc import action_losses, load_split


def make_episode(length=4, split="train"):
    vertices = np.zeros((length + 1, 121, 3), np.float32)
    robot = np.zeros((length + 1, 30), np.float32)
    actions = np.zeros((length, 12), np.float32)
    actions[1, 5] = -1.0
    robot[2:, 14] = 1.0
    stage = np.array([0, 0, 1, 1, 1], np.int64)[:length + 1]
    return StateEpisode(vertices, robot, actions,
                        {"task": "quarter", "kind": "expert", "split": split, "length": length,
                         "success": False, "final_stage": int(stage[-1]), "goal": [0.0] * 6,
                         "anchors0": [[0.0] * 3 for _ in range(4)]},
                        rewards=np.zeros(length, np.float32), terminated=np.zeros(length, np.bool_), stage=stage)


def write_dataset(path):
    store = StateEpisodeStore(path)
    rows = []
    for split in ("train", "test"):
        episode = make_episode(split=split)
        episode.metadata["file"] = store.append(episode).name
        rows.append(episode.metadata)
    (path / "manifest.json").write_text(json.dumps(rows))


def test_audit_and_bc_loader(tmp_path):
    write_dataset(tmp_path)
    summary, errors = audit_dataset(tmp_path, "quarter")
    assert not errors
    assert summary["episodes"] == 2
    assert summary["transitions_by_stage"] == {"0": 4, "1": 4}
    data = load_split(tmp_path, "train")
    assert data["state"].shape == (4, QUARTER.state_dim)
    assert data["action"].shape == (4, QUARTER.action_dim)
    assert data["priority"].max() == 8.0


def test_bc_loader_success_only(tmp_path):
    store = StateEpisodeStore(tmp_path)
    for success in (False, True, True):
        episode = make_episode()
        episode.metadata["success"] = success
        store.append(episode)
    assert load_split(tmp_path, "train")["state"].shape[0] == 12
    assert load_split(tmp_path, "train", success_only=True)["state"].shape[0] == 8
    relabelled = make_episode()
    relabelled.metadata["kind"] = "dagger"
    relabelled.labels = np.ones_like(relabelled.actions)
    store.append(relabelled)
    data = load_split(tmp_path, "train", success_only=True)
    assert data["state"].shape[0] == 12
    assert (data["action"][-4:] == 1.0).all()


def test_gripper_loss_is_weighted():
    prediction = torch.zeros(2, 12)
    target = torch.zeros(2, 12)
    target[:, 5] = -1.0
    loss1, _ = action_losses(prediction, target, 1.0)
    loss3, metrics = action_losses(prediction, target, 3.0)
    assert loss3 > loss1
    assert metrics["gripper_accuracy"] == 0.0


def test_evaluation_summary_failure_funnel():
    base = {"left_grasp": True, "right_grasp": True, "reached_stage1": True,
            "stage1_right_grasp": True, "stage1_placement": True, "fold_score": 0.8, "return": 1.0}
    rows = [{**base, "success": True, "failure": "success"},
            {**base, "success": False, "failure": "stage1_unsettled_or_timeout"}]
    result = summarize(rows)
    assert result["stage0_completion_rate"] == 1.0
    assert result["success_rate"] == 0.5
    assert result["failure_counts"] == {"success": 1, "stage1_unsettled_or_timeout": 1}

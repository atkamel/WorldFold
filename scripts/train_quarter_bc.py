"""Train a two-stage quarter-fold policy by cloning recorded expert actions.

This is deliberately independent of world-model/imagination training:

    .venv/bin/python scripts/train_quarter_bc.py \
        --data outputs/cloth_angles/quarter_expert_v1 [outputs/cloth_angles/quarter_dagger_r1 ...] \
        --output outputs/cloth_angles/quarter_policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloth_angles.data.state_episode import StateEpisodeStore
from cloth_angles.model.actor_critic import Actor, feature_dim, policy_features
from cloth_angles.tasks import QUARTER

JOINT_ACTIONS = torch.tensor([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
GRIPPER_ACTIONS = torch.tensor([5, 11])


def load_split(paths: Path | list[Path], split: str, success_only: bool = False) -> dict[str, torch.Tensor]:
    paths = [paths] if isinstance(paths, Path) else paths
    episodes = [ep for path in paths for ep in StateEpisodeStore(path).load_all() if ep.metadata.get("split") == split]
    if success_only:
        # The scripted expert fails some demos; cloning those teaches bad folds. Episodes
        # driven by anything else are kept: their labels are the expert's corrections.
        episodes = [ep for ep in episodes if ep.metadata.get("kind") != "expert" or ep.metadata.get("success")]
    if not episodes:
        raise ValueError(f"dataset has no {'successful ' if success_only else ''}episodes with split={split!r}")
    for episode in episodes:
        if episode.metadata.get("task") != "quarter":
            raise ValueError("train_quarter_bc.py only accepts quarter-fold episodes")

    states = [ep.states() for ep in episodes]
    current = np.concatenate([s[:-1] for s in states])
    nxt = np.concatenate([s[1:] for s in states])
    actions = np.concatenate([ep.expert_actions() for ep in episodes])
    stages = np.concatenate([ep.stage[:-1] for ep in episodes])
    next_stages = np.concatenate([ep.stage[1:] for ep in episodes])
    goals = np.concatenate([
        np.repeat(np.asarray(ep.metadata["goal"], np.float32)[None], len(ep), axis=0)
        for ep in episodes
    ])

    priorities = []
    for episode, state, next_state in zip(episodes, states, [s[1:] for s in states]):
        priority = np.ones(len(episode), np.float32)
        current_state = state[:-1]
        for arm in QUARTER.arms:
            grasp_changed = ((current_state[:, arm.grasp_index] > 0.5)
                             != (next_state[:, arm.grasp_index] > 0.5))
            command_changed = np.zeros(len(episode), dtype=np.bool_)
            command_changed[1:] = np.abs(np.diff(episode.expert_actions()[:, arm.gripper])) > 0.5
            priority[np.logical_or(grasp_changed, command_changed)] = 5.0
        priority[episode.stage[:-1] != episode.stage[1:]] = 8.0
        priorities.append(priority)
    priority = np.concatenate(priorities)

    return {name: torch.as_tensor(value) for name, value in {
        "state": current, "action": actions, "goal": goals, "stage": stages,
        "priority": priority,
    }.items()}


def normalization(data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mean = data["state"].mean(0)
    scale = data["state"].std(0).clamp(min=1e-4)
    return mean, scale


def action_losses(prediction: torch.Tensor, target: torch.Tensor, gripper_weight: float) -> tuple[torch.Tensor, dict]:
    joint_loss = F.mse_loss(prediction[:, JOINT_ACTIONS], target[:, JOINT_ACTIONS])
    gripper_loss = F.mse_loss(prediction[:, GRIPPER_ACTIONS], target[:, GRIPPER_ACTIONS])
    total = joint_loss + gripper_weight * gripper_loss
    active = target[:, GRIPPER_ACTIONS].abs() >= 0.5
    correct = (prediction[:, GRIPPER_ACTIONS] < 0) == (target[:, GRIPPER_ACTIONS] < 0)
    accuracy = correct[active].float().mean().item() if active.any() else 1.0
    return total, {"loss": total.item(), "joint_loss": joint_loss.item(),
                   "gripper_loss": gripper_loss.item(), "gripper_accuracy": accuracy}


@torch.no_grad()
def validate(actor: Actor, data: dict[str, torch.Tensor], state_mean: torch.Tensor,
             state_scale: torch.Tensor, stage_id: int, gripper_weight: float) -> dict:
    mask = data["stage"] == stage_id
    if not mask.any():
        raise ValueError(f"validation split has no stage {stage_id} transitions")
    features = policy_features(data["state"][mask], data["goal"][mask], state_mean, state_scale,
                               QUARTER, data["stage"][mask].long())
    _, metrics = action_losses(actor(features), data["action"][mask], gripper_weight)
    return metrics


def train_stage(stage_id: int, train: dict[str, torch.Tensor], validation: dict[str, torch.Tensor],
                state_mean: torch.Tensor, state_scale: torch.Tensor, args) -> tuple[Actor, list[dict]]:
    mask = train["stage"] == stage_id
    if not mask.any():
        raise ValueError(f"training split has no stage {stage_id} transitions")
    stage_data = {key: value[mask] for key, value in train.items()}
    probability = stage_data["priority"].numpy().astype(np.float64)
    probability /= probability.sum()

    torch.manual_seed(args.seed + stage_id)
    actor = Actor(feature_dim(QUARTER), QUARTER.action_dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + stage_id)
    history: list[dict] = []
    best_state = None
    best_validation = float("inf")

    for step in range(1, args.steps + 1):
        indexes = torch.as_tensor(rng.choice(len(probability), args.batch_size, replace=True, p=probability))
        features = policy_features(stage_data["state"][indexes], stage_data["goal"][indexes],
                                   state_mean, state_scale, QUARTER, stage_data["stage"][indexes].long())
        loss, train_metrics = action_losses(actor(features), stage_data["action"][indexes], args.gripper_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        optimizer.step()

        if step == 1 or step % args.validate_every == 0 or step == args.steps:
            actor.eval()
            val_metrics = validate(actor, validation, state_mean, state_scale, stage_id, args.gripper_weight)
            row = {"step": step, "stage": stage_id, **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"validation_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            print(" ".join(f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
                           for key, value in row.items()), flush=True)
            if val_metrics["loss"] < best_validation:
                best_validation = val_metrics["loss"]
                best_state = {key: value.detach().clone() for key, value in actor.state_dict().items()}
            actor.train()

    actor.load_state_dict(best_state)
    actor.eval()
    return actor, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000, help="optimizer updates per stage")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gripper-weight", type=float, default=3.0)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true",
                        help="also clone plain expert episodes that did not succeed")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.validate_every < 1:
        raise SystemExit("steps, batch-size and validate-every must be positive")

    torch.set_num_threads(4)
    started = time.perf_counter()
    success_only = not args.include_failures
    train = load_split(args.data, "train", success_only)
    validation = load_split(args.data, "test", success_only)
    state_mean, state_scale = normalization(train)
    actors, history = [], []
    for stage_id in range(len(QUARTER.stages)):
        count = int((train["stage"] == stage_id).sum())
        print(f"training stage {stage_id} on {count} transitions", flush=True)
        actor, stage_history = train_stage(stage_id, train, validation, state_mean, state_scale, args)
        actors.append(actor)
        history.extend(stage_history)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "bc_staged.pt"
    config = {key: [str(v) for v in value] if key == "data" else str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    torch.save({"actors": [actor.state_dict() for actor in actors], "state_mean": state_mean,
                "state_scale": state_scale, "training_config": config}, checkpoint)
    (args.output / "training_history.json").write_text(json.dumps(history, indent=2))
    (args.output / "training_config.json").write_text(json.dumps(config, indent=2))
    print(f"saved {checkpoint} in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

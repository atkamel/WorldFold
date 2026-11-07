"""Clone, then train in imagination, a fold task's actor on a trained world model and its grasp head.

    python -m cloth_angles.train_imagined_actor --model outputs/cloth_angles/isaac_wm_model/state_l1_seed0.pt \
        --data outputs/cloth_angles/isaac_wm_a --output outputs/cloth_angles/isaac_imagined

docs/world_model.md sections 5, 6 and 8, for the checkpoint's task (the half fold on Isaac). The world model and the
grasp head are frozen. Start states come from the task's own training episodes (every kind but play, since the
reward is the task's); play data only trains the model. Only states in a stage the task has are used.

1. Clone. The actor regresses the clone kind's actions (default expert) on its training states, --bc-steps updates.
2. Imagination from the clone (cloth_angles.model.actor_critic.ImaginationTrainer). Batches of real start states
   are rolled --horizon steps through the model, the grasp head setting every arm's flag, and scored by
   FoldReward(task) with the episode's goal (metadata) and starts (its first state's vertices), the stage and the
   cloth at the stage start carried through the rollout. The actor is also pulled toward the clone kind's actions
   with weight --anchor (section 6's anchor, its round 0).

The actor sees the full state and the goal every step. After the clone and every --log-every imagination updates it
logs its deviation from the data actions on the clone kind's held-out episodes, per action dimension (signed: a bias
that walks the arms off the cloth shows here, section 7) and as a mean absolute value. Writes clone.pt and
imagined.pt (actor, the policy-feature normalizer, task) and log.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloth_angles.model.actor_critic import (  # noqa: E402
    Actor, Critic, ImaginationTrainer, feature_dim, policy_features,
)
from cloth_angles.tasks import TASKS  # noqa: E402
from cloth_angles.train_state_predictor import load_checkpoint, load_split  # noqa: E402

BC_BATCH = 512
LEARNING_RATE = 3e-4


def task_transitions(episodes, task):
    """Tensors over every transition whose state is in one of task's stages: cur, prev, act, goal, starts (the
    episode's first-state vertices), stage, stage_start (the vertices at the first state of that stage)."""
    parts = {k: [] for k in ("cur", "prev", "act", "goal", "starts", "stage", "stage_start")}
    for ep in episodes:
        states = ep.states()
        stage = ep.stage if ep.stage is not None else np.zeros(len(ep) + 1, dtype=np.int64)
        keep = stage[:-1] < len(task.stages)
        first = np.array([np.argmax(stage == s) for s in stage[:-1]])
        n = int(keep.sum())
        parts["cur"].append(states[:-1][keep])
        parts["prev"].append(np.concatenate([states[:1], states[:-2]])[keep])
        parts["act"].append(ep.actions[keep])
        parts["goal"].append(np.repeat(np.asarray(ep.metadata["goal"], dtype=np.float32)[None], n, 0))
        parts["starts"].append(np.repeat(ep.vertices[:1], n, 0))
        parts["stage"].append(stage[:-1][keep])
        parts["stage_start"].append(ep.vertices[first[keep]])
    return {k: torch.as_tensor(np.concatenate(v)) for k, v in parts.items()}


def features(model, task, data, idx=slice(None)):
    return policy_features(data["cur"][idx], data["goal"][idx], model.state_mean, model.state_scale, task,
                           data["stage"][idx])


@torch.no_grad()
def action_bias(actor, model, task, data):
    """The actor's deviation from the data actions: mean per action dimension (signed) and mean absolute."""
    diff = actor(features(model, task, data)) - data["act"]
    return {"per_dim": [round(float(v), 4) for v in diff.mean(0)], "mean_abs": round(float(diff.abs().mean()), 4)}


def behavior_clone(actor, model, task, data, steps, seed):
    optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(data["cur"]), BC_BATCH))
        loss = torch.nn.functional.mse_loss(actor(features(model, task, data, idx)), data["act"][idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return loss.item()


def imagine(trainer, data, anchor_data, steps, seed, batch, log_every, held, log):
    rng = np.random.default_rng(seed)
    history = []
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(data["cur"]), batch))
        j = torch.as_tensor(rng.integers(0, len(anchor_data["cur"]), batch))
        anchor = (anchor_data["cur"][j], anchor_data["goal"][j], anchor_data["stage"][j], anchor_data["act"][j])
        history.append(trainer.update(data["cur"][idx], data["prev"][idx], data["goal"][idx], data["starts"][idx],
                                      data["stage"][idx], anchor=anchor, stage_start=data["stage_start"][idx]))
        if step % log_every == 0 or step == steps:
            row = {"stage": "imagination", "step": step,
                   **{k: round(float(np.mean([m[k] for m in history[-log_every:]])), 4) for k in history[-1]},
                   "bias": action_bias(trainer.actor, trainer.world_model, trainer.task, held)}
            log.append(row)
            print(json.dumps(row), flush=True)


def save_actor(path, actor, model, task):
    torch.save({"actor": actor.state_dict(), "state_mean": model.state_mean, "state_scale": model.state_scale,
                "task": task.name}, path)


def load_actor(path):
    """-> (actor in eval mode, state_mean, state_scale, task) from save_actor's file."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    task = TASKS[saved["task"]]
    actor = Actor(feature_dim(task), task.action_dim)
    actor.load_state_dict(saved["actor"])
    return actor.eval(), saved["state_mean"], saved["state_scale"], task


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="world model + grasp head checkpoint")
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="StateEpisodeStore directories")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--clone-kind", default="expert")
    ap.add_argument("--bc-steps", type=int, default=3000)
    ap.add_argument("--imagine-steps", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--anchor", type=float, default=1.0, help="weight of the pull toward the clone kind's actions")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model, head, saved = load_checkpoint(args.model)
    if head is None:
        ap.error(f"{args.model} has no grasp head (variant {saved['variant']})")
    task = TASKS[saved["task"]]
    train_eps, test_eps = load_split(args.data)
    starts = task_transitions([ep for ep in train_eps if ep.metadata.get("kind") != "play"], task)
    clone = task_transitions([ep for ep in train_eps if ep.metadata.get("kind") == args.clone_kind], task)
    held = task_transitions([ep for ep in test_eps if ep.metadata.get("kind") == args.clone_kind], task)
    print(f"task {task.name}: {len(starts['cur'])} start states, {len(clone['cur'])} {args.clone_kind} transitions "
          f"to clone, {len(held['cur'])} held out", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    actor = Actor(feature_dim(task), task.action_dim)
    log = [{"stage": "init", "bias": action_bias(actor, model, task, held)}]
    loss = behavior_clone(actor, model, task, clone, args.bc_steps, args.seed)
    log.append({"stage": "clone", "steps": args.bc_steps, "loss": round(loss, 5),
                "bias": action_bias(actor, model, task, held)})
    print(json.dumps(log[-1]), flush=True)
    save_actor(args.output / "clone.pt", actor, model, task)
    trainer = ImaginationTrainer(model, head, actor, Critic(feature_dim(task)), task, horizon=args.horizon,
                                 bc_coef=args.anchor)
    imagine(trainer, starts, clone, args.imagine_steps, args.seed, args.batch, args.log_every, held, log)
    save_actor(args.output / "imagined.pt", actor, model, task)
    (args.output / "log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {args.output / 'clone.pt'} and {args.output / 'imagined.pt'}", flush=True)


if __name__ == "__main__":
    main()

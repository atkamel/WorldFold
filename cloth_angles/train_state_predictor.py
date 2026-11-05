"""Train the residual state predictor (cloth_angles.model.state_predictor) on stored state episodes.

    python -m cloth_angles.train_state_predictor --data outputs/cloth_angles/isaac_wm_a \
        outputs/cloth_angles/isaac_wm_b --output outputs/cloth_angles/isaac_wm_model/state_l1_seed0.pt

Trains one model on every "train" episode of the given stores (StateEpisodeStore directories, MuJoCo's or Isaac's,
e.g. Isaac play and half-fold episodes together) with the recipe of docs/world_model.md: AdamW 3e-4, random batches
of 512 transitions, 10,000 updates. Variants: state_l1 (cloth vertices and robot state, L1 loss), vertex_l1 (vertices
only), state_mse. The checkpoint carries what the online runner needs: variant, task (its state layout), dimensions,
hidden size, and the normalizer in the state_dict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloth_angles.data.state_episode import StateEpisodeStore  # noqa: E402
from cloth_angles.model.state_predictor import ResidualStatePredictor  # noqa: E402
from cloth_angles.tasks import TASKS, VERTEX_DIM  # noqa: E402

VARIANTS = ("state_l1", "vertex_l1", "state_mse")
BATCH_SIZE = 512
LEARNING_RATE = 3e-4


def load_split(data_dirs):
    """-> (train episodes, test episodes) over every store, by each episode's metadata split."""
    episodes = [ep for d in data_dirs for ep in StateEpisodeStore(d).load_all()]
    if not episodes:
        raise ValueError(f"no episodes under {list(map(str, data_dirs))}")
    return ([ep for ep in episodes if ep.metadata["split"] == "train"],
            [ep for ep in episodes if ep.metadata["split"] == "test"])


def check_layout(episodes, task):
    """Every episode has the task's state and action sizes."""
    for ep in episodes:
        if ep.states().shape[1] != task.state_dim or ep.actions.shape[1] != task.action_dim:
            raise ValueError(f"{ep.metadata.get('file', 'an episode')}: state/action sizes "
                             f"{ep.states().shape[1]}/{ep.actions.shape[1]}, task {task.name} has "
                             f"{task.state_dim}/{task.action_dim}")


def sequence(ep, variant):
    """float32[T+1, S]: the episode's states as the variant sees them."""
    states = ep.states()
    return states[:, :VERTEX_DIM] if variant.startswith("vertex") else states


def transitions(episodes, variant):
    """(state, previous state, action, next state) tensors over every transition; the first previous is the state."""
    cur, prev, act, nxt = [], [], [], []
    for ep in episodes:
        seq = sequence(ep, variant)
        cur.append(seq[:-1])
        prev.append(np.concatenate([seq[:1], seq[:-2]]))
        act.append(ep.actions)
        nxt.append(seq[1:])
    return tuple(torch.as_tensor(np.concatenate(x)) for x in (cur, prev, act, nxt))


def train(episodes, variant="state_l1", steps=10_000, seed=0, hidden_dim=256, device="cpu", log_every=1000):
    """-> (the trained model on the CPU, mean loss of the last 100 updates)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    cur, prev, act, nxt = (x.to(device) for x in transitions(episodes, variant))
    torch.manual_seed(seed)
    model = ResidualStatePredictor(cur.shape[1], act.shape[1], hidden_dim=hidden_dim, loss=variant.split("_")[1])
    model.fit_normalizer(cur.cpu(), nxt.cpu())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, len(cur), BATCH_SIZE), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(cur[idx], prev[idx], act[idx], nxt[idx])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"{variant} seed={seed} step={step}/{steps} loss={np.mean(losses[-log_every:]):.4f}", flush=True)
    return model.cpu().eval(), float(np.mean(losses[-100:]))


def save_checkpoint(model, path, variant, task, steps, seed, hidden_dim):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": variant, "task": task, "state_dim": model.state_mean.shape[0],
                "action_dim": model.net[0].in_features - 2 * model.state_mean.shape[0], "hidden_dim": hidden_dim,
                "steps": steps, "seed": seed, "state_dict": model.state_dict()}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="one or more StateEpisodeStore directories")
    ap.add_argument("--output", type=Path, required=True, help="checkpoint path (.pt)")
    ap.add_argument("--variant", default="state_l1", choices=VARIANTS)
    ap.add_argument("--task", default="half", choices=sorted(TASKS), help="the stores' state layout")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--device", default="cpu", help="a torch device")
    args = ap.parse_args()
    train_eps, _ = load_split(args.data)
    task = TASKS[args.task]
    check_layout(train_eps, task)
    model, loss = train(train_eps, args.variant, args.steps, args.seed, args.hidden_dim, args.device)
    save_checkpoint(model, args.output, args.variant, task.name, args.steps, args.seed, args.hidden_dim)
    print(f"saved {args.output}: {len(train_eps)} training episodes, final loss {loss:.4f}", flush=True)


if __name__ == "__main__":
    main()

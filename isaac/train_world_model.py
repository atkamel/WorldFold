"""Train the residual full-state predictor from Isaac-collected episodes.

Example:
    python isaac/train_world_model.py --data-dir outputs/cloth_angles/isaac_state \
        --output outputs/isaac/state_predictor.pt --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

from cloth_angles.data.state_episode import StateEpisode, StateEpisodeStore  # noqa: E402
from cloth_angles.model.state_predictor import ResidualStatePredictor  # noqa: E402


def _split_episodes(episodes: list[StateEpisode], seed: int):
    train = [ep for ep in episodes if ep.metadata.get("split") == "train"]
    valid = [ep for ep in episodes if ep.metadata.get("split") == "test"]
    if train and valid:
        return train, valid
    if len(episodes) < 2:
        raise ValueError("need at least two episodes for a train/validation split")
    order = np.random.default_rng(seed).permutation(len(episodes))
    n_valid = max(1, round(len(episodes) * 0.2))
    valid_ids = set(order[:n_valid])
    return ([ep for i, ep in enumerate(episodes) if i not in valid_ids],
            [ep for i, ep in enumerate(episodes) if i in valid_ids])


def _transitions(episodes: list[StateEpisode]):
    states, previous, actions, next_states = [], [], [], []
    for episode in episodes:
        flat = episode.states()
        if flat.shape[1] != 378 or episode.actions.shape[1] != 6:
            raise ValueError(f"expected state/action dimensions 378/6, got {flat.shape[1]}/{episode.actions.shape[1]}")
        states.append(flat[:-1])
        previous.append(np.concatenate((flat[:1], flat[:-2]), axis=0))
        actions.append(episode.actions)
        next_states.append(flat[1:])
    return tuple(np.concatenate(part).astype(np.float32, copy=False)
                 for part in (states, previous, actions, next_states))


def _evaluate(model, arrays, device):
    state, prev, action, target = (torch.as_tensor(x, device=device) for x in arrays)
    with torch.inference_mode():
        prediction = model(state, prev, action)
        vertex_mae = float((prediction[:, :363] - target[:, :363]).abs().mean().cpu())
        persistence_mae = float((state[:, :363] - target[:, :363]).abs().mean().cpu())
        loss = float(model.loss(state, prev, action, target).cpu())
    return loss, vertex_mae, persistence_mae


def train(data_dir: Path, output: Path, epochs: int = 100, batch_size: int = 256,
          hidden_dim: int = 256, learning_rate: float = 3e-4, seed: int = 0,
          device_name: str = "auto") -> dict:
    if epochs < 1 or batch_size < 1 or hidden_dim < 1:
        raise ValueError("epochs, batch size, and hidden dimension must be positive")
    episodes = StateEpisodeStore(data_dir).load_all()
    if not episodes:
        raise ValueError(f"no episodes found under {data_dir}")
    train_episodes, valid_episodes = _split_episodes(episodes, seed)
    train_arrays = _transitions(train_episodes)
    valid_arrays = _transitions(valid_episodes)

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(seed)
    model = ResidualStatePredictor(378, 6, hidden_dim=hidden_dim, loss="l1").to(device)
    model.fit_normalizer(torch.as_tensor(train_arrays[0], device=device),
                         torch.as_tensor(train_arrays[3], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_tensors = tuple(torch.as_tensor(x, device=device) for x in train_arrays)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_arrays[0]))
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=device)
            loss = model.loss(*(x[idx] for x in train_tensors))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            valid_loss, valid_mae, baseline_mae = _evaluate(model, valid_arrays, device)
            print(f"epoch={epoch}/{epochs} valid_loss={valid_loss:.6f} "
                  f"vertex_mae_mm={1000 * valid_mae:.3f} "
                  f"persistence_mm={1000 * baseline_mae:.3f}", flush=True)

    valid_loss, valid_mae, baseline_mae = _evaluate(model, valid_arrays, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": "state_l1", "state_dim": 378, "action_dim": 6, "hidden_dim": hidden_dim,
                "state_dict": model.cpu().state_dict(), "episodes": len(episodes),
                "train_episodes": len(train_episodes), "validation_episodes": len(valid_episodes),
                "epochs": epochs, "seed": seed, "valid_loss": valid_loss,
                "valid_vertex_mae_m": valid_mae, "persistence_vertex_mae_m": baseline_mae}, output)
    return {"checkpoint": str(output), "validation_vertex_mae_mm": 1000 * valid_mae,
            "persistence_vertex_mae_mm": 1000 * baseline_mae}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/isaac/state_predictor.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device")
    args = parser.parse_args()
    print(train(args.data_dir, args.output, args.epochs, args.batch_size, args.hidden_dim,
                args.learning_rate, args.seed, args.device), flush=True)


if __name__ == "__main__":
    main()

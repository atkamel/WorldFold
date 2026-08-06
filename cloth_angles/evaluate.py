"""Evaluate a trained cloth-angle RSSM checkpoint against baselines.

Usage:
    python -m cloth_angles.evaluate --config cloth_angles/config.yaml \
        --checkpoint outputs/cloth_angles/checkpoints/checkpoint_010000.pt

Reports one-step MAE, 5/10-step open-loop MAE, per-cell error heatmaps, and a
true-vs-predicted visual grid for one held-out sequence, per the spec's
evaluation section. The RSSM must beat persistence on active held-out
transitions, and open-loop error should grow gradually rather than collapse
after the first action.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloth_angles.data.episode_store import EpisodeStore
from cloth_angles.data.sequence_replay import SequenceReplay
from cloth_angles.model.baselines import persistence_predict
from cloth_angles.model.checkpoint import ExpectedSchema, load_checkpoint
from cloth_angles.model.world_model import WorldModel
from cloth_angles.train import split_episodes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to a train.py checkpoint .pt file")
    parser.add_argument("--output-dir", default=None, help="Where to write plots (default: outputs/cloth_angles/eval)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, device: torch.device, data_cfg: dict) -> WorldModel:
    """Loads a checkpoint, rejecting it (IncompatibleCheckpointError) if its
    schema doesn't match the grid/action/angle settings this eval config
    expects -- e.g. a checkpoint trained on a different grid resolution.
    """
    expected = ExpectedSchema(
        grid_size=data_cfg["grid_size"],
        action_dim=data_cfg["action_dim"],
        angle_unit=data_cfg["angle_unit"],
        angle_convention=data_cfg["angle_convention"],
    )
    model, _ = load_checkpoint(checkpoint_path, expected, device=device)
    model.eval()
    return model


def angle_error(model: WorldModel, angle_hat: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    return model._angle_error(angle_hat, angle).abs()


def one_step_eval(model: WorldModel, replay: SequenceReplay, device, n_sequences: int,
                   active_threshold_rad: float = 0.02):
    """One-step MAE for the RSSM vs persistence, overall AND restricted to
    ACTIVE transitions (per-step mean |wrapped delta| above threshold). The
    gate metric is the active one: on quasi-static steps persistence is
    unbeatable by construction (its error IS the per-step motion), so overall
    MAE mostly measures how much of the data is standing still.
    """
    obs, actions, next_obs, mask, is_first = replay.sample(n_sequences)
    obs_t = torch.as_tensor(obs, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    next_obs_t = torch.as_tensor(next_obs, device=device)
    mask_t = torch.as_tensor(mask, device=device)
    is_first_t = torch.as_tensor(is_first, device=device)

    n = model.grid_size
    target = next_obs_t.reshape(*next_obs_t.shape[:-1], n, n)

    with torch.no_grad():
        angle_hat = model.predicted_angles(obs_t, actions_t, is_first_t)
    rssm_err = angle_error(model, angle_hat, target).mean(dim=(-2, -1))          # [batch, time]

    persistence_hat = persistence_predict(obs_t).reshape(*obs_t.shape[:-1], n, n)
    persistence_err = angle_error(model, persistence_hat, target).mean(dim=(-2, -1))

    # activity = per-step mean |wrapped(next - obs)|; persistence_err IS that quantity
    active_mask = mask_t * (persistence_err > active_threshold_rad).float()

    def masked_mean(err, m):
        return float((err * m).sum() / m.sum().clamp(min=1.0))

    return {
        "rssm_mae_rad": masked_mean(rssm_err, mask_t),
        "persistence_mae_rad": masked_mean(persistence_err, mask_t),
        "rssm_active_mae_rad": masked_mean(rssm_err, active_mask),
        "persistence_active_mae_rad": masked_mean(persistence_err, active_mask),
        "active_fraction": float(active_mask.sum() / mask_t.sum().clamp(min=1.0)),
    }


def open_loop_eval(model: WorldModel, replay: SequenceReplay, device, horizons: list[int], n_sequences: int):
    obs, actions, next_obs, mask, _ = replay.sample(n_sequences)
    obs_t = torch.as_tensor(obs, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    next_obs_t = torch.as_tensor(next_obs, device=device)
    mask_t = torch.as_tensor(mask, device=device)
    n = model.grid_size

    results = {}
    max_horizon = max(horizons)
    with torch.no_grad():
        initial_state = model.initial_state_from_obs(obs_t[:, 0])
        future_actions = actions_t[:, :max_horizon]
        angle_hat = model.imagine_angles(initial_state, future_actions,
                                          initial_obs=obs_t[:, 0])  # [batch, horizon, N, N]

    target = next_obs_t[:, :max_horizon].reshape(-1, max_horizon, n, n)
    horizon_mask = mask_t[:, :max_horizon]

    per_step_mae = []
    for t in range(max_horizon):
        err = angle_error(model, angle_hat[:, t], target[:, t])
        step_mask = horizon_mask[:, t]
        denom = step_mask.sum().clamp(min=1.0)
        per_step_mae.append(float((err.mean(dim=(-2, -1)) * step_mask).sum() / denom))

    for h in horizons:
        results[f"open_loop_{h}step_mae_rad"] = per_step_mae[h - 1] if h - 1 < len(per_step_mae) else float("nan")
    results["per_step_mae_rad"] = per_step_mae
    return results, angle_hat, target, horizon_mask


def save_heatmap(error_grid: np.ndarray, path: Path, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(error_grid, cmap="inferno")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="abs angle error (rad)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_comparison_grid(true_seq: np.ndarray, pred_seq: np.ndarray, path: Path):
    import matplotlib.pyplot as plt

    horizon = true_seq.shape[0]
    fig, axes = plt.subplots(2, horizon, figsize=(2 * horizon, 4))
    for t in range(horizon):
        axes[0, t].imshow(true_seq[t], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[0, t].set_title(f"true t+{t + 1}")
        axes[0, t].axis("off")
        axes[1, t].imshow(pred_seq[t], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[1, t].set_title(f"pred t+{t + 1}")
        axes[1, t].axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    eval_cfg = config["eval"]
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/cloth_angles/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, data_cfg)

    store = EpisodeStore(
        data_cfg["episode_dir"],
        grid_size=data_cfg["grid_size"],
        action_dim=data_cfg["action_dim"],
        angle_unit=data_cfg["angle_unit"],
        angle_convention=data_cfg["angle_convention"],
    )
    episodes = store.load_all()
    if not episodes:
        raise RuntimeError(f"no episodes found under {data_cfg['episode_dir']}")

    # evaluate on the HELD-OUT episodes only, reproducing train.py's episode-level
    # split (same seed) -- previously this evaluated on train+held-out mixed
    train_cfg = config["train"]
    _, held_out_episodes = split_episodes(episodes, data_cfg["held_out_fraction"], train_cfg["seed"])
    if not held_out_episodes:
        raise RuntimeError("held-out split is empty; collect more episodes or raise held_out_fraction")
    replay = SequenceReplay(held_out_episodes, seq_len=data_cfg["seq_len"], seed=123)

    one_step = one_step_eval(model, replay, device, eval_cfg["n_eval_sequences"],
                              active_threshold_rad=eval_cfg.get("active_threshold_rad", 0.02))
    print(f"=== one-step (held-out episodes: {len(held_out_episodes)}) ===")
    for k, v in one_step.items():
        print(f"{k}: {v:.4f}")
    beats_persistence = one_step["rssm_active_mae_rad"] < one_step["persistence_active_mae_rad"]
    print(f"beats persistence (active transitions, the gate): {beats_persistence}")
    print(f"beats persistence (overall): {one_step['rssm_mae_rad'] < one_step['persistence_mae_rad']}")

    open_loop, angle_hat, target, horizon_mask = open_loop_eval(
        model, replay, device, eval_cfg["open_loop_horizons"], eval_cfg["n_eval_sequences"]
    )
    print("\n=== open-loop ===")
    for k, v in open_loop.items():
        if k == "per_step_mae_rad":
            continue
        print(f"{k}: {v:.4f}")
    print(f"per-step mae (rad): {[round(v, 4) for v in open_loop['per_step_mae_rad']]}")

    # Heatmap + comparison grid for the first held-out sequence.
    example_true = target[0].detach().cpu().numpy()
    example_pred = angle_hat[0].detach().cpu().numpy()
    error_grid = np.abs(example_true - example_pred).mean(axis=0)
    save_heatmap(error_grid, output_dir / "error_heatmap.png", "mean abs angle error over horizon")
    save_comparison_grid(example_true, example_pred, output_dir / "true_vs_predicted.png")
    print(f"\nwrote {output_dir / 'error_heatmap.png'}")
    print(f"wrote {output_dir / 'true_vs_predicted.png'}")


if __name__ == "__main__":
    main()

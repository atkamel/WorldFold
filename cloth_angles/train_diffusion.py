"""Train the conditional angle-field DPM on stored episodes.

Usage:
    python -m cloth_angles.train_diffusion --config cloth_angles/config.yaml

Trains ONLY on the train split (same episode-level split as train.py), with
random flap occlusion augmentation. Logs a masked-reconstruction probe on
held-out frames vs the nearest-fill baseline so progress is interpretable:
the DPM earns its keep only where it beats deterministic inpainting on the
HIDDEN cells.
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
from cloth_angles.data.occlusion import half_plane_mask, nearest_fill, sample_mask
from cloth_angles.model.diffusion import AngleFieldDPM, save_dpm_checkpoint, wrapped_abs_error
from cloth_angles.train import split_episodes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def gather_frames(episodes) -> np.ndarray:
    """All observation frames from the given episodes, [M, N*N] float32."""
    return np.concatenate([ep.obs for ep in episodes], axis=0).astype(np.float32)


def make_probe(frames: np.ndarray, grid_size: int, rng: np.random.Generator, n_per_stratum: int = 8):
    """Fixed (frame, mask) probe pairs, half light / half heavy occlusion."""
    idx = rng.choice(len(frames), size=2 * n_per_stratum, replace=False)
    probe = []
    for j, i in enumerate(idx):
        stratum = "light" if j < n_per_stratum else "heavy"
        mask = sample_mask(grid_size, stratum, rng)
        probe.append((frames[i].reshape(grid_size, grid_size), mask, stratum))
    return probe


@torch.no_grad()
def probe_mae(model: AngleFieldDPM, probe, device) -> dict:
    """1-sample DPM reconstruction MAE on HIDDEN cells vs nearest-fill, per stratum."""
    out = {}
    for stratum in ("light", "heavy"):
        dpm_errs, fill_errs = [], []
        for field, mask, s in probe:
            if s != stratum:
                continue
            masked = field.copy()
            masked[mask] = 0.0
            f_t = torch.as_tensor(masked.reshape(-1), device=device)
            m_t = torch.as_tensor(mask.reshape(-1).astype(np.float32), device=device)
            sample = model.sample_k(f_t, m_t, k=1)[0].reshape(model.grid_size, model.grid_size)
            true_t = torch.as_tensor(field, device=device)
            mask_t = torch.as_tensor(mask, device=device)
            dpm_errs.append(float(wrapped_abs_error(sample, true_t)[mask_t].mean()))
            filled = torch.as_tensor(nearest_fill(field, mask), device=device)
            fill_errs.append(float(wrapped_abs_error(filled, true_t)[mask_t].mean()))
        out[f"dpm_{stratum}"] = float(np.mean(dpm_errs))
        out[f"fill_{stratum}"] = float(np.mean(fill_errs))
    return out


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    diff_cfg = config["diffusion"]
    grid_size = data_cfg["grid_size"]

    torch.manual_seed(diff_cfg["seed"])
    rng = np.random.default_rng(diff_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store = EpisodeStore(
        data_cfg["episode_dir"], grid_size=grid_size, action_dim=data_cfg["action_dim"],
        angle_unit=data_cfg["angle_unit"], angle_convention=data_cfg["angle_convention"],
    )
    episodes = store.load_all()
    if not episodes:
        raise RuntimeError(f"no episodes found under {data_cfg['episode_dir']}")
    train_episodes, held_out_episodes = split_episodes(
        episodes, data_cfg["held_out_fraction"], config["train"]["seed"]
    )
    train_frames = gather_frames(train_episodes)
    held_out_frames = gather_frames(held_out_episodes) if held_out_episodes else train_frames

    # varied perception frames (cloth-pose-randomized scripted rollouts) --
    # without them the DPM sees essentially one fold trajectory and collapses
    # on Exp 1's offset configs. collect_perception_frames.py writes this file.
    extra_path = diff_cfg.get("extra_frames_path")
    if extra_path and Path(extra_path).exists():
        extra = np.load(extra_path, allow_pickle=False)["frames"].astype(np.float32)
        n_held = max(1, len(extra) // 10)
        train_frames = np.concatenate([train_frames, extra[:-n_held]], axis=0)
        held_out_frames = np.concatenate([held_out_frames, extra[-n_held:]], axis=0)
        print(f"merged {len(extra)} varied perception frames ({n_held} to held-out)")
    print(f"train frames: {len(train_frames)}, held-out frames: {len(held_out_frames)}")

    probe = make_probe(held_out_frames, grid_size, rng)

    model = AngleFieldDPM(
        grid_size=grid_size, timesteps=diff_cfg["timesteps"],
        hidden_dim=diff_cfg["hidden_dim"], time_embed_dim=diff_cfg["time_embed_dim"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=diff_cfg["learning_rate"])

    checkpoint_dir = Path(diff_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    batch_size = diff_cfg["batch_size"]
    lo, hi = diff_cfg["min_severity"], diff_cfg["max_severity"]

    for step in range(1, diff_cfg["train_steps"] + 1):
        idx = rng.integers(0, len(train_frames), size=batch_size)
        fields = train_frames[idx]                             # [B, N*N]
        masks = np.stack([
            half_plane_mask(grid_size, float(rng.uniform(lo, hi)), rng).reshape(-1)
            for _ in range(batch_size)
        ]).astype(np.float32)
        masked = fields * (1.0 - masks)

        field_t = torch.as_tensor(fields, device=device)
        masked_t = torch.as_tensor(masked, device=device)
        mask_t = torch.as_tensor(masks, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(field_t, masked_t, mask_t)
        loss.backward()
        optimizer.step()

        if step % diff_cfg["eval_every"] == 0 or step == 1:
            model.eval()
            metrics = probe_mae(model, probe, device)
            model.train()
            print(f"step {step}/{diff_cfg['train_steps']} loss={loss.item():.4f} "
                  f"| hidden-cell MAE (rad): dpm_light={metrics['dpm_light']:.4f} "
                  f"fill_light={metrics['fill_light']:.4f} "
                  f"dpm_heavy={metrics['dpm_heavy']:.4f} fill_heavy={metrics['fill_heavy']:.4f}")
            save_dpm_checkpoint(
                checkpoint_dir / f"dpm_{step:06d}.pt", model=model, step=step,
                grid_size=grid_size, angle_unit=data_cfg["angle_unit"],
                angle_convention=data_cfg["angle_convention"],
                model_config={k: diff_cfg[k] for k in ("timesteps", "hidden_dim", "time_embed_dim")},
            )

    print(f"\ntraining complete. checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()

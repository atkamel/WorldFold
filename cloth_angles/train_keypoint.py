"""Train the keypoint head on frozen world-model states over corner-labeled data.

Usage:
    python -m cloth_angles.train_keypoint --config cloth_angles/config.yaml \
        --wm-checkpoint outputs/cloth_angles/checkpoints/checkpoint_020000.pt

Feeds field/action windows through the FROZEN RSSM (deterministic posterior),
regresses per-step corner displacement, reports held-out RMS corner error in
cm. Split by episode. Windows are sampled fully inside episodes with
is_first=True at the window start -- the same partial-context regime the
planner uses when it bootstraps from the current observation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloth_angles.model.checkpoint import ExpectedSchema, load_checkpoint
from cloth_angles.model.keypoint_head import KeypointHead, save_keypoint_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--data", default="outputs/cloth_angles/planner_data.npz")
    return parser.parse_args()


def load_episodes(path: str) -> list[dict]:
    d = np.load(path, allow_pickle=False)
    episodes = []
    for e in np.unique(d["episode_idx"]):
        sel = d["episode_idx"] == e
        episodes.append({
            "fields": d["fields"][sel], "actions": d["actions"][sel],
            "corner_delta": d["corner_delta"][sel],
        })
    return episodes


def sample_batch(episodes, batch_size: int, seq_len: int, rng: np.random.Generator):
    fields = np.zeros((batch_size, seq_len, episodes[0]["fields"].shape[1]), np.float32)
    actions = np.zeros((batch_size, seq_len, episodes[0]["actions"].shape[1]), np.float32)
    deltas = np.zeros((batch_size, seq_len, 12), np.float32)
    is_first = np.zeros((batch_size, seq_len), bool)
    is_first[:, 0] = True
    for b in range(batch_size):
        ep = episodes[int(rng.integers(len(episodes)))]
        t = len(ep["fields"])
        if t <= seq_len:
            start, end = 0, t
        else:
            start = int(rng.integers(0, t - seq_len))
            end = start + seq_len
        n = end - start
        fields[b, :n] = ep["fields"][start:end]
        actions[b, :n] = ep["actions"][start:end]
        deltas[b, :n] = ep["corner_delta"][start:end]
        if n < seq_len:   # tail-pad by repeating the last frame (masked out of loss)
            fields[b, n:] = ep["fields"][end - 1]
            deltas[b, n:] = ep["corner_delta"][end - 1]
    return fields, actions, deltas, is_first


def wm_features(wm, fields_t, actions_t, is_first_t):
    """Frozen deterministic posterior states -> [batch, time, h+z]."""
    with torch.no_grad():
        embeds = wm.encoder(fields_t)
        states = wm.rssm.observe(embeds, actions_t, is_first_t, deterministic=True)
        h = torch.stack([s.h for s in states], dim=1)
        z = torch.stack([s.z for s in states], dim=1)
    return torch.cat([h, z], dim=-1)


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    data_cfg = config["data"]
    kp_cfg = config["keypoint"]

    torch.manual_seed(kp_cfg["seed"])
    rng = np.random.default_rng(kp_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expected = ExpectedSchema(grid_size=data_cfg["grid_size"], action_dim=data_cfg["action_dim"],
                               angle_unit=data_cfg["angle_unit"],
                               angle_convention=data_cfg["angle_convention"])
    wm, _ = load_checkpoint(args.wm_checkpoint, expected, device=device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    latent_dim = wm.rssm.h_dim + wm.rssm.z_dim

    episodes = load_episodes(args.data)
    perm = rng.permutation(len(episodes))
    n_held = max(1, len(episodes) // 5)
    held_out = [episodes[i] for i in perm[:n_held]]
    train = [episodes[i] for i in perm[n_held:]]
    print(f"episodes: {len(train)} train / {len(held_out)} held-out; latent_dim={latent_dim}")

    head = KeypointHead(latent_dim, kp_cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=kp_cfg["learning_rate"])
    seq_len, batch_size = kp_cfg["seq_len"], kp_cfg["batch_size"]

    def eval_rms(eps) -> float:
        f, a, d, i = sample_batch(eps, 64, seq_len, rng)
        feat = wm_features(wm, torch.as_tensor(f, device=device),
                            torch.as_tensor(a, device=device),
                            torch.as_tensor(i, device=device))
        with torch.no_grad():
            pred = head(feat)
        err = (pred - torch.as_tensor(d, device=device)).reshape(-1, 4, 3)
        return float(err.norm(dim=-1).mean()) * 100.0   # mean corner error, cm

    checkpoint_dir = Path(kp_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, kp_cfg["train_steps"] + 1):
        f, a, d, i = sample_batch(train, batch_size, seq_len, rng)
        feat = wm_features(wm, torch.as_tensor(f, device=device),
                            torch.as_tensor(a, device=device),
                            torch.as_tensor(i, device=device))
        pred = head(feat)
        loss = torch.nn.functional.mse_loss(pred, torch.as_tensor(d, device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % kp_cfg["eval_every"] == 0 or step == 1:
            print(f"step {step}/{kp_cfg['train_steps']} loss={loss.item():.6f} "
                  f"| corner err (cm): train={eval_rms(train):.2f} held_out={eval_rms(held_out):.2f}")
            save_keypoint_checkpoint(checkpoint_dir / "keypoint_head.pt", head, step,
                                      latent_dim, kp_cfg["hidden_dim"], str(args.wm_checkpoint))

    print(f"\ndone. checkpoint: {checkpoint_dir / 'keypoint_head.pt'}")


if __name__ == "__main__":
    main()

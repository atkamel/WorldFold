"""Collect VARIED angle-field frames for perception (DPM) training.

Usage:
    python -m cloth_angles.collect_perception_frames --config cloth_angles/config.yaml \
        --rollouts 24 --seed-base 2000

Why this exists: the episode store's fold episodes are collected without
randomization, so they contain essentially ONE fold trajectory -- a DPM
trained on them collapses on Exp 1's evaluation configs, which randomize the
cloth start pose (measured: 2-3 rad hallucinations on offset cloths). This
script runs scripted-fold rollouts WITH the same cloth-pose randomization the
eval uses, at DISJOINT seeds (eval owns 1000-1039), recording every frame's
angle field. Output: a flat [M, N*N] frame bank npz for train_diffusion.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mujuco"))

from cloth_angles.collect_episodes import angle_obs

EVAL_SEEDS = set(range(1000, 1040))   # Exp 1 owns these; never train on them


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollouts", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--output", default="outputs/cloth_angles/perception_frames.npz")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    grid_size = config["data"]["grid_size"]

    from sim_main import ClothFoldEnv, ScriptedFoldPolicy  # noqa: PLC0415

    env = ClothFoldEnv(max_episode_steps=args.max_steps, observation_mode="state")
    frames = []
    for k in range(args.rollouts):
        seed = args.seed_base + k
        if seed in EVAL_SEEDS:
            raise ValueError(f"seed {seed} collides with Exp 1 evaluation seeds")
        rng = np.random.default_rng(seed)
        offset = rng.uniform(-0.03, 0.03, size=2)   # same distribution as Exp 1 configs
        env.reset(seed=seed, options={"cloth_pose": offset})
        policy = ScriptedFoldPolicy(env)
        frames.append(angle_obs(env, grid_size, signed=True))
        for _ in range(args.max_steps):
            _, _, terminated, truncated, _ = env.step(policy.act())
            frames.append(angle_obs(env, grid_size, signed=True))
            if terminated or truncated:
                break
        print(f"rollout {k + 1}/{args.rollouts} (seed={seed}, offset={np.round(offset, 3)}, "
              f"{len(frames)} frames total)")

    env.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=np.stack(frames).astype(np.float32))
    print(f"\nwrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()

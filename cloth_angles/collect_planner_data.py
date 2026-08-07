"""Collect corner-labeled sequences for the keypoint head (M4 planner milestone).

Usage:
    python -m cloth_angles.collect_planner_data --config cloth_angles/config.yaml

Records per step: angle field, action, and the 4 cloth corner positions as
DISPLACEMENTS from the episode's initial corners. The angle field is
translation-invariant (a shifted cloth has identical angles), so absolute
corner positions are not decodable from the latent -- displacements are.
The planner reconstructs absolute corners as corners0 + predicted delta.

Mix: scripted folds with cloth-pose offsets (task-relevant dynamics) +
recovery-style random rollouts (off-policy coverage) + a few still rollouts.
Seeds 4000+ (disjoint from Exp1 evals 1000-1039, perception frames 2000+,
Exp2 3000-3123).
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold-rollouts", type=int, default=20)
    parser.add_argument("--random-rollouts", type=int, default=10)
    parser.add_argument("--still-rollouts", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--output", default="outputs/cloth_angles/planner_data.npz")
    return parser.parse_args()


def corners(env) -> np.ndarray:
    return env.data.xpos[env._corner_ids].reshape(-1).astype(np.float32)   # (12,)


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    grid_size = config["data"]["grid_size"]

    from sim_main import ClothFoldEnv, ScriptedFoldPolicy  # noqa: PLC0415

    env = ClothFoldEnv(max_episode_steps=args.max_steps, observation_mode="state")

    kinds = (["fold"] * args.fold_rollouts + ["random"] * args.random_rollouts
             + ["still"] * args.still_rollouts)
    fields, actions, corner_delta, episode_end, episode_idx = [], [], [], [], []
    corners0_list, goal_list = [], []

    for e, kind in enumerate(kinds):
        seed = args.seed_base + e
        rng = np.random.default_rng(seed)
        offset = rng.uniform(-0.03, 0.03, size=2)
        env.reset(seed=seed, options={"cloth_pose": offset, "task": 0})
        policy = ScriptedFoldPolicy(env)
        c0 = corners(env)
        corners0_list.append(c0)
        goal_list.append(env._goal_corners.reshape(-1).astype(np.float32))

        n_steps = 0
        for t in range(args.max_steps):
            if kind == "fold":
                action = policy.act()
            elif kind == "random":
                action = rng.uniform(-1.0, 1.0, size=14).astype(np.float32)
            else:
                action = np.zeros(14, dtype=np.float32)
            fields.append(angle_obs(env, grid_size, signed=True))
            actions.append(np.asarray(action, dtype=np.float32))
            _, _, terminated, truncated, _ = env.step(action)
            corner_delta.append(corners(env) - c0)
            episode_idx.append(e)
            episode_end.append(False)
            n_steps += 1
            if terminated or truncated:
                break
        episode_end[-1] = True
        print(f"[{kind}] rollout {e + 1}/{len(kinds)} (seed={seed}, {n_steps} steps)")

    env.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fields=np.stack(fields), actions=np.stack(actions),
        corner_delta=np.stack(corner_delta),
        episode_end=np.array(episode_end, dtype=bool),
        episode_idx=np.array(episode_idx, dtype=np.int32),
        corners0=np.stack(corners0_list), goal=np.stack(goal_list),
    )
    print(f"\nwrote {len(fields)} steps / {len(kinds)} episodes to {out}")


if __name__ == "__main__":
    main()

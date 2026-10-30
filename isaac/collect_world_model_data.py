"""Collect Isaac-domain transitions for the full-state residual predictor.

Run from an Isaac Sim 5.1 Python environment. Episodes use bounded random
joint deltas with occasional gripper open/close commands; split assignment is
by episode, so the existing predictor benchmark can evaluate held-out runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cloth_angles.data.state_episode import StateEpisode, StateEpisodeStore
from cloth_fold_rl.fold_env import SingleCornerFoldEnv
from isaac.sim.isaac_env import IsaacClothFoldEnv
from isaac.world_model import observe_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cloth_angles/isaac_state")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--action-scale", type=float, default=0.35,
                        help="max absolute random joint delta; gripper commands are scheduled separately")
    args = parser.parse_args()
    if args.episodes < 2 or not 0 < args.test_episodes < args.episodes:
        parser.error("need at least two episodes and 0 < --test-episodes < --episodes")
    if args.steps < 1 or not 0.0 < args.action_scale <= 1.0:
        parser.error("steps must be positive and action-scale must be in (0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    store = StateEpisodeStore(args.output)
    if store.load_all():
        parser.error(f"{args.output} already contains episodes; choose a fresh output directory")

    env = SingleCornerFoldEnv(base_env=IsaacClothFoldEnv(observation_mode="state"))
    manifest = []
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            rng = np.random.default_rng(seed)
            env.reset(seed=seed)
            base = env.unwrapped
            states = [observe_state(base)]
            actions, rewards, terminated = [], [], []
            for t in range(args.steps):
                action = rng.uniform(-args.action_scale, args.action_scale, size=6).astype(np.float32)
                # Ensure the gripper sees discrete edges during data collection.
                action[5] = -1.0 if (t // 20) % 2 == 0 else 1.0
                _, reward, term, trunc, _ = env.step(action)
                actions.append(action)
                rewards.append(reward)
                terminated.append(term)
                states.append(observe_state(base))
                if term or trunc:
                    break
            states = np.stack(states)
            episode = StateEpisode(
                vertices=states[:, :363].reshape(-1, 121, 3).astype(np.float32),
                robot=states[:, 363:].astype(np.float32),
                actions=np.stack(actions).astype(np.float32),
                metadata={"task": "single", "simulator": "isaacsim-5.1", "kind": "random",
                          "seed": seed, "length": len(actions),
                          "split": "test" if i < args.test_episodes else "train"},
                rewards=np.asarray(rewards, dtype=np.float32),
                terminated=np.asarray(terminated, dtype=np.bool_),
            )
            path = store.append(episode)
            episode.metadata["file"] = path.name
            manifest.append(episode.metadata)
            print(f"{i + 1}/{args.episodes}: {path.name}, transitions={len(episode)}, "
                  f"split={episode.metadata['split']}", flush=True)
    finally:
        env.close()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved Isaac Sim 5.1 data to {args.output}", flush=True)


if __name__ == "__main__":
    main()

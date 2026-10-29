"""Run a trained residual world model alongside a live Isaac fold episode.

Example (inside an Isaac Sim 5.1 Python environment):

    python isaac/run_world_model.py --checkpoint path/to/state_l1_seed0.pt --steps 200

Pass ``--actions`` with an NPZ containing an ``actions`` array to replay the
same six-dimensional controls used by the single-corner task. Without it,
the runner uses zero actions to measure predictor latency without applying
random motion to the scene.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isaac.sim.isaac_env import IsaacClothFoldEnv
from isaac.world_model import IsaacWorldModel, observe_state
from cloth_fold_rl.fold_env import SingleCornerFoldEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Full-state predictor checkpoint")
    parser.add_argument("--actions", type=Path, help="Optional NPZ with an actions[T,6] array")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    actions = None
    if args.actions:
        with np.load(args.actions, allow_pickle=False) as data:
            actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 6 or len(actions) == 0:
            parser.error("actions must have shape [T, 6]")

    env = SingleCornerFoldEnv(base_env=IsaacClothFoldEnv(observation_mode="state"))
    try:
        world_model = IsaacWorldModel(args.checkpoint, device=args.device)
        env.reset(seed=0)
        base = env.unwrapped
        world_model.reset(observe_state(base))
        latencies, errors, loop_times = [], [], []
        for t in range(args.steps):
            action = np.zeros(6, dtype=np.float32) if actions is None else actions[t % len(actions)]
            # Predict from the current observation before Isaac advances. This
            # keeps inference on the live control path without leaking the next state.
            tick = time.perf_counter()
            prediction, predictor_ms = world_model.predict(action)
            _, _, terminated, truncated, info = env.step(action)
            observed = observe_state(base)
            metrics = world_model.observe_prediction(prediction, observed, predictor_ms)
            latencies.append(predictor_ms)
            errors.append(metrics["vertex_mae_m"])
            loop_times.append((time.perf_counter() - tick) * 1000.0)
            if (t + 1) % 25 == 0 or terminated or truncated:
                recent = slice(max(0, len(latencies) - 25), len(latencies))
                mean_loop = float(np.mean(loop_times[recent]))
                sim_fps = 1000.0 / mean_loop if mean_loop > 0 else float("inf")
                print(f"step={t + 1} predictor_ms={np.mean(latencies[recent]):.3f} "
                      f"vertex_mae_mm={1000 * np.mean(errors[recent]):.3f} "
                      f"control_hz={sim_fps:.1f} fold_score={info['fold_score']:.3f}", flush=True)
            if terminated or truncated:
                break
        print(f"completed_steps={len(loop_times)} mean_predictor_ms={np.mean(latencies):.3f} "
              f"mean_control_hz={1000 / np.mean(loop_times):.1f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

"""Expert benchmark on the half fold (design doc M2): success rate, steps, failure reasons.

    python -m imitation.benchmark_expert --episodes 50 --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter

import numpy as np


def _run(seed):
    from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
    from imitation.tasks import HalfFoldEnv

    env = HalfFoldEnv()
    expert = QuarterFoldExpert(env)
    env.reset(seed=seed)
    expert.reset()
    for t in range(env.unwrapped.max_episode_steps):
        _, _, terminated, truncated, info = env.step(expert.act())
        if terminated or truncated:
            break
    return {"seed": seed, "steps": t + 1, "success": bool(info["success"]),
            "fold_score": round(info["fold_score"], 3),
            "reason": info["termination_reason"] or "truncated", "retries": sum(expert.retries.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="optional JSON file for the per-episode rows")
    args = ap.parse_args()
    seeds = range(args.seed_base, args.seed_base + args.episodes)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_run, seeds)
    ok = [r for r in rows if r["success"]]
    print(f"success {len(ok)}/{len(rows)} ({100 * len(ok) / len(rows):.0f}%)")
    print(f"steps: success mean {np.mean([r['steps'] for r in ok]) if ok else float('nan'):.0f} "
          f"max {max((r['steps'] for r in ok), default=0)}")
    print("reasons:", dict(Counter(r["reason"] for r in rows)))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

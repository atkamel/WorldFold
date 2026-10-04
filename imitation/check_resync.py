"""M1 exit check: does the expert still finish the half fold after someone else
has been driving? At a random step the expert hands over to a perturbation
(none / noisy expert / random actions) for `--k` steps, then resyncs its phases
from the sim and finishes the episode.

    python -m imitation.check_resync --episodes 24 --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from collections import Counter, defaultdict

import numpy as np

MODES = ("resync_only", "noisy", "random")


def _run(job):
    seed, mode, k = job
    from imitation.tasks import HalfFoldEnv
    from imitation.teachers import ScriptedTeacher

    rng = np.random.default_rng(10_000 + seed)
    env = HalfFoldEnv()
    teacher = ScriptedTeacher(env)
    env.reset(seed=seed)
    teacher.reset()
    t_switch = int(rng.integers(5, 80))
    info, phase_at_resync = {}, None
    for t in range(env.unwrapped.max_episode_steps):
        if t_switch <= t < t_switch + k and mode != "resync_only":
            a = teacher.act() if mode == "noisy" else rng.uniform(-1, 1, 12)
            if mode == "noisy":
                a = np.clip(a + rng.normal(0, 0.5, 12), -1, 1)
        else:
            if t == t_switch + (0 if mode == "resync_only" else k):
                phase_at_resync = teacher.expert.resync()
            a = teacher.act()
        _, _, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float32))
        if terminated or truncated:
            break
    return {"seed": seed, "mode": mode, "t_switch": t_switch, "success": bool(info["success"]),
            "reason": info["termination_reason"] or "truncated", "phases": phase_at_resync}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = [(s, m, args.k) for m in MODES for s in range(args.episodes)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_run, jobs)
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    for m in MODES:
        rs = by_mode[m]
        print(f"{m:12s} success {sum(r['success'] for r in rs)}/{len(rs)}  "
              f"reasons {dict(Counter(r['reason'] for r in rs))}")
        for r in rs:
            if not r["success"]:
                print(f"    fail seed {r['seed']} t_switch {r['t_switch']} phases {r['phases']} {r['reason']}")


if __name__ == "__main__":
    main()

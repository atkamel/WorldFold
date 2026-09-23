"""Collect scripted-expert half-fold episodes into a new frozen dataset version.

A fraction of episodes are recovery demos (design doc 6.2): the expert is
interrupted by a few random actions, then resyncs and finishes. Failed episodes
are kept (and tagged) -- they matter for failure analysis and offline RL.

    python -m imitation.data.collect --episodes 400 --workers 14 --version v1
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from imitation.data.schema import DatasetWriter
from imitation.rollout import EnvPool, ExpertController, Perturbation, rollout
from imitation.seeds import TRAIN_SEED_BASE

DEFAULT_ROOT = "outputs/imitation/datasets"


def recovery_perturbation(fraction, t_range=(10, 70), k_range=(5, 15)):
    def fn(seed, rng):
        if rng.random() >= fraction:
            return None
        return Perturbation(t=int(rng.integers(*t_range)), k=int(rng.integers(*k_range)))
    return fn


def printer(t0):
    def progress(n, total, ep):
        m = ep.meta
        print(f"  [{n}/{total}] seed {m['seed']} {'ok  ' if m['success'] else 'FAIL'} {ep.steps:3d} steps "
              f"{m['termination_reason']}{' perturbed' if m['perturb'] else ''}  ({time.time() - t0:.0f}s)", flush=True)
    return progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seed-base", type=int, default=TRAIN_SEED_BASE)
    ap.add_argument("--recovery-fraction", type=float, default=0.3)
    args = ap.parse_args()

    config = vars(args) | {"task": "half_fold", "teacher": "QuarterFoldExpert(stage 0)"}
    writer = DatasetWriter(args.root, args.version, config=config)
    seeds = range(args.seed_base, args.seed_base + args.episodes)
    t0 = time.time()
    with EnvPool(args.workers) as pool:
        episodes = rollout(pool, seeds, ExpertController(), perturb_fn=recovery_perturbation(args.recovery_fraction),
                           progress=printer(t0))
    for ep in episodes:
        writer.add(ep, obs_dim=141, action_dim=12)
    m = writer.freeze()
    perturbed = [e for e in episodes if e.meta["perturb"]]
    print(f"froze {args.root}/{args.version}: {m['n_episodes']} episodes, {m['n_steps']} steps, "
          f"success {m['n_success']}/{m['n_episodes']} (clean {sum(e.meta['success'] for e in episodes if not e.meta['perturb'])}"
          f"/{len(episodes) - len(perturbed)}, recovery {sum(e.meta['success'] for e in perturbed)}/{len(perturbed)}), "
          f"hash {m['content_hash'][:12]}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

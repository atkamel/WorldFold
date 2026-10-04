"""Collect scripted-expert half-fold episodes into a new frozen dataset version.

A fraction of episodes are recovery demos (design doc 6.2): the expert is
interrupted by a few random actions, then resyncs and finishes. Failed episodes
are kept but go to their own version `<version>_failures` (they matter for failure
analysis and offline RL, but are not demos); --mixed puts them in `<version>`.

    python -m imitation.data.collect --episodes 400 --workers 14 --version v1
    python -m imitation.data.collect ... --resume     # continue a killed collection

Episodes are written as they finish, so a killed run keeps everything completed.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from imitation.data.schema import DatasetWriter
from imitation.rollout import EnvPool, ExpertController, Perturbation, rollout
from imitation.seeds import TRAIN_SEED_BASE
from imitation.spec import ACTION_DIM, OBS_DIM

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
    ap.add_argument("--mixed", action="store_true", help="keep failures in the demo version")
    ap.add_argument("--resume", action="store_true", help="continue an unfrozen version")
    args = ap.parse_args()

    config = {k: v for k, v in vars(args).items() if k != "resume"} | {
        "task": "half_fold", "teacher": "QuarterFoldExpert(stage 0)"}
    writer = DatasetWriter(args.root, args.version, config=config, resume=args.resume)
    fail_writer = writer if args.mixed else DatasetWriter(
        args.root, f"{args.version}_failures", config=config | {"demos": args.version}, resume=args.resume)
    done = writer.done_seeds | fail_writer.done_seeds
    seeds = [s for s in range(args.seed_base, args.seed_base + args.episodes) if s not in done]
    if done:
        print(f"resuming {args.version}: {len(done)} done, {len(seeds)} to go", flush=True)

    def save(ep):
        (writer if ep.meta["success"] else fail_writer).add(ep, obs_dim=OBS_DIM, action_dim=ACTION_DIM)

    t0 = time.time()
    with EnvPool(args.workers) as pool:
        rollout(pool, seeds, ExpertController(), perturb_fn=recovery_perturbation(args.recovery_fraction),
                progress=printer(t0), on_done=save)
    for w in dict.fromkeys((writer, fail_writer)):
        m = w.freeze()
        print(f"froze {args.root}/{w.version}: {summary(m['episodes'])}, hash {m['content_hash'][:12]}")
    print(f"{time.time() - t0:.0f}s")


def summary(entries):
    clean = [e for e in entries if not e.get("perturb")]
    pert = [e for e in entries if e.get("perturb")]
    ok = lambda es: sum(e["success"] for e in es)
    return (f"{len(entries)} episodes, {sum(e['steps'] for e in entries)} steps, success {ok(entries)}/{len(entries)} "
            f"(clean {ok(clean)}/{len(clean)}, recovery {ok(pert)}/{len(pert)})")

if __name__ == "__main__":
    main()

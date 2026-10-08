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
import hashlib
import time
from pathlib import Path

import numpy as np

from imitation.data.schema import DatasetWriter
from imitation.evaluate import resolve_workers
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


def recovery_perturbation_for(backend, fraction):
    """MuJoCo's (10, 70) onset window scaled by 230/97 for the longer Isaac fold."""
    from imitation.seeds import LEHOME_TASK_BACKENDS
    if backend not in LEHOME_TASK_BACKENDS:      # mujoco and isaac_weld share MuJoCo's episode timing
        return recovery_perturbation(fraction)
    return recovery_perturbation(fraction, t_range=(24, 166), k_range=(5, 15))


def printer(t0):
    def progress(n, total, ep):
        m = ep.meta
        print(f"  [{n}/{total}] seed {m['seed']} {'ok  ' if m['success'] else 'FAIL'} {ep.steps:3d} steps "
              f"{m['termination_reason']}{' perturbed' if m['perturb'] else ''}  ({time.time() - t0:.0f}s)", flush=True)
    return progress


ISAAC_STACK = {"isaacsim": "5.1.0", "lehome_challenge": "a805ad2f7ab52a4583066fc4ee5180459a7f9d15",
               "isaaclab_fork": "69f6fa548c3a3520e3cb26ed24bb8abe60baeef3"}


def parse_cameras(spec):
    """'main=128,left_wrist_cam=64' -> {'main': 128, 'left_wrist_cam': 64}; None/'' -> None."""
    if not spec:
        return None
    return {k: int(v) for k, v in (p.split("=") for p in spec.split(","))}


def build_config(args):
    """Dataset config: the CLI args, task, teacher, and (isaac) the pinned stack + knobs hash."""
    config = {k: v for k, v in vars(args).items() if k != "resume"} | {
        "task": "half_fold", "teacher": "QuarterFoldExpert(stage 0)"}
    from imitation.tasks.half_fold import is_isaac
    if is_isaac(args.backend):
        knobs = Path(__file__).resolve().parents[2] / "isaac" / "isaac_env.py"
        config["teacher"] = f"QuarterFoldExpert(stage 0) [{args.backend} backend]"
        config["isaac_stack"] = dict(ISAAC_STACK, knobs_sha256=hashlib.sha256(knobs.read_bytes()).hexdigest())
    return config


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--workers", type=int, default=None, help="default 14 (mujoco) / N_ISAAC (isaac)")
    ap.add_argument("--backend", choices=("mujoco", "isaac", "isaac_weld", "isaac_friction"), default="mujoco")
    ap.add_argument("--render", action="store_true", help="record camera images at collection")
    ap.add_argument("--cameras", default=None, help="e.g. main=128,left_wrist_cam=64,right_wrist_cam=64")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seed-base", type=int, default=TRAIN_SEED_BASE)
    ap.add_argument("--recovery-fraction", type=float, default=0.3)
    ap.add_argument("--mixed", action="store_true", help="keep failures in the demo version")
    ap.add_argument("--resume", action="store_true", help="continue an unfrozen version")
    return ap


def main():
    args = build_parser().parse_args()
    args.workers = resolve_workers(args.backend, args.workers)
    config = build_config(args)
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
    kw = {}
    if args.backend != "mujoco":
        kw["backend"] = args.backend
    if args.render:
        kw["render"] = True
        kw["cameras"] = parse_cameras(args.cameras)
    with EnvPool(args.workers, kw or None) as pool:
        rollout(pool, seeds, ExpertController(), perturb_fn=recovery_perturbation_for(args.backend, args.recovery_fraction),
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

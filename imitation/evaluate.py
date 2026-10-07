"""Closed-loop evaluation of a checkpoint on fixed seed sets, with failure codes.

    python -m imitation.evaluate --ckpt runs/mlp_v1/final.pt --sets id_easy id_hard recovery --n 48

Metrics per set: success rate, mean final fold score, grasp success (both arms
grasped at some point), termination reasons, failure-code histogram (design doc
8.1), policy inference time. `--save-episodes` also writes the rollouts as a
student-source dataset for failure review.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from imitation.data.schema import ACTOR_PERTURB, Episode
from imitation.rollout import EnvPool, ExpertController, PolicyController, rollout
from imitation.seeds import eval_set

SUCCESS_DIST = 0.05


def resolve_workers(backend, workers):
    """Explicit --workers wins; else 14 for MuJoCo, N_ISAAC for Isaac (one env per process)."""
    if workers is not None:
        return workers
    from imitation.tasks.half_fold import is_isaac
    if is_isaac(backend):
        from imitation.isaac_runtime import N_ISAAC
        return N_ISAAC
    return 14

FAILURE_CODES = {
    "G1": "missed or unstable grasp (an arm never grasped, or dropped its unplaced corner)",
    "M1": "motion error: anchor corner dragged or sim unstable",
    "F1": "fold placed with poor alignment (released off target)",
    "S1": "stalled: timed out still holding / hovering",
}
# Whether a failure followed a disturbance is reported separately (`perturbed_failures`),
# not as its own code: every `recovery` episode is perturbed, so an R1 code that
# short-circuited the checks below made that set's histogram uniformly R1.


def failure_code(ep: Episode) -> str | None:
    """Design-doc failure category of one episode, or None on success."""
    m = ep.meta
    if m["success"]:
        return None
    reason = m["termination_reason"]
    if reason in ("cloth_dragged", "unstable"):
        return "M1"
    ever = ep.grasped.any(axis=0)
    if not ever.all():
        return "G1"
    # an arm let go before its corner got near the goal
    dist = m.get("final_move_distance", [0.0, 0.0])
    for arm in range(2):
        g = ep.grasped[:, arm]
        released = np.flatnonzero(g[:-1] & ~g[1:])
        if len(released) and dist[arm] > 2 * SUCCESS_DIST and not g[-1]:
            return "G1"
    if not ep.grasped[-1].any() and max(dist) > SUCCESS_DIST:
        return "F1"
    return "S1"


def summarize(episodes: list[Episode], infer_ms=None) -> dict:
    n = len(episodes)
    codes = Counter(failure_code(e) for e in episodes)
    codes.pop(None, None)
    succ = [e for e in episodes if e.meta["success"]]
    return {"n": n, "success_rate": len(succ) / n,
            "mean_fold_score": float(np.mean([e.meta["final_fold_score"] for e in episodes])),
            "grasp_success": float(np.mean([e.grasped.any(axis=0).all() for e in episodes])),
            "mean_steps_success": float(np.mean([e.steps for e in succ])) if succ else None,
            "termination": dict(Counter(e.meta["termination_reason"] for e in episodes)),
            "failure_codes": dict(codes),
            "perturbed_failures": sum(1 for e in episodes if not e.meta["success"] and e.meta.get("perturb")),
            "perturbed_steps": int(sum((e.actor == ACTOR_PERTURB).sum() for e in episodes)),
            "policy_infer_ms": infer_ms}


def save_version_name(set_name, ckpt, n) -> str:
    """Dataset version for saved eval rollouts: unique per (set, checkpoint, n), so
    evaluating a second checkpoint into the same directory does not collide."""
    import hashlib
    tag = hashlib.sha1(str(Path(ckpt).resolve() if ckpt != "expert" else ckpt).encode()).hexdigest()[:8]
    return f"eval_{set_name}_n{n}_{tag}"


class _TimedPolicy:
    """Wraps a policy to measure batched inference time (the deployable-rate check)."""

    def __init__(self, policy):
        self.policy, self.obs_horizon, self.chunk = policy, policy.obs_horizon, policy.chunk
        self.needs_images = policy.needs_images
        self.times, self.batch = [], []

    def predict(self, obs, images=None):
        import torch
        t0 = time.perf_counter()
        out = self.policy.predict(obs, images) if self.needs_images else self.policy.predict(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.perf_counter() - t0)
        self.batch.append(len(obs))
        return out


def evaluate(ckpt, sets=("id_easy",), n=48, workers=None, replan_every=8, pool=None, log=print, save_dir=None,
             backend="mujoco", on_set=None):
    from imitation.policies.common import load_policy

    policy = None if ckpt == "expert" else load_policy(ckpt)
    needs_images = policy is not None and policy.needs_images
    cams = dict(policy.cameras) if needs_images else None
    if pool is not None and needs_images and (not pool.render or dict(pool.cameras or {}) != cams):
        raise ValueError(f"an image policy needs a pool made with EnvPool(n, {{'render': True, 'cameras': {cams}}})")
    own = pool is None
    kw = {"render": needs_images, "cameras": cams}
    if backend != "mujoco":
        kw["backend"] = backend
    pool = pool or EnvPool(resolve_workers(backend, workers), kw)
    results = {}
    try:
        if policy is None:
            controller, timed = ExpertController(), None
        else:
            timed = _TimedPolicy(policy)
            controller = PolicyController(timed, replan_every=replan_every)
        for name in sets:
            seeds, reset_options, perturb_fn = eval_set(name, n, backend)
            t0 = time.time()
            eps = rollout(pool, seeds, controller, reset_options=reset_options, perturb_fn=perturb_fn,
                          meta_extra={"eval_set": name, "checkpoint": str(ckpt)})
            infer = None
            if timed and timed.times:
                infer = {"ms_per_call": 1000 * float(np.median(timed.times)),
                         "mean_batch": float(np.mean(timed.batch))}
                timed.times.clear()
                timed.batch.clear()
            results[name] = summarize(eps, infer)
            r = results[name]
            if on_set is not None:         # e.g. persist each set as it finishes (a job limit can cut a long eval)
                on_set(name, r)
            log(f"  {name:9s} success {r['success_rate']:.0%} ({round(r['success_rate'] * r['n'])}/{r['n']})  "
                f"fold {r['mean_fold_score']:.3f}  grasp {r['grasp_success']:.0%}  "
                f"failures {r['failure_codes']}  {time.time() - t0:.0f}s")
            if save_dir:
                from imitation.data.schema import DatasetWriter
                w = DatasetWriter(save_dir, save_version_name(name, ckpt, n), config={"checkpoint": str(ckpt), "set": name})
                for e in eps:
                    w.add(e)
                w.freeze()
    finally:
        if own:
            pool.close()
    return results


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="policy checkpoint, or 'expert' for the scripted teacher")
    ap.add_argument("--sets", nargs="+", default=["id_easy", "id_hard", "recovery"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--workers", type=int, default=None, help="default 14 (mujoco) / N_ISAAC (isaac)")
    ap.add_argument("--backend", choices=("mujoco", "isaac", "isaac_weld"), default="mujoco")
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--out", default=None, help="JSON report path (default: next to the checkpoint)")
    ap.add_argument("--save-episodes", default=None, help="directory to write the rollouts to")
    ap.add_argument("--resume", action="store_true",
                    help="keep the sets already in --out (same checkpoint and replan) and run only the missing ones")
    return ap


def main():
    args = build_parser().parse_args()
    out = Path(args.out or (Path(args.ckpt).with_name(f"eval_r{args.replan_every}.json") if args.ckpt != "expert"
                            else Path("outputs/imitation/eval_expert.json")))
    results = {}
    if args.resume and out.exists():
        prev = json.loads(out.read_text())
        if prev.get("checkpoint") == args.ckpt and prev.get("replan_every") == args.replan_every:
            results = {k: v for k, v in prev["results"].items() if v.get("n") == args.n}
            print(f"resuming {out}: have {sorted(results)}")

    def write(name=None, r=None):
        if name is not None:
            results[name] = r
        order = [s for s in args.sets if s in results] + [s for s in results if s not in args.sets]
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"checkpoint": args.ckpt, "replan_every": args.replan_every,
                       "results": {s: results[s] for s in order}, "failure_codes": FAILURE_CODES}, f, indent=1)

    todo = [s for s in args.sets if s not in results]
    if todo:
        evaluate(args.ckpt, todo, args.n, resolve_workers(args.backend, args.workers), args.replan_every,
                 save_dir=args.save_episodes, backend=args.backend, on_set=write)
    write()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

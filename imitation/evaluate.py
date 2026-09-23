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

FAILURE_CODES = {
    "G1": "missed or unstable grasp (an arm never grasped, or dropped its unplaced corner)",
    "M1": "motion error: anchor corner dragged or sim unstable",
    "F1": "fold placed with poor alignment (released off target)",
    "S1": "stalled: timed out still holding / hovering",
    "R1": "no recovery after a disturbance (perturbed episode that failed)",
}


def failure_code(ep: Episode) -> str | None:
    """Design-doc failure category of one episode, or None on success."""
    m = ep.meta
    if m["success"]:
        return None
    if m.get("perturb") is not None:
        return "R1"
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
            "failure_codes": dict(codes), "perturbed_steps": int(sum((e.actor == ACTOR_PERTURB).sum() for e in episodes)),
            "policy_infer_ms": infer_ms}


class _TimedPolicy:
    """Wraps a policy to measure batched inference time (the deployable-rate check)."""

    def __init__(self, policy):
        self.policy, self.obs_horizon, self.chunk = policy, policy.obs_horizon, policy.chunk
        self.times, self.batch = [], []

    def predict(self, obs):
        import torch
        t0 = time.perf_counter()
        out = self.policy.predict(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.perf_counter() - t0)
        self.batch.append(len(obs))
        return out


def evaluate(ckpt, sets=("id_easy",), n=48, workers=14, replan_every=8, pool=None, log=print, save_dir=None):
    from imitation.policies.common import load_policy

    own = pool is None
    pool = pool or EnvPool(workers)
    results = {}
    try:
        if ckpt == "expert":
            controller, timed = ExpertController(), None
        else:
            timed = _TimedPolicy(load_policy(ckpt))
            controller = PolicyController(timed, replan_every=replan_every)
        for name in sets:
            seeds, reset_options, perturb_fn = eval_set(name, n)
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
            log(f"  {name:9s} success {r['success_rate']:.0%} ({round(r['success_rate'] * r['n'])}/{r['n']})  "
                f"fold {r['mean_fold_score']:.3f}  grasp {r['grasp_success']:.0%}  "
                f"failures {r['failure_codes']}  {time.time() - t0:.0f}s")
            if save_dir:
                from imitation.data.schema import DatasetWriter
                w = DatasetWriter(save_dir, f"eval_{name}", config={"checkpoint": str(ckpt), "set": name})
                for e in eps:
                    w.add(e)
                w.freeze()
    finally:
        if own:
            pool.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="policy checkpoint, or 'expert' for the scripted teacher")
    ap.add_argument("--sets", nargs="+", default=["id_easy", "id_hard", "recovery"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--out", default=None, help="JSON report path (default: next to the checkpoint)")
    ap.add_argument("--save-episodes", default=None, help="directory to write the rollouts to")
    args = ap.parse_args()
    results = evaluate(args.ckpt, args.sets, args.n, args.workers, args.replan_every, save_dir=args.save_episodes)
    out = args.out or (Path(args.ckpt).with_name(f"eval_r{args.replan_every}.json") if args.ckpt != "expert"
                       else Path("outputs/imitation/eval_expert.json"))
    with open(out, "w") as f:
        json.dump({"checkpoint": args.ckpt, "replan_every": args.replan_every, "results": results,
                   "failure_codes": FAILURE_CODES}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

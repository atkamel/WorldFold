"""Recovery episodes on the isaac_weld backend for tuning the expert (Phase W, W3b), outside the eval sets.

Replays what imitation.rollout does for the `recovery` eval set -- a knock of k uniform-random actions at step t
(imitation.seeds.eval_set("recovery", ...)'s draw, from rng([seed, 7919])), then resync -- but on tune-block seeds
(RECOVERY_TUNE_BASE = 610_000 + i) so tuning never touches the MuJoCo eval seeds. One JSON row per episode with a
compact phase timeline. Resumable: rows already in --out are skipped. One Isaac process per call:

    python -u isaac/recovery_replay.py --seeds 610000:610020 --out outputs/isaac/recovery/a.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RECOVERY_TUNE_BASE = 610_000      # inside imitation.seeds' tune range (600_000-700_000), clear of overshoot_measure's


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=f"{RECOVERY_TUNE_BASE}:{RECOVERY_TUNE_BASE + 20}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="isaac_weld", choices=("isaac_weld", "isaac_friction"))
    ap.add_argument("--max-steps", type=int, default=None, help="episode cap (default: the backend's)")
    ap.add_argument("--retries", type=int, default=None, help="QuarterFoldExpert.MAX_RETRIES override (default 2)")
    ap.add_argument("--set", action="append", default=[], metavar="ATTR=VALUE",
                    help="override an IsaacFoldExpert class attribute for this run, e.g. REGRASP_OFFSET=none")
    args = ap.parse_args()
    a, b = (int(v) for v in args.seeds.split(":"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}

    from imitation.seeds import eval_set
    from imitation.spec import ACTION_DIM
    from imitation.tasks import make_env
    from imitation.teachers.scripted import ScriptedTeacher
    from isaac.weld_expert import IsaacFoldExpert
    for kv in args.set:                       # experiment knobs, recorded in every row
        k, v = kv.split("=", 1)
        val = None if v.lower() == "none" else (tuple(v.split("+")) if not v.replace(".", "").isdigit() else float(v))
        setattr(IsaacFoldExpert, k, val)
    import cloth_fold_rl.quarter_fold_expert as qfe
    if args.retries is not None:
        qfe.MAX_RETRIES = args.retries
    # log every retry: when it fires, the arm, and the measured miss it corrects (retry analysis, W3b)
    retry_log = []
    orig_retry = qfe.QuarterFoldExpert._maybe_retry

    def logged_retry(self, key, expert):
        before = self.retries[key]
        orig_retry(self, key, expert)
        if self.retries[key] > before:
            retry_log.append({"t": int(self.base._step_count), "arm": key[1], "n": self.retries[key],
                              "miss": [round(float(v), 4) for v in self.correction[key]]})
    qfe.QuarterFoldExpert._maybe_retry = logged_retry
    _, _, knock = eval_set("recovery", 1, args.backend)
    env = make_env(args.backend, **({"max_episode_steps": args.max_steps} if args.max_steps else {}))
    expert = ScriptedTeacher(env, seed=0).expert
    base = env.unwrapped
    for seed in range(a, b):
        if seed in done:
            continue
        rng = np.random.default_rng([seed, 7919])
        p = knock(seed, rng)
        env.reset(seed=seed)
        expert.reset()
        stale, timeline, prev = False, [], None
        retry_log.clear()
        for t in range(base.max_episode_steps):
            if p.active(t):
                act, tag, stale = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32), "K", True
            else:
                if stale:
                    expert.resync()
                    stale = False
                act, tag = expert.act(), ""
            ph = expert.phases()
            g = "".join(str(int(base.grasp_active(x))) for x in ("left_", "right_"))
            cur = f"{tag}{ph['left_'][:4]}/{ph['right_'][:4]} g{g}"
            if cur != prev:
                timeline.append(f"t{t}:{cur}")
            prev = cur
            _, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        row = {"seed": seed, "knock": [p.t, p.k], "success": bool(info["success"]), "steps": t + 1,
               "reason": info["termination_reason"] or "truncated", "retries": sum(expert.retries.values()),
               "d": [round(float(x), 4) for x in info["move_distance"]], "domain": dict(base._domain_params),
               "knobs": args.set, "max_steps": base.max_episode_steps, "max_retries": qfe.MAX_RETRIES,
               "retry_log": list(retry_log),
               "timeline": timeline}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print("EP", json.dumps({k: v for k, v in row.items() if k != "timeline"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

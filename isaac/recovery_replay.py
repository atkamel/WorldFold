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
    _, _, knock = eval_set("recovery", 1, "isaac_weld")
    env = make_env("isaac_weld")
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

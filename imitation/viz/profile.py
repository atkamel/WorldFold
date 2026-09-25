"""Where does the time go? Profile the pipeline's loops (roadmap M5c.1).

    python -m imitation.viz.profile --episodes 20 --workers 10 --out outputs/imitation/profile.json

Times four representative loops on the same seeds and worker count:
  eval_state     a state policy evaluated (no labels, no cameras)
  eval_vision    a vision policy evaluated (cameras rendered every step)
  dagger_expert  a student rolled out with the scripted expert shadowing + labelling
  dagger_policy  a vision student with a policy teacher labelling on the GPU
and reports, per simulated step: worker physics, worker rendering, worker teacher work
(shadowing), worker teacher labels (the scripted expert's look-ahead simulation),
main-process planning (policy + GPU labels), and main-process waiting. The worker columns
are summed over workers, so they're compared against wall x workers.
Training throughput comes from existing `run.json` files (steps / seconds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imitation.policies.common import load_policy
from imitation.rollout import EnvPool, PolicyController, rollout
from imitation.seeds import EVAL_SEED_BASE

R = Path("outputs/imitation/runs")


def run(name, controller, pool_kwargs, seeds, workers):
    stats = {}
    with EnvPool(workers, pool_kwargs) as pool:
        rollout(pool, seeds, controller, stats=stats)
    steps = max(stats.get("steps", 0), 1)
    wall = stats["wall_s"]
    row = {"loop": name, "episodes": len(seeds), "steps": steps, "wall_s": round(wall, 1),
           "ms_per_step_wall": round(1000 * wall / steps, 2)}
    for k in ("physics_s", "render_s", "teacher_s", "label_s"):   # worker-side, summed over workers
        row[k.replace("_s", "_share")] = round(stats.get(k, 0.0) / (wall * workers), 3)
    for k in ("plan_s", "wait_s"):                             # main process
        row[k.replace("_s", "_share_main")] = round(stats.get(k, 0.0) / wall, 3)
    row["ms_per_step_physics"] = round(1000 * stats.get("physics_s", 0.0) / steps, 2)
    row["ms_per_step_render"] = round(1000 * stats.get("render_s", 0.0) / steps, 2)
    return row


def training_rows():
    rows = []
    for rj in sorted(R.glob("*/run.json")) + sorted(R.glob("*/round_*/run.json")):
        info = json.loads(rj.read_text())
        t = info.get("train", {})
        if t.get("seconds") and t.get("steps"):
            rows.append({"run": str(rj.parent.relative_to(R)), "kind": info["policy"]["kind"],
                         "steps": t["steps"], "seconds": t["seconds"],
                         "steps_per_s": round(t["steps"] / t["seconds"], 1),
                         "teacher": info["dataset"].get("teacher") is not None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--state", default=str(R / "dagger_diff/round_1/final.pt"))
    ap.add_argument("--vision", default=str(R / "vision_bc_128/final.pt"))
    ap.add_argument("--out", default="outputs/imitation/profile.json")
    args = ap.parse_args()
    seeds = list(range(EVAL_SEED_BASE["id_easy"], EVAL_SEED_BASE["id_easy"] + args.episodes))
    state, vision = load_policy(args.state), load_policy(args.vision)
    cams = {"render": True, "cameras": dict(vision.cameras)}
    rows = [run("eval_state", PolicyController(state), {}, seeds, args.workers),
            run("eval_vision", PolicyController(vision), cams, seeds, args.workers),
            run("dagger_expert", PolicyController(state, beta=0.3, label=True, source="dagger"), {}, seeds,
                args.workers),
            run("dagger_policy", PolicyController(vision, beta=0.3, label=True, source="dagger",
                                                  teacher_policy=state), cams, seeds, args.workers)]
    out = {"workers": args.workers, "episodes": args.episodes, "loops": rows, "training": training_rows()}
    Path(args.out).write_text(json.dumps(out, indent=1))
    for r in rows:
        print(r)
    for t in out["training"]:
        print(t)


if __name__ == "__main__":
    main()

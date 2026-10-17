"""QD-2 at scale (2026-10-16): the half fold's settled placement miss on isaac_friction, signed, per arm.

Runs the scripted expert on clean half-fold episodes (tune-block seeds, no knock) and, when the episode ends, records
for each stage-0 move where its carried corner settled relative to the move's goal (the goal vertex's position at the
stage start), in cm, table frame. The placement overshoot (isaac.fold_expert.OVERSHOOT_FRICTION, 5 cm along y) was
calibrated against the old release's fling; this is the number to recalibrate it from if the gentle release
(WORLDFOLD_GRASP_KNOBS / WORLDFOLD_EXPERT_PARAMS, honoured here through the env and the expert) lands short.
Resumable (rows already in --out are skipped). One Isaac process per call.

    .venv-isaac/Scripts/python.exe -u isaac/place_miss.py --seeds 610000:610020 --out outputs/.../half_miss_base.jsonl
    .venv/Scripts/python.exe isaac/place_miss.py --summary a.jsonl [b.jsonl ...]     # no Isaac: the table only
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HELD_DIST = 0.10      # a corner further than this from its goal was dropped or dragged: left out of the mean miss


def summarise(paths):
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        print(f"{path}: n = {len(rows)}, success {sum(r['success'] for r in rows)}/{len(rows)}, "
              f"mean steps {np.mean([r['steps'] for r in rows]):.0f}")
        print("  arm     held   placed(5cm)  mean miss x,y (cm)   SD x,y   median |miss|")
        for arm in ("left_", "right_"):
            m = [mv for r in rows for mv in r["moves"] if mv["arm"] == arm]
            held = [mv for mv in m if mv["dist_cm"] < 100 * HELD_DIST]
            if not held:
                print(f"  {arm:7s} 0/{len(m)}")
                continue
            xy = np.array([mv["miss_xy_cm"] for mv in held])
            print(f"  {arm:7s} {len(held):2d}/{len(m):2d}   {sum(mv['placed'] for mv in m):2d}/{len(m):2d}       "
                  f"{xy[:, 0].mean():+5.1f}, {xy[:, 1].mean():+5.1f}        {xy[:, 0].std():.1f}, {xy[:, 1].std():.1f}"
                  f"     {np.median(np.linalg.norm(xy, axis=1)):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="610000:610020")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--summary", nargs="*", default=None, help="summarise these jsonl files and exit (no Isaac)")
    args = ap.parse_args()
    if args.summary is not None:
        summarise(args.summary)
        return
    a, b = (int(v) for v in args.seeds.split(":"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["seed"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}

    from imitation.tasks import make_env
    from imitation.teachers.scripted import ScriptedTeacher
    from imitation.thermal import wait_while_hot
    env = make_env("isaac_friction", task="half_fold", **({"max_episode_steps": args.max_steps} if args.max_steps else {}))
    expert = ScriptedTeacher(env, seed=0).expert
    base = env.unwrapped
    knobs = {k: os.environ.get(k, "") for k in ("WORLDFOLD_GRASP_KNOBS", "WORLDFOLD_EXPERT_PARAMS")}
    for seed in range(a, b):
        if seed in done:
            continue
        wait_while_hot()
        env.reset(seed=seed)
        expert.reset()
        moves = env.stages[0].moves
        goals = {i: np.array(env.goal(m), dtype=float).copy() for i, m in enumerate(moves)}
        info, t = {}, 0
        for t in range(base.max_episode_steps):
            _, _, term, trunc, info = env.step(expert.act())
            if term or trunc:
                break
        cloth = base.cloth_positions()
        rows = []
        for i, m in enumerate(moves):
            for c in m.corners:
                d = (np.asarray(cloth[c], dtype=float) - goals[i])
                rows.append({"arm": m.prefix, "corner": int(c), "miss_xy_cm": [round(100 * float(v), 2) for v in d[:2]],
                             "dz_cm": round(100 * float(d[2]), 2), "dist_cm": round(100 * float(np.linalg.norm(d)), 2),
                             "placed": bool(np.linalg.norm(d) < 0.05)})
        row = {"seed": seed, "success": bool(info.get("success", False)), "steps": t + 1,
               "reason": info.get("termination_reason") or "truncated", "retries": sum(expert.retries.values()),
               "knobs": knobs, "moves": rows}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print("EP", json.dumps(row), flush=True)
    summarise([str(out)])


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

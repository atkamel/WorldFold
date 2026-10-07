"""Friction-grasp bench (track G, roadmap IG.1-IG.2): the friction expert's first attempt on each arm, scored per arm
as acquired / held / placed / released plus anchor drift (definitions in isaac/grasp_metrics.py).

Each episode is the Isaac half fold on the "lehome" profile (friction grasp, CPU device) driven by the pipeline's
ScriptedTeacher (IsaacArmExpert in QuarterFoldExpert) with retries off, so a failed grasp is a failure and not a
second chance. It stops once both arms have retreated and waited SETTLE_WAIT steps, or the env ends the episode.
Every step's phase, gripperframe site, corner, jaw angle and anchor per arm goes to traces/<seed>.json; one row per
episode goes to rows.jsonl. The run resumes: seeds already in rows.jsonl are skipped.

    (Isaac venv, cwd = the checkout that holds Assets/)
    python -u <repo>/isaac/grasp_bench.py --seeds 600000:600020 --out <repo>/outputs/isaac/grasp/baseline
    python -u <repo>/isaac/grasp_bench.py ... --knobs particle_friction=1.0 --expert PINCH_INSET=0.01
    python isaac/grasp_bench.py --summarize outputs/isaac/grasp/baseline [--rescore]     (any venv, no Isaac)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaac.grasp_metrics import SETTLE_WAIT, arm_metrics, summarize  # noqa: E402

CAP = 400                 # = imitation.isaac_runtime.ISAAC_MAX_STEPS


def _rows(out):
    path = Path(out) / "rows.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_summary(out, rescore=False):
    rows = _rows(out)
    if rescore:            # recompute the per-arm flags from the stored traces (after a metric definition changes)
        for r in rows:
            tr = json.loads((Path(out) / "traces" / f"{r['seed']}.json").read_text(encoding="utf-8"))
            r["arms"] = {p: arm_metrics(a, a["goal"], a["rest_z"]) for p, a in tr["arms"].items()}
    s = summarize(rows)
    s["config"] = rows[0].get("config") if rows else None
    s["reasons"] = {}
    for r in rows:
        s["reasons"][r["reason"]] = s["reasons"].get(r["reason"], 0) + 1
    (Path(out) / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


def print_summary(s):
    print(f"n = {s['n']}  reasons {s['reasons']}")
    for p, st in s["arms"].items():
        cells = "  ".join(f"{m} {st[m]['k']}/{st[m]['n']} [{st[m]['wilson'][0]}, {st[m]['wilson'][1]}]"
                          for m in ("acquired", "held", "placed", "released", "released_strict"))
        print(f"  {p:7s} {cells}  drift mean {st['anchor_drift_m']['mean']} max {st['anchor_drift_m']['max']}"
              f"  first failure {st['first_failure']}")


def parse_kv(text):
    """'a=1.5,b.c=50' -> {"a": 1.5, "b": {"c": 50}} (shell-safe: no JSON quoting through PowerShell and cmd)."""
    out = {}
    for item in filter(None, (s.strip() for s in (text or "").split(","))):
        key, val = item.split("=", 1)
        val = json.loads(val)
        if "." in key:
            outer, inner = key.split(".", 1)
            out.setdefault(outer, {})[inner] = val
        else:
            out[key] = val
    return out


def run(args):
    import cloth_fold_rl.quarter_fold_expert as qfe
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS
    from imitation.tasks.half_fold import HalfFoldEnv
    from imitation.isaac_runtime import ISAAC_CLOTH_JITTER
    from isaac.isaac_env import IsaacClothFoldEnv
    from isaac.fold_expert import OVERSHOOT_FRICTION, IsaacArmExpert
    from imitation.teachers.scripted import ScriptedTeacher

    qfe.MAX_RETRIES = 0
    if args.overshoot:                       # stage-0 placement offsets, a candidate (fold_expert.OVERSHOOT_FRICTION)
        lx, ly, rx, ry = (float(v) for v in args.overshoot.split(","))
        OVERSHOOT_FRICTION[(0, "left_")][:] = (lx, ly, 0.0)
        OVERSHOOT_FRICTION[(0, "right_")][:] = (rx, ry, 0.0)
    knobs = parse_kv(args.knobs)
    expert_over = parse_kv(args.expert)
    for k, v in expert_over.items():
        if not hasattr(IsaacArmExpert, k):
            raise ValueError(f"IsaacArmExpert has no {k}")
        setattr(IsaacArmExpert, k, v)
    sys.argv = sys.argv[:1]                  # AppLauncher reads sys.argv
    base = IsaacClothFoldEnv(observation_mode="state", max_episode_steps=CAP, grasp_corners=GRASP_CORNERS,
                             grasp_radius=GRASP_RADIUS, profile="lehome", grasp_knobs=knobs or None)
    env = HalfFoldEnv(base_env=base, max_episode_steps=CAP, cloth_jitter=ISAAC_CLOTH_JITTER)
    teacher = ScriptedTeacher(env)
    qf = teacher.expert
    config = {"knobs": base.grasp_knobs, "expert": {k: getattr(IsaacArmExpert, k) for k in
                                                    ("PINCH_HEIGHT", "PINCH_INSET", "PLACE_HEIGHT", "ARC_HEIGHT",
                                                     "CLOSE_DWELL", "OPEN_DWELL")},
              "overshoot": {f"{s}{p}": np.round(v, 4).tolist() for (s, p), v in OVERSHOOT_FRICTION.items() if s == 0},
              "code": str(Path(__file__).resolve())}
    print("CONFIG", json.dumps(config), flush=True)

    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    done = {r["seed"] for r in _rows(out)}
    a, b = (int(v) for v in args.seeds.split(":"))
    for seed in range(a, b):
        if seed in done:
            continue
        t0 = time.time()
        env.reset(seed=seed)
        teacher.reset()
        arms = {key[1]: e for key, e in qf.experts.items() if key[0] == 0}
        moves = {p: qf.moves[(0, p)] for p in arms}
        goals = {p: env.goal(moves[p]) for p in arms}
        rest_z = {p: float(arms[p]._corner()[2]) for p in arms}
        tr = {p: {"phase": [], "site": [], "quat": [], "corner": [], "jaw": [], "anchor": []} for p in arms}

        def record():
            for p, e in arms.items():
                tr[p]["phase"].append(e.PHASES[e.phase])
                pos, quat = base.gripper_pose(p)
                tr[p]["site"].append(np.round(pos, 4).tolist())
                tr[p]["quat"].append(np.round(quat, 4).tolist())
                tr[p]["corner"].append(np.round(e._corner(), 4).tolist())
                tr[p]["jaw"].append(round(float(base.joint_positions(p)[5]), 4))
                tr[p]["anchor"].append(np.round(env._vertex(moves[p].goal), 4).tolist())

        info, t = {}, 0
        for t in range(CAP):
            record()
            _, _, term, trunc, info = env.step(teacher.act())
            if term or trunc:
                break
            if all(e.PHASES[e.phase] == "done" for e in arms.values()) and \
                    all(qf.done_steps[(0, p)] >= SETTLE_WAIT + 2 for p in arms):
                break
        record()
        metrics = {p: arm_metrics(tr[p], goals[p], rest_z[p]) for p in arms}
        row = {"seed": seed, "steps": t + 1, "reason": info.get("termination_reason") or ("truncated" if t + 1 >= CAP
                                                                                          else "bench_stop"),
               "success": bool(info.get("success")), "fold_score": round(float(info.get("fold_score", 0.0)), 3),
               "arms": metrics, "wall_s": round(time.time() - t0, 1), "config": config}
        trace = {"seed": seed, "arms": {p: dict(tr[p], goal=np.round(goals[p], 4).tolist(), rest_z=rest_z[p])
                                        for p in arms}}
        (out / "traces" / f"{seed}.json").write_text(json.dumps(trace), encoding="utf-8")
        with open(out / "rows.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        short = {p: {k: m[k] for k in ("acquired", "held", "placed", "released")} for p, m in metrics.items()}
        print("ROW", seed, row["reason"], row["steps"], row["wall_s"], json.dumps(short), flush=True)
    print_summary(write_summary(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="600000:600020", help="start:end (imitation.seeds TUNE_SEED_BASE block)")
    ap.add_argument("--out", help="output directory (absolute when cwd is another checkout)")
    ap.add_argument("--knobs", default=None, help="k=v,..., isaac_env.FRICTION_GRASP overrides (gripper_drive.stiffness=50)")
    ap.add_argument("--expert", default=None, help="k=v,..., IsaacArmExpert class-attribute overrides")
    ap.add_argument("--overshoot", default=None, help="stage-0 'lx,ly,rx,ry' (m) placement offsets")
    ap.add_argument("--summarize", default=None, help="only summarize this output directory (no Isaac)")
    ap.add_argument("--rescore", action="store_true", help="with --summarize: recompute flags from the traces")
    args = ap.parse_args()
    if args.summarize:
        print_summary(write_summary(args.summarize, rescore=args.rescore))
        return
    run(args)


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        main()
        sys.exit(0)
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

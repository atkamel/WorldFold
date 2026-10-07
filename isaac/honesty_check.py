"""Grasp honesty check (Phase F, F1): does the cloth move only through the jaws' contact?

Runs the scripted expert (the pipeline's ScriptedTeacher) for N episodes on one Isaac backend and checks every step:
  pins           no particle is pinned (env._pinned and the scene's pin table are empty)
  attachments    the stage holds no PhysX attachment prims (PhysxPhysicsAttachment / AutoAttachment), at reset and end
  jaw_target     while the gripper is commanded closed, the jaw joint's target is the closed target, not forced open
  lifted_shut    whenever an arm's grasp corner is LIFT_MIN above its rest height and within HOLD_DIST of the gripper,
                 that arm's jaw is physically shut (joint within JAW_SETTLED_TOL of the closed target)
A friction backend must have zero violations of every check. The weld backend is the negative control: its jaws are
forced open while its pinned corner rides in the air, so jaw_target and lifted_shut must fire there.

    (Isaac venv, cwd = the checkout with Assets/)
    python -u isaac/honesty_check.py --backend isaac_friction --n 20 --out outputs/isaac/honesty/friction.json
    python -u isaac/honesty_check.py --backend isaac_weld --n 5 --out outputs/isaac/honesty/weld.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LIFT_MIN = 0.02            # = isaac.grasp_metrics.LIFT_MIN
CHECKS = ("pins", "attachments", "jaw_target", "lifted_shut")


def attachment_prims(stage):
    hits = []
    for prim in stage.Traverse():
        name = prim.GetTypeName()
        schemas = " ".join(prim.GetAppliedSchemas())
        if "Attachment" in name or "Attachment" in schemas:
            hits.append(str(prim.GetPath()))
    return hits


def run(args):
    from imitation.tasks.half_fold import make_env
    from imitation.teachers.scripted import ScriptedTeacher
    from isaac.isaac_env import HOLD_DIST, JAW_SETTLED_TOL

    sys.argv = sys.argv[:1]
    env = make_env(args.backend)
    base = env.unwrapped.env if hasattr(env.unwrapped, "env") else env.unwrapped
    teacher = ScriptedTeacher(env)
    counts = {c: 0 for c in CHECKS}
    examples = {c: [] for c in CHECKS}
    lifted_steps = 0
    episodes = []

    def flag(check, seed, t, detail):
        counts[check] += 1
        if len(examples[check]) < 5:
            examples[check].append({"seed": seed, "t": t, **detail})

    for seed in range(args.seed0, args.seed0 + args.n):
        t0 = time.time()
        env.reset(seed=seed)
        teacher.reset()
        rest = {p: float(np.mean([base.cloth_positions()[v][2] for v in base.grasp_corners[p]])) for p in base.prefixes}
        for hit in attachment_prims(base.lab.sim.stage):
            flag("attachments", seed, 0, {"prim": hit})
        info, t = {}, 0
        for t in range(env.max_episode_steps if hasattr(env, "max_episode_steps") else 400):
            _, _, term, trunc, info = env.step(teacher.act())
            if any(base._pinned[p] for p in base.prefixes) or any(cp.pins for cp in base.lab.copies):
                flag("pins", seed, t, {})
            targets = base._targets().cpu().numpy()[0]
            allc = base.cloth_positions()
            for i, p in enumerate(base.prefixes):
                jaw_t = float(targets[5 + 6 * i])
                jaw_q = float(base.joint_positions(p)[5])
                if base._gripper_closed[p] and abs(jaw_t - base._closed_q) > 1e-6:
                    flag("jaw_target", seed, t, {"arm": p, "target": round(jaw_t, 3)})
                site = base.gripper_position(p)
                for v in base.grasp_corners[p]:
                    c = allc[v]
                    if c[2] > rest[p] + LIFT_MIN and np.linalg.norm(c - site) < HOLD_DIST:
                        lifted_steps += 1
                        if jaw_q > base._closed_q + JAW_SETTLED_TOL:
                            flag("lifted_shut", seed, t, {"arm": p, "jaw": round(jaw_q, 3),
                                                          "corner_z_up_cm": round(100 * (c[2] - rest[p]), 1)})
            if term or trunc:
                break
        for hit in attachment_prims(base.lab.sim.stage):
            flag("attachments", seed, t, {"prim": hit})
        episodes.append({"seed": seed, "steps": t + 1, "success": bool(info.get("success")),
                         "wall_s": round(time.time() - t0, 1)})
        print("EP", seed, episodes[-1]["success"], t + 1, json.dumps(counts), flush=True)

    report = {"backend": args.backend, "profile": base.profile, "grasp_mode": base.grasp_mode, "n": args.n,
              "seeds": [args.seed0, args.seed0 + args.n], "lifted_steps": lifted_steps, "violations": counts,
              "examples": examples, "episodes": episodes,
              "successes": sum(e["success"] for e in episodes),
              "honest": all(v == 0 for v in counts.values()) and lifted_steps > 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("HONEST" if report["honest"] else "NOT HONEST", json.dumps(counts), "lifted_steps", lifted_steps, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="isaac_friction", choices=("isaac", "isaac_weld", "isaac_friction"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=100_000, help="first seed (default: id_easy's)")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


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

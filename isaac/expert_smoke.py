"""One or a few half-fold episodes on Isaac driven by the imitation pipeline's ScriptedTeacher (IsaacArmExpert), with
phase transitions printed. A debugging aid for Phase I, I2.1; the measured rates come from imitation.evaluate.

    python -u isaac/expert_smoke.py --seeds 100000 100001     (Isaac venv)
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[100000])
    args = ap.parse_args()
    from imitation.tasks import make_env
    from imitation.teachers.scripted import ScriptedTeacher
    env = make_env(backend="isaac")
    teacher = ScriptedTeacher(env)
    for seed in args.seeds:
        env.reset(seed=seed)
        teacher.reset()
        last, info, t = None, {}, 0
        for t in range(env.unwrapped.max_episode_steps):
            phases = teacher.phases()
            if phases != last:
                print(f"  t={t:3d} {phases}", flush=True)
                last = phases
            _, _, term, trunc, info = env.step(teacher.act())
            if term or trunc:
                break
        print("EPISODE", json.dumps({"seed": seed, "steps": t + 1, "success": bool(info.get("success")),
                                     "reason": info.get("termination_reason"),
                                     "fold_score": round(float(info.get("fold_score", 0)), 3),
                                     "move_distance": [round(float(d), 3) for d in info.get("move_distance", [])],
                                     "anchor_drift": round(float(info.get("anchor_drift", 0)), 3)}), flush=True)


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

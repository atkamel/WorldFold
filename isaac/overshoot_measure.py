"""Spring-back of a placed corner on the Isaac weld profile (Phase W, W3), to set isaac.weld_expert.OVERSHOOT_ISAAC
the way cloth_fold_rl.quarter_fold_expert.OVERSHOOT was set on MuJoCo: place every corner exactly on its goal (zero
overshoot, no retries), wait SETTLE_WAIT steps after release, and record where it settled relative to the goal.
The overshoot is minus the mean miss. Run in the Isaac venv from the repo root (one Isaac process per call):

    python -u isaac/overshoot_measure.py --seeds 600000:600010 --out outputs/isaac/overshoot/a.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="600000:600010", help="start:end, from imitation.seeds' tune block")
    ap.add_argument("--out", required=True)
    ap.add_argument("--place-height", type=float, default=None, help="override IsaacFoldExpert.PLACE_HEIGHT (m)")
    ap.add_argument("--overshoot", default=None, help="stage-0 'lx,ly,rx,ry' (m) instead of zero: a candidate to check")
    ap.add_argument("--weld-tau", type=float, default=None, help="soft weld time constant (s); default rigid")
    ap.add_argument("--retries", type=int, default=0, help="QuarterFoldExpert.MAX_RETRIES (0: measure the first placement)")
    args = ap.parse_args()
    a, b = (int(v) for v in args.seeds.split(":"))

    import cloth_fold_rl.quarter_fold_expert as qfe
    from imitation.tasks import make_env
    from isaac.weld_expert import IsaacFoldExpert
    qfe.MAX_RETRIES = args.retries
    if args.place_height is not None:
        IsaacFoldExpert.PLACE_HEIGHT = args.place_height
    env = make_env("isaac_weld")
    if args.weld_tau is not None:                 # default: the profile's calibrated soft weld
        env.unwrapped.lab.weld_tau = args.weld_tau or None
    overshoot = {key: np.zeros(3) for key in qfe.OVERSHOOT}
    if args.overshoot:
        lx, ly, rx, ry = (float(v) for v in args.overshoot.split(","))
        overshoot[(0, "left_")], overshoot[(0, "right_")] = np.array([lx, ly, 0.0]), np.array([rx, ry, 0.0])
    expert = qfe.QuarterFoldExpert(env, seed=0, expert_cls=IsaacFoldExpert, overshoot=overshoot)
    rows = []
    for seed in range(a, b):
        env.reset(seed=seed)
        expert.reset()
        settled, placed_at = {}, {}
        for t in range(env.unwrapped.max_episode_steps):
            for key, e in expert._arms().items():
                name = e.PHASES[e.phase]
                if name == "release" and key not in placed_at:
                    placed_at[key] = np.mean([env._vertex(c) for c in expert.moves[(0, key)].corners], axis=0)
                if name == "done" and expert.done_steps[(0, key)] >= qfe.SETTLE_WAIT and key not in settled:
                    m = expert.moves[(0, key)]
                    settled[key] = np.mean([env._vertex(c) for c in m.corners], axis=0) - env.goal(m)
            _, _, term, trunc, info = env.step(expert.act())
            if len(settled) == 2 and args.retries == 0:
                break
            if term or trunc:
                # a success terminates once the placement has held (HalfFoldEnv's settle), often before SETTLE_WAIT:
                # a released corner's miss is read here, or the sample would only hold the failures
                for key in placed_at:
                    if key not in settled:
                        m = expert.moves[(0, key)]
                        settled[key] = np.mean([env._vertex(c) for c in m.corners], axis=0) - env.goal(m)
                break
        row = {"seed": seed, "steps": t + 1, "success": bool(info["success"]), "retries": sum(expert.retries.values()),
               "domain": dict(env.unwrapped._domain_params),
               "miss": {k: np.round(v, 4).tolist() for k, v in settled.items()},
               "miss_at_release": {k: np.round(v - env.goal(expert.moves[(0, k)]), 4).tolist() for k, v in placed_at.items()}}
        rows.append(row)
        print("ROW", json.dumps(row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))


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

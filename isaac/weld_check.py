"""Physical check of the weld grasp on Isaac (Phase W, W1). Run in the Isaac venv from the repo root:

    python -u isaac/weld_check.py --out outputs/isaac/weld

Left arm only: position-only IK (PinchIK, orientation_weight=0) to just above cloth_10, close (the weld must engage),
lift 10 cm, carry 10 cm sideways, hold with a neutral gripper command (hysteresis), open (the corner must drop),
then repeat the approach with cloth_10 removed from weld_mask (nothing may pin). Writes weld.json: per-step
tracking error of every pinned particle vs gripperframe composed with its offset, the corner's height, step cost.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORNER = 10           # cloth_10, the left arm's half-fold corner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/isaac/weld")
    ap.add_argument("--device", default="cuda:0", help="the weld needs the GPU pipeline's particle tensor view")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS
    from isaac.isaac_env import IsaacClothFoldEnv, _matrix_from_quat
    from isaac.pinch import PinchIK
    from mujuco.cloth_params import JOINT_DELTA_SCALE
    env = IsaacClothFoldEnv(observation_mode="state", max_episode_steps=1000, grasp_corners=GRASP_CORNERS,
                            grasp_radius=GRASP_RADIUS, grasp_mode="weld", device=args.device)
    ik = PinchIK()
    report = {"phases": {}, "device": args.device}

    def act(q_target, grip):
        a = np.zeros(14, dtype=np.float32)
        d = (np.asarray(q_target) - env.joint_positions("left_")[:5]) / JOINT_DELTA_SCALE
        a[0:5] = np.clip(d, -1, 1)
        a[6] = grip
        a[13] = 1.0
        env.step(a)

    def go(target, grip, steps, tag, track=False):
        q, err = ik.solve("left_", target, np.array([0.0, 1.0, 0.0]), list(env.joint_positions("left_")[:5]),
                          orientation_weight=0.0)
        errs, zs, t0 = [], [], time.time()
        for _ in range(steps):
            act(q, grip)
            if track and env._pinned["left_"]:
                site, quat = env.gripper_pose("left_")
                R = _matrix_from_quat(quat)
                e = max(float(np.max(np.linalg.norm(env._particles[idx] - (site + off @ R.T), axis=1)))
                        for idx, off in env._pinned["left_"].values())
                errs.append(e)
            zs.append(float(env.corner_positions()[1][2]))
        report["phases"][tag] = {"ik_err_mm": round(1000 * err, 2), "steps": steps,
                                 "s_per_step": round((time.time() - t0) / steps, 3),
                                 "max_track_mm": round(1000 * max(errs), 3) if errs else None,
                                 "corner_z_start": round(zs[0], 4), "corner_z_end": round(zs[-1], 4),
                                 "grasp_active": env.grasp_active("left_")}

    env.reset(seed=0)
    env.weld_mask = {"left_": None, "right_": None}
    cloth = env.cloth_positions()
    report["cloth_rest_z"] = [round(float(cloth[:, 2].min()), 4), round(float(cloth[:, 2].max()), 4)]
    corner = cloth[CORNER]
    go(corner + [0, 0, 0.06], 1.0, 40, "approach")
    go(corner + [0, 0, 0.005], 1.0, 30, "descend")
    go(corner + [0, 0, 0.005], -1.0, 5, "close", track=True)
    report["engaged"] = env.grasp_active("left_")
    report["n_pinned_particles"] = int(sum(len(i) for i, _ in env._pinned["left_"].values()))
    go(corner + [0, 0, 0.105], -1.0, 40, "lift", track=True)
    go(corner + [0.10, 0, 0.105], -1.0, 40, "carry", track=True)
    go(corner + [0.10, 0, 0.105], 0.0, 10, "hold_neutral", track=True)     # hysteresis: 0 keeps it closed
    report["held_on_neutral"] = env.grasp_active("left_")
    go(corner + [0.10, 0, 0.105], 1.0, 30, "release")
    report["released"] = not env.grasp_active("left_")

    env.reset(seed=1)
    env.weld_mask = {"left_": set(), "right_": None}                          # cloth_10 not allowed
    corner = env.cloth_positions()[CORNER]
    go(corner + [0, 0, 0.06], 1.0, 40, "mask_approach")
    go(corner + [0, 0, 0.005], 1.0, 30, "mask_descend")
    go(corner + [0, 0, 0.005], -1.0, 5, "mask_close")
    report["masked_engaged"] = env.grasp_active("left_")

    ph = report["phases"]
    report["checks"] = {
        "engaged": report["engaged"],
        "tracking_under_3mm": all((ph[k]["max_track_mm"] or 0) < 3.0 for k in ("lift", "carry", "hold_neutral")),
        "corner_rose_8cm": ph["lift"]["corner_z_end"] - ph["lift"]["corner_z_start"] > 0.08,
        "held_on_neutral": report["held_on_neutral"],
        "released_and_fell": report["released"] and ph["release"]["corner_z_end"] < ph["release"]["corner_z_start"] - 0.03,
        "mask_respected": not report["masked_engaged"],
        "cloth_rests_on_table": 0.41 < report["cloth_rest_z"][0] and report["cloth_rest_z"][1] < 0.45,
    }
    report["passed"] = all(report["checks"].values())
    (out / "weld.json").write_text(json.dumps(report, indent=1))
    print("WELD", json.dumps(report), flush=True)


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

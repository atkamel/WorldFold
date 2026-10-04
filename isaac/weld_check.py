"""Physical check of the weld grasp on Isaac (Phase W, W1). Run in the Isaac venv from the repo root:

    python -u isaac/weld_check.py --out outputs/isaac/weld

Left arm only: position-only IK (PinchIK, orientation_weight=0) to just above cloth_10, close (the weld must engage),
lift 10 cm, carry 10 cm sideways, hold with a neutral gripper command (hysteresis), open (the corner must drop),
then repeat the approach with cloth_10 removed from weld_mask (nothing may pin). Writes weld.json: per-step
tracking error of every pinned particle vs the gripperframe site plus its offset, the corner's height, step cost.
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
# the same script on MuJoCo's own weld (sim_main, default solref): max lag of the held corner behind the gripper hand
# composed with its captured offset, lift / carry (W3, docs/results.md). The soft weld is held to it; a rigid pin
# (--weld-tau 0) tracks within 3 mm but stores the fold's tension and snaps the corner back on release.
MUJOCO_MAX_TRACK_MM = 27.2
TRACK_PARITY = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/isaac/weld")
    ap.add_argument("--device", default="cuda:0", help="the weld needs the GPU pipeline's particle tensor view")
    ap.add_argument("--weld-tau", type=float, default=None,
                    help="soft weld time constant (s); default the weld profile's WELD_TAU, 0 = rigid pin")
    ap.add_argument("--weld-mass", type=float, default=None, help="soft weld pinned-mass multiple; default WELD_MASS")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS
    from isaac.isaac_env import IsaacClothFoldEnv
    from isaac.pinch import PinchIK
    from mujuco.cloth_params import JOINT_DELTA_SCALE
    env = IsaacClothFoldEnv(observation_mode="state", max_episode_steps=1000, grasp_corners=GRASP_CORNERS,
                            grasp_radius=GRASP_RADIUS, grasp_mode="weld", device=args.device)
    from isaac.isaac_env import WELD_MASS, WELD_TAU
    env.lab.weld_tau = WELD_TAU if args.weld_tau is None else (args.weld_tau or None)
    env.lab.weld_mass = WELD_MASS if args.weld_mass is None else args.weld_mass
    ik = PinchIK()
    report = {"phases": {}, "device": args.device, "weld_tau": env.lab.weld_tau, "weld_mass": env.lab.weld_mass}

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
                site, _ = env.gripper_pose("left_")
                e = 0.0
                for idx, off in env._pinned["left_"].values():
                    target = site + off                     # world-axis offsets (W3)
                    target[:, 2] = np.maximum(target[:, 2], env.lab.pin_floor)
                    e = max(e, float(np.max(np.linalg.norm(env._particles[idx] - target, axis=1))))
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
        "tracking_like_mujoco": all((ph[k]["max_track_mm"] or 0) < TRACK_PARITY * MUJOCO_MAX_TRACK_MM
                                    for k in ("lift", "carry", "hold_neutral")),
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

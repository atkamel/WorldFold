"""Physical check of the weld profile on Isaac (Phase W, W2; settings carried over from the MuJoCo setup). Run in the Isaac venv from the repo root:

    python -u isaac/profile_check.py --out outputs/isaac/profile

One IsaacClothFoldEnv(profile="weld") process. Writes profile.json:
- arm tracking: from home, each arm joint gets a one-step 0.05 rad command (action 1.0); the fraction of it reached
  after one control step (MuJoCo's drives reach it in one step; LeHome's ~0.78)
- cloth placement: centred at (0, 0) on a zero offset, and offset by exactly the requested cloth_pose
- dynamics DR: values drawn per seed, identical on a repeat, different across seeds, inside DR_RANGE; the particle
  masses and the particle material read back scaled by them; randomization=False restores the spawned values
- the material write reaches the solver: the same drop with damping x0 vs x40 falls measurably differently
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STEP_RAD = 0.05
# the same measurement on MuJoCo's ClothFoldEnv (left arm, joints 0-4; 2026-10-03, docs/results.md W2). Above 1
# because the commanded target, not the sagging measured pose, moves 0.05 rad: both simulators hold it there.
MUJOCO_STEP_FRACTION = {0: 1.287, 1: 1.395, 2: 1.413, 3: 1.33, 4: 1.297}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/isaac/profile")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS
    from isaac.isaac_env import DR_RANGE, MUJOCO_ARM_DRIVE, IsaacClothFoldEnv
    env = IsaacClothFoldEnv(observation_mode="state", max_episode_steps=1000, grasp_corners=GRASP_CORNERS,
                            grasp_radius=GRASP_RADIUS, profile="weld")
    lab = env.lab
    report = {"profile": "weld", "device": env.sim_device, "grasp_mode": env.grasp_mode,
              "arm_drive": MUJOCO_ARM_DRIVE}

    # ---- arm tracking ----
    env.domain_randomization = False
    track = {}
    for j in range(5):
        env.reset(seed=0)
        for _ in range(5):                                 # hold home so the arm starts at rest
            env.step(np.zeros(14, dtype=np.float32))
        q0 = env.joint_positions("left_")[:5].copy()
        a = np.zeros(14, dtype=np.float32)
        a[j] = 1.0
        env.step(a)
        dq = env.joint_positions("left_")[:5] - q0
        track[j] = round(float(dq[j] / STEP_RAD), 3)
    report["arm_step_fraction"] = track
    report["arm_step_fraction_mujoco"] = MUJOCO_STEP_FRACTION

    # ---- cloth placement ----
    env.reset(seed=1, options={"cloth_pose": [0.0, 0.0, 0.0]})
    c0 = env._particles[:, :2].mean(0)
    env.reset(seed=1, options={"cloth_pose": [0.02, -0.03, 0.0]})
    c1 = env._particles[:, :2].mean(0)
    report["cloth_centroid_zero"] = np.round(c0, 4).tolist()
    report["cloth_offset_measured"] = np.round(c1 - c0, 4).tolist()
    report["cloth_rest_z"] = [round(float(env._particles[:, 2].min()), 4), round(float(env._particles[:, 2].max()), 4)]

    # ---- dynamics DR ----
    env.domain_randomization = True
    pv = lab.cloth._cloth_prim_view._physics_view
    mat = lab.cloth.particle_material
    draws = {}
    for s in (5, 5, 6):
        env.reset(seed=s)
        dp = dict(env._domain_params)
        mass_ratio = float((pv.get_masses() / lab._rest_masses).flatten()[0])
        draws.setdefault(s, []).append({
            "params": dp, "mass_ratio": round(mass_ratio, 5),
            "friction_ratio": round(float(mat.get_friction()) / lab._rest_material[0], 5),
            "damping_ratio": round(float(mat.get_damping()) / lab._rest_material[1], 5)})
    env.reset(seed=5, options={"randomization": False})
    restored = {"params": dict(env._domain_params),
                "mass_ratio": round(float((pv.get_masses() / lab._rest_masses).flatten()[0]), 5),
                "friction": float(mat.get_friction()), "damping": float(mat.get_damping())}
    report["dr"] = {"seed5": draws[5], "seed6": draws[6], "unrandomized": restored,
                    "rest_material": list(lab._rest_material)}

    # ---- the material write reaches the solver: a 30 cm flat drop, damping x0 vs x40 (a mechanism test: inside the
    # DR range the effect over 3 steps is ~1e-4 m). Friction goes through the same material view. ----
    env.domain_randomization = False
    fall = {}
    for scale in (0.0, 40.0):
        env.reset(seed=2)
        lab.reset_cloth(np.zeros(2), 0.30, (0.0, 0.0, 0.0))
        lab.set_dynamics(1.0, 1.0, scale)
        env._particles = lab.particle_positions()
        for _ in range(3):
            env._advance()
        fall[scale] = round(float(env._particles[:, 2].mean()), 5)
    lab.set_dynamics(1.0, 1.0, 1.0)
    report["drop_mean_z_after_3_steps"] = {str(k): v for k, v in fall.items()}

    s5a, s5b, s6 = draws[5][0], draws[5][1], draws[6][0]
    keys = ("cloth_mass_scale", "table_friction_scale", "cloth_damping_scale")
    readback_ok = all(abs(d["mass_ratio"] - d["params"]["cloth_mass_scale"]) < 1e-3
                      and abs(d["friction_ratio"] - d["params"]["table_friction_scale"]) < 1e-3
                      and abs(d["damping_ratio"] - d["params"]["cloth_damping_scale"]) < 1e-3 for d in (s5a, s5b, s6))
    report["checks"] = {
        "arm_reaches_90pct_in_one_step": all(v >= 0.9 for v in track.values()),
        "cloth_centred": bool(np.all(np.abs(c0) < 0.003)),
        "cloth_offset_applied": bool(np.allclose(c1 - c0, [0.02, -0.03], atol=0.003)),
        "dr_deterministic_per_seed": s5a["params"] == s5b["params"],
        "dr_differs_across_seeds": s5a["params"] != s6["params"],
        "dr_in_range": all(DR_RANGE[0] <= d["params"][k] <= DR_RANGE[1] for d in (s5a, s6) for k in keys),
        "dr_read_back": readback_ok,
        "dr_off_restores": restored["params"] == {} and abs(restored["mass_ratio"] - 1.0) < 1e-6
                           and abs(restored["friction"] - lab._rest_material[0]) < 1e-6
                           and abs(restored["damping"] - lab._rest_material[1]) < 1e-6,
        "damping_takes_effect": abs(fall[0.0] - fall[40.0]) > 1e-3,
    }
    report["passed"] = all(report["checks"].values())
    (out / "profile.json").write_text(json.dumps(report, indent=1))
    print("PROFILE", json.dumps(report), flush=True)


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

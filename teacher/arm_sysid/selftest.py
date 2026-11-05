"""Sanity tests for every models.py variant -> models_selftest.json.

1. Gravity hold: reset at a pose, command the same pose, simulate 2 s, report
   sag = q(2 s) - q_cmd per joint [deg] and the gravity torque at the pose.
2. Step: from the zero pose (held 0.5 s), step shoulder_lift or elbow_flex by
   +-30 deg, simulate 1.5 s. Response y(t) = q(t) - q(t_step). Rise time is
   10%->90% of the final response y(end); overshoot = (max over-travel past
   y(end)) / |y(end)|; steady-state error = q(end) - q_cmd.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models  # noqa: E402

DT = 0.002
POSES_DEG = {
    "zero": [0, 0, 0, 0, 0, 0],
    "extended_forward": [0, 90, -90, 0, 0, 0],  # arm horizontal, max shoulder_lift load
    "mixed": [20, -30, 40, 60, 30, 20],
    "folded": [0, -100, 95, 70, 0, 0],
}
STEP_JOINTS = ["shoulder_lift", "elbow_flex"]
STEP_DEG = 30.0


def gravity_torque(sim, q):
    m, d = sim.model, mujoco.MjData(sim.model)
    d.qpos[sim.qadr] = q
    mujoco.mj_forward(m, d)
    return d.qfrc_bias[sim.vadr].copy()


def hold_test(sim, q_deg):
    q = np.radians(q_deg)
    sim.reset(q)
    sim.set_target(q)
    sim.advance(2.0)
    qe = sim.qpos()
    return {
        "pose_deg": q_deg,
        "gravity_torque_Nm": np.round(gravity_torque(sim, q), 4).tolist(),
        "sag_deg": np.round(np.degrees(qe - q), 4).tolist(),
        "residual_speed_deg_s": np.round(np.degrees(sim.qvel()), 4).tolist(),
    }


def step_metrics(t, y, y_cmd):
    y_end = y[-1]
    a = abs(y_end)
    s = np.sign(y_end) if y_end != 0 else 1.0
    yn = s * y  # positive-going response
    def first_cross(level):
        idx = np.nonzero(yn >= level)[0]
        return float(t[idx[0]]) if len(idx) else None
    t10, t90 = first_cross(0.1 * a), first_cross(0.9 * a)
    over = float(max(0.0, yn.max() - a) / a * 100.0) if a > 0 else None
    # 2% settling around the final value
    band = 0.02 * a
    out = np.nonzero(np.abs(yn - a) > band)[0]
    t_settle = float(t[out[-1] + 1]) if len(out) and out[-1] + 1 < len(t) else (0.0 if not len(out) else None)
    r = lambda x, n: None if x is None else round(float(x), n)
    return {
        "final_response_deg": r(y_end, 3),
        "commanded_deg": r(y_cmd, 3),
        "steady_state_error_deg": r(y_end - y_cmd, 3),
        "t_10pct_ms": r(None if t10 is None else 1000 * t10, 1),
        "rise_time_10_90_ms": r(None if (t10 is None or t90 is None) else 1000 * (t90 - t10), 1),
        "overshoot_pct": r(over, 2),
        "settle_2pct_ms": r(None if t_settle is None else 1000 * t_settle, 1),
        "peak_speed_deg_s": None,
    }


def step_test(sim, joint, sign):
    j = models.JOINT_NAMES.index(joint)
    q0 = np.zeros(6)
    sim.reset(q0)
    sim.set_target(q0)
    sim.advance(0.5)
    q_start = sim.qpos()[j]
    q_cmd = q0.copy()
    q_cmd[j] += sign * np.radians(STEP_DEG)
    sim.set_target(q_cmd)
    n = int(round(1.5 / DT))
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    v = np.zeros(n + 1)
    for k in range(1, n + 1):
        sim.step(1)
        t[k] = k * DT
        y[k] = sim.qpos()[j] - q_start
        v[k] = sim.qvel()[j]
    met = step_metrics(t, np.degrees(y), np.degrees(q_cmd[j] - q_start))
    met["peak_speed_deg_s"] = round(float(np.degrees(np.abs(v).max())), 1)
    met["start_sag_deg"] = round(float(np.degrees(q_start - q0[j])), 3)
    met["end_error_vs_command_deg"] = round(float(np.degrees(sim.qpos()[j] - q_cmd[j])), 3)
    return met


def main():
    out = {
        "dt": DT,
        "notes": {
            "geometry": "Menagerie robotstudio_so101 (commit 62ab00f), TRS CAD masses, total 0.6455 kg incl. camera mount",
            "solver": "implicitfast, contacts disabled, gravity on, noslip_iterations=1, same for all variants",
            "bam": "C/D: STS3215 7.4 V params, firmware kp 16, vin 7.4 V, command_delay + max_velocity rate limit on",
        },
        "variants": {},
    }
    for v in models.VARIANTS:
        sim = models.make_sim(v, dt=DT)
        res = {"param_table": sim.param_table}
        if getattr(sim, "bam_params", None):
            res["bam_params"] = sim.bam_params
        t0 = time.time()
        res["hold_2s"] = {name: hold_test(sim, q) for name, q in POSES_DEG.items()}
        res["step_30deg_from_zero"] = {
            f"{jn}_{'+' if s > 0 else '-'}{int(STEP_DEG)}": step_test(sim, jn, s)
            for jn in STEP_JOINTS
            for s in (+1, -1)
        }
        # speed: wall seconds per simulated second
        sim.reset(np.zeros(6))
        tw = time.time()
        sim.advance(2.0)
        res["wall_s_per_sim_s"] = (time.time() - tw) / 2.0
        out["variants"][v] = res
        print(f"== {v}  ({time.time()-t0:.1f} s)")
        for name, h in res["hold_2s"].items():
            print(f"   hold {name:17s} sag deg {h['sag_deg']}")
        for name, sm in res["step_30deg_from_zero"].items():
            print(
                f"   step {name:18s} rise {sm['rise_time_10_90_ms']} ms, overshoot {sm['overshoot_pct']:.2f} %,"
                f" t10 {sm['t_10pct_ms']} ms, ss_err {sm['steady_state_error_deg']:.3f} deg, peak {sm['peak_speed_deg_s']:.0f} deg/s"
            )
    vj = os.path.join(HERE, "validate_bam_actuator.json")
    if os.path.exists(vj):
        with open(vj) as f:
            val = json.load(f)
        out["bam_actuator_validation"] = {
            m: {k: {"mean_mae_deg": r["mean_mae_deg"], "ratio_to_bam_reference_same_dt": r.get("ratio_to_bam_reference_same_dt"),
                    "ratio_to_bam_reference_5ms": r.get("ratio_to_bam_reference_5ms")}
                for k, r in val[m].items()}
            for m in ("m1", "m6")
        }
        out["bam_actuator_validation"]["logdir"] = val["logdir"]
        out["bam_actuator_validation"]["n_logs"] = val["n_logs"]
    path = os.path.join(HERE, "models_selftest.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()

"""Replay the 8 held-out Anchor-Lab motions on the SO-101 inside Isaac (LeHome's robot USD, LeHome's dt 1/90, CPU
physics), with the same protocol and score as anchor_fit.eval_item for whole motions (start from the measured state,
box-filtered command targets, 0.5 s warm-up, per-joint RMSE in degrees, arm = mean over the 5 arm joints).
Two robots in one scene:
  lehome  LeHome's actuators as shipped (stiffness 17.8, damping 0.6, effort 10, arm gravity disabled)
  fitted  our fit ported (oracle/so101_mj_actuator.py) + gravity on + Menagerie masses (+23.1 g camera)
Reference: the same motions in MuJoCo with the fit = 0.284 deg arm mean; NVIDIA params, our sim = 0.385.

    .venv/bin/python isaac_arm_replay.py --data <dir with heldout-*.npz> --out <json> [--vf 0.15]
"""
import argparse
import glob
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--vf", type=float, nargs="+", default=[0.15])
ap.add_argument("--dt", type=float, default=1.0 / 90.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
import so101_mj_actuator as MJ  # noqa: E402

JOINTS = MJ.JOINTS
WARM, VEL_HALF, CAM_KG = 0.5, 0.02, 0.0231
MENAGERIE = json.load(open(os.path.join(HERE, "results", "menagerie_inertials.json")))
BODY_MAP = {"base": "base", "shoulder": "shoulder", "upper_arm": "upper_arm", "lower_arm": "lower_arm",
            "wrist": "wrist", "gripper": "gripper", "jaw": "moving_jaw_so101_v1"}


def event_targets(t_cmd, qc, t_start, nsteps, dt):
    """anchor_fit.event_targets with delay 0: box-filtered per-step targets of the command held until the next one."""
    b = t_cmd
    seg = np.diff(b)
    icum = np.concatenate([np.zeros((1, qc.shape[1])), np.cumsum(qc[:-1] * seg[:, None], axis=0)])
    e = t_start + np.arange(nsteps + 1) * dt
    k = np.clip(np.searchsorted(b, e, side="right") - 1, 0, len(b) - 1)
    integ = icum[k] + qc[k] * (e - b[k])[:, None]
    return np.diff(integ, axis=0) / dt


def start_state(d, t_start):
    i0 = min(max(int(np.searchsorted(d["t_state"], t_start)), 0), len(d["t_state"]) - 1)
    t0 = float(d["t_state"][i0])
    sel = np.abs(d["t_state"] - t0) <= VEL_HALF
    tt, qq = d["t_state"][sel] - t0, d["q"][sel]
    v0 = np.linalg.lstsq(np.stack([tt, np.ones_like(tt)], 1), qq, rcond=None)[0][0] if len(tt) >= 3 else np.zeros(6)
    return t0, d["q"][i0].copy(), v0


def make_robot(name, x, fitted, vf):
    cfg = SO101_FOLLOWER_CFG.replace(prim_path=f"/World/{name}")
    cfg.init_state.pos = (x, 0.0, 0.0)
    if fitted:
        cfg.actuators = MJ.fitted_actuators(v_friction=vf)
        cfg.spawn.rigid_props.disable_gravity = False
    return Articulation(cfg)


def port_masses(robot):
    view = robot.root_physx_view
    m = view.get_masses().clone()
    before = {}
    for i, b in enumerate(robot.body_names):
        key = next((v for k, v in BODY_MAP.items() if b == k or b.startswith(k)), None)
        before[b] = round(float(m[0, i]), 5)
        if key and key in MENAGERIE:
            m[:, i] = MENAGERIE[key]["mass"] + (CAM_KG if key == "gripper" else 0.0)
    view.set_masses(m, torch.arange(m.shape[0]))
    return before, {b: round(float(m[0, i]), 5) for i, b in enumerate(robot.body_names)}


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device="cpu"))
    variants = [("lehome", None)] + [(f"fitted_vf{int(round(vf * 1000))}mrad", vf) for vf in args.vf]
    robots = {n: make_robot(n, 1.0 * i, vf is not None, vf) for i, (n, vf) in enumerate(variants)}
    sim.reset()
    masses = {}
    for n, vf in variants:
        if vf is not None:
            masses[n] = port_masses(robots[n])
    print("MASSES", json.dumps(masses), flush=True)
    order = {n: [r.joint_names.index(j) for j in JOINTS] for n, r in robots.items()}
    results = {n: {} for n, _ in variants}
    for path in sorted(glob.glob(os.path.join(args.data, "heldout-*.npz"))):
        name = os.path.basename(path)[:-4]
        d = dict(np.load(path))
        t_start = float(max(d["t_state"][0], d["t_cmd"][0]))
        t_end = float(d["t_state"][-1])
        t0, q0, v0 = start_state(d, t_start)
        nsteps = int(math.ceil((t_end - t0) / args.dt)) + 1
        U = event_targets(d["t_cmd"], d["qc"], t0, nsteps, args.dt)
        Q = {n: np.empty((nsteps + 1, 6)) for n in robots}
        for n, r in robots.items():
            qi, vi = torch.zeros(1, r.num_joints), torch.zeros(1, r.num_joints)
            qi[0, order[n]] = torch.tensor(q0, dtype=torch.float32)
            vi[0, order[n]] = torch.tensor(v0, dtype=torch.float32)
            r.write_joint_state_to_sim(qi, vi)
            r.set_joint_position_target(qi)
            Q[n][0] = q0
        for k in range(nsteps):
            for n, r in robots.items():
                tgt = torch.zeros(1, r.num_joints)
                tgt[0, order[n]] = torch.tensor(U[k], dtype=torch.float32)
                r.set_joint_position_target(tgt)
                r.write_data_to_sim()
            sim.step(render=False)
            for n, r in robots.items():
                r.update(args.dt)
                Q[n][k + 1] = r.data.joint_pos[0, order[n]].cpu().numpy()
        i_lo = int(np.searchsorted(d["t_state"], t0 + WARM))
        ts, real = d["t_state"][i_lo:], d["q"][i_lo:]
        x = (ts - t0) / args.dt
        n0 = np.clip(np.floor(x).astype(int), 0, nsteps - 1)
        fr = (x - n0)[:, None]
        for n in robots:
            pred = Q[n][n0] * (1 - fr) + Q[n][n0 + 1] * fr
            err = np.nan_to_num(np.degrees(pred - real), nan=1e3)
            rm = np.sqrt((err ** 2).mean(0))
            results[n][name] = {"rmse": [round(float(v), 4) for v in rm], "arm": round(float(rm[:5].mean()), 4),
                                "jaw": round(float(rm[5]), 4)}
        print("MOTION", name, json.dumps({n: results[n][name]["arm"] for n in robots}), flush=True)
    summary = {n: {"arm_mean_deg": round(float(np.mean([v["arm"] for v in res.values()])), 4),
                   "jaw_mean_deg": round(float(np.mean([v["jaw"] for v in res.values()])), 4)} for n, res in results.items()}
    summary["reference"] = {"mujoco_fitted": 0.284, "mujoco_nvidia_params": 0.385}
    print("SUMMARY", json.dumps(summary), flush=True)
    json.dump({"summary": summary, "per_motion": results, "masses": masses, "dt": args.dt}, open(args.out, "w"), indent=1)


main()
# results are written and printed above; Isaac can hang on close (2026-10-18, ~20 min), so exit hard
sys.stdout.flush()
os._exit(0)

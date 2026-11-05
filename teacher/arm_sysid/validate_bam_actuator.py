"""Validate models.BamServoBank (the per-step BAM STS3215 actuator used by the
C_bam_m1 / D_bam_m6 arm variants) against BAM's own evaluation on BAM's raw
single-servo pendulum logs (7.4 V, firmware kp 16).

Compared, per param file (m1, m6), mean over logs of MAE |q_sim - q_log|:
  1. bam_reference : bam.simulate.Simulator (BAM's hand-written integrator), exactly
                     what `python -m bam.mae` computes.
  2. bam_mujoco    : bam.mujoco.Simulator(command_delay=True) (BAM's own MuJoCo
                     backend, = `bam.mae --mujoco`), with a guard for the no-load
                     logs (mass = length = 0) on which BAM's MuJoCo pendulum
                     builder divides by zero.
  3. ours_*        : models.BamServoBank on BAM's MuJoCo pendulum spec, driven
                     exactly like the arm (set_target at log rate, physics substeps).

Usage: python validate_bam_actuator.py [logdir] [out.json]
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

bam, load_model, MujocoController = models._import_bam()
from bam.logs import Logs  # noqa: E402
from bam import simulate as bam_simulate  # noqa: E402
from bam import mujoco as bam_mujoco  # noqa: E402
from bam import testbench_mujoco  # noqa: E402

DEFAULT_LOGDIR = (
    r"C:/Users/valen/AppData/Local/Temp/claude/C--Users-valen-Documents-vla-distill/"
    r"c1e14d5c-c49d-4f52-b957-d99b0501209c/scratchpad/dl74/kp16"
)

# ---------------------------------------------------------------------------
# Guard for no-load logs: BAM's testbench_mujoco.Pendulum.inertial_params does
# com_z = -(m*l + ma*l/2)/(m+ma) -> ZeroDivisionError when m = ma = 0. The
# reference testbench then has zero link inertia and zero gravity torque, so a
# negligible body (1e-9 kg at the pivot, 1e-12 kg m^2) reproduces it: the joint
# inertia is the BAM armature (~0.02 kg m^2).
_orig_inertial = testbench_mujoco.Pendulum.inertial_params


def _guarded_inertial(self):
    if self.mass + self.arm_mass <= 0.0:
        return 1e-9, 0.0, 1e-12
    return _orig_inertial(self)


testbench_mujoco.Pendulum.inertial_params = _guarded_inertial

# Same logs also have length = 0, which makes the visual capsule degenerate
# ("fromto points too close"). With zero mass the length has no physical effect,
# so a 1 mm visual length is used for those logs only.
_orig_init = testbench_mujoco.Pendulum.__init__


def _guarded_init(self, log):
    _orig_init(self, log)
    if self.mass + self.arm_mass <= 0.0 and self.length <= 0.0:
        self.length = 1e-3


testbench_mujoco.Pendulum.__init__ = _guarded_init


def mae(positions, log) -> float:
    ref = np.array([e["position"] for e in log["entries"]])
    return float(np.mean(np.abs(np.asarray(positions, dtype=float).reshape(-1) - ref)))


def rollout_reference(params_json, log):
    model = load_model(params_json)
    sim = bam_simulate.Simulator(model)
    pos, _, _ = sim.rollout_log(log, simulate_control=True)
    return pos


def rollout_bam_mujoco(params_json, log):
    model = load_model(params_json)
    sim = bam_mujoco.Simulator(model, command_delay=True)
    pos, _, _ = sim.rollout_log(log)
    return pos


def upsample(log, n):
    """Same log at dt/n: every goal held (zero-order hold) for n sub-samples."""
    out = {k: v for k, v in log.items() if k != "entries"}
    out["dt"] = log["dt"] / n
    out["entries"] = [e for e in log["entries"] for _ in range(n)]
    return out


def rollout_reference_dt(params_json, log, n):
    pos = rollout_reference(params_json, upsample(log, n))
    return np.asarray(pos)[::n]


def rollout_bam_mujoco_dt(params_json, log, n):
    pos = rollout_bam_mujoco(params_json, upsample(log, n))
    return np.asarray(pos)[::n]


def rollout_ours(params_json, log, phys_dt, integrator):
    """models.BamServoBank on BAM's MuJoCo pendulum, driven like the arm:
    set_target once per log sample, then physics steps until the next sample
    time (running time goal, so phys_dt need not divide the log dt)."""
    with open(params_json) as f:
        q_offset = json.load(f).get("q_offset", 0.0)
    pend = testbench_mujoco.Pendulum(
        {"mass": log["mass"], "arm_mass": log.get("arm_mass", 0.0), "length": log["length"]}
    )
    spec = pend.build_spec("pendulum", q_offset=q_offset)
    spec.option.timestep = phys_dt
    spec.option.integrator = integrator
    m = spec.compile()
    d = mujoco.MjData(m)
    bank = models.BamServoBank(
        m, d, ["pendulum"], params_json, kp_fw=log["kp"], vin=log.get("vin", 7.4), command_delay=True
    )
    e0 = log["entries"][0]
    d.qpos[0] = e0["position"]
    d.qvel[0] = e0.get("speed", 0.0)
    mujoco.mj_forward(m, d)
    bank.reset()
    log_dt = log["dt"]
    pos = []
    t_goal = 0.0
    for e in log["entries"]:
        pos.append(float(d.qpos[0]))
        bank.torque_enable = bool(e["torque_enable"])
        bank.set_target([e["goal_position"]])
        t_goal += log_dt
        for _ in range(int(round((t_goal - d.time) / phys_dt))):
            bank.before_step()
            mujoco.mj_step(m, d)
    return pos


def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOGDIR
    out_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "validate_bam_actuator.json")
    logs = Logs(logdir).logs
    print(f"{len(logs)} logs from {logdir}")
    EUL = mujoco.mjtIntegrator.mjINT_EULER
    IMP = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    configs = {
        # BAM's own code at the identification step (5 ms = log dt)
        "bam_reference": lambda pj, lg: rollout_reference(pj, lg),
        "bam_mujoco": lambda pj, lg: rollout_bam_mujoco(pj, lg),
        # ours, same step and integrator as BAM's MuJoCo backend
        "ours_dt5ms_euler": lambda pj, lg: rollout_ours(pj, lg, 0.005, EUL),
        # BAM's own code and ours at finer steps (log goals held, ZOH)
        "bam_reference_dt2.5ms": lambda pj, lg: rollout_reference_dt(pj, lg, 2),
        "bam_mujoco_dt2.5ms": lambda pj, lg: rollout_bam_mujoco_dt(pj, lg, 2),
        "ours_dt2.5ms_euler": lambda pj, lg: rollout_ours(pj, lg, 0.0025, EUL),
        "bam_reference_dt1ms": lambda pj, lg: rollout_reference_dt(pj, lg, 5),
        "bam_mujoco_dt1ms": lambda pj, lg: rollout_bam_mujoco_dt(pj, lg, 5),
        "ours_dt1ms_euler": lambda pj, lg: rollout_ours(pj, lg, 0.001, EUL),
        # ours exactly as the arm runs it (implicitfast; 2 ms does not divide 5 ms)
        "ours_dt5ms_implicitfast": lambda pj, lg: rollout_ours(pj, lg, 0.005, IMP),
        "ours_dt2ms_implicitfast_ARM_DEFAULT": lambda pj, lg: rollout_ours(pj, lg, 0.002, IMP),
    }
    results = {"logdir": logdir, "n_logs": len(logs), "files": [os.path.basename(l["filename"]) for l in logs]}
    for mname in ("m1", "m6"):
        pj = models.bam_params_path(mname)
        results[mname] = {}
        for cname, fn in configs.items():
            t0 = time.time()
            per = [mae(fn(pj, lg), lg) for lg in logs]
            results[mname][cname] = {
                "mean_mae_rad": float(np.mean(per)),
                "mean_mae_deg": float(np.degrees(np.mean(per))),
                "per_log_rad": per,
                "wall_s": time.time() - t0,
            }
            print(f"  {mname} {cname:32s} MAE = {np.degrees(np.mean(per)):.4f} deg ({np.mean(per)*1000:.2f} mrad)  [{time.time()-t0:.1f} s]")
        for cname in configs:
            per = np.array(results[mname][cname]["per_log_rad"])
            # compare with BAM's reference simulator at the same physics step
            if "dt2.5ms" in cname:
                rname = "bam_reference_dt2.5ms"
            elif "dt1ms" in cname:
                rname = "bam_reference_dt1ms"
            else:
                rname = "bam_reference"
            ref = np.array(results[mname][rname]["per_log_rad"])
            ref5 = np.array(results[mname]["bam_reference"]["per_log_rad"])
            results[mname][cname]["compared_to"] = rname
            results[mname][cname]["ratio_to_bam_reference_same_dt"] = float(per.mean() / ref.mean())
            results[mname][cname]["ratio_to_bam_reference_5ms"] = float(per.mean() / ref5.mean())
            results[mname][cname]["max_abs_perlog_diff_deg_same_dt"] = float(np.degrees(np.max(np.abs(per - ref))))
            print(f"    {cname:38s} ratio vs {rname}: {per.mean()/ref.mean():.4f}   vs 5 ms reference: {per.mean()/ref5.mean():.4f}")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", out_json)


if __name__ == "__main__":
    main()

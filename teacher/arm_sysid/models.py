"""SO-101 MuJoCo models that differ ONLY in motor / joint parameters.

All variants share the MuJoCo Menagerie ``robotstudio_so101`` geometry and the
TheRobotStudio CAD masses/inertias (assets/menagerie_so101/so101.xml, Menagerie
commit 62ab00f). They also share the same solver settings (implicitfast,
contacts disabled for free-space replay, gravity on, 1 noslip iteration as in
BAM's own MuJoCo testbench). Only actuator + joint parameters change:

  A_stock   Menagerie / TheRobotStudio values exactly as in the file:
            <position kp=998.22 kv=2.731 forcerange=+-2.94>, joint damping 0.60,
            frictionloss 0.052, armature 0.028 (same on all 6 joints).
  B_nvidia  NVIDIA Isaac Sim 6.1 so101_new_calib_SysID per-joint values, parsed
            from assets/nvidia_sysid/physics.usda and converted to SI per radian
            (see NVIDIA_CONVERSION below). Position actuator kp/kv + forcerange,
            joint damping / frictionloss / armature.
  C_bam_m1  Feetech STS3215 7.4 V, Rhoban BAM model M1, firmware Kp=16 (LeRobot
            P_Coefficient). Torque (motor) actuator whose torque is computed every
            physics step by BAM's own code (bam.mujoco.MujocoController ->
            STS3215Actuator.compute_control/compute_torque, Model.compute_frictions),
            plus BAM's command_delay applied to the incoming targets.
  D_bam_m6  Same with BAM model M6 (Stribeck + load-dependent directional
            quadratic friction), which BAM's MujocoController supports per step.

Usage::

    from models import make_sim
    sim = make_sim("C_bam_m1", dt=0.002)
    sim.reset(q0)                  # 6 joint angles, rad, sim joint convention
    for q_cmd in commands:         # at the recording rate (e.g. 30 Hz)
        sim.set_target(q_cmd)
        sim.advance(1 / 30)
        q = sim.qpos()

Angles are in the MuJoCo model's joint convention (radians). Mapping LeRobot
"*.pos" values (degrees / 0..100 gripper) to these joints is the caller's job.
"""

from __future__ import annotations

import collections
import math
import os
import re
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE_XML = os.path.join(HERE, "assets", "menagerie_so101", "so101.xml")
NVIDIA_USDA = os.path.join(HERE, "assets", "nvidia_sysid", "physics.usda")

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

RAD_PER_DEG = math.pi / 180.0
DEG_PER_RAD = 180.0 / math.pi

VARIANTS = ("A_stock", "B_nvidia", "C_bam_m1", "D_bam_m6")

# --------------------------------------------------------------------------
# BAM import (published code, reused, not re-derived)
# --------------------------------------------------------------------------
_BAM_CANDIDATES = [
    os.environ.get("BAM_SRC", ""),
    # session scratchpad clone of github.com/Rhoban/bam main e9a619d
    r"C:/Users/valen/AppData/Local/Temp/claude/C--Users-valen-Documents-vla-distill/"
    r"c1e14d5c-c49d-4f52-b957-d99b0501209c/scratchpad/src/bam",
    # unmodified copy of the same commit, kept next to this file
    os.path.join(HERE, "vendor", "bam_e9a619d"),
]


def _import_bam():
    for cand in _BAM_CANDIDATES:
        if cand and os.path.isfile(os.path.join(cand, "bam", "mujoco.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            break
    import bam  # noqa: F401
    from bam.model import load_model
    from bam.mujoco import MujocoController

    return bam, load_model, MujocoController


def bam_params_path(model_name: str, motor_dir: str = "feetech_sts3215_7_4V") -> str:
    bam, _, _ = _import_bam()
    return os.path.join(os.path.dirname(bam.__file__), "params", motor_dir, f"{model_name}.json")


# --------------------------------------------------------------------------
# NVIDIA SysID USD parsing + unit conversion
# --------------------------------------------------------------------------
NVIDIA_CONVERSION = {
    "drive:angular:physics:stiffness": "N*m/deg (UsdPhysics DriveAPI doc) -> x 180/pi -> actuator kp [N*m/rad]",
    "drive:angular:physics:damping": "N*m*s/deg (UsdPhysics DriveAPI doc) -> x 180/pi -> actuator kv [N*m*s/rad]",
    "drive:angular:physics:maxForce": "N*m (torque units, not per-degree) -> actuator forcerange +-maxForce",
    "newton:damping": "effort*s/deg (NewtonJointAPI doc; Newton importer multiplies by 180/pi) -> joint damping [N*m*s/rad]",
    "newton:friction": "effort = N*m (direct) -> joint frictionloss",
    "newton:armature": "kg*m^2 (direct) -> joint armature",
    "physics:lowerLimit/upperLimit": "deg -> rad (identical to the Menagerie ranges, so Menagerie ranges are kept)",
}


def load_nvidia_sysid(path: str = NVIDIA_USDA) -> dict:
    """Parse the per-joint drive / Newton attributes from NVIDIA's physics.usda.

    Plain-text parse of ``def PhysicsRevoluteJoint "<name>" { ... }`` blocks so
    the numbers come straight from the downloaded file (no pxr dependency).
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {}
    for m in re.finditer(r'def PhysicsRevoluteJoint "(\w+)"\s*\([^)]*\)\s*\{(.*?)\n        \}', text, re.S):
        name, body = m.group(1), m.group(2)
        vals = {}
        for key in (
            "drive:angular:physics:damping",
            "drive:angular:physics:maxForce",
            "drive:angular:physics:stiffness",
            "newton:armature",
            "newton:damping",
            "newton:friction",
            "newton:velocityLimit",
            "physics:lowerLimit",
            "physics:upperLimit",
        ):
            mm = re.search(r"float " + re.escape(key) + r" = ([-0-9.eE+]+)", body)
            vals[key] = float(mm.group(1)) if mm else None
        out[name] = vals
    missing = [j for j in JOINT_NAMES if j not in out]
    if missing:
        raise RuntimeError(f"joints {missing} not found in {path}")
    return out


def nvidia_si_params(path: str = NVIDIA_USDA) -> dict:
    raw = load_nvidia_sysid(path)
    si = {}
    for j in JOINT_NAMES:
        r = raw[j]
        si[j] = {
            "kp": r["drive:angular:physics:stiffness"] * DEG_PER_RAD,
            "kv": r["drive:angular:physics:damping"] * DEG_PER_RAD,
            "forcerange": r["drive:angular:physics:maxForce"],
            "damping": r["newton:damping"] * DEG_PER_RAD,
            "frictionloss": r["newton:friction"],
            "armature": r["newton:armature"],
            "range_rad": (r["physics:lowerLimit"] * RAD_PER_DEG, r["physics:upperLimit"] * RAD_PER_DEG),
        }
    return si


# --------------------------------------------------------------------------
# Delayed target buffer (BAM command_delay)
# --------------------------------------------------------------------------
class DelayedTarget:
    """BAM's command delay, applied online at the physics rate.

    The commanded target is held (zero-order hold) between set_target() calls and
    sampled once per physics step into u[n]. The value fed to the firmware model
    at step n is BAM's fractional delay shift of that sequence:

        d = delay/dt, d0 = floor(d), frac = d - d0
        u_delayed[n] = (1-frac)*u[n-d0] + frac*u[n-d0-1]   (indices < 0 -> u[0])

    which is exactly ``bam.simulate.fractional_delay_shift`` (the function BAM's
    reference and MuJoCo simulators use) applied to the physics-rate sequence.
    """

    def __init__(self, delay: float, dt: float):
        self.delay = float(delay)
        self.dt = float(dt)
        d = self.delay / self.dt
        self.d0 = int(math.floor(d))
        self.frac = d - self.d0
        self.buf: collections.deque = collections.deque(maxlen=self.d0 + 2)
        self.cmd = None

    def reset(self, q) -> None:
        self.cmd = np.array(q, dtype=float)
        self.buf.clear()

    def push(self, q) -> None:
        self.cmd = np.array(q, dtype=float)

    def next(self) -> np.ndarray:
        """Sample the held command for this physics step; return the delayed value."""
        self.buf.append(self.cmd.copy())
        n = len(self.buf) - 1  # index of the newest sample in buf
        i0 = max(n - self.d0, 0)
        i1 = max(n - self.d0 - 1, 0)
        return (1.0 - self.frac) * self.buf[i0] + self.frac * self.buf[i1]


# --------------------------------------------------------------------------
# BAM servo bank: BAM per-step torque for a set of MuJoCo joints
# --------------------------------------------------------------------------
class BamServoBank:
    """Drives MuJoCo joints with BAM's STS3215 actuator model, every physics step.

    The torque law and friction model are BAM's own code, called through
    ``bam.mujoco.MujocoController.update()``:

    * STS3215Actuator.compute_control: internal target rate-limited at
      ``max_velocity`` (stateful), duty = clip((q_smooth - q)*kp*0.166*error_gain_ratio,
      +-0.97), volts = vin*duty
    * VoltageControlledActuator.compute_torque: tau = kt*V/R - kt^2*dq/R
    * Model.compute_frictions -> written into dof_frictionloss / dof_damping
      (friction_viscous); MuJoCo's solver applies the friction.
    * armature written into dof_armature.

    BAM applies ``command_delay`` outside the controller (it shifts the goal
    sequence in bam.simulate / bam.mujoco.Simulator); here it is applied by
    :class:`DelayedTarget` (BAM's fractional shift, online, at the physics rate)
    to the targets passed to :meth:`set_target`.
    The joints' MuJoCo actuators must be ``motor`` actuators named like the joints.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        joint_names,
        params_json: str,
        kp_fw: float = 16.0,
        vin: float = 7.4,
        command_delay: bool = True,
    ):
        _, load_model, MujocoController = _import_bam()
        self.m, self.d = mj_model, mj_data
        self.joint_names = list(joint_names)
        self.params_json = params_json
        self.bam_model = load_model(params_json)
        self.bam_model.actuator.kp = float(kp_fw)
        self.bam_model.actuator.vin = float(vin)
        # MujocoController sets dof_armature (and mj_setConst) for these joints
        self.controller = MujocoController(self.bam_model, self.joint_names, mj_model, mj_data)
        delay = self.bam_model.command_delay.value if command_delay else 0.0
        self.target = DelayedTarget(delay, mj_model.opt.timestep)
        self.torque_enable = True
        self.target.reset(mj_data.qpos[self.controller.qpos_indexes])

    # parameters actually in use (for reporting)
    def param_summary(self) -> dict:
        bm, act = self.bam_model, self.bam_model.actuator
        kt, R = bm.kt.value, bm.R.value
        p = {k: v.value for k, v in bm.get_parameters().items()}
        p.update(
            {
                "model": bm.name,
                "firmware_kp": act.kp,
                "vin": act.vin,
                "error_gain": act.error_gain,
                "max_pwm": act.max_pwm,
                "equiv_position_kp_Nm_per_rad": act.error_gain * bm.error_gain_ratio.value * act.kp * act.vin * kt / R,
                "stall_torque_Nm(max_pwm*vin*kt/R)": act.max_pwm * act.vin * kt / R,
                "backemf_damping_Nms_per_rad(kt^2/R)": kt * kt / R,
                "params_json": self.params_json,
            }
        )
        p.pop("q_offset", None)  # pendulum-testbench calibration, not used on a robot
        return p

    def reset(self) -> None:
        """Call after the MuJoCo state was reset (qpos set, mj_forward done)."""
        self.controller.reset()
        self.target.reset(self.d.qpos[self.controller.qpos_indexes])

    def set_target(self, q_target) -> None:
        self.target.push(np.asarray(q_target, dtype=float))

    def before_step(self) -> None:
        """Call exactly once before every mj_step (the firmware model is stateful)."""
        self.controller.q_target = self.target.next()
        self.controller.update()
        if not self.torque_enable:
            self.d.ctrl[self.controller.act_indexes] = 0.0


# --------------------------------------------------------------------------
# Arm simulation
# --------------------------------------------------------------------------
def _hinge_damping(value: float) -> np.ndarray:
    # MjsJoint.damping is a (3, 1) array in MuJoCo 3.14; a hinge uses entry 0
    d = np.zeros((3, 1))
    d[0, 0] = float(value)
    return d


class SO101Sim:
    joint_names = JOINT_NAMES

    def __init__(
        self,
        variant: str,
        dt: float = 0.002,
        vin: float = 7.4,
        kp_fw: float = 16.0,
        contacts: bool = False,
        noslip_iterations: int = 1,
        xml_path: str = MENAGERIE_XML,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        self.variant = variant
        self.dt = float(dt)
        spec = mujoco.MjSpec.from_file(xml_path)
        spec.option.timestep = self.dt
        spec.option.noslip_iterations = int(noslip_iterations)
        if not contacts:
            spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        joints = {j.name: j for j in spec.joints}
        acts = {a.name: a for a in spec.actuators}

        self.param_table: dict = {}
        if variant == "A_stock":
            pass  # file values untouched
        elif variant == "B_nvidia":
            si = nvidia_si_params()
            for jn in JOINT_NAMES:
                p = si[jn]
                j, a = joints[jn], acts[jn]
                j.damping = _hinge_damping(p["damping"])
                j.frictionloss = p["frictionloss"]
                j.armature = p["armature"]
                g = np.zeros(10)
                g[0] = p["kp"]
                b = np.zeros(10)
                b[1], b[2] = -p["kp"], -p["kv"]
                a.gainprm = g
                a.biasprm = b
                a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                a.forcerange = [-p["forcerange"], p["forcerange"]]
        else:  # BAM variants: position actuators -> motor (torque) actuators
            for jn in JOINT_NAMES:
                a = acts[jn]
                a.set_to_motor()
                a.gear = [1.0, 0, 0, 0, 0, 0]
                a.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
                a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
                j = joints[jn]
                j.damping = _hinge_damping(0.0)  # rewritten every step by BAM
                j.frictionloss = 0.0  # rewritten every step by BAM

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINT_NAMES])
        self.vadr = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in JOINT_NAMES])
        self.aid = np.array([self.model.actuator(n).id for n in JOINT_NAMES])

        self.bank = None
        if variant in ("C_bam_m1", "D_bam_m6"):
            name = "m1" if variant == "C_bam_m1" else "m6"
            self.bank = BamServoBank(
                self.model,
                self.data,
                JOINT_NAMES,
                bam_params_path(name),
                kp_fw=kp_fw,
                vin=vin,
                command_delay=True,
            )
        self._build_param_table()
        self.reset(np.zeros(6))

    # ---------------------------------------------------------------- params
    def _build_param_table(self) -> None:
        m = self.model
        tab = {}
        for k, jn in enumerate(JOINT_NAMES):
            jid, did, aid = m.joint(jn).id, self.vadr[k], self.aid[k]
            row = {
                "range_rad": [float(x) for x in m.jnt_range[jid]],
                "armature": float(m.dof_armature[did]),
            }
            if self.bank is None:
                row.update(
                    {
                        "actuator": "position",
                        "kp_Nm_per_rad": float(m.actuator_gainprm[aid, 0]),
                        "kv_Nms_per_rad": float(-m.actuator_biasprm[aid, 2]),
                        "forcerange_Nm": [float(x) for x in m.actuator_forcerange[aid]],
                        "joint_damping_Nms_per_rad": float(m.dof_damping[did]),
                        "frictionloss_Nm": float(m.dof_frictionloss[did]),
                        "command_delay_s": 0.0,
                        "target_rate_limit_rad_s": None,
                    }
                )
            else:
                bp = self.bank.param_summary()
                row.update(
                    {
                        "actuator": "motor (BAM per-step torque)",
                        "equiv_kp_Nm_per_rad": bp["equiv_position_kp_Nm_per_rad"],
                        "stall_torque_Nm": bp["stall_torque_Nm(max_pwm*vin*kt/R)"],
                        "backemf_damping_Nms_per_rad": bp["backemf_damping_Nms_per_rad(kt^2/R)"],
                        "joint_damping_Nms_per_rad(friction_viscous)": bp["friction_viscous"],
                        "frictionloss_Nm(base; M6 adds load/Stribeck terms per step)": bp["friction_base"],
                        "command_delay_s": bp["command_delay"],
                        "target_rate_limit_rad_s": bp["max_velocity"],
                    }
                )
            tab[jn] = row
        self.param_table = tab
        if self.bank is not None:
            self.bam_params = self.bank.param_summary()

    # ------------------------------------------------------------ interface
    def reset(self, q_rad_6) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q = np.asarray(q_rad_6, dtype=float)
        self.data.qpos[self.qadr] = q
        self.data.qvel[:] = 0.0
        if self.bank is None:
            self.data.ctrl[self.aid] = q
        mujoco.mj_forward(self.model, self.data)
        if self.bank is not None:
            self.bank.reset()
        self._t_goal = float(self.data.time)

    def set_target(self, q_target_rad_6) -> None:
        q = np.asarray(q_target_rad_6, dtype=float)
        if self.bank is None:
            self.data.ctrl[self.aid] = q  # MuJoCo clamps to ctrlrange
        else:
            self.bank.set_target(q)

    def set_torque_enable(self, on: bool) -> None:
        if self.bank is None:
            raise NotImplementedError("torque disable only modelled for BAM variants")
        self.bank.torque_enable = bool(on)

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            if self.bank is not None:
                self.bank.before_step()
            mujoco.mj_step(self.model, self.data)

    def advance(self, seconds: float) -> None:
        """Advance by ``seconds`` of sim time. A running time goal is kept so a
        recording period that is not a multiple of dt (e.g. 1/30 s at dt=2 ms)
        does not drift: each call steps to the physics step nearest the goal."""
        self._t_goal += float(seconds)
        n = int(round((self._t_goal - self.data.time) / self.dt))
        self.step(max(n, 0))

    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def qvel(self) -> np.ndarray:
        return self.data.qvel[self.vadr].copy()

    @property
    def time(self) -> float:
        return float(self.data.time)


def make_sim(variant: str, dt: float = 0.002, **kwargs) -> SO101Sim:
    """Build one SO-101 variant. kwargs: vin (BAM, default 7.4), kp_fw (BAM,
    default 16 = LeRobot P_Coefficient), contacts (default False),
    noslip_iterations (default 1)."""
    return SO101Sim(variant, dt=dt, **kwargs)


if __name__ == "__main__":
    import json

    for v in VARIANTS:
        s = make_sim(v)
        print(v, json.dumps(s.param_table["shoulder_lift"], indent=None))

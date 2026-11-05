"""Newton replay harness for the SO-101 arm: does Newton 1.6.1 reproduce the MuJoCo replay?

Purpose
-------
The fitted arm (so101_fitted.json, written by the fit step) must later run inside Newton, where the
cloth is simulated. This script loads the Menagerie SO-101 into Newton, applies per-joint parameters
from a JSON file, replays the same TEST windows as replay.py (same window list, same loading code,
same 30 Hz zero-order-hold command timing) and compares, frame by frame,

    Newton (SolverMuJoCo)  vs  a models.py-style MuJoCo sim built from the SAME JSON  vs  the real arm.

It must run in the isolated Newton venv (Newton 1.6.1 pins mujoco~=3.12, the main env has mujoco 3.14,
and installing Newton there would change the baseline):

    <scratchpad>/newton_venv/Scripts/python.exe newton_check.py               # standard suite
    <scratchpad>/newton_venv/Scripts/python.exe newton_check.py --quick       # 1 window/dataset smoke test
    <scratchpad>/newton_venv/Scripts/python.exe newton_check.py --write-default-json   # params/b_nvidia.json

Writes results/newton_check.json and results/newton_check.md. No file used by replay.py / models.py is
modified (models.py and replay.py are imported read-only).

Parameter JSON schema ASSUMED (the fit agent's so101_fitted.json does not exist yet; until it does, the
equivalent JSON for B_nvidia is generated from models.nvidia_si_params() into params/b_nvidia.json).
The loader is tolerant: every key is optional, aliases are accepted, and a missing value keeps the
Menagerie MJCF value (A_stock) and is listed in the report under "defaulted".

    {
      "schema": "so101_params/v1",                 # optional
      "variant": "B_nvidia",                       # free text
      "actuator": "position",                      # only position-servo models are supported here
      "sim": {"dt": 0.002, "noslip_iterations": 1},          # optional overrides
      "joints": {                                  # or the joint dicts directly at top level
        "<joint>": {
          "kp": N*m/rad          (alias ke, stiffness)
          "kv": N*m*s/rad        (alias kd)          actuator velocity gain (MuJoCo position kv)
          "damping": N*m*s/rad                       joint (passive) damping
          "frictionloss": N*m    (alias friction)
          "armature": kg*m^2
          "forcerange": N*m      scalar f -> [-f, f], or a [lo, hi] pair (alias effort_limit)
          "delay_s": s           (alias delay)       per-joint transport delay of the command
          "deadband_rad": rad    (alias deadband)    optional servo dead zone (see below)
          "range_rad": [lo, hi]                      optional joint (hard-stop) range
          "ctrlrange_rad": [lo, hi]                  optional command clamp; defaults to range_rad
        }, ...
      },
      "per_dataset": {                              # optional, keyed by dataset short name
        "<dataset>": {"delta_s": s,                 # added to every joint's delay_s
                      "time_scale": 1.0,            # real frame period = time_scale / 30 s
                      "range_rad": {"<joint>": [lo, hi]}, "ctrlrange_rad": {...},
                      "state_offset_rad": {"<joint>": rad},    # sim = lerobot_rad + offset
                      "action_offset_rad": {"<joint>": rad}}
      }
    }

Semantics used for the optional parameters (both sims get IDENTICAL per-physics-step targets, so the
Newton-vs-MuJoCo comparison does not depend on these choices):
  * delay: command k (sent at frame k) becomes the servo target at time k*T + delay, switching at the
    physics step nearest that time (round((k*T + delay)/dt)); at delay 0 this is exactly replay.py's
    SO101Sim.advance() step pattern (17/17/16 steps per 1/30 s at dt = 2 ms).
  * deadband: target_eff = q + dz(target - q) with dz(e) = sign(e)*max(|e| - db, 0), recomputed every
    physics step from that sim's own q (a shifted dead zone: zero P-torque inside +-db).
  * range: joint limit and, unless ctrlrange_rad is given, the actuator ctrlrange (Menagerie has
    ctrlrange == range). limits='wide' switches both off for the 5 arm joints (as replay.py 'wide').

How the JSON reaches Newton (Newton 1.6.1 API, VERIFIED by reading the installed source):
  ModelBuilder.add_mjcf(so101.xml)  -> the 6 <position> actuators become JOINT_TARGET drives
  builder.joint_target_ke / joint_target_kd   <- kp / kv   (-> MuJoCo gainprm[0] / biasprm[1:3])
  builder.joint_damping / joint_friction / joint_armature <- damping / frictionloss / armature
  builder.joint_effort_limit <- forcerange (-> MuJoCo jnt_actfrcrange) AND the importer's per-actuator
      custom attribute mujoco:actuator_forcerange (otherwise the MJCF's +-2.94 stays as a 2nd clamp)
  builder.joint_limit_lower/upper + mujoco:actuator_ctrlrange / ctrllimited <- ranges
  control.joint_target_q <- the per-step target
  SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True); then mj_model.opt.noslip_iterations
      is set directly (Newton has no noslip option; mujoco_warp raises NotImplementedError for noslip>0).

Schema v2 (anchor_fit.fitted_json; "schema": "so101_params/v2") adds, and both sims honour:
  sim{dt, integrator, noslip_iterations, iterations, ls_iterations, solimpfriction (5-list|null),
      solreffriction (2-list|null), friction_setting}, top-level delay_s, bodies.camera_mount{mass, ipos, inertia},
  per-joint range_rad (gripper upper ~2.094 rad); actuator ctrl limits OFF (as anchor_fit.apply_opts does).
  Newton route (VERIFIED in newton/_src/solvers/mujoco/solver_mujoco.py, Newton 1.6.1):
    solimpfriction / solreffriction -> builder custom attributes mujoco:solimpfriction (vec5) / mujoco:solreffriction
        (vec2), frequency JOINT_DOF (register_custom_attributes, lines 1239-1260) -> MjSpec joint solimp_friction /
        solref_friction (_convert_to_mjc, lines 5923-5924, 6911-6914, 7023-7026) -> mj_model.dof_solimp / dof_solref;
        runtime updates via update_dof_properties_kernel (lines 8337-8364) into mjw_model.dof_solimp/dof_solref.
    iterations / ls_iterations / integrator -> SolverMuJoCo(...) constructor arguments (lines 5766-5814).
    camera_mount -> builder.body_mass/body_com/body_inertia of the Newton body whose label ends in 'camera_mount'
        (add_mjcf keeps it as its own body on a fixed joint: collapse_fixed_joints defaults to False).
    noslip_iterations -> no Newton option at all; set on solver.mj_model.opt for the MuJoCo-C backend only.

Anchor mode (--anchor): replays whole NVIDIA Anchor-Lab motions (anchor_data.py) with the per-step targets of
anchor_fit.event_targets (JSON delay_s, sim.dt), start state anchor_fit.start_state, scored at the measured state
times after 0.5 s by linear interpolation, exactly as anchor_fit.eval_item. The MuJoCo reference is
anchor_fit.get_pos_sim(opts-from-JSON, params) (the fit's own model); Newton's MjModel is asserted equal to it.
Writes results/newton_check_anchor.json/.md.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import models  # noqa: E402  (read-only: nvidia_si_params, MENAGERIE_XML, _hinge_damping, make_sim)

MJCF = models.MENAGERIE_XML
JOINTS = list(models.JOINT_NAMES)
ARM = 5
FPS = 30
RESULTS = os.path.join(HERE, "results")
PARAMS_DIR = os.path.join(HERE, "params")
FITTED_JSON = os.path.join(HERE, "so101_fitted.json")
DEFAULT_JSON = os.path.join(PARAMS_DIR, "b_nvidia.json")
PRIMARY = ("dobri420", "HashtagRobotics", "DecisionFacts", "Chaenn", "dharak22", "vasco281204")
PROVENANCE_VERIFIED = ("dobri420", "HashtagRobotics", "DecisionFacts")

_ALIASES = {
    "kp": ("kp", "ke", "stiffness", "kp_Nm_per_rad"),
    "kv": ("kv", "kd", "kv_Nms_per_rad"),
    "damping": ("damping", "joint_damping", "joint_damping_Nms_per_rad"),
    "frictionloss": ("frictionloss", "friction", "frictionloss_Nm"),
    "armature": ("armature",),
    "forcerange": ("forcerange", "effort_limit", "forcerange_Nm"),
    "delay_s": ("delay_s", "delay", "command_delay_s"),
    "deadband_rad": ("deadband_rad", "deadband"),
    "range_rad": ("range_rad", "range"),
    "ctrlrange_rad": ("ctrlrange_rad", "ctrlrange"),
}


# ================================================================================================
# Parameters
# ================================================================================================
def b_nvidia_params() -> dict:
    """B_nvidia expressed in the JSON schema (values from models.nvidia_si_params(), i.e. exactly
    what models.make_sim('B_nvidia') writes into the MjSpec)."""
    si = models.nvidia_si_params()
    joints = {}
    for j in JOINTS:
        p = si[j]
        joints[j] = {"kp": p["kp"], "kv": p["kv"], "damping": p["damping"], "frictionloss": p["frictionloss"],
                     "armature": p["armature"], "forcerange": p["forcerange"], "delay_s": 0.0, "deadband_rad": 0.0}
    return {
        "schema": "so101_params/v1",
        "variant": "B_nvidia",
        "actuator": "position",
        "source": "models.nvidia_si_params() <- assets/nvidia_sysid/physics.usda (NVIDIA Isaac Sim 6.1 "
                  "so101_new_calib_SysID), converted to SI per radian (models.NVIDIA_CONVERSION). Ranges/ctrlranges "
                  "are not given, so the Menagerie values are used (as models.py does).",
        "units": "kp N*m/rad, kv N*m*s/rad, damping N*m*s/rad, frictionloss N*m, armature kg*m^2, forcerange N*m "
                 "(symmetric), delay_s s, deadband_rad rad",
        "sim": {"dt": 0.002, "noslip_iterations": 1},
        "joints": joints,
        "per_dataset": {},
    }


def write_default_json(path: str = DEFAULT_JSON) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(b_nvidia_params(), fh, indent=1)
    return path


def _pick(d: dict, key: str):
    for k in _ALIASES.get(key, (key,)):
        if k in d and d[k] is not None:
            return d[k]
    return None


def _mjcf_defaults() -> dict:
    """Per-joint values as they are in so101.xml (A_stock), used for anything the JSON omits."""
    import mujoco

    m = mujoco.MjSpec.from_file(MJCF).compile()
    out = {}
    for j in JOINTS:
        jid, aid = m.joint(j).id, m.actuator(j).id
        did = m.jnt_dofadr[jid]
        out[j] = {"kp": float(m.actuator_gainprm[aid, 0]), "kv": float(-m.actuator_biasprm[aid, 2]),
                  "damping": float(m.dof_damping[did]), "frictionloss": float(m.dof_frictionloss[did]),
                  "armature": float(m.dof_armature[did]),
                  "forcerange": [float(x) for x in m.actuator_forcerange[aid]],
                  "delay_s": 0.0, "deadband_rad": 0.0,
                  "range_rad": [float(x) for x in m.jnt_range[jid]],
                  "ctrlrange_rad": [float(x) for x in m.actuator_ctrlrange[aid]]}
    return out


def load_params(path: str) -> dict:
    """Normalise a parameter JSON (see module docstring) -> {"joints": {j: {...all keys...}}, ...}."""
    with open(path) as fh:
        raw = json.load(fh)
    act = str(raw.get("actuator", "position")).lower()
    if "bam" in act or "motor" in act:
        raise NotImplementedError(f"{path}: actuator={raw.get('actuator')!r}; this harness replays position-servo "
                                  "models only (BAM per-step torque: see bam_newton_sketch() below)")
    jsrc = raw.get("joints", raw)
    base = _mjcf_defaults()
    joints, defaulted, ignored = {}, [], []
    for j in JOINTS:
        src = jsrc.get(j, {}) or {}
        row = {}
        for key in _ALIASES:
            v = _pick(src, key)
            if v is None:
                v = base[j][key]
                if key not in ("delay_s", "deadband_rad", "range_rad", "ctrlrange_rad"):
                    defaulted.append(f"{j}.{key}")
            row[key] = v
        f = row["forcerange"]
        row["forcerange"] = [-abs(float(f)), abs(float(f))] if np.isscalar(f) else [float(f[0]), float(f[1])]
        for k in ("kp", "kv", "damping", "frictionloss", "armature", "delay_s", "deadband_rad"):
            row[k] = float(row[k])
        row["range_rad"] = [float(x) for x in row["range_rad"]]
        if _pick(src, "ctrlrange_rad") is None and _pick(src, "range_rad") is not None:
            row["ctrlrange_rad"] = list(row["range_rad"])  # Menagerie convention: ctrlrange == range
        row["ctrlrange_rad"] = [float(x) for x in row["ctrlrange_rad"]]
        known = {a for al in _ALIASES.values() for a in al}
        ignored += [f"{j}.{k}" for k in src if k not in known]
        joints[j] = row
    sim = dict(raw.get("sim", {}) or {})
    schema = str(raw.get("schema", "so101_params/v1"))
    v2 = schema.endswith("/v2")
    return {"path": os.path.abspath(path), "variant": raw.get("variant", os.path.basename(path)),
            "joints": joints, "per_dataset": raw.get("per_dataset", {}) or {},
            "dt": float(sim.get("dt", 0.002)), "noslip_iterations": int(sim.get("noslip_iterations", 1)),
            "defaulted": defaulted, "ignored_keys": ignored,
            # schema v2 extras (None / {} / True for v1, i.e. the v1 behaviour is unchanged)
            "schema": schema, "is_v2": v2,
            "iterations": sim.get("iterations"), "ls_iterations": sim.get("ls_iterations"),
            "integrator": sim.get("integrator"),
            "solimpfriction": sim.get("solimpfriction"), "solreffriction": sim.get("solreffriction"),
            "friction_setting": sim.get("friction_setting"),
            "bodies": raw.get("bodies") or {}, "delay_s": raw.get("delay_s"),
            "ctrllimited": not v2}       # v2: actuator ctrl limits off (anchor_fit.apply_opts)


def dataset_overrides(P: dict, name: str) -> dict:
    """Effective per-dataset settings: ranges, ctrlranges, per-joint delays, offsets, time scale."""
    pd_ = P["per_dataset"].get(name, {}) or {}
    rng = {j: list(P["joints"][j]["range_rad"]) for j in JOINTS}
    crng = {j: list(P["joints"][j]["ctrlrange_rad"]) for j in JOINTS}
    for j, v in (pd_.get("range_rad") or {}).items():
        rng[j] = [float(v[0]), float(v[1])]
        if j not in (pd_.get("ctrlrange_rad") or {}):
            crng[j] = [float(v[0]), float(v[1])]
    for j, v in (pd_.get("ctrlrange_rad") or {}).items():
        crng[j] = [float(v[0]), float(v[1])]
    delta = float(pd_.get("delta_s", 0.0) or 0.0)
    delays = np.array([P["joints"][j]["delay_s"] + delta for j in JOINTS])
    so = np.array([float((pd_.get("state_offset_rad") or {}).get(j, 0.0)) for j in JOINTS])
    ao = np.array([float((pd_.get("action_offset_rad") or {}).get(j, 0.0)) for j in JOINTS])
    return {"range": rng, "ctrlrange": crng, "delays": delays, "state_offset": so, "action_offset": ao,
            "time_scale": float(pd_.get("time_scale", 1.0) or 1.0), "delta_s": delta}


# ================================================================================================
# Command schedule (shared by every simulator, so they get bit-identical targets)
# ================================================================================================
def step_schedule(A, ts, n_frames, dt, delays, time_scale=1.0):
    """Per-physics-step servo targets for a replay that starts at frame ts.

    A: (T, 6) commands in sim radians (full episode). Command A[k] is sent at frame k; for joint i it
    becomes the target at relative time (k - ts)*T + delays[i] (T = time_scale/FPS), switching at physics
    step round(((k - ts)*T + delay)/dt). Commands before frame 0 are clamped to A[0].
    Returns targets (n_steps, 6) and the step counts after which frame ts+1 .. ts+n_frames is sampled
    (round((m+1)*T/dt), m = 0..n_frames-1; replay.py's running time goal)."""
    T = time_scale / FPS
    sample_steps = np.array([int(round((m + 1) * T / dt)) for m in range(n_frames)])
    n_steps = int(sample_steps[-1])
    steps = np.arange(n_steps)
    tgt = np.empty((n_steps, 6))
    Tlen = len(A)
    for i in range(6):
        d = float(delays[i])
        jmin = -int(math.ceil(d / T)) - 2
        js = np.arange(jmin, n_frames + 1)
        B = np.rint((js * T + d) / dt).astype(np.int64)
        idx = np.searchsorted(B, steps, side="right") - 1
        k = np.clip(ts + js[np.maximum(idx, 0)], 0, Tlen - 1)
        tgt[:, i] = A[k, i]
    return tgt, sample_steps


def _deadband(target, q, db):
    e = target - q
    return q + np.sign(e) * np.maximum(np.abs(e) - db, 0.0)


# ================================================================================================
# MuJoCo reference (models.py-style, built from the JSON)
# ================================================================================================
class MjArm:
    """models.SO101Sim('B_nvidia')-style MuJoCo model whose joint/actuator values come from the JSON.

    Same construction as models.py: MjSpec.from_file(so101.xml); timestep dt; noslip_iterations;
    contacts disabled; per joint: damping/frictionloss/armature on the joint, gainprm[0]=kp,
    biasprm[1:3]=(-kp,-kv), forcelimited with +-forcerange on the position actuator."""

    def __init__(self, P, dt, limits, ov, noslip=None):
        import mujoco

        self.mujoco = mujoco
        spec = mujoco.MjSpec.from_file(MJCF)
        spec.option.timestep = dt
        spec.option.noslip_iterations = int(P["noslip_iterations"] if noslip is None else noslip)
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        joints = {j.name: j for j in spec.joints}
        acts = {a.name: a for a in spec.actuators}
        for k, jn in enumerate(JOINTS):
            p = P["joints"][jn]
            j, a = joints[jn], acts[jn]
            j.damping = models._hinge_damping(p["damping"])
            j.frictionloss = p["frictionloss"]
            j.armature = p["armature"]
            g = np.zeros(10)
            g[0] = p["kp"]
            b = np.zeros(10)
            b[1], b[2] = -p["kp"], -p["kv"]
            a.gainprm = g
            a.biasprm = b
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.forcerange = list(p["forcerange"])
            j.range = list(ov["range"][jn])
            a.ctrlrange = list(ov["ctrlrange"][jn])
        self.model = spec.compile()
        if limits == "wide":
            for k, jn in enumerate(JOINTS[:ARM]):
                self.model.jnt_limited[self.model.joint(jn).id] = 0
                self.model.actuator_ctrllimited[self.model.actuator(jn).id] = 0
        self.data = mujoco.MjData(self.model)
        apply_v2_to_mjmodel(self.model, self.data, P)
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINTS])
        self.vadr = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in JOINTS])
        self.aid = np.array([self.model.actuator(n).id for n in JOINTS])

    def run(self, q0, v0, targets, sample_steps, deadband=None):
        mj, m, d = self.mujoco, self.model, self.data
        mj.mj_resetData(m, d)
        d.qpos[self.qadr] = q0
        d.qvel[:] = 0.0
        d.ctrl[self.aid] = q0
        mj.mj_forward(m, d)
        d.qvel[self.vadr] = v0
        mj.mj_forward(m, d)
        out = np.empty((len(sample_steps), 6))
        k = 0
        for n in range(len(targets)):
            t = targets[n]
            if deadband is not None:
                t = _deadband(t, d.qpos[self.qadr], deadband)
            d.ctrl[self.aid] = t
            mj.mj_step(m, d)
            while k < len(sample_steps) and sample_steps[k] == n + 1:
                out[k] = d.qpos[self.qadr]
                k += 1
        return out


_INTEGRATORS = {"euler": "mjINT_EULER", "rk4": "mjINT_RK4", "implicit": "mjINT_IMPLICIT",
                "implicitfast": "mjINT_IMPLICITFAST"}


def apply_v2_to_mjmodel(m, d, P):
    """Schema-v2 options on a compiled MjModel: body inertials, joint friction softness, solver iterations,
    integrator, ctrl limits off. For a fitted_json() file this is what anchor_fit.apply_opts does (camera inertia
    is diagonal in the body frame; for the isotropic camera of fitted_json the iquat is irrelevant)."""
    import mujoco

    if not P.get("is_v2"):
        return
    for name, b in P["bodies"].items():
        bid = m.body(name).id
        m.body_mass[bid] = float(b["mass"])
        m.body_ipos[bid] = [float(x) for x in b["ipos"]]
        m.body_inertia[bid] = [float(x) for x in b["inertia"]]
        m.body_iquat[bid] = [1.0, 0.0, 0.0, 0.0]
    vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in JOINTS])
    if P["solimpfriction"] is not None:
        m.dof_solimp[vadr] = np.asarray(P["solimpfriction"], float)
    if P["solreffriction"] is not None:
        m.dof_solref[vadr] = np.asarray(P["solreffriction"], float)
    if P["iterations"] is not None:
        m.opt.iterations = int(P["iterations"])
    if P["ls_iterations"] is not None:
        m.opt.ls_iterations = int(P["ls_iterations"])
    if P["integrator"] is not None:
        m.opt.integrator = int(getattr(mujoco.mjtIntegrator, _INTEGRATORS[str(P["integrator"]).lower()]))
    if not P["ctrllimited"]:
        m.actuator_ctrllimited[:] = 0
    mujoco.mj_setConst(m, d)


class ModelsPyArm:
    """The untouched models.make_sim('B_nvidia') object driven with the same per-step targets
    (cross-check that the JSON route rebuilds exactly the models.py model)."""

    def __init__(self, dt, limits):
        import mujoco

        self.mujoco = mujoco
        self.sim = models.make_sim("B_nvidia", dt=dt)
        if limits == "wide":
            for k, j in enumerate(JOINTS[:ARM]):
                self.sim.model.jnt_limited[self.sim.model.joint(j).id] = 0
                self.sim.model.actuator_ctrllimited[self.sim.aid[k]] = 0

    def run(self, q0, v0, targets, sample_steps, deadband=None):
        s = self.sim
        s.reset(q0)
        s.data.qvel[s.vadr] = v0
        self.mujoco.mj_forward(s.model, s.data)
        out = np.empty((len(sample_steps), 6))
        k = 0
        for n in range(len(targets)):
            s.data.ctrl[s.aid] = targets[n]
            self.mujoco.mj_step(s.model, s.data)
            while k < len(sample_steps) and sample_steps[k] == n + 1:
                out[k] = s.qpos()
                k += 1
        return out


# ================================================================================================
# Newton
# ================================================================================================
def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mujoco_body_inertials(mj_model) -> dict:
    """{body name: (mass, com in body frame, 3x3 inertia about com in body frame)} from a compiled MjModel."""
    import mujoco

    out = {}
    for b in range(1, mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b)
        R = _quat_to_mat(mj_model.body_iquat[b])
        I = R @ np.diag(mj_model.body_inertia[b]) @ R.T
        out[name] = (float(mj_model.body_mass[b]), np.array(mj_model.body_ipos[b], float), I)
    return out


def _attr_set(attr, row, value):
    vals = attr.values
    if isinstance(vals, dict):
        vals[row] = value
    else:
        vals[row] = value


class NewtonArm:
    """SO-101 in Newton 1.6.1 (ModelBuilder.add_mjcf + SolverMuJoCo), parameters from the JSON."""

    def __init__(self, P, dt, limits, ov, backend="mujoco_cpu", inertia="mujoco", noslip=None,
                 update_data_interval=1, ref_inertials=None):
        import warp as wp

        wp.config.quiet = True
        import newton
        from newton._src.core.types import MAXVAL

        from newton._src.core.types import vec5

        self.wp, self.newton = wp, newton
        self.dt = dt
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.add_mjcf(MJCF, floating=False)
        short = [lab.rsplit("/", 1)[-1] for lab in b.joint_label]
        jidx = {n: i for i, n in enumerate(short)}
        self.dofs = [b.joint_qd_start[jidx[j]] for j in JOINTS]
        self.coords = [b.joint_q_start[jidx[j]] for j in JOINTS]
        ca = b.custom_attributes
        trn = ca["mujoco:actuator_trnid"].values
        trn_list = [trn[i] for i in range(len(trn))] if not isinstance(trn, dict) else [trn[i] for i in sorted(trn)]
        row_of_dof = {int(t[0]): r for r, t in enumerate(trn_list)}
        self.applied = {}
        sip, srf = P.get("solimpfriction"), P.get("solreffriction")
        ctrl_on = P.get("ctrllimited", True)
        for k, jn in enumerate(JOINTS):
            p, d = P["joints"][jn], self.dofs[k]
            b.joint_target_ke[d] = p["kp"]
            b.joint_target_kd[d] = p["kv"]
            b.joint_damping[d] = p["damping"]
            b.joint_friction[d] = p["frictionloss"]
            b.joint_armature[d] = p["armature"]
            b.joint_effort_limit[d] = max(abs(p["forcerange"][0]), abs(p["forcerange"][1]))
            lo, hi = ov["range"][jn]
            clo, chi = ov["ctrlrange"][jn]
            wide = limits == "wide" and k < ARM
            b.joint_limit_lower[d] = -MAXVAL if wide else lo
            b.joint_limit_upper[d] = MAXVAL if wide else hi
            r = row_of_dof[d]
            _attr_set(ca["mujoco:actuator_forcerange"], r, wp.vec2(*p["forcerange"]))
            _attr_set(ca["mujoco:actuator_has_forcerange"], r, 1)
            _attr_set(ca["mujoco:actuator_forcelimited"], r, 1)
            _attr_set(ca["mujoco:actuator_ctrlrange"], r, wp.vec2(clo, chi))
            _attr_set(ca["mujoco:actuator_has_ctrlrange"], r, 1)
            _attr_set(ca["mujoco:actuator_ctrllimited"], r, 0 if (wide or not ctrl_on) else 1)
            # v2 joint friction softness (per DOF; default = MuJoCo's solimp 0.9 0.95 0.001 0.5 2 / solref 0.02 1)
            if sip is not None:
                _attr_set(ca["mujoco:solimpfriction"], d, vec5(*[float(x) for x in sip]))
            if srf is not None:
                _attr_set(ca["mujoco:solreffriction"], d, wp.vec2(*[float(x) for x in srf]))
        if inertia == "json":
            # importer inertials for every body, then the JSON's bodies (v2: camera_mount) through the builder
            labels = {lab.rsplit("/", 1)[-1]: i for i, lab in enumerate(b.body_label)}
            for name, bj in (P.get("bodies") or {}).items():
                if name not in labels:
                    raise KeyError(f"JSON body {name!r} not among Newton bodies {sorted(labels)}")
                i = labels[name]
                m_ = float(bj["mass"])
                I_ = np.diag([float(x) for x in bj["inertia"]])
                b.body_mass[i] = m_
                b.body_inv_mass[i] = 1.0 / m_ if m_ > 0 else 0.0
                b.body_com[i] = wp.vec3(*[float(x) for x in bj["ipos"]])
                b.body_inertia[i] = wp.mat33(*I_.flatten())
                b.body_inv_inertia[i] = wp.mat33(*np.linalg.inv(I_).flatten()) if m_ > 0 else wp.mat33(0.0)
        if inertia == "mujoco":
            # Newton's importer derives the camera_mount inertial from its geoms differently from
            # MuJoCo's compiler (no <inertial> on that body); copy MuJoCo's compiled inertials so both
            # sims have identical mass properties.
            for i, lab in enumerate(b.body_label):
                name = lab.rsplit("/", 1)[-1]
                if name not in ref_inertials:
                    continue
                m_, c_, I_ = ref_inertials[name]
                b.body_mass[i] = m_
                b.body_inv_mass[i] = 1.0 / m_ if m_ > 0 else 0.0
                b.body_com[i] = wp.vec3(*c_)
                b.body_inertia[i] = wp.mat33(*I_.flatten())
                b.body_inv_inertia[i] = wp.mat33(*np.linalg.inv(I_).flatten()) if m_ > 0 else wp.mat33(0.0)
        self.model = b.finalize(device="cpu")
        ns = int(P["noslip_iterations"] if noslip is None else noslip)
        # v2 solver options as SolverMuJoCo constructor arguments (None -> MJCF <option> / Newton default)
        skw = {k: P[k] for k in ("iterations", "ls_iterations") if P.get(k) is not None}
        if P.get("integrator") is not None:
            skw["integrator"] = str(P["integrator"]).lower()
        if backend == "mujoco_cpu":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_cpu=True, disable_contacts=True, update_data_interval=update_data_interval,
                **skw)
            self.solver.mj_model.opt.noslip_iterations = ns
        else:
            if ns:
                raise ValueError("mujoco_warp backend: noslip not implemented (mujoco_warp io.py raises); use noslip=0")
            # disable_contacts=True sets nconmax=0, and mujoco_warp 3.12's CPU fallback in
            # solver._update_gradient uses dim_block = naconmax -> ZeroDivisionError. Keep a contact
            # buffer and switch contacts off with the same MuJoCo disable bit instead (Python-level flag,
            # read by mujoco_warp at every step call).
            import mujoco

            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_cpu=False, disable_contacts=False, update_data_interval=update_data_interval,
                **skw)
            self.solver.mjw_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        # Newton writes opt.timestep = dt on every step(); set it now so the built model can be compared
        self.solver.mj_model.opt.timestep = dt
        self.backend = backend
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.tq_idx = np.array([int(self.model.joint_target_q_start.numpy()[jidx[j]]) for j in JOINTS]) \
            if hasattr(self.model, "joint_target_q_start") else np.array(self.dofs)
        self._tq = self.control.joint_target_q.numpy().copy()
        self.coords = np.array(self.coords)
        self.dofs = np.array(self.dofs)

    def mj_model(self):
        return self.solver.mj_model  # the MjModel Newton built (also kept for the warp backend)

    def ncon(self):
        if self.backend == "mujoco_cpu":
            return int(self.solver.mj_data.ncon)
        return int(self.solver.mjw_data.nacon.numpy().sum())

    def run(self, q0, v0, targets, sample_steps, deadband=None):
        wp = self.wp
        q = self.model.joint_q.numpy().copy()
        qd = self.model.joint_qd.numpy().copy()
        q[self.coords] = q0
        qd[:] = 0.0
        qd[self.dofs] = v0
        self.model.joint_q.assign(q)
        self.model.joint_qd.assign(qd)
        self.solver.reset(self.s0)          # clears qacc_warmstart/ctrl/act/qfrc_applied, state <- model.joint_q/qd
        self.s1.joint_q.assign(q)
        self.s1.joint_qd.assign(qd)
        tq = self._tq
        out = np.empty((len(sample_steps), 6))
        k, last = 0, None
        s0, s1 = self.s0, self.s1
        for n in range(len(targets)):
            t = targets[n]
            if deadband is not None:
                t = _deadband(t, s0.joint_q.numpy()[self.coords].astype(np.float64), deadband)
            if deadband is not None or last is None or not np.array_equal(t, last):
                tq[self.tq_idx] = t
                self.control.joint_target_q.assign(tq)
                last = t
            self.solver.step(s0, s1, self.control, None, self.dt)
            s0, s1 = s1, s0
            while k < len(sample_steps) and sample_steps[k] == n + 1:
                out[k] = s0.joint_q.numpy()[self.coords]
                k += 1
        self.s0, self.s1 = s0, s1
        return out

    def run_steps(self, q0, v0, U):
        """Every physics step: Q[0] = q0, Q[n+1] = Newton State.joint_q after step n with target U[n]
        (the layout of anchor_fit.rollout_q). Arrays live on the CPU device, so .numpy() is a zero-copy view."""
        q = self.model.joint_q.numpy().copy()
        qd = self.model.joint_qd.numpy().copy()
        q[self.coords] = q0
        qd[:] = 0.0
        qd[self.dofs] = v0
        self.model.joint_q.assign(q)
        self.model.joint_qd.assign(qd)
        self.solver.reset(self.s0)
        self.s1.joint_q.assign(q)
        self.s1.joint_qd.assign(qd)
        tq = self.control.joint_target_q.numpy()      # view: writes reach the solver
        Q = np.empty((len(U) + 1, 6))
        Q[0] = q0
        s0, s1 = self.s0, self.s1
        step, dt, idx, coords = self.solver.step, self.dt, self.tq_idx, self.coords
        ctrl = self.control
        for n in range(len(U)):
            tq[idx] = U[n]
            step(s0, s1, ctrl, None, dt)
            s0, s1 = s1, s0
            Q[n + 1] = s0.joint_q.numpy()[coords]
        self.s0, self.s1 = s0, s1
        return Q


MATCH_RTOL = 1e-6      # float32 storage in Newton: relative rounding <= 6e-8; 1e-6 leaves margin for re-derivations
MATCH_REQUIRED = ("dof_solimp", "dof_solref", "dof_frictionloss", "dof_damping", "dof_armature",
                  "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_forcelimited",
                  "actuator_ctrllimited", "body_mass", "body_com", "jnt_range", "jnt_limited",
                  "opt.noslip_iterations", "opt.iterations", "opt.ls_iterations", "opt.timestep", "opt.integrator")


def _by_suffix(m, objtype, n):
    """Id of the joint/body whose name is n, or ends in '_' + n (Newton's path-prefixed names); must be unique."""
    import mujoco

    count, obj = {"joint": (m.njnt, mujoco.mjtObj.mjOBJ_JOINT), "body": (m.nbody, mujoco.mjtObj.mjOBJ_BODY)}[objtype]
    names = [mujoco.mj_id2name(m, obj, i) or "" for i in range(count)]
    hits = [i for i, s in enumerate(names) if s == n or s.endswith("_" + n)]
    if len(hits) != 1:
        raise KeyError(f"{objtype} {n!r}: {len(hits)} matches")
    return hits[0]


def model_match(ref, new) -> dict:
    """Element-wise comparison of the reference MjModel (anchor_fit.get_pos_sim) with the MjModel Newton built,
    matched by joint / actuator / body name (Newton prefixes names with the body path). Returns
    {"items": {name: {max_abs, rel, ok, required}}, "ok": all required ok}. Float items pass when
    max_abs / max(|ref|) <= MATCH_RTOL; integer items must be equal."""
    import mujoco

    J = [(_by_suffix(ref, "joint", j), _by_suffix(new, "joint", j)) for j in JOINTS]
    D = [(ref.jnt_dofadr[a], new.jnt_dofadr[b]) for a, b in J]
    A = []
    for a, b in J:
        ua = [i for i in range(ref.nu) if ref.actuator_trnid[i, 0] == a]
        ub = [i for i in range(new.nu) if new.actuator_trnid[i, 0] == b]
        assert len(ua) == 1 and len(ub) == 1, (ua, ub)
        A.append((ua[0], ub[0]))
    bodies = [mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, ref.nbody)]
    B = [(i + 1, _by_suffix(new, "body", n)) for i, n in enumerate(bodies)]
    ib_ref, ib_new = mujoco_body_inertials(ref), mujoco_body_inertials(new)
    inv_nb = {j: mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_BODY, j) for _, j in B}

    def pair(arr_a, arr_b, idx, cols=None):
        a = np.array([np.atleast_1d(arr_a[i])[:cols] if cols else np.atleast_1d(arr_a[i]) for i, _ in idx], float)
        b = np.array([np.atleast_1d(arr_b[j])[:cols] if cols else np.atleast_1d(arr_b[j]) for _, j in idx], float)
        return a, b

    rows = {
        "dof_solimp": pair(ref.dof_solimp, new.dof_solimp, D),
        "dof_solref": pair(ref.dof_solref, new.dof_solref, D),
        "dof_frictionloss": pair(ref.dof_frictionloss, new.dof_frictionloss, D),
        "dof_damping": pair(ref.dof_damping, new.dof_damping, D),
        "dof_armature": pair(ref.dof_armature, new.dof_armature, D),
        "actuator_gainprm": pair(ref.actuator_gainprm, new.actuator_gainprm, A, 3),
        "actuator_biasprm": pair(ref.actuator_biasprm, new.actuator_biasprm, A, 3),
        "actuator_forcerange": pair(ref.actuator_forcerange, new.actuator_forcerange, A),
        "actuator_forcelimited": pair(ref.actuator_forcelimited, new.actuator_forcelimited, A),
        "actuator_ctrllimited": pair(ref.actuator_ctrllimited, new.actuator_ctrllimited, A),
        "actuator_gaintype/biastype/dyntype": (
            np.array([[ref.actuator_gaintype[i], ref.actuator_biastype[i], ref.actuator_dyntype[i]] for i, _ in A], float),
            np.array([[new.actuator_gaintype[j], new.actuator_biastype[j], new.actuator_dyntype[j]] for _, j in A], float)),
        "body_mass": pair(ref.body_mass, new.body_mass, B),
        "body_com": (np.array([ib_ref[n][1] for n in bodies]), np.array([ib_new[inv_nb[j]][1] for _, j in B])),
        "body_inertia_full": (np.array([ib_ref[n][2] for n in bodies]), np.array([ib_new[inv_nb[j]][2] for _, j in B])),
        "jnt_range": pair(ref.jnt_range, new.jnt_range, J),
        "jnt_limited": pair(ref.jnt_limited, new.jnt_limited, J),
        "jnt_solref(limit)": pair(ref.jnt_solref, new.jnt_solref, J),
        "jnt_solimp(limit)": pair(ref.jnt_solimp, new.jnt_solimp, J),
        "jnt_margin": pair(ref.jnt_margin, new.jnt_margin, J),
        # Newton also clamps the summed actuator force per joint (actfrclimited, +-joint_effort_limit); the
        # reference has no such clamp. With one actuator per joint it is redundant iff it equals the actuator
        # forcerange: compared here against the REFERENCE actuator forcerange.
        "newton_jnt_actfrcrange_vs_ref_forcerange": (
            np.array([ref.actuator_forcerange[i] for i, _ in A], float),
            np.array([new.jnt_actfrcrange[b] if new.jnt_actfrclimited[b] else [-np.inf, np.inf] for _, b in J], float)),
    }
    for f in ("noslip_iterations", "iterations", "ls_iterations", "timestep", "integrator", "cone", "solver",
              "impratio", "tolerance", "ls_tolerance", "jacobian"):
        rows["opt." + f] = (np.array([float(getattr(ref.opt, f))]), np.array([float(getattr(new.opt, f))]))
    rows["opt.gravity"] = (np.array(ref.opt.gravity, float), np.array(new.opt.gravity, float))
    cb = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    rows["opt.contacts_disabled"] = (np.array([float(bool(ref.opt.disableflags & cb))]),
                                     np.array([float(bool(new.opt.disableflags & cb))]))
    integer = {"actuator_forcelimited", "actuator_ctrllimited", "actuator_gaintype/biastype/dyntype", "jnt_limited",
               "opt.noslip_iterations", "opt.iterations", "opt.ls_iterations", "opt.integrator", "opt.cone",
               "opt.solver", "opt.jacobian", "opt.contacts_disabled"}
    items = {}
    for k, (a, b) in rows.items():
        dmax = float(np.max(np.abs(a - b))) if a.size else 0.0
        scale = float(np.max(np.abs(a))) if a.size else 0.0
        rel = dmax / scale if scale > 0 else dmax
        ok = (dmax == 0.0) if k in integer else (rel <= MATCH_RTOL)
        items[k] = {"max_abs": dmax, "rel": rel, "ok": bool(ok), "required": k in MATCH_REQUIRED}
        if k in integer or a.size <= 12:
            items[k]["ref"] = a.tolist()
            items[k]["newton"] = b.tolist()
    return {"items": items, "ok": all(v["ok"] for v in items.values() if v["required"]),
            "all_ok": all(v["ok"] for v in items.values()), "rtol": MATCH_RTOL}


def compare_mj_models(ref, new) -> dict:
    """Largest differences between the reference MjModel and the one Newton builds (same joint order)."""
    import mujoco

    def jid(m, j):
        for i in range(m.njnt):
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i).endswith(j):
                return i
        raise KeyError(j)

    out = {}
    opt = {}
    for f in ("timestep", "integrator", "cone", "solver", "iterations", "ls_iterations", "noslip_iterations",
              "impratio", "tolerance", "ls_tolerance", "disableflags", "jacobian"):
        a, b = getattr(ref.opt, f), getattr(new.opt, f)
        opt[f] = [float(a) if not isinstance(a, int) else int(a), float(b) if not isinstance(b, int) else int(b)]
    opt["gravity"] = [list(map(float, ref.opt.gravity)), list(map(float, new.opt.gravity))]
    out["opt_ref_vs_newton"] = opt
    rows = {}
    for j in JOINTS:
        a, b = jid(ref, j), jid(new, j)
        da, db = ref.jnt_dofadr[a], new.jnt_dofadr[b]
        ua = [i for i in range(ref.nu) if ref.actuator_trnid[i, 0] == a][0]
        ub = [i for i in range(new.nu) if new.actuator_trnid[i, 0] == b][0]
        rows[j] = {
            "armature": [float(ref.dof_armature[da]), float(new.dof_armature[db])],
            "damping": [float(ref.dof_damping[da]), float(new.dof_damping[db])],
            "frictionloss": [float(ref.dof_frictionloss[da]), float(new.dof_frictionloss[db])],
            "range": [list(map(float, ref.jnt_range[a])), list(map(float, new.jnt_range[b]))],
            "limited": [int(ref.jnt_limited[a]), int(new.jnt_limited[b])],
            "kp": [float(ref.actuator_gainprm[ua, 0]), float(new.actuator_gainprm[ub, 0])],
            "kv": [float(-ref.actuator_biasprm[ua, 2]), float(-new.actuator_biasprm[ub, 2])],
            "forcerange": [list(map(float, ref.actuator_forcerange[ua])), list(map(float, new.actuator_forcerange[ub]))],
            "ctrlrange": [list(map(float, ref.actuator_ctrlrange[ua])), list(map(float, new.actuator_ctrlrange[ub]))],
            "ctrllimited": [int(ref.actuator_ctrllimited[ua]), int(new.actuator_ctrllimited[ub])],
            "jnt_actfrcrange_newton": list(map(float, new.jnt_actfrcrange[b])),
            "jnt_solref": [list(map(float, ref.jnt_solref[a])), list(map(float, new.jnt_solref[b]))],
            "dof_solref": [list(map(float, ref.dof_solref[da])), list(map(float, new.dof_solref[db]))],
        }
    out["joints"] = rows
    # mass properties per body (by short name)
    def bodies(m):
        r = {}
        for i in range(1, m.nbody):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            r[n.split("_worldbody_")[-1].split("_")[-1] if False else n] = i
        return r
    ib_ref = mujoco_body_inertials(ref)
    ib_new = mujoco_body_inertials(new)
    worst = {}
    for nr, (m1, c1, I1) in ib_ref.items():
        cand = [k for k in ib_new if k == nr or k.endswith("_" + nr)]
        if not cand:
            continue
        m2, c2, I2 = ib_new[cand[0]]
        worst[nr] = {"dmass": abs(m1 - m2), "dcom_mm": float(1e3 * np.abs(c1 - c2).max()),
                     "dinertia_rel": float(np.abs(I1 - I2).max() / max(np.abs(I1).max(), 1e-12))}
    out["bodies"] = worst
    return out


# ================================================================================================
# Windows / data
# ================================================================================================
def select_windows(dsets, W, per_dataset, names):
    import replay

    sel = []
    for ds in dsets:
        if ds["name"] not in names:
            continue
        wins = replay.make_windows(ds, W)
        if per_dataset and len(wins) > per_dataset:
            idx = np.unique(np.round(np.linspace(0, len(wins) - 1, per_dataset)).astype(int))
            wins = [wins[i] for i in idx]
        sel += [(ds["name"], ep, t0) for ep, t0 in wins]
    return sel


_CACHE = {}


def _sims(cfg, P, ds_name, ov):
    key = (cfg["name"], ds_name if P["per_dataset"].get(ds_name) else "")
    if key in _CACHE:
        return _CACHE[key]
    t = time.perf_counter()
    ref = MjArm(P, cfg["dt"], cfg["limits"], ov, noslip=cfg.get("noslip_ref", cfg.get("noslip")))
    build_ref = time.perf_counter() - t
    sims = {"mujoco": ref}
    t = time.perf_counter()
    if cfg.get("newton", True):
        sims["newton"] = NewtonArm(P, cfg["dt"], cfg["limits"], ov, backend=cfg.get("backend", "mujoco_cpu"),
                                   inertia=cfg.get("inertia", "mujoco"), noslip=cfg.get("noslip"),
                                   update_data_interval=cfg.get("update_data_interval", 1),
                                   ref_inertials=mujoco_body_inertials(ref.model))
    build_newton = time.perf_counter() - t
    if cfg.get("models_py", False):
        sims["models_py"] = ModelsPyArm(cfg["dt"], cfg["limits"])
    if cfg.get("alt_ref"):
        sims["mujoco_alt"] = MjArm(P, cfg["dt"], cfg["limits"], ov, noslip=cfg["alt_ref"]["noslip"])
    _CACHE[key] = (sims, {"build_ref_s": build_ref, "build_newton_s": build_newton})
    return _CACHE[key]


def run_job(job):
    import replay

    cfg, P = job["cfg"], job["P"]
    dsets = {d["name"]: d for d in job["dsets"]}
    W, warm = cfg["W"], cfg.get("warm", 0)
    recs, cpu, builds = [], {}, {}
    for ds_name, ep, t0 in job["windows"]:
        ov = dataset_overrides(P, ds_name)
        sims, b = _sims(cfg, P, ds_name, ov)
        builds = b
        e = dsets[ds_name]["episodes"][ep]
        A = e["A"] + ov["action_offset"]
        S = e["S"] + ov["state_offset"]
        N = int(round(W * FPS))
        ts = max(t0 - warm, 0)
        nf = (t0 - ts) + N
        q0 = S[ts].copy()
        v0 = (S[ts] - S[ts - 1]) * FPS / ov["time_scale"] if ts > 0 else np.zeros(6)
        if cfg["limits"] == "sim":
            lo = np.array([ov["range"][j][0] for j in JOINTS])
            hi = np.array([ov["range"][j][1] for j in JOINTS])
        else:
            lo = np.array([ov["range"][j][0] if k >= ARM else -np.inf for k, j in enumerate(JOINTS)])
            hi = np.array([ov["range"][j][1] if k >= ARM else np.inf for k, j in enumerate(JOINTS)])
        qc = np.clip(q0, lo, hi)
        v0 = np.where(qc != q0, 0.0, v0)
        q0 = qc
        tgt, samp = step_schedule(A, ts, nf, cfg["dt"], ov["delays"], ov["time_scale"])
        db = np.array([P["joints"][j]["deadband_rad"] for j in JOINTS])
        db = db if cfg.get("deadband", True) and np.any(db > 0) else None
        preds = {}
        for name, sim in sims.items():
            c = time.process_time()
            preds[name] = sim.run(q0, v0, tgt, samp, deadband=db)[t0 - ts:]
            cpu[name] = cpu.get(name, 0.0) + time.process_time() - c
        real = S[t0 + 1:t0 + N + 1]
        rec = {"dataset": ds_name, "episode": int(ep), "t0": int(t0), "n": N, "ts": int(ts)}
        if "newton" in sims:
            rec["newton_ncon_end"] = sims["newton"].ncon()
        for name, Pq in preds.items():
            err = np.degrees(Pq - real)
            rec["sse_real|" + name] = (err ** 2).sum(0).tolist()
        Dn = np.degrees(preds.get("newton", preds["mujoco"]) - preds["mujoco"])
        rec["maxabs_newton_vs_mujoco"] = np.abs(Dn).max(0).tolist()
        rec["sse_newton_vs_mujoco"] = (Dn ** 2).sum(0).tolist()
        if "models_py" in preds:
            rec["maxabs_modelspy_vs_mujoco"] = np.abs(np.degrees(preds["models_py"] - preds["mujoco"])).max(0).tolist()
        if "mujoco_alt" in preds:
            Da = np.degrees(preds["mujoco_alt"] - preds["mujoco"])
            rec["maxabs_alt_vs_mujoco"] = np.abs(Da).max(0).tolist()
            rec["sse_alt_vs_mujoco"] = (Da ** 2).sum(0).tolist()
        # frame index where Newton-vs-MuJoCo difference is largest (for diagnosis)
        rec["argmax_frame_newton_vs_mujoco"] = int(np.abs(Dn).max(1).argmax())
        recs.append(rec)
    sim_s = len(job["windows"]) * (W + warm / FPS)
    return {"recs": recs, "cpu": cpu, "sim_s": sim_s, "builds": builds, "pid": os.getpid()}


# ================================================================================================
# Aggregation
# ================================================================================================
def aggregate(recs, models_present, names):
    import pandas as pd  # noqa: F401

    out = {"n_windows": len(recs), "n_frames": int(sum(r["n"] for r in recs))}
    n = out["n_frames"]
    mx = np.max([r["maxabs_newton_vs_mujoco"] for r in recs], 0)
    rms = np.sqrt(np.sum([r["sse_newton_vs_mujoco"] for r in recs], 0) / n)
    out["newton_vs_mujoco"] = {"max_deg": dict(zip(JOINTS, mx.round(6).tolist())),
                               "rms_deg": dict(zip(JOINTS, rms.round(6).tolist())),
                               "max_deg_arm": float(mx[:ARM].max()), "rms_deg_arm_mean": float(rms[:ARM].mean())}
    worst = max(recs, key=lambda r: max(r["maxabs_newton_vs_mujoco"][:ARM]))
    out["newton_vs_mujoco"]["worst_window"] = {k: worst[k] for k in ("dataset", "episode", "t0")} | {
        "frame": worst["argmax_frame_newton_vs_mujoco"]}
    if "maxabs_modelspy_vs_mujoco" in recs[0]:
        out["models_py_vs_json_mujoco_max_deg"] = float(np.max([r["maxabs_modelspy_vs_mujoco"] for r in recs]))
    if "maxabs_alt_vs_mujoco" in recs[0]:
        mxa = np.max([r["maxabs_alt_vs_mujoco"] for r in recs], 0)
        rmsa = np.sqrt(np.sum([r["sse_alt_vs_mujoco"] for r in recs], 0) / n)
        out["alt_ref_vs_mujoco"] = {"max_deg": dict(zip(JOINTS, mxa.round(6).tolist())),
                                    "rms_deg": dict(zip(JOINTS, rmsa.round(6).tolist()))}
    # vs real: pooled RMSE per dataset, equal-weight means over the 6 primary and the 3 provenance-verified
    vs_real = {}
    for m in models_present:
        per_ds = {}
        for name in names:
            rr = [r for r in recs if r["dataset"] == name]
            if not rr:
                continue
            nn = sum(r["n"] for r in rr)
            sse = np.sum([r["sse_real|" + m] for r in rr], 0)
            per_ds[name] = np.sqrt(sse / nn)
        ent = {"per_dataset_arm_mean": {k: round(float(v[:ARM].mean()), 4) for k, v in per_ds.items()}}
        for label, group in (("primary6", PRIMARY), ("provenance3", PROVENANCE_VERIFIED)):
            g = [per_ds[k] for k in group if k in per_ds]
            if g:
                mean = np.mean(g, 0)
                ent[label] = {"per_joint": dict(zip(JOINTS, mean.round(4).tolist())),
                              "arm_mean": round(float(mean[:ARM].mean()), 4), "n_datasets": len(g)}
        vs_real[m] = ent
    out["vs_real"] = vs_real
    return out


def crosscheck_windows_csv(recs, limits):
    """Our MuJoCo reference (venv, mujoco 3.12) vs replay.py's B_nvidia per-window RMSE (main env,
    mujoco 3.14) for the same windows. Valid only for B_nvidia params, warm 0, no delay."""
    import pandas as pd

    path = os.path.join(RESULTS, "windows.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df = df[(df["W"] == 4) & (df["limits"] == limits)].set_index(["dataset", "episode", "t0"])
    diffs = []
    for r in recs:
        key = (r["dataset"], r["episode"], r["t0"])
        if key not in df.index:
            continue
        row = df.loc[key]
        ours = np.sqrt(np.array(r["sse_real|mujoco"]) / r["n"])
        theirs = np.array([row[f"rmse|B_nvidia|{j}"] for j in JOINTS])
        diffs.append(np.abs(ours - theirs))
    if not diffs:
        return None
    d = np.max(diffs, 0)
    return {"n_windows": len(diffs), "max_abs_diff_window_rmse_deg": dict(zip(JOINTS, d.round(8).tolist()))}


# ================================================================================================
# BAM in Newton (optional sketch)
# ================================================================================================
def bam_newton_sketch():
    """How a BAM servo (torque computed per physics step in Python) would run in Newton.  SKETCH, not run.

    Newton side (verified API, not executed with BAM):
      * Load the MJCF, then for the 6 joints set builder.joint_target_mode[dof] = JointTargetMode.NONE and
        joint_target_ke = joint_target_kd = 0 (no MuJoCo position actuator is created), armature from BAM.
      * Each physics step: tau = BAM(q, qd, delayed target) for every joint (bam.model.load_model(...)
        .actuator.compute_control/compute_torque, as models.BamServoBank does through MujocoController),
        written to control.joint_f[dof] (SolverMuJoCo copies Control.joint_f into MuJoCo qfrc_applied).
      * BAM friction (M1: constant + viscous; M6: load/Stribeck dependent) changes every step. Options:
          (a) use_mujoco_cpu: write solver.mj_model.dof_frictionloss / dof_damping directly, exactly as
              BAM's MujocoController does (Newton does not overwrite them unless notify_model_changed);
          (b) Newton-native (also GPU): model.joint_friction / model.joint_damping .assign(...) followed by
              solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES) every step (costly), or
              fold M1's viscous part into joint_damping once and only update frictionloss.
      * Difference to models.BamServoBank: there the torque goes to motor actuators (ctrl) and is not part of
        the implicit integration either, so qfrc_applied is equivalent for implicitfast (motor actuators
        have no velocity derivative). BAM's back-EMF term -kt^2/R*dq is then explicit in both.
      * On the GPU the per-step Python round trip would dominate; BAM's torque law is a few lines and would
        have to be ported to a Warp kernel for parallel worlds.
    """
    return bam_newton_sketch.__doc__


# ================================================================================================
# Report
# ================================================================================================
def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def write_md(report, path):
    L = ["# Newton 1.6.1 vs MuJoCo replay of the SO-101 (newton_check.py)", ""]
    L.append(f"Generated {report['generated']}. Params: `{report['params']['path']}` (variant "
             f"{report['params']['variant']}). Environment: {report['env']}.")
    L.append("")
    L.append("All angles in degrees, sim joint convention, TEST windows only (replay.py window list, W = 4 s, "
             "subset = evenly spaced windows per primary dataset). 'mujoco' = models.py-style MuJoCo built from the "
             "same JSON (mujoco " + report["mujoco_version"] + " in the Newton venv); 'newton' = Newton "
             "ModelBuilder.add_mjcf + SolverMuJoCo.")
    L.append("")
    L.append("## 1. Newton vs MuJoCo (same JSON, same per-step targets)")
    L.append("")
    L.append("| config | windows | max diff arm (deg) | RMS diff arm-mean | per-joint max: " + " / ".join(JOINTS)
             + " | models.py vs JSON-MuJoCo max | Newton CPU s per sim s | MuJoCo CPU s per sim s |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in report["configs"]:
        a = c["agg"]
        nm = a["newton_vs_mujoco"]
        rt = c["runtime"]
        L.append(f"| {c['cfg']['name']} | {a['n_windows']} | {nm['max_deg_arm']:.2e} | {nm['rms_deg_arm_mean']:.2e} | "
                 + " / ".join(f"{nm['max_deg'][j]:.1e}" for j in JOINTS)
                 + f" | {_f(a.get('models_py_vs_json_mujoco_max_deg'), 1) if a.get('models_py_vs_json_mujoco_max_deg') is None else format(a['models_py_vs_json_mujoco_max_deg'], '.1e')}"
                 + f" | {_f(rt.get('newton_cpu_per_sim_s'), 3)} | {_f(rt.get('mujoco_cpu_per_sim_s'), 3)} |")
    L.append("")
    L.append("## 2. Both vs the real arm (pooled RMSE; equal-weight mean over datasets)")
    L.append("")
    L.append("| config | sim | primary6 arm mean | provenance3 arm mean | primary6 per joint: " + " / ".join(JOINTS) + " |")
    L.append("|---|---|---|---|---|")
    for c in report["configs"]:
        for m, e in c["agg"]["vs_real"].items():
            p6, p3 = e.get("primary6"), e.get("provenance3")
            L.append(f"| {c['cfg']['name']} | {m} | {_f(p6 and p6['arm_mean'])} | {_f(p3 and p3['arm_mean'])} | "
                     + (" / ".join(f"{p6['per_joint'][j]:.3f}" for j in JOINTS) if p6 else "-") + " |")
    L.append("")
    L.append("## 3. Cross-check of the venv MuJoCo reference against replay.py's results/windows.csv (mujoco 3.14)")
    L.append("")
    for c in report["configs"]:
        cc = c.get("crosscheck_windows_csv")
        if cc:
            L.append(f"- {c['cfg']['name']}: {cc['n_windows']} windows, max |per-window RMSE difference| per joint = "
                     + ", ".join(f"{j} {v:.1e}" for j, v in cc["max_abs_diff_window_rmse_deg"].items()))
    L.append("")
    L.append("## 4. Model comparison (reference MjModel vs the MjModel Newton builds)")
    L.append("")
    mc = report.get("model_compare")
    if mc:
        L.append("Options (ref, newton): " + "; ".join(f"{k} {v[0]} / {v[1]}" for k, v in mc["opt_ref_vs_newton"].items()))
        L.append("")
        L.append("Bodies (after the inertial sync): " + "; ".join(
            f"{k} dmass {v['dmass']:.1e} dcom {v['dcom_mm']:.1e} mm dI {v['dinertia_rel']:.1e}" for k, v in mc["bodies"].items()))
        L.append("")
        if report.get("model_compare_importer_inertia"):
            L.append("Bodies with Newton's own importer inertials (no sync): " + "; ".join(
                f"{k} dcom {v['dcom_mm']:.2f} mm dI {v['dinertia_rel']:.2f}" for k, v in
                report["model_compare_importer_inertia"]["bodies"].items() if v["dcom_mm"] > 1e-3 or v["dinertia_rel"] > 1e-3))
            L.append("")
    L.append("## 5. Findings")
    L.append("")
    for line in report.get("findings", []):
        L.append("- " + line)
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# ================================================================================================
# Anchor mode (NVIDIA Anchor-Lab whole motions, schema v2)
# ================================================================================================
ANCHOR_MAX_DEG = 0.1        # pass: per joint, max |Newton - MuJoCo| < 0.1 deg ...
ANCHOR_RMS_DEG = 0.01       # ... and RMS < 0.01 deg (at the scored state samples)
MJ_DEFAULT_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2.0]
MJ_DEFAULT_SOLREF = [0.02, 1.0]
PREREG = os.path.join(RESULTS, "anchor_prereg.json")
ANCHOR_CACHE = os.path.join(RESULTS, "anchor_cache")


def _sha256(path):
    import hashlib

    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def anchor_setup(P):
    """Schema-v2 JSON -> (anchor_fit.opts dict, params 6-lists, delay, problems).

    problems lists every JSON setting that anchor_fit.opts()/get_pos_sim() cannot represent (then the reference
    would not be the JSON's model); json_vs_reference() afterwards checks the compiled reference against the JSON."""
    import anchor_fit as AF

    probs = []
    if not P["is_v2"]:
        probs.append(f"schema {P['schema']!r} is not so101_params/v2")

    def same(a, b):
        return (a is None and b is None) or (a is not None and b is not None
                                             and np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-12))

    def fric(e):
        return (int(e["noslip"]), None if e["solimp"] is None else [e["solimp"], e["solimp"], 0.001, 0.5, 2.0],
                None if e["solref"] is None else [e["solref"], 1.0])

    want = (int(P["noslip_iterations"]), P["solimpfriction"], P["solreffriction"])
    match = [k for k, e in AF.FRICTION.items()
             if fric(e)[0] == want[0] and same(fric(e)[1], want[1]) and same(fric(e)[2], want[2])]
    fs = P["friction_setting"]
    if fs in match:
        friction = fs
    elif match:
        friction = match[0]
        probs.append(f"note: friction_setting {fs!r} does not match sim.noslip/solimpfriction/solreffriction; "
                     f"the values match anchor_fit.FRICTION[{friction!r}], which is used")
    else:
        friction = "F0_default"
        probs.append(f"NOT REPRESENTABLE: noslip={want[0]}, solimpfriction={want[1]}, solreffriction={want[2]} is no "
                     f"anchor_fit.FRICTION entry (opts() only knows solimp=[x,x,0.001,0.5,2], solref=[x,1])")
    extra = [n for n in P["bodies"] if n != "camera_mount"]
    if extra:
        probs.append(f"NOT REPRESENTABLE: bodies {extra} (opts() only sets camera_mount)")
    cam = P["bodies"].get("camera_mount")
    camera_g = round(float(cam["mass"]) * 1e3, 9) if cam else AF.CAM_MENAGERIE_G
    if cam and abs(camera_g - AF.CAM_MENAGERIE_G) > 1e-9:
        m_ = camera_g * 1e-3
        if not np.allclose(cam["ipos"], AF.CAM_IPOS, rtol=0, atol=1e-12) or \
                not np.allclose(cam["inertia"], [0.4 * m_ * 0.008 ** 2] * 3, rtol=1e-9, atol=0):
            probs.append("NOT REPRESENTABLE: camera_mount ipos/inertia differ from anchor_fit's point camera "
                         "(CAM_IPOS, 0.4*m*0.008^2)")
    if P["integrator"] is not None and str(P["integrator"]).lower() != "implicitfast":
        probs.append(f"NOT REPRESENTABLE: integrator {P['integrator']!r} (models.py/so101.xml use implicitfast)")
    params = {k: [float(P["joints"][j][k]) for j in JOINTS] for k in ("kp", "kv", "damping", "frictionloss", "armature")}
    params["forcerange"] = [float(P["joints"][j]["forcerange"][1]) for j in JOINTS]
    if any(abs(P["joints"][j]["forcerange"][0] + P["joints"][j]["forcerange"][1]) > 1e-12 for j in JOINTS):
        probs.append("NOT REPRESENTABLE: asymmetric forcerange (get_pos_sim uses +-f)")
    delay = float(P["delay_s"]) if P["delay_s"] is not None else float(P["joints"][JOINTS[0]]["delay_s"])
    if any(abs(P["joints"][j]["delay_s"] - delay) > 1e-12 for j in JOINTS):
        probs.append("NOT REPRESENTABLE: per-joint delay_s differ from delay_s (event_targets takes one delay)")
    if any(P["joints"][j]["deadband_rad"] > 0 for j in JOINTS):
        probs.append("NOT REPRESENTABLE: deadband_rad > 0 (the fit has no dead band)")
    if P["per_dataset"]:
        probs.append("note: per_dataset entries are ignored in anchor mode")
    if P["ctrllimited"]:
        probs.append("NOT REPRESENTABLE: actuator ctrl limits on (apply_opts switches them off)")
    o = AF.opts(friction=friction, dt=P["dt"], camera_g=camera_g,
                iterations=int(P["iterations"]) if P["iterations"] is not None else 10,
                ls_iterations=int(P["ls_iterations"]) if P["ls_iterations"] is not None else 20,
                jaw_hi_deg=round(math.degrees(P["joints"]["gripper"]["range_rad"][1]), 9))
    return o, params, delay, probs


def json_vs_reference(P, s):
    """Every v2 value in the JSON vs the compiled reference (anchor_fit.get_pos_sim): proves the opts() mapping
    rebuilt the JSON's model. Returns a list of mismatch strings (empty = identical)."""
    m, v, a = s.model, s.vadr, s.aid
    bad = []

    def chk(label, ref, want, tol=1e-12):
        ref, want = np.asarray(ref, float), np.asarray(want, float)
        d = float(np.max(np.abs(ref - want))) if ref.size else 0.0
        if not d <= tol * max(1.0, float(np.max(np.abs(want)))):
            bad.append(f"{label}: reference {ref.tolist()} vs JSON {want.tolist()} (max diff {d:.3e})")

    J = P["joints"]
    chk("kp", m.actuator_gainprm[a, 0], [J[j]["kp"] for j in JOINTS])
    chk("kv", -m.actuator_biasprm[a, 2], [J[j]["kv"] for j in JOINTS])
    chk("forcerange", m.actuator_forcerange[a], [J[j]["forcerange"] for j in JOINTS])
    chk("damping", m.dof_damping[v], [J[j]["damping"] for j in JOINTS])
    chk("frictionloss", m.dof_frictionloss[v], [J[j]["frictionloss"] for j in JOINTS])
    chk("armature", m.dof_armature[v], [J[j]["armature"] for j in JOINTS])
    chk("jnt_range", m.jnt_range[[m.joint(j).id for j in JOINTS]], [J[j]["range_rad"] for j in JOINTS])
    chk("dof_solimp", m.dof_solimp[v], [P["solimpfriction"] or MJ_DEFAULT_SOLIMP] * 6)
    chk("dof_solref", m.dof_solref[v], [P["solreffriction"] or MJ_DEFAULT_SOLREF] * 6)
    chk("opt.timestep", [m.opt.timestep], [P["dt"]])
    chk("opt.noslip_iterations", [m.opt.noslip_iterations], [P["noslip_iterations"]])
    if P["iterations"] is not None:
        chk("opt.iterations", [m.opt.iterations], [P["iterations"]])
    if P["ls_iterations"] is not None:
        chk("opt.ls_iterations", [m.opt.ls_iterations], [P["ls_iterations"]])
    chk("actuator_ctrllimited", m.actuator_ctrllimited[a], [0 if not P["ctrllimited"] else 1] * 6)
    inert = mujoco_body_inertials(m)
    for name, bj in P["bodies"].items():
        mass, com, I = inert[name]
        chk(f"{name}.mass", [mass], [bj["mass"]])
        chk(f"{name}.ipos", com, bj["ipos"])
        chk(f"{name}.inertia(full)", I, np.diag(bj["inertia"]), tol=1e-9)
    return bad


def _interp(Q, ts, t_start, dt):
    """anchor_fit.eval_item's linear interpolation of a per-step trajectory at the state times ts."""
    x = (ts - t_start) / dt
    n0 = np.clip(np.floor(x).astype(int), 0, len(Q) - 2)
    fr = (x - n0)[:, None]
    return Q[n0] * (1 - fr) + Q[n0 + 1] * fr


def anchor_motion_setup(AF, d, name, dt, delay, seconds=None):
    """anchor_fit.eval_item for a ("whole", name) item up to the targets U; seconds limits the span (warp check)."""
    t_req, t_end = AF.item_span(d, ("whole", name))
    i0, t_start, q0, v0 = AF.start_state(d, t_req)
    if seconds is not None:
        t_end = min(t_end, t_start + seconds)
    i_lo = int(np.searchsorted(d["t_state"], t_start + AF.WARM))
    i_hi = int(np.searchsorted(d["t_state"], t_end, side="right"))
    nsteps = int(math.ceil((t_end - t_start) / dt)) + 1
    U = AF.event_targets(d["t_cmd"], d["qc"], float(delay), t_start, nsteps, dt)
    return {"t_start": t_start, "t_end": t_end, "q0": q0, "v0": v0, "i_lo": i_lo, "i_hi": i_hi, "U": U,
            "ts": d["t_state"][i_lo:i_hi], "real": d["q"][i_lo:i_hi]}


def _jd(arr, nd=6):
    return {j: round(float(x), nd) for j, x in zip(JOINTS, arr)}


def run_anchor_job(job):
    """One worker: build the reference (anchor_fit.get_pos_sim) and Newton once, assert the models match, replay
    each motion through both."""
    import anchor_data as AD
    import anchor_fit as AF

    P, o, params, delay = job["P"], job["opts"], job["params"], job["delay"]
    dt = o["dt"]
    ref = AF.get_pos_sim(o, params)
    t = time.perf_counter()
    na = NewtonArm(P, dt, "sim", dataset_overrides(P, ""), backend="mujoco_cpu", inertia="json",
                   update_data_interval=job["udi"])
    build_s = time.perf_counter() - t
    match = model_match(ref.model, na.mj_model())
    if job["assert"] and not match["ok"]:
        bad = {k: v for k, v in match["items"].items() if v["required"] and not v["ok"]}
        raise AssertionError(f"Newton mj_model != reference MjModel: {json.dumps(bad, default=float)}")
    fp = None
    if job.get("final_preds"):
        fp = np.load(job["final_preds"])
    recs = {}
    for name in job["motions"]:
        d = AD.load(name)
        S = anchor_motion_setup(AF, d, name, dt, delay)
        c = time.process_time()
        Qm = AF.rollout_q(ref, S["q0"], S["v0"], S["U"])
        cpu_m = time.process_time() - c
        c = time.process_time()
        Qn = na.run_steps(S["q0"], S["v0"], S["U"])
        cpu_n = time.process_time() - c
        pm = _interp(Qm, S["ts"], S["t_start"], dt)
        pn = _interp(Qn, S["ts"], S["t_start"], dt)
        em = np.nan_to_num(np.degrees(pm - S["real"]), nan=1e3, posinf=1e3, neginf=-1e3)
        en = np.nan_to_num(np.degrees(pn - S["real"]), nan=1e3, posinf=1e3, neginf=-1e3)
        dnm = np.nan_to_num(np.abs(np.degrees(pn - pm)), nan=1e3, posinf=1e3, neginf=1e3)
        dstep = np.nan_to_num(np.abs(np.degrees(Qn - Qm)), nan=1e3, posinf=1e3, neginf=1e3)
        n = len(S["ts"])
        mx, rms = dnm.max(0), np.sqrt((dnm ** 2).mean(0))
        r_m, r_n = np.sqrt((em ** 2).mean(0)), np.sqrt((en ** 2).mean(0))
        k = int(dnm.max(1).argmax())
        rec = {"motion": name, "n_samples": n, "n_steps": int(len(S["U"])), "t_start": S["t_start"],
               "t_end": S["t_end"], "i_lo": S["i_lo"], "i_hi": S["i_hi"],
               "newton_vs_mujoco": {"max_deg": _jd(mx, 8), "rms_deg": _jd(rms, 8),
                                    "max_deg_all_steps": _jd(dstep.max(0), 8),
                                    "worst_sample_t_s": float(S["ts"][k] - S["t_start"])},
               "sse_newton_vs_mujoco": (dnm ** 2).sum(0).tolist(),
               "pass": bool(np.all(mx < ANCHOR_MAX_DEG) and np.all(rms < ANCHOR_RMS_DEG)),
               "vs_real_rmse_deg": {"mujoco": _jd(r_m, 4) | {"arm_mean": round(float(r_m[:ARM].mean()), 4)},
                                    "newton": _jd(r_n, 4) | {"arm_mean": round(float(r_n[:ARM].mean()), 4)}},
               "cpu_s": {"mujoco": cpu_m, "newton": cpu_n},
               "newton_ms_per_step": 1e3 * cpu_n / len(S["U"])}
        # the fit's own scoring path on the same motion: must give the same MuJoCo prediction and SSE
        ev = AF.eval_item(AF.cfg_pos(o, params, delay), ("whole", name), want_pred=True)
        rec["eval_item_crosscheck"] = {
            "same_span": [ev["i_lo"], ev["i_hi"]] == [S["i_lo"], S["i_hi"]],
            "max_abs_pred_deg": float(np.max(np.abs(ev["pred_deg"].astype(np.float64) - np.degrees(pm)))),
            "max_rel_sse": float(np.max(np.abs(np.array(ev["sse"]) - (em ** 2).sum(0)) / np.maximum((em ** 2).sum(0), 1e-30)))}
        if fp is not None:
            rec["final_preds_compare"] = compare_final_pred(fp, job["final_spans"], name, S, pm)
        recs[name] = rec
    return {"recs": recs, "match": match, "build_newton_s": build_s, "pid": os.getpid()}


def compare_final_pred(fp, spans, name, S, pm):
    """venv MuJoCo 3.12 prediction vs the main env's (MuJoCo 3.14) anchor_fit stage `final` prediction."""
    key = f"fitted__{name}"
    if key not in fp.files or name not in spans:
        return {"available": False, "why": f"{key} not in final_preds.npz or no span in final.json"}
    i_lo, i_hi, t0 = spans[name]
    if [int(i_lo), int(i_hi)] != [S["i_lo"], S["i_hi"]] or abs(float(t0) - S["t_start"]) > 1e-12:
        return {"available": False, "why": f"span mismatch: final {spans[name]} vs here "
                                           f"{[S['i_lo'], S['i_hi'], S['t_start']]}"}
    dd = np.abs(fp[key].astype(np.float64) - np.degrees(pm))
    return {"available": True, "max_deg": _jd(dd.max(0), 8), "rms_deg": _jd(np.sqrt((dd ** 2).mean(0)), 8),
            "note": "final_preds are float32 deg (storage step ~1e-5 deg at 100 deg)"}


def warp_cpu_check(P, o, params, delay, motion, seconds, udi):
    """Newton's MuJoCo-Warp path (use_mujoco_cpu=False) on Warp's CPU device for one short span of a motion,
    against Newton's MuJoCo-C path and the reference. Returns numbers, or the exact error."""
    import traceback

    import anchor_data as AD
    import anchor_fit as AF
    import warp as wp

    out = {"motion": motion, "seconds": seconds, "warp_device": str(wp.get_device()),
           "cuda_available": bool(wp.is_cuda_available())}
    dt = o["dt"]
    d = AD.load(motion)
    S = anchor_motion_setup(AF, d, motion, dt, delay, seconds=seconds)
    ref = AF.get_pos_sim(o, params)
    Qm = AF.rollout_q(ref, S["q0"], S["v0"], S["U"])
    ov = dataset_overrides(P, "")
    nc = NewtonArm(P, dt, "sim", ov, backend="mujoco_cpu", inertia="json", update_data_interval=udi)
    Qc = nc.run_steps(S["q0"], S["v0"], S["U"])
    out["n_steps"] = int(len(S["U"]))
    try:
        t = time.perf_counter()
        nw = NewtonArm(P, dt, "sim", ov, backend="mujoco_warp", inertia="json", update_data_interval=udi)
        out["build_s"] = time.perf_counter() - t
        out["model_match_cpu_mjmodel"] = {k: v for k, v in model_match(ref.model, nw.mj_model())["items"].items()
                                          if v["required"]}
        mw = nw.solver.mjw_model
        vw = [nw.mj_model().jnt_dofadr[_by_suffix(nw.mj_model(), "joint", j)] for j in JOINTS]
        out["mjw_model"] = {
            "dof_solimp_max_abs_vs_ref": float(np.abs(mw.dof_solimp.numpy()[0][vw] - ref.model.dof_solimp[ref.vadr]).max()),
            "dof_solref_max_abs_vs_ref": float(np.abs(mw.dof_solref.numpy()[0][vw] - ref.model.dof_solref[ref.vadr]).max()),
            "dof_frictionloss_max_abs_vs_ref": float(np.abs(mw.dof_frictionloss.numpy()[0][vw]
                                                            - ref.model.dof_frictionloss[ref.vadr]).max())}
        for f in ("iterations", "ls_iterations", "tolerance", "ls_tolerance", "noslip_iterations"):
            try:
                val = getattr(mw.opt, f)
                out["mjw_model"]["opt." + f] = float(val.numpy().ravel()[0]) if hasattr(val, "numpy") else float(val)
            except Exception as e:  # field absent in this mujoco_warp version
                out["mjw_model"]["opt." + f] = f"n/a ({type(e).__name__})"
        t = time.perf_counter()
        nw.run_steps(S["q0"], S["v0"], S["U"][:2])          # first call compiles/loads kernels
        out["first_steps_s"] = time.perf_counter() - t
        t = time.perf_counter()
        Qw = nw.run_steps(S["q0"], S["v0"], S["U"])
        out["run_s"] = time.perf_counter() - t
        out["ms_per_step_cpu_device"] = 1e3 * out["run_s"] / len(S["U"])
        dw_c = np.abs(np.degrees(Qw - Qc))
        dw_m = np.abs(np.degrees(Qw - Qm))
        dc_m = np.abs(np.degrees(Qc - Qm))
        out["ok"] = bool(np.all(np.isfinite(Qw)))
        out["warp_vs_newton_cpu"] = {"max_deg": _jd(dw_c.max(0), 8), "rms_deg": _jd(np.sqrt((dw_c ** 2).mean(0)), 8)}
        out["warp_vs_mujoco"] = {"max_deg": _jd(dw_m.max(0), 8), "rms_deg": _jd(np.sqrt((dw_m ** 2).mean(0)), 8)}
        out["newton_cpu_vs_mujoco"] = {"max_deg": _jd(dc_m.max(0), 8), "rms_deg": _jd(np.sqrt((dc_m ** 2).mean(0)), 8)}
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]
    return out


def run_warp_job(job):
    """--warp-whole worker: Newton's MuJoCo-Warp path (on wp.get_device(): cuda:0 on the L4) over whole motions,
    scored against the real arm exactly like run_anchor_job, next to the reference MuJoCo and Newton's CPU path."""
    import traceback

    import anchor_data as AD
    import anchor_fit as AF
    import warp as wp

    P, o, params, delay = job["P"], job["opts"], job["params"], job["delay"]
    dt = o["dt"]
    out = {"device": str(wp.get_device()), "cuda_available": bool(wp.is_cuda_available()), "recs": {}}
    try:
        ref = AF.get_pos_sim(o, params)
        ov = dataset_overrides(P, "")
        nc = NewtonArm(P, dt, "sim", ov, backend="mujoco_cpu", inertia="json", update_data_interval=job["udi"])
        t = time.perf_counter()
        nw = NewtonArm(P, dt, "sim", ov, backend="mujoco_warp", inertia="json", update_data_interval=job["udi"])
        out["build_s"] = time.perf_counter() - t
        mw = nw.solver.mjw_model
        vw = [nw.mj_model().jnt_dofadr[_by_suffix(nw.mj_model(), "joint", j)] for j in JOINTS]
        out["mjw_model"] = {
            "dof_solimp_max_abs_vs_ref": float(np.abs(mw.dof_solimp.numpy()[0][vw] - ref.model.dof_solimp[ref.vadr]).max()),
            "dof_solref_max_abs_vs_ref": float(np.abs(mw.dof_solref.numpy()[0][vw] - ref.model.dof_solref[ref.vadr]).max()),
            "dof_frictionloss_max_abs_vs_ref": float(np.abs(mw.dof_frictionloss.numpy()[0][vw]
                                                            - ref.model.dof_frictionloss[ref.vadr]).max()),
            "dof_damping_max_abs_vs_ref": float(np.abs(mw.dof_damping.numpy()[0][vw] - ref.model.dof_damping[ref.vadr]).max()),
            "dof_armature_max_abs_vs_ref": float(np.abs(mw.dof_armature.numpy()[0][vw] - ref.model.dof_armature[ref.vadr]).max())}
        for f in ("iterations", "ls_iterations", "tolerance", "ls_tolerance"):
            try:
                val = getattr(mw.opt, f)
                out["mjw_model"]["opt." + f] = float(val.numpy().ravel()[0]) if hasattr(val, "numpy") else float(val)
            except Exception as e:
                out["mjw_model"]["opt." + f] = f"n/a ({type(e).__name__})"
        nw.run_steps(np.zeros(6), np.zeros(6), np.zeros((2, 6)))     # compile / load kernels once
        for name in job["motions"]:
            d = AD.load(name)
            S = anchor_motion_setup(AF, d, name, dt, delay, seconds=job.get("seconds"))
            Qm = AF.rollout_q(ref, S["q0"], S["v0"], S["U"])
            Qc = nc.run_steps(S["q0"], S["v0"], S["U"])
            t = time.perf_counter()
            Qw = nw.run_steps(S["q0"], S["v0"], S["U"])
            run_s = time.perf_counter() - t
            finite = bool(np.all(np.isfinite(Qw)))
            pr = {}
            for k, Q in (("warp", Qw), ("newton_cpu", Qc), ("mujoco", Qm)):
                e = np.nan_to_num(np.degrees(_interp(Q, S["ts"], S["t_start"], dt) - S["real"]),
                                  nan=1e3, posinf=1e3, neginf=-1e3)
                r = np.sqrt((e ** 2).mean(0))
                pr[k] = _jd(r, 4) | {"arm_mean": round(float(r[:ARM].mean()), 4)}
            dwm = np.nan_to_num(np.abs(np.degrees(Qw - Qm)), nan=1e3, posinf=1e3, neginf=1e3)
            out["recs"][name] = {
                "n_steps": int(len(S["U"])), "seconds": float(S["t_end"] - S["t_start"]), "finite": finite,
                "ms_per_step": 1e3 * run_s / len(S["U"]), "vs_real_rmse_deg": pr,
                "warp_minus_mujoco_arm_rmse": round(pr["warp"]["arm_mean"] - pr["mujoco"]["arm_mean"], 5),
                "warp_vs_mujoco_max_deg_all_steps": _jd(dwm.max(0), 6),
                "warp_vs_mujoco_rms_deg_all_steps": _jd(np.sqrt((dwm ** 2).mean(0)), 6)}
            print(f"[warp {out['device']}] {name}: arm vs real warp {pr['warp']['arm_mean']:.4f} / mujoco "
                  f"{pr['mujoco']['arm_mean']:.4f} / newton-cpu {pr['newton_cpu']['arm_mean']:.4f}; "
                  f"{out['recs'][name]['ms_per_step']:.2f} ms/step; finite {finite}", flush=True)
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]
    return out


def write_anchor_md(R, path):
    L = ["# Newton 1.6.1 vs MuJoCo replay of NVIDIA Anchor-Lab motions (newton_check.py --anchor)", ""]
    p = R["params"]
    L.append(f"Generated {R['generated']}. Params `{p['path']}` (schema {p['schema']}, variant {p['variant']}, "
             f"sha256 `{p['sha256'][:16]}`), friction `{R['anchor_opts']['friction']}`, delay {1e3 * p['delay_s']:.1f} ms, "
             f"dt {R['anchor_opts']['dt']} s. {R['env']}.")
    L.append("")
    L.append("Reference = `anchor_fit.get_pos_sim(opts, params)` (the fit's MuJoCo model) in the Newton venv; "
             "Newton = `ModelBuilder.add_mjcf` + v2 parameters + `SolverMuJoCo(use_mujoco_cpu=True)`. Same per-step "
             "targets (`anchor_fit.event_targets`), same start state, scored at the measured state times after 0.5 s "
             "by linear interpolation (as `anchor_fit.eval_item`). Angles in degrees.")
    L.append("")
    L.append(f"Pass: every joint max < {ANCHOR_MAX_DEG} deg and RMS < {ANCHOR_RMS_DEG} deg (Newton vs MuJoCo).")
    L.append("")
    L.append("## 1. Newton vs MuJoCo per motion (deg, scored samples)")
    L.append("")
    L.append("| motion | pass | max: " + " / ".join(JOINTS) + " | RMS: " + " / ".join(JOINTS)
             + " | max over all steps (arm) | Newton ms/step |")
    L.append("|---|---|---|---|---|---|")
    for n, r in R["motions"].items():
        x = r["newton_vs_mujoco"]
        L.append(f"| {n} | {'yes' if r['pass'] else 'NO'} | " + " / ".join(f"{x['max_deg'][j]:.1e}" for j in JOINTS)
                 + " | " + " / ".join(f"{x['rms_deg'][j]:.1e}" for j in JOINTS)
                 + f" | {max(list(x['max_deg_all_steps'].values())[:ARM]):.1e} | {r['newton_ms_per_step']:.3f} |")
    s = R["summary"]
    L.append(f"| **all** | {'yes' if s['all_pass'] else 'NO'} | "
             + " / ".join(f"{s['newton_vs_mujoco_max_deg'][j]:.1e}" for j in JOINTS) + " | "
             + " / ".join(f"{s['newton_vs_mujoco_rms_deg'][j]:.1e}" for j in JOINTS) + " | | |")
    L.append("")
    L.append("## 2. Each sim vs the real arm (RMSE deg; arm = mean of 5 arm joints)")
    L.append("")
    L.append("| motion | sim | arm | " + " | ".join(JOINTS) + " |")
    L.append("|---|---|---|" + "---|" * 6)
    for n, r in R["motions"].items():
        for simn in ("mujoco", "newton"):
            e = r["vs_real_rmse_deg"][simn]
            L.append(f"| {n} | {simn} | {e['arm_mean']:.3f} | " + " | ".join(f"{e[j]:.3f}" for j in JOINTS) + " |")
    for simn in ("mujoco", "newton"):
        e = s["vs_real_mean_over_motions"][simn]
        L.append(f"| **mean** | {simn} | {e['arm_mean']:.3f} | " + " | ".join(f"{e[j]:.3f}" for j in JOINTS) + " |")
    L.append("")
    L.append("## 3. Model checks")
    L.append("")
    L.append(f"JSON -> reference (opts() mapping reproduces every v2 value): "
             + ("identical" if not R["json_vs_reference"] else "; ".join(R["json_vs_reference"])))
    L.append("")
    L.append("Not representable via `anchor_fit.opts()`: " + ("none" if not R["not_representable"]
                                                              else "; ".join(R["not_representable"])))
    L.append("")
    mm = R["model_match"]
    L.append(f"Newton mj_model vs reference MjModel (asserted, rtol {mm['rtol']:g} for floats, equality for "
             f"integers): required items {'all match' if mm['ok'] else 'MISMATCH'}; all items "
             f"{'match' if mm['all_ok'] else 'see below'}.")
    L.append("")
    L.append("| item | required | max abs diff | relative | ok |")
    L.append("|---|---|---|---|---|")
    for k, v in mm["items"].items():
        L.append(f"| {k} | {'yes' if v['required'] else ''} | {v['max_abs']:.3e} | {v['rel']:.2e} | {'yes' if v['ok'] else 'NO'} |")
    L.append("")
    X = [r["eval_item_crosscheck"] for r in R["motions"].values()]
    L.append(f"Reference prediction vs `anchor_fit.eval_item` (the fit's own scoring call): same spans "
             f"{all(x['same_span'] for x in X)}, max |pred diff| {max(x['max_abs_pred_deg'] for x in X):.1e} deg "
             f"(eval_item stores float32), max relative SSE diff {max(x['max_rel_sse'] for x in X):.1e}.")
    L.append("")
    w = R.get("warp_cpu_check")
    L.append("## 4. GPU path (MuJoCo-Warp, use_mujoco_cpu=False) on Warp's CPU device")
    L.append("")
    if not w:
        L.append("Not run (--warp-check 0).")
    elif w.get("ok"):
        L.append(f"Ran: {w['motion']} first {w['seconds']} s ({w['n_steps']} steps), device {w['warp_device']}, "
                 f"{w['ms_per_step_cpu_device']:.2f} ms/step. Max |warp - Newton CPU| per joint: "
                 + ", ".join(f"{j} {v:.1e}" for j, v in w["warp_vs_newton_cpu"]["max_deg"].items())
                 + "; max |warp - MuJoCo|: " + ", ".join(f"{j} {v:.1e}" for j, v in w["warp_vs_mujoco"]["max_deg"].items())
                 + f". mjw_model: {json.dumps(w['mjw_model'])}. Must still be checked on the L4 (CUDA).")
    else:
        L.append(f"Failed: `{w.get('error')}`. Must be checked on the L4. Traceback tail: "
                 + " | ".join(w.get("traceback_tail", [])))
    L.append("")
    ww = R.get("warp_whole")
    if ww:
        L.append(f"## 4b. GPU path (MuJoCo-Warp) over whole motions, device `{ww['device']}`")
        L.append("")
        L.append(f"ok {ww['ok']}, all finite {ww['all_finite']}, {ww['procs']} procs, {ww['wall_s']:.0f} s wall. "
                 f"Pass: per motion |warp - MuJoCo| arm RMSE vs real <= 0.03 deg: {ww['pass_per_motion_0p03']} "
                 f"(max {ww['max_abs_motion_diff_deg']}); mean within 0.01 deg: {ww['pass_mean_0p01']} "
                 f"(mean arm vs real {ww['mean_arm_vs_real']}). mjw_model: {json.dumps(ww['mjw_model'])}")
        if ww["errors"]:
            L.append(f"Errors: {json.dumps(ww['errors'])}")
        L.append("")
        L.append("| motion | warp arm vs real | MuJoCo arm vs real | Newton-CPU arm vs real | warp - MuJoCo | "
                 "max |warp-MuJoCo| any arm joint, all steps | ms/step | finite |")
        L.append("|---|---|---|---|---|---|---|---|")
        for n, r in ww["motions"].items():
            v = r["vs_real_rmse_deg"]
            L.append(f"| {n} | {v['warp']['arm_mean']:.4f} | {v['mujoco']['arm_mean']:.4f} | "
                     f"{v['newton_cpu']['arm_mean']:.4f} | {r['warp_minus_mujoco_arm_rmse']:+.4f} | "
                     f"{max(list(r['warp_vs_mujoco_max_deg_all_steps'].values())[:ARM]):.2e} | "
                     f"{r['ms_per_step']:.2f} | {r['finite']} |")
        L.append("")
    fpc = R.get("final_preds_compare")
    L.append("## 5. MuJoCo 3.12 (venv) vs 3.14 (anchor_fit stage final)")
    L.append("")
    if not fpc:
        L.append("Not requested (--compare-final-preds).")
    else:
        L.append(fpc["status"])
        for n, r in R["motions"].items():
            c = r.get("final_preds_compare")
            if c and c.get("available"):
                L.append(f"- {n}: max " + ", ".join(f"{j} {v:.1e}" for j, v in c["max_deg"].items())
                         + "; RMS " + ", ".join(f"{j} {v:.1e}" for j, v in c["rms_deg"].items()))
            elif c:
                L.append(f"- {n}: {c['why']}")
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main_anchor(args):
    import anchor_data as AD
    import anchor_fit as AF
    import mujoco
    import newton
    import warp as wp

    t_wall = time.perf_counter()
    ppath = args.params or (FITTED_JSON if os.path.isfile(FITTED_JSON) else None)
    if ppath is None:
        raise SystemExit("--anchor needs a schema-v2 --params JSON (so101_fitted.json does not exist yet)")
    P = load_params(ppath)
    o, params, delay, probs = anchor_setup(P)
    if any(p.startswith("NOT REPRESENTABLE") for p in probs) and not args.allow_unrepresentable:
        raise SystemExit("JSON not representable by anchor_fit.opts():\n  " + "\n  ".join(probs))
    sp = AD.split()
    motions = args.anchor_motions or sp["heldout"]
    held = [m_ for m_ in motions if m_ in sp["heldout"] or m_.startswith("heldout-")]
    if held and not os.path.isfile(PREREG) and not args.allow_heldout_before_prereg:
        raise SystemExit(f"refusing to load held-out motions {held}: {os.path.relpath(PREREG, HERE)} does not exist "
                         "yet (the held-out set is pre-registered and scored once, after `anchor_fit.py freeze`). "
                         "Develop with --anchor-motions train-step-response train-velocity-ramp.")
    print(f"params {ppath}: schema {P['schema']}, friction {o['friction']}, delay {1e3 * delay:.1f} ms, dt {o['dt']}; "
          f"opts {o}; problems {probs or 'none'}")
    ref = AF.get_pos_sim(o, params)
    jvr = json_vs_reference(P, ref)
    print("JSON vs reference:", "identical" if not jvr else jvr)

    final_preds, final_spans, fp_status = None, None, None
    if args.compare_final_preds:
        fpp, fjp = os.path.join(ANCHOR_CACHE, "final_preds.npz"), os.path.join(ANCHOR_CACHE, "final.json")
        if os.path.isfile(fpp) and os.path.isfile(fjp):
            with open(fjp, encoding="utf-8") as fh:
                fin = json.load(fh)
            final_spans = fin["models"]["fitted"].get("spans", {})
            final_preds = fpp
            same = fin.get("sha256_fitted") == _sha256(ppath)
            fp_status = (f"{os.path.relpath(fpp, HERE)} found; params sha256 "
                         + ("equals" if same else "DIFFERS FROM") + " final.json sha256_fitted"
                         + ("" if same else " (not like-for-like: the comparison uses a different model)"))
        else:
            fp_status = f"{os.path.relpath(fpp, HERE)} not found (anchor_fit.py final has not run): skipped"
        print(fp_status)

    procs = max(1, min(args.procs, len(motions)))
    chunks = [motions[i::procs] for i in range(procs)]
    jobs = [{"P": P, "opts": o, "params": params, "delay": delay, "motions": c, "udi": args.update_data_interval,
             "assert": not args.no_assert, "final_preds": final_preds, "final_spans": final_spans} for c in chunks if c]
    t = time.perf_counter()
    if len(jobs) == 1:
        res = [run_anchor_job(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            res = pool.map(run_anchor_job, jobs)
    wall = time.perf_counter() - t
    recs = {}
    for r in res:
        recs.update(r["recs"])
    recs = {n: recs[n] for n in motions}
    match = res[0]["match"]
    print(f"model match (required items): {'OK' if match['ok'] else 'MISMATCH'}; "
          + "; ".join(f"{k} {v['max_abs']:.2e}" for k, v in match["items"].items() if v["required"]))

    ntot = sum(r["n_samples"] for r in recs.values())
    mx = np.max([[r["newton_vs_mujoco"]["max_deg"][j] for j in JOINTS] for r in recs.values()], 0)
    rms = np.sqrt(np.sum([r["sse_newton_vs_mujoco"] for r in recs.values()], 0) / ntot)
    vr = {}
    for simn in ("mujoco", "newton"):
        a = np.mean([[r["vs_real_rmse_deg"][simn][j] for j in JOINTS] for r in recs.values()], 0)
        vr[simn] = _jd(a, 4) | {"arm_mean": round(float(a[:ARM].mean()), 4)}
    summary = {"n_motions": len(recs), "n_samples": ntot, "all_pass": all(r["pass"] for r in recs.values()),
               "newton_vs_mujoco_max_deg": _jd(mx, 8), "newton_vs_mujoco_rms_deg": _jd(rms, 8),
               "vs_real_mean_over_motions": vr}
    for n, r in recs.items():
        x = r["newton_vs_mujoco"]
        print(f"[{n}] pass {r['pass']}: N-M max {max(x['max_deg'].values()):.2e} deg, RMS {max(x['rms_deg'].values()):.2e} "
              f"(worst joint); vs real arm MuJoCo {r['vs_real_rmse_deg']['mujoco']['arm_mean']:.3f} Newton "
              f"{r['vs_real_rmse_deg']['newton']['arm_mean']:.3f}; Newton {r['newton_ms_per_step']:.3f} ms/step; "
              f"eval_item x-check {r['eval_item_crosscheck']}")
    warp = None
    warp_whole = None
    if args.warp_whole:
        wprocs = max(1, min(args.procs, len(motions)))
        wjobs = [{"P": P, "opts": o, "params": params, "delay": delay, "motions": motions[i::wprocs],
                  "udi": args.update_data_interval, "seconds": (args.warp_check or None)} for i in range(wprocs)]
        t = time.perf_counter()
        if len(wjobs) == 1:
            wres = [run_warp_job(wjobs[0])]
        else:
            with mp.get_context("spawn").Pool(len(wjobs)) as pool:
                wres = pool.map(run_warp_job, wjobs)
        wrecs = {}
        for r in wres:
            wrecs.update(r["recs"])
        wrecs = {n: wrecs[n] for n in motions if n in wrecs}
        d_ = [abs(r["warp_minus_mujoco_arm_rmse"]) for r in wrecs.values()]
        mean = {k: round(float(np.mean([r["vs_real_rmse_deg"][k]["arm_mean"] for r in wrecs.values()])), 4)
                for k in ("warp", "mujoco", "newton_cpu")} if wrecs else {}
        warp_whole = {"ok": all(r["ok"] for r in wres) and len(wrecs) == len(motions),
                      "errors": [{k: r.get(k) for k in ("error", "traceback_tail")} for r in wres if not r["ok"]],
                      "device": wres[0]["device"], "mjw_model": wres[0].get("mjw_model"),
                      "procs": len(wjobs), "wall_s": time.perf_counter() - t, "motions": wrecs,
                      "mean_arm_vs_real": mean,
                      "max_abs_motion_diff_deg": max(d_) if d_ else None,
                      "pass_per_motion_0p03": bool(d_) and max(d_) <= 0.03,
                      "pass_mean_0p01": bool(mean) and abs(mean["warp"] - mean["mujoco"]) <= 0.01,
                      "all_finite": all(r["finite"] for r in wrecs.values())}
        print("warp whole-motion check:", json.dumps({k: v for k, v in warp_whole.items() if k != "motions"},
                                                     default=float)[:2000])
    elif args.warp_check > 0:
        warp = warp_cpu_check(P, o, params, delay, motions[0], args.warp_check, args.update_data_interval)
        print("warp CPU-device check:", json.dumps(warp, default=float)[:1500])
    R = {"generated": time.strftime("%Y-%m-%d %H:%M"),
         "env": f"python {sys.version.split()[0]}, newton {newton.__version__}, warp {wp.__version__}, "
                f"mujoco {mujoco.__version__}, CPU only",
         "params": {"path": P["path"], "sha256": _sha256(ppath), "schema": P["schema"], "variant": P["variant"],
                    "friction_setting": P["friction_setting"], "delay_s": delay, "dt": P["dt"]},
         "anchor_opts": o, "anchor_params": params, "not_representable": probs, "json_vs_reference": jvr,
         "model_match": match, "criteria": {"max_deg": ANCHOR_MAX_DEG, "rms_deg": ANCHOR_RMS_DEG},
         "update_data_interval": args.update_data_interval, "motions": recs, "summary": summary,
         "warp_cpu_check": warp,
         "warp_whole": warp_whole,
         "final_preds_compare": {"status": fp_status} if args.compare_final_preds else None,
         "runtime": {"wall_s": time.perf_counter() - t_wall, "replay_wall_s": wall, "procs": len(jobs),
                     "newton_build_s": float(np.mean([r["build_newton_s"] for r in res]))}}
    out_dir = args.out_dir or RESULTS
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, args.out or "newton_check_anchor")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=float)
    write_anchor_md(R, base + ".md")
    print(f"all pass: {summary['all_pass']}; wrote {base}.json and .md in {R['runtime']['wall_s']:.0f} s")


def make_v2_json(out, friction, delay, note="dev"):
    """A schema-v2 JSON from anchor_fit.fitted_json(opts(friction), base_params(), delay, note) (development)."""
    import anchor_fit as AF

    js = AF.fitted_json(AF.opts(friction=friction), AF.base_params(), float(delay), note)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(js, fh, indent=1)
    return out


# ================================================================================================
# Main
# ================================================================================================
def default_suite(dt):
    base = {"dt": dt, "W": 4, "warm": 0, "backend": "mujoco_cpu", "inertia": "mujoco", "update_data_interval": 1}
    return [
        base | {"name": "B_sim_warm0", "limits": "sim", "models_py": True},
        base | {"name": "B_wide_warm0", "limits": "wide", "models_py": True},
        base | {"name": "B_wide_warm8", "limits": "wide", "warm": 8},
        base | {"name": "B_wide_noslip0_both", "limits": "wide", "noslip": 0, "alt_ref": {"noslip": 1}},
        base | {"name": "B_wide_importer_inertia", "limits": "wide", "inertia": "newton"},
        base | {"name": "B_wide_update_interval0", "limits": "wide", "update_data_interval": 0},
        # Exercises the optional JSON paths (per-joint delay + deadband, per-dataset delta/range/offset).
        # The values are ARBITRARY test values, not a model: only its Newton-vs-MuJoCo column is meaningful.
        base | {"name": "PATHTEST_delay_deadband_perdataset", "limits": "sim", "warm": 8,
                "joint_overrides": {"delay_s": 0.020, "deadband_rad": math.radians(0.2)},
                "per_dataset_overrides": {"dobri420": {"delta_s": 0.010,
                                                       "range_rad": {"shoulder_lift": [-1.95, 1.75],
                                                                     "wrist_flex": [-1.75, 1.85]},
                                                       "state_offset_rad": {"elbow_flex": 0.01},
                                                       "action_offset_rad": {"elbow_flex": 0.01}}}},
    ]


def patch_params(P, cfg):
    import copy

    if not cfg.get("joint_overrides") and not cfg.get("per_dataset_overrides"):
        return P
    Q = copy.deepcopy(P)
    for j in JOINTS:
        Q["joints"][j].update(cfg.get("joint_overrides", {}))
    Q["per_dataset"] = {**Q["per_dataset"], **cfg.get("per_dataset_overrides", {})}
    Q["variant"] = P["variant"] + "+PATHTEST"
    return Q


def make_findings(report):
    """Data-driven summary lines (numbers taken from this run)."""
    C = {c["cfg"]["name"]: c for c in report["configs"]}
    F = []

    def nm(name):
        return C[name]["agg"]["newton_vs_mujoco"] if name in C else None

    for name in ("B_sim_warm0", "B_wide_warm0", "B_wide_warm8"):
        if name in C:
            a = C[name]["agg"]
            x = a["newton_vs_mujoco"]
            line = (f"{name}: Newton (SolverMuJoCo, MuJoCo-C backend) vs JSON-built MuJoCo over {a['n_windows']} TEST windows: "
                    f"max {x['max_deg_arm']:.1e} deg, RMS {x['rms_deg_arm_mean']:.1e} deg (arm joints); target was < 0.05 deg. "
                    f"vs real arm-mean RMSE: MuJoCo {a['vs_real']['mujoco']['primary6']['arm_mean']:.3f}, Newton "
                    f"{a['vs_real']['newton']['primary6']['arm_mean']:.3f} (6 primary), MuJoCo "
                    f"{a['vs_real']['mujoco']['provenance3']['arm_mean']:.3f}, Newton {a['vs_real']['newton']['provenance3']['arm_mean']:.3f} (3 provenance-verified).")
            if "models_py_vs_json_mujoco_max_deg" in a:
                line += f" models.make_sim('B_nvidia') vs JSON-built MuJoCo: max {a['models_py_vs_json_mujoco_max_deg']:.1e} deg."
            cc = C[name].get("crosscheck_windows_csv")
            if cc:
                line += (f" venv MuJoCo {report['mujoco_version']} reproduces replay.py's windows.csv (mujoco 3.14) per-window "
                         f"RMSE to {max(cc['max_abs_diff_window_rmse_deg'].values()):.1e} deg.")
            F.append(line)
    if "B_wide_update_interval0" in C and "B_wide_warm0" in C:
        F.append(f"Source of the residual: with update_data_interval=0 (Newton never re-writes MuJoCo qpos/qvel from its float32 "
                 f"State) the difference drops from {nm('B_wide_warm0')['max_deg_arm']:.1e} to "
                 f"{nm('B_wide_update_interval0')['max_deg_arm']:.1e} deg, so it is the per-step float32 round trip of the joint state "
                 f"(plus float32 model parameters), not a dynamics difference.")
    if "B_wide_importer_inertia" in C:
        F.append(f"Inertia: Newton's MJCF importer derives camera_mount (13.4 g, no <inertial>) inertia from geoms differently from "
                 f"MuJoCo's compiler (see section 4); without the sync the max difference is {nm('B_wide_importer_inertia')['max_deg_arm']:.1e} deg "
                 f"(RMS {nm('B_wide_importer_inertia')['rms_deg_arm_mean']:.1e}) vs {nm('B_wide_warm0')['max_deg_arm']:.1e} "
                 f"(RMS {nm('B_wide_warm0')['rms_deg_arm_mean']:.1e}) with it: small, but newton_check syncs it.")
    if "B_wide_noslip0_both" in C:
        a = C["B_wide_noslip0_both"]["agg"]
        alt = a["alt_ref_vs_mujoco"]
        F.append(f"noslip_iterations matters: MuJoCo noslip=1 (models.py) vs noslip=0 differ by up to "
                 f"{max(list(alt['max_deg'].values())[:ARM]):.2f} deg (RMS per joint "
                 + ", ".join(f"{j} {v:.3f}" for j, v in list(alt['rms_deg'].items())[:ARM])
                 + f"); vs real arm-mean: noslip=0 {a['vs_real']['mujoco']['primary6']['arm_mean']:.3f}, noslip=1 "
                 f"{a['vs_real']['mujoco_alt']['primary6']['arm_mean']:.3f}. Newton has no noslip option (set on mj_model directly "
                 f"for the MuJoCo-C backend) and mujoco_warp 3.12 raises NotImplementedError for noslip>0, so a fit meant for "
                 f"Newton on GPU should be done with noslip_iterations=0. Newton vs MuJoCo at noslip 0: max "
                 f"{a['newton_vs_mujoco']['max_deg_arm']:.1e} deg.")
    for name, c in C.items():
        if c["cfg"].get("backend") == "mujoco_warp":
            x = c["agg"]["newton_vs_mujoco"]
            rt = c["runtime"]
            F.append(f"{name}: Newton with the mujoco_warp backend on the CPU device (float32 physics, as on GPU) vs MuJoCo-C "
                     f"(noslip 0) over {c['agg']['n_windows']} windows: max {x['max_deg_arm']:.1e} deg, RMS {x['rms_deg_arm_mean']:.1e}; "
                     f"{rt.get('newton_cpu_per_sim_s', float('nan')):.2f} CPU s per simulated s (CPU device; a GPU run is not "
                     f"measured here). Needed workaround: disable_contacts=True -> nconmax=0 -> ZeroDivisionError in mujoco_warp's "
                     f"CPU fallback (solver._update_gradient); instead keep contacts allocated and set mjDSBL_CONTACT on mjw_model.opt.")
        if name.startswith("PATHTEST"):
            x = c["agg"]["newton_vs_mujoco"]
            F.append(f"{name} (arbitrary delay 20 ms + deadband 0.2 deg on all joints, per-dataset delta/range/offset on dobri420, "
                     f"warm 8): Newton vs MuJoCo max {x['max_deg_arm']:.1e} deg, RMS {x['rms_deg_arm_mean']:.1e}. Its vs-real numbers mean nothing.")
    rt = C.get("B_wide_warm0", {}).get("runtime", {})
    if rt:
        F.append(f"Runtime (this laptop, CPU): Newton MuJoCo-C backend {rt['newton_cpu_per_sim_s']:.3f} CPU s per simulated s "
                 f"(~{rt['newton_cpu_per_sim_s'] / 500 * 1e3:.2f} ms per 2 ms step; ~7 Warp CPU kernel launches per step dominate), "
                 f"plain MuJoCo {rt['mujoco_cpu_per_sim_s']:.4f}; Newton model build {rt['newton_build_s']:.1f} s per process.")
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help="parameter JSON (default: so101_fitted.json if present, else params/b_nvidia.json)")
    ap.add_argument("--per-dataset", type=int, default=12, help="TEST windows per primary dataset (0 = all)")
    ap.add_argument("--procs", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--configs", nargs="*", default=None, help="subset of suite config names")
    ap.add_argument("--warp-backend-windows", type=int, default=1,
                    help="windows for the mujoco_warp-on-CPU timing/agreement run (0 = skip)")
    ap.add_argument("--write-default-json", action="store_true")
    ap.add_argument("--out", default=None, help="output basename (default newton_check / newton_check_anchor)")
    # Anchor mode
    ap.add_argument("--anchor", action="store_true", help="replay whole NVIDIA Anchor-Lab motions (schema-v2 JSON)")
    ap.add_argument("--anchor-motions", nargs="*", default=None,
                    help="motion names (default: the 8 held-out motions of anchor_data.split())")
    ap.add_argument("--allow-heldout-before-prereg", action="store_true",
                    help="load held-out motions although results/anchor_prereg.json does not exist")
    ap.add_argument("--allow-unrepresentable", action="store_true",
                    help="run although the JSON has settings anchor_fit.opts() cannot represent")
    ap.add_argument("--no-assert", action="store_true", help="report, but do not abort on, a model mismatch")
    ap.add_argument("--update-data-interval", type=int, default=1,
                    help="SolverMuJoCo update_data_interval (1 = Newton default: State -> mj_data every step)")
    ap.add_argument("--warp-whole", action="store_true",
                    help="run the MuJoCo-Warp path over every --anchor motion (whole, or the first --warp-check s), "
                         "on wp.get_device() (cuda:0 when a GPU is present), scored vs real")
    ap.add_argument("--warp-check", type=float, default=0.0,
                    help="seconds of the first motion to replay with use_mujoco_cpu=False on Warp's CPU device")
    ap.add_argument("--compare-final-preds", action="store_true",
                    help="compare with results/anchor_cache/final_preds.npz (anchor_fit stage final, MuJoCo 3.14)")
    ap.add_argument("--out-dir", default=None, help="output directory (default results/)")
    ap.add_argument("--make-v2-json", default=None, metavar="OUT",
                    help="write a dev schema-v2 JSON from anchor_fit.fitted_json(opts(--friction), base_params(), "
                         "--delay) and exit")
    ap.add_argument("--friction", default="F1_imp990", help="with --make-v2-json: an anchor_fit.FRICTION key")
    ap.add_argument("--delay", type=float, default=0.03, help="with --make-v2-json: delay_s")
    args = ap.parse_args()

    if args.make_v2_json:
        print("wrote", make_v2_json(args.make_v2_json, args.friction, args.delay,
                                    f"dev: anchor_fit.fitted_json(opts(friction={args.friction!r}), base_params(), "
                                    f"{args.delay}) via newton_check.py --make-v2-json"))
        return
    if args.anchor:
        return main_anchor(args)
    args.out = args.out or "newton_check"

    if args.write_default_json or not os.path.isfile(DEFAULT_JSON):
        print("wrote", write_default_json())
        if args.write_default_json:
            return
    ppath = args.params or (FITTED_JSON if os.path.isfile(FITTED_JSON) else DEFAULT_JSON)
    P = load_params(ppath)
    print(f"params {ppath}: variant {P['variant']}, defaulted {P['defaulted'] or 'none'}, ignored {P['ignored_keys'] or 'none'}")

    import replay
    import mujoco
    import newton
    import warp as wp

    t_wall = time.perf_counter()
    dsets = [d for d in replay.load_datasets() if d["name"] in PRIMARY]
    names = [d["name"] for d in dsets]
    per = 1 if args.quick else args.per_dataset
    wins = select_windows(dsets, 4, per, names)
    print(f"{len(wins)} TEST windows (W=4 s) from {len(names)} datasets; loaded in {time.perf_counter() - t_wall:.1f} s")

    suite = default_suite(P["dt"])
    if args.configs:
        suite = [c for c in suite if c["name"] in args.configs]
    if args.warp_backend_windows and not args.configs:
        suite.append({"dt": P["dt"], "W": 4, "warm": 0, "limits": "wide", "backend": "mujoco_warp", "inertia": "mujoco",
                      "update_data_interval": 1, "noslip": 0, "name": "B_wide_mujoco_warp_cpu_noslip0",
                      "n_windows": args.warp_backend_windows})

    # slim episode payload per job
    slim = [{"name": d["name"], "episodes": d["episodes"]} for d in dsets]
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "params": {k: P[k] for k in ("path", "variant", "defaulted", "ignored_keys", "dt", "noslip_iterations")},
              "params_joints": P["joints"],
              "env": f"python {sys.version.split()[0]}, newton {newton.__version__}, warp {wp.__version__}, mujoco {mujoco.__version__}, CPU only",
              "mujoco_version": mujoco.__version__, "configs": []}

    # MjModel comparison (main process, once)
    ov0 = dataset_overrides(P, names[0])
    ref0 = MjArm(P, P["dt"], "sim", ov0)
    t = time.perf_counter()
    na = NewtonArm(P, P["dt"], "sim", ov0, ref_inertials=mujoco_body_inertials(ref0.model))
    report["newton_build_s_first"] = time.perf_counter() - t
    report["model_compare"] = compare_mj_models(ref0.model, na.mj_model())
    na2 = NewtonArm(P, P["dt"], "sim", ov0, inertia="newton")
    report["model_compare_importer_inertia"] = compare_mj_models(ref0.model, na2.mj_model())
    del na, na2

    ctx = mp.get_context("spawn")
    for cfg in suite:
        Pc = patch_params(P, cfg)
        w = wins if "n_windows" not in cfg else wins[:: max(1, len(wins) // cfg["n_windows"])][: cfg["n_windows"]]
        procs = min(args.procs, len(w))
        chunks = [w[i::procs] for i in range(procs)]
        jobs = [{"cfg": cfg, "P": Pc, "dsets": slim, "windows": c} for c in chunks if c]
        t = time.perf_counter()
        with ctx.Pool(len(jobs)) as pool:
            res = pool.map(run_job, jobs)
        wall = time.perf_counter() - t
        recs = [r for x in res for r in x["recs"]]
        recs.sort(key=lambda r: (r["dataset"], r["episode"], r["t0"]))
        present = [k.split("|", 1)[1] for k in recs[0] if k.startswith("sse_real|")]
        agg = aggregate(recs, present, names)
        sim_s = sum(x["sim_s"] for x in res)
        cpu = {}
        for x in res:
            for k, v in x["cpu"].items():
                cpu[k] = cpu.get(k, 0.0) + v
        runtime = {f"{k}_cpu_per_sim_s": v / sim_s for k, v in cpu.items()}
        runtime.update({"sim_s": sim_s, "wall_s": wall, "procs": len(jobs),
                        "newton_build_s": float(np.mean([x["builds"].get("build_newton_s", 0) for x in res])),
                        "real_time_factor_newton_single_core": (sim_s / cpu["newton"]) if cpu.get("newton") else None})
        entry = {"cfg": cfg, "agg": agg, "runtime": runtime,
                 "newton_ncon_max": int(max(r.get("newton_ncon_end", 0) for r in recs))}
        if cfg["warm"] == 0 and Pc["variant"] == "B_nvidia" and not any(Pc["joints"][j]["delay_s"] for j in JOINTS) \
                and cfg.get("noslip", 1) == 1:
            entry["crosscheck_windows_csv"] = crosscheck_windows_csv(recs, cfg["limits"])
        report["configs"].append(entry)
        nm = agg["newton_vs_mujoco"]
        print(f"[{cfg['name']}] {len(recs)} windows, wall {wall:.0f} s: Newton-vs-MuJoCo max {nm['max_deg_arm']:.2e} deg, "
              f"RMS {nm['rms_deg_arm_mean']:.2e}; per joint max {nm['max_deg']}; "
              + "; ".join(f"{m} vs real p6 {e['primary6']['arm_mean']:.3f} p3 {e.get('provenance3', {}).get('arm_mean', float('nan')):.3f}"
                          for m, e in agg["vs_real"].items() if 'primary6' in e)
              + f"; runtime {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in runtime.items()})}")
        if "models_py_vs_json_mujoco_max_deg" in agg:
            print(f"    models.py vs JSON-built MuJoCo max diff {agg['models_py_vs_json_mujoco_max_deg']:.2e} deg")
        if "alt_ref_vs_mujoco" in agg:
            print(f"    alt ref vs ref: {agg['alt_ref_vs_mujoco']}")
        if entry.get("crosscheck_windows_csv"):
            print(f"    windows.csv cross-check: {entry['crosscheck_windows_csv']}")

    report["bam_newton_sketch"] = bam_newton_sketch()
    report["wall_s"] = time.perf_counter() - t_wall
    if args.quick:
        print("quick run: nothing written")
        return
    report["findings"] = make_findings(report) + STATIC_FINDINGS
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, args.out + ".json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    write_md(report, os.path.join(RESULTS, args.out + ".md"))
    print(f"wrote results/{args.out}.json and .md in {report['wall_s']:.0f} s")


STATIC_FINDINGS: list[str] = [
    "Install (VERIFIED): newton 1.6.1 is a py3-none wheel, but its [sim] extra pins mujoco~=3.12 / mujoco-warp~=3.12 "
    "(cp314 wheels exist). Installing it into the main Python 3.14 user site would downgrade mujoco 3.14 -> 3.12 under "
    "models.py/replay.py, so it lives in an isolated venv: `python -m venv <scratchpad>/newton_venv` then "
    "`pip install \"newton[sim]==1.6.1\" numpy pandas pyarrow trimesh scipy` (trimesh is needed by add_mjcf for the STL "
    "meshes). Resolved: newton 1.6.1, warp-lang 1.18.0 (CPU only, no CUDA driver), mujoco 3.12.0, mujoco-warp 3.12.0.",
    "Drive mapping (VERIFIED in newton/_src/solvers/mujoco/solver_mujoco.py): a JOINT_TARGET POSITION drive becomes a MuJoCo "
    "general actuator gainprm=[ke], biasprm=[0,-ke,-kd] (= MuJoCo <position kp kv>); joint_effort_limit becomes the joint's "
    "actfrcrange (clamps the summed actuator force) and the MJCF actuator forcerange/ctrlrange are re-attached from the "
    "importer's mujoco:actuator_* custom attributes; joint_damping/joint_friction/joint_armature -> dof_damping/"
    "dof_frictionloss/dof_armature. Joint limits imported without solreflimit keep MuJoCo's default (0.02, 1) "
    "(SOLREF_MODE_MJCF_DEFAULT) unless joint_limit_ke/kd are edited.",
    "Integrator/options: Newton reads <option> from so101.xml (implicitfast, elliptic cone, iterations 10, ls_iterations 20, "
    "impratio 10) and sets opt.timestep = dt on every step(); tolerance is 1e-8 on the MuJoCo-C backend but 1e-6 in "
    "mujoco_warp. Gravity is float32 (-9.8100004).",
    "Velocity limits (joint_velocity_limit) are NOT supported by SolverMuJoCo (docstring), so a fitted max-velocity / "
    "target rate limit (BAM max_velocity) would have to be applied to the command in Python.",
]


if __name__ == "__main__":
    main()

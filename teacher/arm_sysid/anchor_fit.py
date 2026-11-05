"""Fit the SO-101 arm to NVIDIA's real Anchor-Lab test motions; score held-out once; cross-check on 6 teleop arms.

Recipe follows NVIDIA's anchor-bench actuator_sys_id (position-PD servo per joint: stiffness, damping, armature,
dry friction, viscous friction; effort limit 3.35; dt 4 ms; CMA-ES on joint-position error), with these changes:
  * commands are held from their LOGGED time plus one global transport delay (fitted), box-filtered per physics step
    (NVIDIA interpolates commands onto a 50 Hz grid, which anticipates every step; that "native" replay is kept to
    score their fit fairly);
  * kv and joint damping act as one -c*qd in free space: c is fitted per joint, NVIDIA's split ratio is kept;
  * the jaw is in physical radians (data 0..1 -> -10 + 127.09 deg * x). NVIDIA fitted it in 0..1 units, so their
    jaw friction and effort limit are multiplied by 2.218 rad/unit to reproduce their jaw in physical units;
  * camera: Menagerie's 13.44 g camera_mount is replaced by NVIDIA's 23.1 g point camera (same place);
  * MuJoCo joint friction "creeps" during holds without noslip, and Newton on GPU has no noslip, so the friction
    constraint softness (dof_solimp / dof_solref) is chosen first (stage frictionsel) on VAL;
  * an effort-limit multiplier (servo load saturates at 700 = 70 % in PRBS) and the delay are fitted.

Data split (anchor_data.py): FIT 36 train motions (seeded 6.5 s windows), VAL 6 train motions (model selection),
HELD-OUT 8 motions, scored once in stage `final` after the fitted JSON hash is written to results/anchor_prereg.json.

Stages:  profile | baselines | frictionsel | fit_arm | fit_jaw | freeze | final | teleop | report
Run with low priority + 6 workers; every CMA generation is checkpointed (--resume continues a stage).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# BAM main e9a619d + PR#20 (STS3215 12 V params); must be set before models.py is imported anywhere
_BAM20 = (r"C:/Users/valen/AppData/Local/Temp/claude/C--Users-valen-Documents-vla-distill/"
          r"c1e14d5c-c49d-4f52-b957-d99b0501209c/scratchpad/src/bam20")
if os.path.isdir(_BAM20):
    os.environ.setdefault("BAM_SRC", _BAM20)

import numpy as np  # noqa: E402

import anchor_data as AD  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(RESULTS, "anchor_cache")
LOG = os.path.join(RESULTS, "anchor_log.txt")
FITTED_JSON = os.path.join(HERE, "so101_fitted.json")
NPROC = max(1, min(int(os.environ.get("ARM_SYSID_PROCS", "6")), (os.cpu_count() or 2)))

JOINTS = AD.JOINTS
ARM = 5
DT = 0.004
NATIVE_CTRL_DT = 0.02
WARM = AD.WIN_WARM
VEL_HALF = 0.02                       # s, half-width of the linear fit for the start velocity
HOLD_LOOKBACK = 1.0                   # s; a joint is "holding" if its command moved < HOLD_TOL over the last 1 s
HOLD_TOL = math.radians(0.05)
JAW_SCALE = math.radians(AD.JAW_SPAN_DEG)       # 2.218 rad per 0..1 unit
CAM_MENAGERIE_G = 13.44
CAM_NVIDIA_G = 23.1
CAM_IPOS = (-0.003913, 0.052410, -0.042000)    # from NVIDIA's camera USD (gripper COM shift), gripper frame
DELAY_MAX = 0.08
DELAY_GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
PURE_DELAY_GRID = [0.005 * i for i in range(0, 41)]

FRICTION = {   # dof_solimp[0:2] (dmin = dmax) and dof_solref[0] for the frictionloss constraint
    "F0_default": {"noslip": 0, "solimp": None, "solref": None},
    "F1_imp990": {"noslip": 0, "solimp": 0.99, "solref": None},
    "F2_imp995": {"noslip": 0, "solimp": 0.995, "solref": None},
    "F3_imp990_ref008": {"noslip": 0, "solimp": 0.99, "solref": 0.008},
    # added after the first frictionsel pass (VAL only): F2 and F3 each helped and creep was still 0.5 deg;
    # solref 0.008 = 2*dt is MuJoCo's stiffest allowed time constant at dt 4 ms
    "F4_imp995_ref008": {"noslip": 0, "solimp": 0.995, "solref": 0.008},
    "FN_noslip1": {"noslip": 1, "solimp": None, "solref": None},   # reference only: Newton on GPU has no noslip
}


def opts(friction="F0_default", dt=DT, camera_g=CAM_NVIDIA_G, iterations=10, ls_iterations=20, jaw_hi_deg=120.0):
    f = FRICTION[friction]
    return {"friction": friction, "dt": dt, "noslip": f["noslip"], "solimp": f["solimp"], "solref": f["solref"],
            "camera_g": camera_g, "iterations": iterations, "ls_iterations": ls_iterations, "jaw_hi_deg": jaw_hi_deg}


# ================================================================================================ log / io
def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def low_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:   # non-Windows or no psutil: best effort
        pass


# ================================================================================================ parameters
def base_params(variant="B_nvidia"):
    """Per-joint SI params (6-lists). B_nvidia jaw friction / effort scaled to physical radians (see docstring)."""
    import models
    if variant == "B_nvidia":
        si = models.nvidia_si_params()
        p = {k: [float(si[j][k]) for j in JOINTS] for k in ("kp", "kv", "damping", "frictionloss", "armature",
                                                             "forcerange")}
        p["frictionloss"][5] *= JAW_SCALE
        p["forcerange"][5] *= JAW_SCALE
        return p
    if variant == "B_nvidia_rawjaw":
        si = models.nvidia_si_params()
        return {k: [float(si[j][k]) for j in JOINTS] for k in ("kp", "kv", "damping", "frictionloss", "armature",
                                                                "forcerange")}
    if variant == "A_stock":
        s = models.make_sim("A_stock", dt=0.002)
        m = s.model
        return {"kp": m.actuator_gainprm[s.aid, 0].tolist(), "kv": (-m.actuator_biasprm[s.aid, 2]).tolist(),
                "damping": m.dof_damping[s.vadr].tolist(), "frictionloss": m.dof_frictionloss[s.vadr].tolist(),
                "armature": m.dof_armature[s.vadr].tolist(), "forcerange": m.actuator_forcerange[s.aid, 1].tolist()}
    raise ValueError(variant)


# ================================================================================================ sims (worker side)
_SIMS = {}


def _okey(o):
    return tuple(sorted((k, v) for k, v in o.items()))


def apply_opts(s, o):
    """Camera mass, jaw range, friction softness, solver iterations on a compiled models.SO101Sim."""
    import mujoco
    m = s.model
    bid = m.body("camera_mount").id
    if not hasattr(s, "_cam0"):
        s._cam0 = (float(m.body_mass[bid]), m.body_ipos[bid].copy(), m.body_inertia[bid].copy())
    g = float(o["camera_g"])
    if abs(g - CAM_MENAGERIE_G) < 1e-9:
        m.body_mass[bid], m.body_ipos[bid], m.body_inertia[bid] = s._cam0[0], s._cam0[1], s._cam0[2]
    else:
        mass = max(g, 1e-6) * 1e-3
        m.body_mass[bid] = mass
        m.body_ipos[bid] = CAM_IPOS
        m.body_inertia[bid] = 0.4 * mass * 0.008 ** 2      # small solid ~8 mm radius (point-like)
    jid = m.joint("gripper").id
    m.jnt_range[jid, 1] = math.radians(o["jaw_hi_deg"])
    m.actuator_ctrllimited[:] = 0
    v = s.vadr
    if o["solimp"] is not None:
        m.dof_solimp[v, 0] = o["solimp"]
        m.dof_solimp[v, 1] = o["solimp"]
    if o["solref"] is not None:
        m.dof_solref[v, 0] = o["solref"]
    m.opt.noslip_iterations = int(o["noslip"])
    m.opt.iterations = int(o["iterations"])
    m.opt.ls_iterations = int(o["ls_iterations"])
    mujoco.mj_setConst(m, s.data)


def get_pos_sim(o, params):
    import mujoco
    import models
    key = ("pos", _okey(o))
    if key not in _SIMS:
        s = models.make_sim("B_nvidia", dt=o["dt"], noslip_iterations=int(o["noslip"]))
        apply_opts(s, o)
        s._arm = None
        _SIMS[key] = s
    s = _SIMS[key]
    m = s.model
    a, v = s.aid, s.vadr
    m.actuator_gainprm[a, 0] = params["kp"]
    m.actuator_biasprm[a, 1] = -np.asarray(params["kp"])
    m.actuator_biasprm[a, 2] = -np.asarray(params["kv"])
    m.actuator_forcelimited[a] = 1
    m.actuator_forcerange[a, 0] = -np.asarray(params["forcerange"])
    m.actuator_forcerange[a, 1] = np.asarray(params["forcerange"])
    m.dof_damping[v] = params["damping"]
    m.dof_frictionloss[v] = params["frictionloss"]
    arm = np.asarray(params["armature"], float)
    if s._arm is None or not np.array_equal(s._arm, arm):
        m.dof_armature[v] = arm
        mujoco.mj_setConst(m, s.data)
        s._arm = arm.copy()
    return s


def get_bam_sim(o, model_name):
    import fit as F
    F.DT = o["dt"]
    s = F.get_sim({"kind": "bam", "model": model_name, "set": "12V", "vin": 12.3,
                   "fric_scale": [1.0] * 6, "egr_scale": [1.0] * 6})
    key = ("bamopts", model_name, _okey(o))
    if key not in _SIMS:
        apply_opts(s, o)
        _SIMS[key] = True
    return s


# ================================================================================================ schedules
def event_targets(t_cmd, qc, delay, t_start, nsteps, dt):
    """Box-filtered per-step targets of the command held from (logged time + delay) until the next one."""
    b = t_cmd + delay
    seg = np.diff(b)
    icum = np.concatenate([np.zeros((1, qc.shape[1])), np.cumsum(qc[:-1] * seg[:, None], axis=0)])
    e = t_start + np.arange(nsteps + 1) * dt
    k = np.clip(np.searchsorted(b, e, side="right") - 1, 0, len(b) - 1)
    integ = icum[k] + qc[k] * (e - b[k])[:, None]
    return np.diff(integ, axis=0) / dt


def native_targets(t_cmd, qc, t_start, nsteps, dt):
    """NVIDIA anchor-bench replay: commands np.interp'd onto a 50 Hz grid, held per 20 ms control step."""
    n_ctrl = int(math.ceil(nsteps * dt / NATIVE_CTRL_DT)) + 1
    g = t_start + np.arange(n_ctrl) * NATIVE_CTRL_DT
    c = np.stack([np.interp(g, t_cmd, qc[:, j]) for j in range(qc.shape[1])], 1)
    idx = np.minimum(np.floor(np.arange(nsteps) * dt / NATIVE_CTRL_DT + 1e-9).astype(int), n_ctrl - 1)
    return c[idx]


def zoh(t_cmd, qc, t):
    k = np.clip(np.searchsorted(t_cmd, t, side="right") - 1, 0, len(t_cmd) - 1)
    return qc[k]


# ================================================================================================ simulate one item
def start_state(d, t_start):
    i0 = int(np.searchsorted(d["t_state"], t_start))
    i0 = min(max(i0, 0), len(d["t_state"]) - 1)
    t0 = float(d["t_state"][i0])
    sel = np.abs(d["t_state"] - t0) <= VEL_HALF
    tt = d["t_state"][sel] - t0
    qq = d["q"][sel]
    if len(tt) >= 3 and np.ptp(tt) > 0:
        A = np.stack([tt, np.ones_like(tt)], 1)
        v0 = np.linalg.lstsq(A, qq, rcond=None)[0][0]
    else:
        v0 = np.zeros(6)
    return i0, t0, d["q"][i0].copy(), v0


def rollout_q(s, q0, v0, U):
    import mujoco
    from mujoco import rollout
    m, d = s.model, s.data
    mujoco.mj_resetData(m, d)
    d.qpos[s.qadr] = q0
    d.qvel[s.vadr] = v0
    d.ctrl[s.aid] = U[0]
    mujoco.mj_forward(m, d)
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    st0 = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, st0, spec)
    ctrl = np.zeros((len(U), m.nu))
    ctrl[:, s.aid] = U
    st, _ = rollout.rollout(m, d, st0, ctrl)
    Q = np.empty((len(U) + 1, 6))
    Q[0] = q0
    Q[1:] = st[0][:, 1 + s.qadr]
    return Q


def loop_q(s, q0, v0, U, bam=False):
    import mujoco
    m, d = s.model, s.data
    mujoco.mj_resetData(m, d)
    d.qpos[s.qadr] = q0
    d.qvel[s.vadr] = v0
    if not bam:
        d.ctrl[s.aid] = U[0]
    mujoco.mj_forward(m, d)
    if bam:
        s.bank.reset()
    Q = np.empty((len(U) + 1, 6))
    Q[0] = q0
    step = mujoco.mj_step
    for n in range(len(U)):
        if bam:
            s.bank.set_target(U[n])
            s.bank.before_step()
        else:
            d.ctrl[s.aid] = U[n]
        step(m, d)
        Q[n + 1] = d.qpos[s.qadr]
    return Q


def item_span(d, item):
    kind, name = item[0], item[1]
    if kind == "win":
        t_start = float(item[2])
        t_end = t_start + AD.WIN_WARM + AD.WIN_SCORED
    else:
        t_start = float(max(d["t_state"][0], d["t_cmd"][0]))
        t_end = float(d["t_state"][-1])
    return t_start, t_end


def eval_item(cfg, item, want_pred=False):
    d = AD.load(item[1])
    t_req, t_end = item_span(d, item)
    i0, t_start, q0, v0 = start_state(d, t_req)
    i_lo = int(np.searchsorted(d["t_state"], t_start + WARM))
    i_hi = int(np.searchsorted(d["t_state"], t_end, side="right"))
    ts = d["t_state"][i_lo:i_hi]
    real = d["q"][i_lo:i_hi]
    kind = cfg["kind"]
    if kind == "nophys":
        b = cfg["baseline"]
        if b == "hold_start":
            pred = np.repeat(q0[None], len(ts), 0)
        elif b == "perfect":
            pred = zoh(d["t_cmd"], d["qc"], ts)
        elif b == "delay":
            pred = zoh(d["t_cmd"], d["qc"], ts - cfg["delay"])
        else:
            raise ValueError(b)
    else:
        o = cfg["opts"]
        dt = o["dt"]
        nsteps = int(math.ceil((t_end - t_start) / dt)) + 1
        if cfg.get("sched", "event") == "native":
            U = native_targets(d["t_cmd"], d["qc"], t_start, nsteps, dt)
        else:
            U = event_targets(d["t_cmd"], d["qc"], float(cfg["delay"]), t_start, nsteps, dt)
        if kind == "pos":
            s = get_pos_sim(o, cfg["params"])
            Q = rollout_q(s, q0, v0, U)
        elif kind == "bam":
            s = get_bam_sim(o, cfg["bam_model"])
            Q = loop_q(s, q0, v0, U, bam=True)
        else:
            raise ValueError(kind)
        x = (ts - t_start) / dt
        n0 = np.clip(np.floor(x).astype(int), 0, len(Q) - 2)
        fr = (x - n0)[:, None]
        pred = Q[n0] * (1 - fr) + Q[n0 + 1] * fr
    err = np.degrees(pred - real)
    if not np.all(np.isfinite(err)):
        err = np.nan_to_num(err, nan=1e3, posinf=1e3, neginf=-1e3)
    c_now = zoh(d["t_cmd"], d["qc"], ts)
    c_back = zoh(d["t_cmd"], d["qc"], ts - HOLD_LOOKBACK)
    hold = np.abs(c_now - c_back) < HOLD_TOL
    e2 = err ** 2
    rec = {"motion": item[1], "kind": item[0], "t_start": t_start,
           "sse": e2.sum(0).tolist(), "n": int(len(ts)),
           "sse_hold": (e2 * hold).sum(0).tolist(), "n_hold": hold.sum(0).tolist(),
           "sse_move": (e2 * ~hold).sum(0).tolist(), "n_move": (~hold).sum(0).tolist()}
    if want_pred:
        rec["pred_deg"] = np.degrees(pred).astype(np.float32)
        rec["i_lo"], rec["i_hi"] = i_lo, i_hi
    return rec


def job_fn(job):
    t = time.process_time()
    out = [eval_item(job["cfg"], it, job.get("want_pred", False)) for it in job["items"]]
    return {"tag": job["tag"], "recs": out, "cpu": time.process_time() - t}


def _init_worker():
    low_priority()


_POOL = None


def pool():
    global _POOL
    if _POOL is None:
        _POOL = mp.get_context("spawn").Pool(NPROC, initializer=_init_worker)
    return _POOL


def run_many(cfgs, items, chunk=10, want_pred=False):
    jobs = []
    for ci, cfg in enumerate(cfgs):
        for i in range(0, len(items), chunk):
            jobs.append({"cfg": cfg, "items": items[i:i + chunk], "tag": ci, "want_pred": want_pred})
    res = [[] for _ in cfgs]
    for r in pool().imap_unordered(job_fn, jobs):
        res[r["tag"]] += r["recs"]
    return res


# ================================================================================================ aggregation
def aggregate(recs, jaw_motions=None):
    """Equal weight per motion: per-motion per-joint RMSE (deg), arm mean over 5 joints; jaw over jaw_motions."""
    per = {}
    for r in recs:
        p = per.setdefault(r["motion"], {k: np.zeros(6) for k in ("sse", "n", "sse_hold", "n_hold", "sse_move",
                                                                  "n_move")})
        for k in p:
            p[k] = p[k] + np.asarray(r[k], float)
    out = {}
    for mname, p in per.items():
        rm = np.sqrt(p["sse"] / np.maximum(p["n"], 1))
        rh = np.where(p["n_hold"] > 0, np.sqrt(p["sse_hold"] / np.maximum(p["n_hold"], 1)), np.nan)
        rv = np.where(p["n_move"] > 0, np.sqrt(p["sse_move"] / np.maximum(p["n_move"], 1)), np.nan)
        out[mname] = {"rmse": rm.tolist(), "arm": float(rm[:ARM].mean()), "jaw": float(rm[5]),
                      "rmse_hold": rh.tolist(), "rmse_move": rv.tolist(),
                      "arm_hold": float(np.nanmean(rh[:ARM])) if np.any(np.isfinite(rh[:ARM])) else None,
                      "arm_move": float(np.nanmean(rv[:ARM])) if np.any(np.isfinite(rv[:ARM])) else None,
                      "n": int(p["n"][0])}
    names = sorted(out)
    jm = [n for n in names if (jaw_motions is None or n in jaw_motions)]
    summ = {"arm": float(np.mean([out[n]["arm"] for n in names])),
            "joint": np.mean([out[n]["rmse"] for n in names], 0).tolist(),
            "jaw": float(np.mean([out[n]["jaw"] for n in jm])) if jm else None,
            "arm_move": float(np.nanmean([out[n]["arm_move"] if out[n]["arm_move"] is not None else np.nan
                                          for n in names])),
            "arm_hold": float(np.nanmean([out[n]["arm_hold"] if out[n]["arm_hold"] is not None else np.nan
                                          for n in names])),
            "n_motions": len(names)}
    return {"summary": summ, "per_motion": out}


def win_items(names):
    sp = AD.split()
    return [("win", n, t) for n in names for t in sp["windows"][n]]


def whole_items(names):
    return [("whole", n) for n in names]


# ================================================================================================ CMA (checkpointed)
def run_cma(tag, x0, stds, lower, upper, evaluate, popsize, max_gens, budget_s, seed=1, resume=False):
    import cma
    ck = os.path.join(CACHE, f"{tag}_es.pkl")
    if resume and os.path.exists(ck):
        with open(ck, "rb") as fh:
            st = pickle.load(fh)
        es, best, hist, g, spent = st["es"], st["best"], st["hist"], st["g"], st["spent"]
        log(f"  [{tag}] resumed at gen {g} (best {best[0]:.4f})")
    else:
        es = cma.CMAEvolutionStrategy(list(x0), 1.0, {
            "bounds": [list(lower), list(upper)], "CMA_stds": list(stds), "popsize": popsize, "seed": seed,
            "verbose": -9, "tolfun": 1e-5, "tolx": 1e-5})
        best, hist, g, spent = (float("inf"), list(x0)), [], 0, 0.0
    t0 = time.perf_counter() - spent
    while not es.stop() and g < max_gens and time.perf_counter() - t0 < budget_s:
        X = es.ask()
        f = evaluate(X)
        es.tell(X, f)
        g += 1
        i = int(np.argmin(f))
        if f[i] < best[0]:
            best = (float(f[i]), list(map(float, X[i])))
        hist.append({"gen": g, "best": float(np.min(f)), "median": float(np.median(f)), "sigma": float(es.sigma),
                     "t": round(time.perf_counter() - t0, 1)})
        tmp = ck + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"es": es, "best": best, "hist": hist, "g": g, "spent": time.perf_counter() - t0}, fh)
        os.replace(tmp, ck)
        if g % 5 == 0 or g == 1:
            log(f"  [{tag}] gen {g:4d} best {np.min(f):.4f} median {np.median(f):.4f} sigma {es.sigma:.3f} "
                f"overall-best {best[0]:.4f} t={time.perf_counter() - t0:.0f}s")
    why = {k: str(v) for k, v in (es.stop() or {}).items()} or {"stopped": "budget/max_gens"}
    log(f"  [{tag}] stopped after {g} generations ({time.perf_counter() - t0:.0f} s): {why}; best {best[0]:.4f}")
    return best, hist, why


# ================================================================================================ decoders
ARM_DIM = 22     # 20 log-mults (kp, c=kv+damping, frictionloss, armature) x 5 arm joints, delay, effort


def decode_arm(x, base):
    p = {k: list(v) for k, v in base.items()}
    for j in range(ARM):
        p["kp"][j] = base["kp"][j] * math.exp(x[j])
        cm = math.exp(x[ARM + j])
        p["kv"][j] = base["kv"][j] * cm
        p["damping"][j] = base["damping"][j] * cm
        p["frictionloss"][j] = base["frictionloss"][j] * math.exp(x[2 * ARM + j])
        p["armature"][j] = base["armature"][j] * math.exp(x[3 * ARM + j])
        p["forcerange"][j] = base["forcerange"][j] * math.exp(x[21])
    delay = float(np.clip(x[20], 0.0, 1.0)) * DELAY_MAX
    return p, delay


def arm_bounds():
    lower = [-2.3] * 20 + [0.0, -1.2]
    upper = [2.3] * 20 + [1.0, 0.4]
    stds = [0.3] * 20 + [0.2, 0.2]
    return lower, upper, stds


def decode_jaw(x, base):
    p = {k: list(v) for k, v in base.items()}
    j = 5
    p["kp"][j] = base["kp"][j] * math.exp(x[0])
    p["kv"][j] = base["kv"][j] * math.exp(x[1])
    p["damping"][j] = base["damping"][j] * math.exp(x[1])
    p["frictionloss"][j] = base["frictionloss"][j] * math.exp(x[2])
    p["armature"][j] = base["armature"][j] * math.exp(x[3])
    p["forcerange"][j] = base["forcerange"][j] * math.exp(x[4])
    return p


def cfg_pos(o, params, delay, sched="event"):
    return {"kind": "pos", "opts": o, "params": params, "delay": float(delay), "sched": sched}


# ================================================================================================ stages
def stage_profile():
    """rollout vs Python loop equality, then time one CMA-sized generation (16 candidates x FIT windows)."""
    sp = AD.split()
    o = opts()
    base = base_params()
    cfg = cfg_pos(o, base, 0.03)
    item = ("win", sp["fit"][0], sp["windows"][sp["fit"][0]][0])
    d = AD.load(item[1])
    t_req, t_end = item_span(d, item)
    _, t_start, q0, v0 = start_state(d, t_req)
    nsteps = int(math.ceil((t_end - t_start) / DT)) + 1
    U = event_targets(d["t_cmd"], d["qc"], 0.03, t_start, nsteps, DT)
    s = get_pos_sim(o, base)
    Qr = rollout_q(s, q0, v0, U)
    Ql = loop_q(s, q0, v0, U)
    diff = float(np.max(np.abs(Qr - Ql)))
    log(f"profile: rollout vs loop max |dq| = {diff:.3e} rad over {nsteps} steps")
    items = win_items(sp["fit"])
    rng = np.random.default_rng(0)
    X = [rng.normal(0, 0.2, ARM_DIM) for _ in range(16)]
    for x in X:
        x[20] = 0.4
        x[21] = 0.0
    cfgs = [cfg_pos(o, *decode_arm(x, base)) for x in X]
    t = time.perf_counter()
    res = run_many(cfgs, items)
    wall = time.perf_counter() - t
    steps = sum(int(math.ceil((AD.WIN_WARM + AD.WIN_SCORED) / DT)) + 1 for _ in items) * len(cfgs)
    log(f"profile: 1 generation = 16 x {len(items)} windows = {steps / 1e6:.2f} M steps in {wall:.1f} s wall "
        f"({NPROC} procs, {1e6 * wall * NPROC / steps:.2f} us/step/proc)")
    jdump({"rollout_vs_loop_max_rad": diff, "gen_wall_s": wall, "steps": steps, "nproc": NPROC},
          os.path.join(CACHE, "profile.json"))
    return diff, wall


def pure_delay_fit(items):
    cfgs = [{"kind": "nophys", "baseline": "delay", "delay": dl} for dl in PURE_DELAY_GRID]
    res = run_many(cfgs, items, chunk=40)
    sc = [aggregate(r)["summary"]["arm"] for r in res]
    i = int(np.argmin(sc))
    return PURE_DELAY_GRID[i], sc


def model_delay_fit(make_cfg, items, grid=DELAY_GRID):
    res = run_many([make_cfg(dl) for dl in grid], items)
    sc = [aggregate(r)["summary"]["arm"] for r in res]
    i = int(np.argmin(sc))
    return grid[i], sc


def stage_baselines():
    sp = AD.split()
    fit_items = win_items(sp["fit"])
    val_whole = whole_items(sp["val"])
    jaw = set(sp["jaw_active"])
    out = {"delay_choice_on": "FIT windows", "models": {}}
    log(f"=== baselines: delays chosen on FIT windows ({len(fit_items)}), scored on FIT windows + VAL whole ===")
    d_pure, sc = pure_delay_fit(fit_items)
    log(f"  pure delay on FIT: best {1e3 * d_pure:.0f} ms (arm {min(sc):.3f})")
    rows = {
        "hold_start": {"kind": "nophys", "baseline": "hold_start"},
        "perfect_tracking": {"kind": "nophys", "baseline": "perfect"},
        "pure_delay": {"kind": "nophys", "baseline": "delay", "delay": d_pure},
    }
    base_b = base_params("B_nvidia")
    for cam in (CAM_NVIDIA_G, CAM_MENAGERIE_G, 0.0):
        o = opts(camera_g=cam)
        dl, sc = model_delay_fit(lambda x, o=o: cfg_pos(o, base_b, x), fit_items)
        rows[f"B_nvidia_cam{cam:g}"] = cfg_pos(o, base_b, dl)
        log(f"  B_nvidia camera {cam:g} g: delay {1e3 * dl:.0f} ms on FIT (arm {min(sc):.3f})")
    # NVIDIA's own replay: 50 Hz interpolated commands, no delay, MJWarp-like 5/2 solver iterations, camera 23.1 g.
    # Their jaw ran in 0..1 units; the x2.218 jaw friction/effort scaling in base_params reproduces it in radians.
    o_nat = opts(iterations=5, ls_iterations=2)
    rows["B_nvidia_native"] = cfg_pos(o_nat, base_b, 0.0, sched="native")
    o_a = opts(dt=0.002, camera_g=CAM_MENAGERIE_G)
    base_a = base_params("A_stock")
    dl, sc = model_delay_fit(lambda x: cfg_pos(o_a, base_a, x), fit_items)
    rows["A_stock"] = cfg_pos(o_a, base_a, dl)
    log(f"  A_stock (dt 2 ms): delay {1e3 * dl:.0f} ms (arm {min(sc):.3f})")
    for bm in ("m1", "m6"):
        o_b = opts(dt=0.002)
        dl, sc = model_delay_fit(lambda x, bm=bm: {"kind": "bam", "opts": o_b, "bam_model": bm, "delay": x,
                                                   "sched": "event"}, fit_items, grid=[0.0, 0.02, 0.04, 0.06])
        rows[f"BAM_{bm}_12V"] = {"kind": "bam", "opts": o_b, "bam_model": bm, "delay": dl, "sched": "event"}
        log(f"  BAM {bm} 12V (vin 12.3, dt 2 ms): extra delay {1e3 * dl:.0f} ms (arm {min(sc):.3f})")
    names = list(rows)
    res_fit = run_many([rows[n] for n in names], fit_items)
    res_val = run_many([rows[n] for n in names], val_whole, chunk=1)
    for n, rf, rv in zip(names, res_fit, res_val):
        out["models"][n] = {"cfg": rows[n], "fit_windows": aggregate(rf, jaw), "val_whole": aggregate(rv, jaw)}
        log(f"  {n:22s} FIT-win arm {out['models'][n]['fit_windows']['summary']['arm']:.3f}  "
            f"VAL-whole arm {out['models'][n]['val_whole']['summary']['arm']:.3f} "
            f"(move {out['models'][n]['val_whole']['summary']['arm_move']:.3f}, "
            f"hold {out['models'][n]['val_whole']['summary']['arm_hold']:.3f})  "
            f"jaw {out['models'][n]['val_whole']['summary']['jaw'] or float('nan'):.2f}")
    out["pure_delay_s"] = d_pure
    jdump(out, os.path.join(CACHE, "baselines.json"))


def creep_test(o, params, delay=0.0, seconds=30.0):
    """Hold a fixed command with the elbow started 2.9 deg off target (gravity side); return drift (deg) of each
    arm joint from 1 s to the end (the real arm drifts < 0.1 deg in train-static-holding)."""
    import mujoco
    s = get_pos_sim(o, params)
    q_home = np.radians([-1.0, -40.0, 3.0, 0.0, 0.0, 30.0])
    mujoco.mj_resetData(s.model, s.data)
    s.data.qpos[s.qadr] = q_home
    mujoco.mj_forward(s.model, s.data)
    g = s.data.qfrc_bias[s.vadr].copy()
    q0 = q_home.copy()
    q0[2] -= np.sign(g[2]) * math.radians(2.9)     # sagged side: real holds show cmd - q with the sign of gravity
    nsteps = int(seconds / o["dt"])
    U = np.repeat(q_home[None], nsteps, 0)
    Q = rollout_q(s, q0, np.zeros(6), U)
    i1 = int(1.0 / o["dt"])
    drift = np.degrees(np.abs(Q[-1] - Q[i1]))
    return {"drift_deg_1s_to_end": drift[:ARM].tolist(), "max_drift_deg": float(drift[:ARM].max()),
            "elbow_err_deg_at_1s": float(np.degrees(q_home[2] - Q[i1, 2])),
            "elbow_err_deg_at_end": float(np.degrees(q_home[2] - Q[-1, 2]))}


def fit_arm(tag, o, x0, gens, budget_min, items, resume=False, stds_scale=1.0, seed=1):
    base = base_params()
    lower, upper, stds = arm_bounds()
    stds = [s_ * stds_scale for s_ in stds]

    def evaluate(X):
        cfgs = [cfg_pos(o, *decode_arm(x, base)) for x in X]
        return [aggregate(r)["summary"]["arm"] for r in run_many(cfgs, items)]

    f0 = evaluate([x0])[0]
    log(f"  [{tag}] start objective (FIT windows, arm-mean RMSE over motions) {f0:.4f}")
    best, hist, why = run_cma(tag, x0, stds, lower, upper, evaluate, 16, gens, budget_min * 60, seed=seed,
                              resume=resume)
    if best[0] > f0:
        best = (f0, list(x0))
    p, dl = decode_arm(best[1], base)
    out = {"tag": tag, "opts": o, "x": best[1], "objective_fit": best[0], "start_objective": f0, "params": p,
           "delay_s": dl, "history": hist, "stop": why}
    jdump(out, os.path.join(CACHE, f"{tag}.json"))
    return out


def stage_frictionsel(gens=60, resume=False):
    sp = AD.split()
    fit_items = win_items(sp["fit"])
    val_whole = whole_items(sp["val"])
    bl = jload(os.path.join(CACHE, "baselines.json"))
    d_b = bl["models"][f"B_nvidia_cam{CAM_NVIDIA_G:g}"]["cfg"]["delay"]
    base = base_params()
    out = {}
    log(f"=== frictionsel: {gens}-gen fits per friction setting, chosen on VAL whole motions ===")
    for fname in FRICTION:
        o = opts(friction=fname)
        cr0 = creep_test(o, base)
        x0 = [0.0] * 20 + [d_b / DELAY_MAX, 0.0]
        f = fit_arm(f"fsel_{fname}", o, x0, gens, 12.0, fit_items, resume=resume)
        cr1 = creep_test(o, f["params"])
        rv = aggregate(run_many([cfg_pos(o, f["params"], f["delay_s"])], val_whole, chunk=1)[0])
        rb = aggregate(run_many([cfg_pos(o, base, d_b)], val_whole, chunk=1)[0])
        out[fname] = {"creep_B": cr0, "creep_fitted": cr1, "fit_obj": f["objective_fit"],
                      "val_whole_fitted": rv["summary"], "val_whole_B": rb["summary"], "x": f["x"],
                      "delay_s": f["delay_s"]}
        log(f"  {fname:18s} creep B {cr0['max_drift_deg']:.2f} / fitted {cr1['max_drift_deg']:.2f} deg; "
            f"FIT {f['objective_fit']:.3f}; VAL whole fitted {rv['summary']['arm']:.3f} "
            f"(move {rv['summary']['arm_move']:.3f} hold {rv['summary']['arm_hold']:.3f}) vs B {rb['summary']['arm']:.3f}")
        jdump(out, os.path.join(CACHE, "frictionsel.json"))
    usable = [k for k in out if FRICTION[k]["noslip"] == 0]
    choice = min(usable, key=lambda k: (out[k]["val_whole_fitted"]["arm"], out[k]["creep_fitted"]["max_drift_deg"]))
    o = opts(friction=choice)
    f = jload(os.path.join(CACHE, f"fsel_{choice}.json"))
    iters = {}
    for it, ls in ((5, 2), (10, 20)):
        oi = opts(friction=choice, iterations=it, ls_iterations=ls)
        rv = aggregate(run_many([cfg_pos(oi, f["params"], f["delay_s"])], val_whole, chunk=1)[0])
        iters[f"{it}/{ls}"] = rv["summary"]
        log(f"  {choice} solver iterations {it}/{ls}: VAL whole arm {rv['summary']['arm']:.4f}")
    out["_choice"] = {"friction": choice, "rule": "lowest VAL whole-motion arm RMSE among noslip=0 settings "
                      "(Newton GPU cannot run noslip); tie -> least creep", "solver_iterations_check": iters}
    jdump(out, os.path.join(CACHE, "frictionsel.json"))
    log(f"  chosen friction setting: {choice}")


def stage_fit_arm(gens=250, budget_min=25.0, resume=False):
    sp = AD.split()
    fs = jload(os.path.join(CACHE, "frictionsel.json"))
    choice = fs["_choice"]["friction"]
    o = opts(friction=choice)
    log(f"=== fit_arm: {choice}, warm start from its frictionsel optimum, <= {gens} gens / {budget_min} min ===")
    f = fit_arm("fit_arm", o, fs[choice]["x"], gens, budget_min, win_items(sp["fit"]), resume=resume,
                stds_scale=0.6)
    rv = aggregate(run_many([cfg_pos(o, f["params"], f["delay_s"])], whole_items(sp["val"]), chunk=1)[0])
    f["val_whole"] = rv
    jdump(f, os.path.join(CACHE, "fit_arm.json"))
    log(f"  fit_arm: FIT {f['objective_fit']:.4f}, VAL whole arm {rv['summary']['arm']:.4f} "
        f"(move {rv['summary']['arm_move']:.4f}, hold {rv['summary']['arm_hold']:.4f}), delay {1e3 * f['delay_s']:.1f} ms")


def stage_fit_jaw(gens=80, budget_min=10.0, resume=False):
    sp = AD.split()
    fa = jload(os.path.join(CACHE, "fit_arm.json"))
    o = fa["opts"]
    base = fa["params"]
    delay = fa["delay_s"]
    jm = [n for n in sp["fit"] if n in sp["jaw_active"]]
    items = win_items(jm)
    log(f"=== fit_jaw: 5 params on {len(jm)} jaw-active FIT motions ({len(items)} windows), delay fixed ===")

    def evaluate(X):
        cfgs = [cfg_pos(o, decode_jaw(x, base), delay) for x in X]
        return [aggregate(r)["summary"]["jaw"] for r in run_many(cfgs, items)]

    x0 = [0.0] * 5
    f0 = evaluate([x0])[0]
    log(f"  [fit_jaw] start jaw RMSE {f0:.3f} deg")
    best, hist, why = run_cma("fit_jaw", x0, [0.3] * 5, [-2.3] * 4 + [-1.5], [2.3] * 4 + [1.5], evaluate, 12, gens,
                              budget_min * 60, resume=resume)
    if best[0] > f0:
        best = (f0, x0)
    p = decode_jaw(best[1], base)
    vj = [n for n in sp["val"] if n in sp["jaw_active"]]
    rv = aggregate(run_many([cfg_pos(o, p, delay), cfg_pos(o, base, delay)], whole_items(vj), chunk=1)[0])
    out = {"x": best[1], "objective_fit_jaw": best[0], "start": f0, "params": p, "history": hist, "stop": why,
           "val_jaw_whole": rv["summary"]}
    jdump(out, os.path.join(CACHE, "fit_jaw.json"))
    log(f"  fit_jaw: FIT jaw {best[0]:.3f} deg (start {f0:.3f}); VAL whole jaw {rv['summary']['jaw']:.3f}")


def fitted_json(o, p, delay, note):
    import models
    s = models.make_sim("B_nvidia", dt=o["dt"])
    m = s.model
    joints = {}
    for k, j in enumerate(JOINTS):
        lo, hi = (float(x) for x in m.jnt_range[m.joint(j).id])
        if j == "gripper":
            hi = math.radians(o["jaw_hi_deg"])
        joints[j] = {"kp": p["kp"][k], "kv": p["kv"][k], "damping": p["damping"][k],
                     "frictionloss": p["frictionloss"][k], "armature": p["armature"][k],
                     "forcerange": p["forcerange"][k], "delay_s": delay, "deadband_rad": 0.0,
                     "range_rad": [lo, hi]}
    f = FRICTION[o["friction"]]
    cam = float(o["camera_g"]) * 1e-3
    return {
        "schema": "so101_params/v2",
        "variant": "anchor_fit",
        "actuator": "position",
        "source": note,
        "units": "kp N*m/rad, kv/damping N*m*s/rad, frictionloss N*m, armature kg*m^2, forcerange N*m, delay_s s",
        "sim": {"dt": o["dt"], "integrator": "implicitfast", "noslip_iterations": f["noslip"],
                "iterations": o["iterations"], "ls_iterations": o["ls_iterations"],
                "solimpfriction": None if f["solimp"] is None else [f["solimp"], f["solimp"], 0.001, 0.5, 2.0],
                "solreffriction": None if f["solref"] is None else [f["solref"], 1.0],
                "friction_setting": o["friction"]},
        "delay_s": delay,
        "delay_steps": {f"{hz}Hz": int(round(delay * hz)) for hz in (50, 100, 125, 250)},
        "bodies": {"camera_mount": {"mass": cam, "ipos": list(CAM_IPOS),
                                    "inertia": [0.4 * cam * 0.008 ** 2] * 3}},
        "jaw_map": {"offset_deg": AD.JAW_ZERO_DEG, "span_deg": AD.JAW_SPAN_DEG,
                    "note": "theta_rad = radians(offset_deg + span_deg * x), x = LeRobot/Anchor 0..1 jaw value"},
        "joints": joints,
        "randomization": {
            "kp": [0.8, 1.2], "kv_and_damping": [0.8, 1.2], "frictionloss": [0.8, 1.2], "armature": [0.8, 1.2],
            "delay_s": [max(0.0, delay - 0.02), delay + 0.02], "camera_mount_mass_kg": [0.015, 0.031],
            "note": ("multipliers on the fitted values; OmniReset-style (arXiv 2603.15789: gains x0.8-1.2, delay "
                     "0-2 control steps). Fitted to ONE arm (NVIDIA's) in one session; the arm-to-arm spread is not "
                     "measured. Camera range = Fermi estimate for the team's Yahboom 2MP wrist camera + mount."),
        },
    }


def usd_units(js):
    """Per-degree USD/Newton-attribute view of a v2 JSON (anchor-bench usd.py convention): drive stiffness,
    drive damping and newton:damping x pi/180; friction (N*m) and armature unchanged."""
    k = math.pi / 180.0
    out = {}
    for j, r in js["joints"].items():
        out[j] = {"drive:angular:physics:stiffness": r["kp"] * k, "drive:angular:physics:damping": r["kv"] * k,
                  "drive:angular:physics:maxForce": r["forcerange"], "newton:damping": r["damping"] * k,
                  "newton:friction": r["frictionloss"], "newton:armature": r["armature"],
                  "mujoco:solimpfriction": js["sim"]["solimpfriction"],
                  "mujoco:solreffriction": js["sim"]["solreffriction"]}
    return {"units": "per degree (USD DriveAPI / NewtonJointAPI)", "delay_s_apply_to_commands": js["delay_s"],
            "joints": out}


def stage_freeze():
    fa = jload(os.path.join(CACHE, "fit_arm.json"))
    fj = jload(os.path.join(CACHE, "fit_jaw.json"))
    o = fa["opts"]
    p = fa["params"]
    for k in p:
        p[k][5] = fj["params"][k][5]
    note = ("CMA-ES fit to nvidia/Anchor-Lab rev 647edd57 (36 FIT train motions, VAL 6 for model choice), "
            "started from NVIDIA Isaac Sim 6.1 so101_new_calib_SysID; anchor_fit.py")
    js = fitted_json(o, p, fa["delay_s"], note)
    jdump(js, FITTED_JSON)
    jdump(usd_units(js), os.path.join(HERE, "params", "so101_fitted_usd_units.json"))
    jdump(fitted_json(opts(friction=o["friction"]), base_params(), fa["delay_s"],
                      "NVIDIA so101_new_calib_SysID (jaw friction/effort x2.218 to physical rad), same sim options"),
          os.path.join(HERE, "params", "b_nvidia_v2.json"))
    with open(FITTED_JSON, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    bl = jload(os.path.join(CACHE, "baselines.json"))
    pre = {"fitted_json": os.path.basename(FITTED_JSON), "sha256": h, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "heldout_touched_before": os.path.exists(os.path.join(CACHE, "final.json")),
           "models_to_score": ["hold_start", "perfect_tracking", "pure_delay", "A_stock", "B_nvidia_cam23.1",
                               "B_nvidia_cam13.44", "B_nvidia_cam0", "B_nvidia_native", "BAM_m1_12V", "BAM_m6_12V",
                               "B_nvidia_same_opts", "fitted"],
           "pure_delay_s": bl["pure_delay_s"], "friction": o["friction"]}
    jdump(pre, os.path.join(RESULTS, "anchor_prereg.json"))
    log(f"frozen: {FITTED_JSON} sha256 {h}")


def final_rows():
    bl = jload(os.path.join(CACHE, "baselines.json"))
    rows = {n: v["cfg"] for n, v in bl["models"].items()}
    fa = jload(os.path.join(CACHE, "fit_arm.json"))
    o = fa["opts"]
    js = jload(FITTED_JSON)
    p = {k: [js["joints"][j][k] for j in JOINTS] for k in ("kp", "kv", "damping", "frictionloss", "armature",
                                                            "forcerange")}
    rows["fitted"] = cfg_pos(o, p, js["delay_s"])
    rows["B_nvidia_same_opts"] = cfg_pos(o, base_params(), rows["B_nvidia_cam23.1"]["delay"])
    return rows


def stage_final():
    pre = jload(os.path.join(RESULTS, "anchor_prereg.json"))
    with open(FITTED_JSON, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    if h != pre["sha256"]:
        raise SystemExit("so101_fitted.json changed after pre-registration; refusing to score held-out")
    sp = AD.split()
    items = whole_items(sp["heldout"])
    jaw = [n for n in sp["heldout"] if np.ptp(AD.rad_to_jaw(AD.load(n)["qc"][:, 5])) > 0.05]
    rows = final_rows()
    names = list(rows)
    log(f"=== final: {len(names)} models x {len(items)} held-out whole motions (scored once) ===")
    res = run_many([rows[n] for n in names], items, chunk=1, want_pred=True)
    out = {"jaw_active_heldout": jaw, "models": {}}
    preds = {}
    for n, r in zip(names, res):
        out["models"][n] = aggregate(r, jaw)
        for rec in r:
            preds[f"{n}|{rec['motion']}"] = rec.pop("pred_deg")
            out["models"][n].setdefault("spans", {})[rec["motion"]] = [rec["i_lo"], rec["i_hi"], rec["t_start"]]
        s = out["models"][n]["summary"]
        log(f"  {n:22s} held-out arm {s['arm']:.3f} (move {s['arm_move']:.3f}, hold {s['arm_hold']:.3f}) "
            f"jaw {s['jaw'] if s['jaw'] is not None else float('nan'):.2f}  joints "
            + " ".join(f"{v:.2f}" for v in s["joint"]))
    out["sha256_fitted"] = h
    jdump(out, os.path.join(CACHE, "final.json"))
    np.savez_compressed(os.path.join(CACHE, "final_preds.npz"), **{k.replace("|", "__"): v for k, v in preds.items()})


# ------------------------------------------------------------------------------------------------ teleop check
def _teleop_job(job):
    import fit as F
    o = job["cfg"]["opts"]
    F.DT = o["dt"]
    dsets = F.datasets()
    s = get_pos_sim(o, job["cfg"]["params"])
    jid = [s.model.joint(j).id for j in JOINTS]
    out = []
    for (dn, ep, t0, W) in job["items"]:
        nu = job["nuis"][dn]
        N = int(round(W * F.FPS))
        nsteps, i1, i2, w, n0, fr = F.schedule(float(nu["delta_s"]), float(nu["period_s"]), F.K_WARM + N)
        C, q0, v0, real, off, lo, hi = F.window_arrays(dsets[dn], ep, t0, N, nu)
        s.model.jnt_range[jid, 0] = lo
        s.model.jnt_range[jid, 1] = hi
        U = C[i1] * w[:, None] + C[i2] * (1.0 - w)[:, None]
        Q = rollout_q(s, q0, v0, U[:nsteps])
        Qo = Q[n0] * (1 - fr)[:, None] + Q[np.minimum(n0 + 1, nsteps)] * fr[:, None]
        err = np.degrees(Qo[F.K_WARM:F.K_WARM + N] - off - real)
        if not np.all(np.isfinite(err)):
            err = np.nan_to_num(err, nan=1e3, posinf=1e3, neginf=-1e3)
        out.append({"ds": dn, "ep": int(ep), "t0": int(t0), "W": W, "n": N, "sse": (err ** 2).sum(0).tolist()})
    return {"tag": job["tag"], "recs": out}


def stage_teleop():
    import fit as F
    prep = F.jload(os.path.join(F.CACHE, "prep.json"))
    fa = jload(os.path.join(CACHE, "fit_arm.json"))
    o = fa["opts"]
    js = jload(FITTED_JSON)
    p_fit = {k: [js["joints"][j][k] for j in JOINTS] for k in ("kp", "kv", "damping", "frictionloss", "armature",
                                                                "forcerange")}
    models_ = {"B_nvidia": base_params(), "fitted": p_fit}
    offs = {dn: prep["datasets"][dn]["fk_offsets_deg"] for dn in F.PRIMARY}
    it_train = F.items_for(prep, "fit")
    it_test = F.items_for(prep, ("test", 4))
    grid = [0.01 * i for i in range(0, 13)]       # total command delay per arm, 0..120 ms (nuisance)
    log(f"=== teleop cross-check: per-arm total delay grid on TRAIN subset ({len(it_train)} windows), "
        f"TEST {len(it_test)} windows; friction {o['friction']}, dt {o['dt']} ===")
    out = {}

    def run(params, deltas, items):
        nuis = F.nuis_from(prep, offsets=offs, deltas=deltas)
        jobs = [{"cfg": {"opts": o, "params": params}, "nuis": nuis, "items": items[i:i + 10], "tag": 0}
                for i in range(0, len(items), 10)]
        recs = []
        for r in pool().imap_unordered(_teleop_job, jobs):
            recs += r["recs"]
        return F.aggregate(recs)

    for name, params in models_.items():
        best = {}
        scores = {dn: [] for dn in F.PRIMARY}
        for dl in grid:
            ag = run(params, {dn: dl for dn in F.PRIMARY}, it_train)
            for dn in F.PRIMARY:
                scores[dn].append(ag["per_ds"][dn]["arm"])
        for dn in F.PRIMARY:
            best[dn] = grid[int(np.argmin(scores[dn]))]
        test = run(params, best, it_test)
        out[name] = {"delay_per_arm_s": best, "test": test}
        log(f"  {name:10s} TEST arm-mean {test['OVERALL']['arm']:.3f} (verified-3 {test['VERIFIED']['arm']:.3f}); "
            f"per arm " + " ".join(f"{dn[:6]} {test['per_ds'][dn]['arm']:.2f}" for dn in F.PRIMARY))
    jdump(out, os.path.join(CACHE, "teleop.json"))


# ------------------------------------------------------------------------------------------------ report
def sign_test_p(wins, n):
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def stage_report():
    fin = jload(os.path.join(CACHE, "final.json"))
    bl = jload(os.path.join(CACHE, "baselines.json"))
    fs = jload(os.path.join(CACHE, "frictionsel.json"))
    fa = jload(os.path.join(CACHE, "fit_arm.json"))
    fj = jload(os.path.join(CACHE, "fit_jaw.json"))
    pre = jload(os.path.join(RESULTS, "anchor_prereg.json"))
    tele = jload(os.path.join(CACHE, "teleop.json")) if os.path.exists(os.path.join(CACHE, "teleop.json")) else None
    M = fin["models"]
    motions = sorted(M["fitted"]["per_motion"])
    L = ["# SO-101 arm fit on NVIDIA Anchor-Lab (held-out, scored once)", ""]
    L.append(f"Fitted JSON sha256 `{pre['sha256'][:16]}…` frozen {pre['frozen_at']} before held-out was read. "
             f"Friction setting `{pre['friction']}`; delay {1e3 * fa['delay_s']:.1f} ms; dt {fa['opts']['dt']} s.")
    L.append("")
    L.append("## Held-out, whole motions (deg RMSE; equal weight per motion; arm = mean of 5 arm joints)")
    L.append("")
    L.append("| model | arm | moving | holding | pan | lift | elbow | w_flex | w_roll | jaw |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(M, key=lambda n: M[n]["summary"]["arm"])
    for n in order:
        s = M[n]["summary"]
        L.append(f"| {n} | {s['arm']:.3f} | {s['arm_move']:.3f} | {s['arm_hold']:.3f} | "
                 + " | ".join(f"{v:.2f}" for v in s["joint"][:5]) + f" | {s['jaw'] if s['jaw'] is not None else float('nan'):.2f} |")
    L.append("")
    L.append("## Per motion (arm deg): fitted vs NVIDIA fit vs plain delay")
    L.append("")
    L.append("| motion | fitted | B_nvidia (same opts) | B_nvidia (cam 23.1) | pure delay | perfect tracking |")
    L.append("|---|---|---|---|---|---|")
    for mn in motions:
        L.append(f"| {mn} | {M['fitted']['per_motion'][mn]['arm']:.3f} | "
                 f"{M['B_nvidia_same_opts']['per_motion'][mn]['arm']:.3f} | "
                 f"{M['B_nvidia_cam23.1']['per_motion'][mn]['arm']:.3f} | {M['pure_delay']['per_motion'][mn]['arm']:.3f} | "
                 f"{M['perfect_tracking']['per_motion'][mn]['arm']:.3f} |")
    L.append("")
    stats = {}
    for ref in ("B_nvidia_same_opts", "B_nvidia_cam23.1", "pure_delay"):
        for key in ("arm", "arm_move"):
            w = sum(1 for mn in motions if (M["fitted"]["per_motion"][mn][key] or 9e9)
                    < (M[ref]["per_motion"][mn][key] or 9e9))
            stats[f"{ref}:{key}"] = {"wins": w, "n": len(motions), "p_one_sided": sign_test_p(w, len(motions))}
    L.append("## Sign tests (fitted better on k of 8 held-out motions; one-sided)")
    L.append("")
    for k, v in stats.items():
        L.append(f"- vs {k}: {v['wins']}/{v['n']}, p = {v['p_one_sided']:.3f}")
    L.append("")
    L.append("## Model selection evidence (VAL whole motions; held-out not used)")
    L.append("")
    for k, v in fs.items():
        if k.startswith("_"):
            continue
        L.append(f"- {k}: VAL arm fitted {v['val_whole_fitted']['arm']:.3f} vs B {v['val_whole_B']['arm']:.3f}; "
                 f"creep (30 s hold) B {v['creep_B']['max_drift_deg']:.2f} deg, fitted "
                 f"{v['creep_fitted']['max_drift_deg']:.2f} deg")
    L.append(f"- chosen: {fs['_choice']['friction']} ({fs['_choice']['rule']})")
    L.append(f"- jaw fit: FIT jaw {fj['objective_fit_jaw']:.3f} deg from {fj['start']:.3f}; VAL jaw "
             f"{fj['val_jaw_whole']['jaw']:.3f}")
    L.append("")
    if tele:
        L.append("## Teleop cross-check (6 public arms, unknown voltage; robustness only; per-arm delay fitted on TRAIN)")
        L.append("")
        for n, v in tele.items():
            L.append(f"- {n}: TEST arm-mean {v['test']['OVERALL']['arm']:.3f} deg "
                     f"(verified-3 {v['test']['VERIFIED']['arm']:.3f})")
        L.append("")
    js = jload(FITTED_JSON)
    L.append("## Fitted parameters (SI per rad)")
    L.append("")
    L.append("| joint | kp | kv | damping | frictionloss | armature | forcerange |")
    L.append("|---|---|---|---|---|---|---|")
    for j in JOINTS:
        r = js["joints"][j]
        L.append(f"| {j} | {r['kp']:.2f} | {r['kv']:.3f} | {r['damping']:.3f} | {r['frictionloss']:.3f} | "
                 f"{r['armature']:.4f} | {r['forcerange']:.2f} |")
    with open(os.path.join(RESULTS, "anchor_table.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    jdump(stats, os.path.join(CACHE, "stats.json"))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["profile", "baselines", "frictionsel", "fit_arm", "fit_jaw", "freeze", "final",
                                      "teleop", "report"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--budget", type=float, default=None, help="minutes")
    a = ap.parse_args()
    low_priority()
    os.makedirs(CACHE, exist_ok=True)
    t = time.perf_counter()
    if a.stage == "profile":
        stage_profile()
    elif a.stage == "baselines":
        stage_baselines()
    elif a.stage == "frictionsel":
        stage_frictionsel(gens=a.gens or 60, resume=a.resume)
    elif a.stage == "fit_arm":
        stage_fit_arm(gens=a.gens or 250, budget_min=a.budget or 25.0, resume=a.resume)
    elif a.stage == "fit_jaw":
        stage_fit_jaw(gens=a.gens or 80, budget_min=a.budget or 10.0, resume=a.resume)
    elif a.stage == "freeze":
        stage_freeze()
    elif a.stage == "final":
        stage_final()
    elif a.stage == "teleop":
        stage_teleop()
    elif a.stage == "report":
        stage_report()
    log(f"stage {a.stage} finished in {time.perf_counter() - t:.0f} s")


if __name__ == "__main__":
    main()

"""Step 4: fit the SO-101 arm models to REAL TRAIN data and report held-out TEST error.

Re-uses replay.py (data loading, split, units) and models.py (make_sim, BamServoBank) unchanged; the old
replay.simulate_window path is untouched, so results/baseline_table.md stays reproducible. Everything new
lives here and writes only results/fit_*, results/fit_cache/ and so101_fitted.json.

Protocol ("new protocol"), fixing the verifier issues
-------------------------------------------------------
Fix 1  Ranges. Per dataset and arm joint, a REAL hard stop is detected on TRAIN states: the robust extreme
       (0.02 / 99.98 percentile) is within 15 deg of (or beyond) the CAD limit at that end AND at least
       max(30, 0.2 %) of TRAIN frames sit within 0.5 deg of it (rest-pose stops: shoulder_lift ~-103,
       elbow ~+96, wrist_flex ~+100, wrist_roll ~+166). A detected stop becomes the sim joint limit at that
       end and also clamps the commanded target (the servo holds Min/Max_Position_Limit). Ends without a stop
       get a limit 10 deg beyond both the CAD limit and the observed extreme (i.e. never binding).
       Mid-range obstructions (table contact, e.g. DecisionFacts elbow at -59) are NOT turned into limits.
Fix 2  Sub-frame transport delay delta. action[t] is applied delta s after state[t] is read
       (command held piecewise-constant from t*T+delta to (t+1)*T+delta; per physics step the target is the
       box-filtered average, so the objective is continuous in delta). delta in [0, 33.3] ms is a per-dataset
       nuisance fitted on TRAIN. Before the fit it is SHARED by all models (grid 0..30 ms, argmin of the mean
       over A-D TRAIN error); fitted models re-fit it jointly with their parameters. Frame period T = 1/30 s
       is an assumption (timestamps are synthetic).
Fix 4  Offsets. q_sim = q_lerobot + offset (state AND action) for shoulder_lift, elbow_flex, wrist_flex,
       wrist_roll, per dataset. Before the fit: offsets from the flat-table FK test alone (pick-moment jaw
       z flat vs reach and near table level; minimum-norm solution with a prior). After the fit: refined on
       TRAIN tracking error within +-5 deg, constrained to keep the FK test inside its 2-sigma band.
Fix 5  Warm start. Every window starts K=8 frames early from the real state at t0-K and is driven open loop
       by action[t0-K .. t0+N-1]; only state[t0+1 .. t0+N] is scored. Windows are tiled from frame K.
Fix 6  Reported for all 6 primary datasets and for the provenance-verified subset (dobri420,
       HashtagRobotics, DecisionFacts).

Run stages:  python fit.py prep | before | fit1 | offsets | fit1db | fit2 | final | report   (or: all)
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = (r"C:/Users/valen/AppData/Local/Temp/claude/C--Users-valen-Documents-vla-distill/"
           r"c1e14d5c-c49d-4f52-b957-d99b0501209c/scratchpad")
BAM20 = SCRATCH + "/src/bam20"   # Rhoban/bam main e9a619d + PR#20 (adds vin arg + STS3215 12V params only)
if os.path.isdir(BAM20):
    os.environ.setdefault("BAM_SRC", BAM20)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
# one math thread per worker process (spawned workers inherit this); set before numpy is imported
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(RESULTS, "fit_cache")
LOG = os.path.join(RESULTS, "fit_log.txt")

FPS = 30
PERIOD = 1.0 / FPS
DT = 0.002
K_WARM = 8
ARM = 5
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PRIMARY = ["dobri420", "HashtagRobotics", "DecisionFacts", "Chaenn", "dharak22", "vasco281204"]
VERIFIED = ["dobri420", "HashtagRobotics", "DecisionFacts"]
CAD_DEG = np.array([[-110.0, 110.0], [-100.0, 100.0], [-96.8296, 96.8296], [-95.0, 95.0],
                    [-157.2113, 162.7887], [-10.0, 100.0]])
OFFSET_JOINTS = [1, 2, 3, 4]
DELTA_MAX = PERIOD
DELTA_GRID_MS = [0, 5, 10, 15, 20, 25, 30]
N_FIT_WIN = 20      # TRAIN W=4 windows per dataset used during search (fixed random subset)
N_FIT_WIN_BAM = 12  # subset of those for the BAM fit
# capped: the 2026-10-15 run at 16 procs / 100 % CPU ended in a laptop crash
NPROC = max(1, min(int(os.environ.get("ARM_SYSID_PROCS", "6")), (os.cpu_count() or 2)))
FRIC_PARAMS = ["friction_base", "friction_stribeck", "load_friction_motor", "load_friction_external",
               "load_friction_motor_stribeck", "load_friction_external_stribeck", "load_friction_motor_quad",
               "load_friction_external_quad", "friction_viscous"]


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ================================================================================================
# Data
# ================================================================================================
_DS = None


def datasets():
    global _DS
    if _DS is None:
        import replay
        _DS = {d["name"]: d for d in replay.load_datasets() if d["name"] in PRIMARY}
    return _DS


def make_windows(ds, split, W):
    N = int(round(W * FPS))
    out = []
    for ep in ds[split]:
        T = ds["episodes"][ep]["T"]
        t0 = K_WARM
        while t0 + N <= T - 1:
            out.append((int(ep), int(t0)))
            t0 += N
    return out


# ================================================================================================
# Nuisance 1: real hard stops (TRAIN only)
# ================================================================================================
def detect_ranges(ds):
    S = np.degrees(np.concatenate([ds["episodes"][e]["S"] for e in ds["train"]]))
    A = np.degrees(np.concatenate([ds["episodes"][e]["A"] for e in ds["train"]]))
    n = len(S)
    lo = CAD_DEG[:, 0].copy()
    hi = CAD_DEG[:, 1].copy()
    info = {}
    for j in range(ARM):
        rmin, rmax = np.percentile(S[:, j], [0.02, 99.98])
        rec = {}
        for end, ext, cad in (("lo", rmin, CAD_DEG[j, 0]), ("hi", rmax, CAD_DEG[j, 1])):
            near = (S[:, j] <= ext + 0.5) if end == "lo" else (S[:, j] >= ext - 0.5)
            beyond = (A[:, j] < S[:, j] - 1.0) if end == "lo" else (A[:, j] > S[:, j] + 1.0)
            dist_cad = (ext - cad) if end == "lo" else (cad - ext)   # >0: inside CAD range by that much
            n_near = int(near.sum())
            stop = (dist_cad <= 15.0) and (n_near >= max(30, int(0.002 * n)))
            if stop:
                val = float(ext)
            else:
                val = float(min(cad, ext) - 10.0) if end == "lo" else float(max(cad, ext) + 10.0)
            rec[end] = {"stop": bool(stop), "limit_deg": round(val, 3), "robust_extreme_deg": round(float(ext), 3),
                        "deg_inside_cad": round(float(dist_cad), 2), "frames_near": n_near,
                        "frames_near_and_cmd_beyond": int((near & beyond).sum())}
            if end == "lo":
                lo[j] = val
            else:
                hi[j] = val
        info[JOINTS[j]] = rec
    return lo, hi, info


# ================================================================================================
# Nuisance 4: FK flat-table test (TRAIN pick moments) -> offsets
# ================================================================================================
_FK = None


def _fk_model():
    global _FK
    if _FK is None:
        import mujoco
        import models
        m = mujoco.MjModel.from_xml_path(models.MENAGERIE_XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        _FK = (m, d, m.site("gripperframe").id, d.xpos[m.body("shoulder").id][:2].copy(),
               np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS]))
    return _FK


def pick_events(ds):
    """First grasp onset per TRAIN episode (gripper state - action > 5 for >= 10 frames): state in deg."""
    import replay
    out = []
    for ep in ds["train"]:
        e = ds["episodes"][ep]
        g = e["grasp"]
        idx = np.flatnonzero(g)
        if len(idx) == 0:
            continue
        out.append(np.degrees(e["S"][idx[0]]))
    return np.array(out)


def fk_stats(qdeg, offset_deg):
    import mujoco
    m, d, sid, pan_xy, qadr = _fk_model()
    z, r = [], []
    for q in qdeg:
        qq = q + offset_deg
        d.qpos[:] = 0
        d.qpos[qadr] = np.radians(qq)
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[sid]
        z.append(p[2])
        r.append(math.hypot(p[0] - pan_xy[0], p[1] - pan_xy[1]))
    z, r = np.array(z), np.array(r)
    slope, icpt = np.polyfit(r, z, 1)
    resid = z - (slope * r + icpt)
    se = float(resid.std(ddof=2) / (r.std() * math.sqrt(len(r)))) if len(r) > 3 else float("inf")
    return {"n": int(len(z)), "z_med_cm": float(100 * np.median(z)), "slope": float(slope), "slope_se": se,
            "reach_p10_p90_cm": [float(100 * x) for x in np.percentile(r, [10, 90])],
            "resid_std_cm": float(100 * resid.std())}


def fk_usable(st):
    return st["n"] >= 20 and (st["reach_p10_p90_cm"][1] - st["reach_p10_p90_cm"][0]) >= 3.0 and st["resid_std_cm"] < 3.0


Z_SIGMA_CM = 1.5      # unknown object height / gripperframe height above the table at a pick
PRIOR_SIGMA = np.array([0, 3.0, 3.0, 3.0, 5.0, 0])


def fk_chi2(picks, off6, se_ref):
    st = fk_stats(picks, off6)
    return (st["slope"] / se_ref) ** 2 + (st["z_med_cm"] / Z_SIGMA_CM) ** 2, st


def fk_offsets(picks):
    """Minimum-norm offsets (lift, elbow, wflex, roll) that make the pick plane flat (slope 0, z ~ 0)."""
    from scipy.optimize import minimize
    st0 = fk_stats(picks, np.zeros(6))
    if not fk_usable(st0):
        return np.zeros(6), st0, st0, False
    se = max(st0["slope_se"], 0.005)

    def obj(x):
        off = np.zeros(6)
        off[OFFSET_JOINTS] = x
        c, _ = fk_chi2(picks, off, se)
        return c + float(np.sum((x / PRIOR_SIGMA[OFFSET_JOINTS]) ** 2))

    simplex = np.vstack([np.zeros(4), np.eye(4)])     # 1-deg initial simplex (default is 0.00025 at x0=0)
    r = minimize(obj, np.zeros(4), method="Nelder-Mead",
                 options={"xatol": 0.01, "fatol": 1e-5, "maxiter": 2000, "initial_simplex": simplex})
    off = np.zeros(6)
    off[OFFSET_JOINTS] = r.x
    return off, st0, fk_stats(picks, off), True


def fk_band_penalty(picks, off6, se_ref):
    """0 inside the 2-sigma band of the FK test, quadratic outside (used when refining offsets)."""
    st = fk_stats(picks, off6)
    a = max(0.0, abs(st["slope"]) / se_ref - 2.0)
    b = max(0.0, abs(st["z_med_cm"]) / Z_SIGMA_CM - 2.0)
    return a * a + b * b, st


# ================================================================================================
# Window preparation (sim coordinates = data + offset)
# ================================================================================================
def window_arrays(ds, ep, t0, N, nu):
    """-> C (K+N+1, 6) clamped commands in sim coords (row 0 = command before the window start),
    q0, v0 (sim coords), real (N, 6) data coords, scored state indices t0+1..t0+N."""
    e = ds["episodes"][ep]
    A, S = e["A"], e["S"]
    off = np.radians(np.asarray(nu["offset_deg"]))
    lo = np.radians(np.asarray(nu["lo_deg"])) + off
    hi = np.radians(np.asarray(nu["hi_deg"])) + off
    ts = t0 - K_WARM
    cmd = A[ts:t0 + N] + off
    prev = (A[ts - 1] + off) if ts >= 1 else (S[ts] + off)
    C = np.clip(np.vstack([prev[None], cmd]), lo, hi)
    q0 = S[ts] + off
    v0 = (S[ts] - S[ts - 1]) * FPS if ts >= 1 else np.zeros(6)
    qc = np.clip(q0, lo, hi)
    v0 = np.where(qc != q0, 0.0, v0)
    return C, qc, v0, S[t0 + 1:t0 + N + 1], off, lo, hi


def schedule(delta, period, n_frames):
    """Per physics step: indices into C (row 0 = previous command) and blend weight, box-filtered so the
    target is continuous in delta. Also observation interpolation for state reads at (i+1)*period."""
    nsteps = int(math.ceil(n_frames * period / DT)) + 1
    a = np.arange(nsteps) * DT - delta
    ia = np.floor(a / period).astype(int)              # frame index whose command is active at step start
    boundary = (ia + 1) * period
    w = np.clip((boundary - a) / DT, 0.0, 1.0)         # fraction of the step still in frame ia
    i1 = np.clip(ia + 1, 0, n_frames)
    i2 = np.clip(ia + 2, 0, n_frames)
    tobs = (np.arange(n_frames) + 1) * period
    nf = tobs / DT
    n0 = np.floor(nf).astype(int)
    fr = nf - n0
    return nsteps, i1, i2, w, n0, fr


# ================================================================================================
# Simulation configs (worker side)
# ================================================================================================
_SIMS = {}


def _base_pos_params(variant):
    import models
    s = models.make_sim(variant, dt=DT)
    m = s.model
    p = {"kp": m.actuator_gainprm[s.aid, 0].copy(), "kv": -m.actuator_biasprm[s.aid, 2].copy(),
         "forcerange": m.actuator_forcerange[s.aid, 1].copy(), "damping": m.dof_damping[s.vadr].copy(),
         "frictionloss": m.dof_frictionloss[s.vadr].copy(), "armature": m.dof_armature[s.vadr].copy()}
    return s, p


def base_pos_params(variant):
    _, p = _base_pos_params(variant)
    return {k: v.tolist() for k, v in p.items()}


def get_sim(cfg):
    """cfg: {"kind": "pos", "variant": "A_stock"|"B_nvidia", "params": {...6-arrays...}, "deadband_deg": 6 or None}
         or {"kind": "bam", "model": "m1"|"m6", "set": "7.4V"|"12V", "vin": float, "fric_scale": 6, "egr_scale": 6}"""
    import mujoco
    import models
    if cfg["kind"] == "pos":
        key = ("pos", cfg["variant"])
        if key not in _SIMS:
            s, _ = _base_pos_params(cfg["variant"])
            for k, j in enumerate(JOINTS):
                s.model.actuator_ctrllimited[s.aid[k]] = 0     # targets are clamped in numpy (stops)
            _SIMS[key] = s
        s = _SIMS[key]
        m = s.model
        p = cfg["params"]
        m.actuator_gainprm[s.aid, 0] = p["kp"]
        m.actuator_biasprm[s.aid, 1] = -np.asarray(p["kp"])
        m.actuator_biasprm[s.aid, 2] = -np.asarray(p["kv"])
        m.actuator_forcerange[s.aid, 0] = -np.asarray(p["forcerange"])
        m.actuator_forcerange[s.aid, 1] = np.asarray(p["forcerange"])
        m.dof_damping[s.vadr] = p["damping"]
        m.dof_frictionloss[s.vadr] = p["frictionloss"]
        if not np.allclose(m.dof_armature[s.vadr], p["armature"]):
            m.dof_armature[s.vadr] = p["armature"]
            mujoco.mj_setConst(m, s.data)
        return s
    key = ("bam", cfg["model"], cfg["set"])
    if key not in _SIMS:
        variant = "C_bam_m1" if cfg["model"] == "m1" else "D_bam_m6"
        s = models.make_sim(variant, dt=DT, vin=float(cfg["vin"]))
        if cfg["set"] == "12V":
            path = models.bam_params_path(cfg["model"], "feetech_sts3215_12V")
            s.bank = models.BamServoBank(s.model, s.data, JOINTS, path, kp_fw=16.0, vin=float(cfg["vin"]))
        bm = s.bank.bam_model
        s._base = {"egr": float(bm.error_gain_ratio.value)}
        # Per-joint friction scale: BAM's friction budget (Model.compute_frictions) is linear in all of its
        # friction parameters, so scaling its OUTPUT per joint == scaling every friction parameter of that
        # joint. The parameters themselves must stay scalar: compute_frictions does
        # `frictionloss = self.friction_base.value; frictionloss += ...`, which would mutate a numpy array
        # parameter in place (M6 then accumulates every step).
        s._fric_scale = np.ones(6)
        orig = bm.compute_frictions

        def scaled_frictions(motor_torque, external_torque, dtheta, _orig=orig, _s=s):
            fl, dmp = _orig(motor_torque, external_torque, dtheta)
            return fl * _s._fric_scale, dmp * _s._fric_scale

        bm.compute_frictions = scaled_frictions
        _SIMS[key] = s
    s = _SIMS[key]
    bm = s.bank.bam_model
    bm.actuator.vin = float(cfg["vin"])
    # error_gain_ratio enters compute_control only as a factor (no in-place op), so a per-joint array is safe
    bm.error_gain_ratio.value = s._base["egr"] * np.asarray(cfg.get("egr_scale", np.ones(6)), float)
    s._fric_scale = np.asarray(cfg.get("fric_scale", np.ones(6)), float)
    return s


def run_window(s, C, q0, v0, nsteps, i1, i2, w, deadband=None):
    import mujoco
    m, d = s.model, s.data
    s.reset(q0)
    d.qvel[0:6] = v0
    mujoco.mj_forward(m, d)
    if s.bank is not None:
        s.bank.reset()
    U = C[i1] * w[:, None] + C[i2] * (1.0 - w)[:, None]
    Q = np.empty((nsteps + 1, 6))
    Q[0] = d.qpos[0:6]
    step = mujoco.mj_step
    if s.bank is not None:
        bank = s.bank
        for n in range(nsteps):
            bank.set_target(U[n])
            bank.before_step()
            step(m, d)
            Q[n + 1] = d.qpos[0:6]
    elif deadband is not None:
        db = deadband
        for n in range(nsteps):
            q = d.qpos[0:6]
            e = U[n] - q
            d.ctrl[0:6] = q + np.sign(e) * np.maximum(np.abs(e) - db, 0.0)
            step(m, d)
            Q[n + 1] = d.qpos[0:6]
    else:
        for n in range(nsteps):
            d.ctrl[0:6] = U[n]
            step(m, d)
            Q[n + 1] = d.qpos[0:6]
    return Q


def sim_job(job):
    """job = {cfg, nuis: {ds: nu}, items: [(ds, ep, t0, W)], want_pred, tag} -> per-window SSE (deg^2)."""
    dsets = datasets()
    cfg = job["cfg"]
    s = get_sim(cfg)
    db = None
    if cfg["kind"] == "pos" and cfg.get("deadband_deg") is not None and np.any(np.asarray(cfg["deadband_deg"]) > 0):
        db = np.radians(np.asarray(cfg["deadband_deg"], float))
    out = []
    t_cpu = time.process_time()
    sched = {}
    for (dn, ep, t0, W) in job["items"]:
        ds = dsets[dn]
        nu = job["nuis"][dn]
        N = int(round(W * FPS))
        key = (dn, N)
        if key not in sched:
            sched[key] = schedule(float(nu["delta_s"]), float(nu.get("period_s", PERIOD)), K_WARM + N)
        nsteps, i1, i2, w, n0, fr = sched[key]
        C, q0, v0, real, off, lo, hi = window_arrays(ds, ep, t0, N, nu)
        jid = [s.model.joint(j).id for j in JOINTS]
        s.model.jnt_range[jid, 0] = lo
        s.model.jnt_range[jid, 1] = hi
        Q = run_window(s, C, q0, v0, nsteps, i1, i2, w, db)
        Qo = Q[n0] * (1 - fr)[:, None] + Q[np.minimum(n0 + 1, nsteps)] * fr[:, None]
        pred = Qo[K_WARM:K_WARM + N] - off                 # data coords
        err = np.degrees(pred - real)
        if not np.all(np.isfinite(err)):
            err = np.nan_to_num(err, nan=1e3, posinf=1e3, neginf=-1e3)
        rec = {"ds": dn, "ep": int(ep), "t0": int(t0), "W": W, "n": N, "sse": (err ** 2).sum(0).tolist()}
        if job.get("want_pred"):
            rec["pred_deg"] = np.degrees(Qo - off).tolist()          # includes warm-up frames
        out.append(rec)
    return {"tag": job.get("tag"), "recs": out, "cpu": time.process_time() - t_cpu}


def _init_worker():
    datasets()


_POOL = None


def pool():
    global _POOL
    if _POOL is None:
        _POOL = mp.get_context("spawn").Pool(NPROC, initializer=_init_worker)
    return _POOL


def run_many(cfgs_nuis, items, chunk=12, want_pred=False):
    """Evaluate several (cfg, nuis) pairs on the same window items in one pool pass.
    -> list (per pair) of record lists."""
    jobs = []
    for ci, (cfg, nuis) in enumerate(cfgs_nuis):
        for i in range(0, len(items), chunk):
            jobs.append({"cfg": cfg, "nuis": nuis, "items": items[i:i + chunk], "want_pred": want_pred, "tag": ci})
    res = [[] for _ in cfgs_nuis]
    for r in pool().imap_unordered(sim_job, jobs):
        res[r["tag"]] += r["recs"]
    return res


# ================================================================================================
# Aggregation
# ================================================================================================
def aggregate(recs, names=PRIMARY):
    """pooled RMSE per dataset per joint; arm mean; OVERALL (6) and VERIFIED (3) equal-weight means."""
    per = {}
    for dn in names:
        rr = [r for r in recs if r["ds"] == dn]
        if not rr:
            continue
        sse = np.sum([r["sse"] for r in rr], 0)
        n = sum(r["n"] for r in rr)
        rm = np.sqrt(sse / n)
        per[dn] = {"joint": rm.tolist(), "arm": float(rm[:ARM].mean()), "n_windows": len(rr), "n_frames": int(n)}
    out = {"per_ds": per}
    for label, sub in (("OVERALL", PRIMARY), ("VERIFIED", VERIFIED)):
        v = [per[d] for d in sub if d in per]
        if v:
            out[label] = {"arm": float(np.mean([x["arm"] for x in v])),
                          "joint": np.mean([x["joint"] for x in v], 0).tolist(), "n_datasets": len(v)}
    return out


def objective_from_recs(recs):
    a = aggregate(recs)
    return a["OVERALL"]["arm"], a


# ================================================================================================
# Baselines (no physics), with the same per-dataset stops (clip) as the sims
# ================================================================================================
def frac_delayed(A, d):
    d0 = int(math.floor(d))
    f = d - d0
    idx0 = np.maximum(np.arange(len(A)) - d0, 0)
    idx1 = np.maximum(np.arange(len(A)) - d0 - 1, 0)
    return (1 - f) * A[idx0] + f * A[idx1]


def baseline_recs(ds, wins, W, nu, kind, d=None, clip=True):
    N = int(round(W * FPS))
    lo = np.radians(np.asarray(nu["lo_deg"]))
    hi = np.radians(np.asarray(nu["hi_deg"]))
    out = []
    for ep, t0 in wins:
        e = ds["episodes"][ep]
        A, S = e["A"], e["S"]
        if kind == "perfect_tracking":
            P = A[t0:t0 + N]
        else:
            P = frac_delayed(A, d)[t0 + 1:t0 + N + 1]
        if clip:
            P = np.clip(P, lo, hi)
        err = np.degrees(P - S[t0 + 1:t0 + N + 1])
        out.append({"ds": ds["name"], "ep": ep, "t0": t0, "W": W, "n": N, "sse": (err ** 2).sum(0).tolist()})
    return out


def fit_frac_delay(ds, wins, nu, clip=True):
    grid = np.round(np.arange(0, 15.0001, 0.05), 3)
    best = None
    curve = []
    for d in grid:
        a = aggregate(baseline_recs(ds, wins, 4, nu, "delay", d, clip), [ds["name"]])["per_ds"][ds["name"]]["arm"]
        curve.append(a)
        if best is None or a < best[1]:
            best = (float(d), a)
    return best[0], best[1]


# ================================================================================================
# Stage: prep
# ================================================================================================
def stage_prep():
    log("=== stage prep: ranges, FK offsets, windows, fractional-delay baseline (TRAIN only) ===")
    dsets = datasets()
    prep = {"datasets": {}}
    rng = np.random.default_rng(0)
    for dn in PRIMARY:
        ds = dsets[dn]
        lo, hi, rinfo = detect_ranges(ds)
        picks = pick_events(ds)
        if len(picks) >= 5:
            off, st0, st1, usable = fk_offsets(picks)
        else:
            off, st0, st1, usable = np.zeros(6), {"n": int(len(picks))}, {"n": int(len(picks))}, False
        nu = {"offset_deg": off.tolist(), "delta_s": 0.0, "period_s": PERIOD, "lo_deg": lo.tolist(), "hi_deg": hi.tolist()}
        wtr4 = make_windows(ds, "train", 4)
        sub = sorted(rng.choice(len(wtr4), size=min(N_FIT_WIN, len(wtr4)), replace=False).tolist())
        fitw = [wtr4[i] for i in sub]
        d_clip, a_clip = fit_frac_delay(ds, wtr4, nu, clip=True)
        d_raw, a_raw = fit_frac_delay(ds, wtr4, nu, clip=False)
        prep["datasets"][dn] = {
            "ranges": rinfo, "lo_deg": lo.tolist(), "hi_deg": hi.tolist(),
            "fk_usable": usable, "fk_identity": st0, "fk_after_offsets": st1, "fk_offsets_deg": off.tolist(),
            "n_picks": int(len(picks)), "fit_windows": fitw, "fit_windows_bam": fitw[:N_FIT_WIN_BAM],
            "best_pure_delay_frames": {"clipped": d_clip, "unclipped": d_raw},
            "best_pure_delay_train_arm": {"clipped": a_clip, "unclipped": a_raw},
            "n_train_windows_W4": len(wtr4),
        }
        stops = [f"{j}:{e}={r[e]['limit_deg']}" for j, r in rinfo.items() for e in ("lo", "hi") if r[e]["stop"]]
        log(f"{dn:16s} stops {stops}")
        log(f"{'':16s} FK usable={usable} picks={len(picks)} identity slope {st0.get('slope', float('nan')):+.3f} "
            f"(se {st0.get('slope_se', float('nan')):.3f}) z {st0.get('z_med_cm', float('nan')):+.2f} cm -> offsets "
            f"{np.round(off[1:5], 2).tolist()} deg; best pure delay {d_clip:.2f} fr (clipped) {d_raw:.2f} (raw)")
    jdump(prep, os.path.join(CACHE, "prep.json"))
    return prep


def nuis_from(prep, offsets=None, deltas=None):
    nu = {}
    for dn in PRIMARY:
        p = prep["datasets"][dn]
        nu[dn] = {"offset_deg": (offsets or {}).get(dn, p["fk_offsets_deg"]),
                  "delta_s": float((deltas or {}).get(dn, 0.0)), "period_s": PERIOD,
                  "lo_deg": p["lo_deg"], "hi_deg": p["hi_deg"]}
    return nu


def items_for(prep, which):
    dsets = datasets()
    it = []
    for dn in PRIMARY:
        if which == "fit":
            it += [(dn, ep, t0, 4) for ep, t0 in prep["datasets"][dn]["fit_windows"]]
        elif which == "fit_bam":
            it += [(dn, ep, t0, 4) for ep, t0 in prep["datasets"][dn]["fit_windows_bam"]]
        else:
            split, W = which
            it += [(dn, ep, t0, W) for ep, t0 in make_windows(dsets[dn], split, W)]
    # long jobs first is handled by chunking; interleave datasets for balance
    return it


# ================================================================================================
# Model configs
# ================================================================================================
def cfg_pos(variant, params=None, deadband=None):
    return {"kind": "pos", "variant": variant, "params": params or base_pos_params(variant),
            "deadband_deg": deadband}


def cfg_bam(model, pset, vin, fric=None, egr=None):
    return {"kind": "bam", "model": model, "set": pset, "vin": float(vin),
            "fric_scale": list(fric) if fric is not None else [1.0] * 6,
            "egr_scale": list(egr) if egr is not None else [1.0] * 6}


def before_cfgs():
    return {
        "A_stock": cfg_pos("A_stock"),
        "B_nvidia": cfg_pos("B_nvidia"),
        "C_bam_m1": cfg_bam("m1", "7.4V", 7.4),
        "D_bam_m6": cfg_bam("m6", "7.4V", 7.4),
        "D_bam_m6_vin5": cfg_bam("m6", "7.4V", 5.0),
        "M6_12Vset_vin12": cfg_bam("m6", "12V", 12.0),
    }


# A_stock is left out of the shared-delta choice: kp 998 with no delay makes it respond within ~1 frame, so its
# TRAIN error falls monotonically with delta and it would push the shared value to the grid edge.
SHARED_DELTA_MODELS = ("B_nvidia", "C_bam_m1", "D_bam_m6")


# ================================================================================================
# Stage: before (TRAIN only: shared delta grid + TRAIN numbers of the stock models)
# ================================================================================================
def stage_before():
    prep = jload(os.path.join(CACHE, "prep.json"))
    log("=== stage before: shared delta grid on the TRAIN fit subset (all variants) ===")
    cfgs = before_cfgs()
    items = items_for(prep, "fit")
    pairs, keys = [], []
    for name, cfg in cfgs.items():
        for dms in DELTA_GRID_MS:
            nu = nuis_from(prep, deltas={dn: dms / 1000.0 for dn in PRIMARY})
            pairs.append((cfg, nu))
            keys.append((name, dms))
    t = time.perf_counter()
    res = run_many(pairs, items, chunk=10)
    log(f"delta grid: {len(pairs)} (model, delta) x {len(items)} windows in {time.perf_counter() - t:.0f} s")
    curves = {}
    for (name, dms), recs in zip(keys, res):
        a = aggregate(recs)
        curves.setdefault(name, {})[dms] = {dn: a["per_ds"][dn]["arm"] for dn in PRIMARY}
        curves[name][dms]["OVERALL"] = a["OVERALL"]["arm"]
    shared = {}
    for dn in PRIMARY:
        # mean over models of each model's curve normalised by its own minimum (no model dominates by level)
        mean_curve = [np.mean([curves[m][dms][dn] / min(curves[m][g][dn] for g in DELTA_GRID_MS)
                               for m in SHARED_DELTA_MODELS]) for dms in DELTA_GRID_MS]
        shared[dn] = DELTA_GRID_MS[int(np.argmin(mean_curve))] / 1000.0
    own = {m: {dn: DELTA_GRID_MS[int(np.argmin([curves[m][dms][dn] for dms in DELTA_GRID_MS]))] / 1000.0
               for dn in PRIMARY} for m in cfgs}
    for m in cfgs:
        log(f"  {m:16s} TRAIN-fit-subset OVERALL arm vs delta(ms): " +
            " ".join(f"{dms}:{curves[m][dms]['OVERALL']:.3f}" for dms in DELTA_GRID_MS) +
            f" | own best per ds (ms) {[int(1000 * own[m][dn]) for dn in PRIMARY]}")
    log(f"  shared delta per dataset (ms): {{{', '.join(f'{dn}: {1000 * shared[dn]:.0f}' for dn in PRIMARY)}}}")
    jdump({"delta_grid_ms": DELTA_GRID_MS, "curves_train_fit_subset": curves, "shared_delta_s": shared,
           "own_delta_s": own}, os.path.join(CACHE, "before.json"))


# ================================================================================================
# CMA-ES helpers
# ================================================================================================
POS_KEYS = ["kp", "kv", "damping", "frictionloss", "armature"]


def decode_fit1(x, base, with_db):
    """x = [25 log-mults (key-major: kp*5, kv*5, ...), 6 delta in units of DELTA_MAX, (5 deadband in deg)]"""
    p = {k: list(base[k]) for k in base}
    for ki, k in enumerate(POS_KEYS):
        for j in range(ARM):
            p[k][j] = base[k][j] * math.exp(x[ki * ARM + j])
    deltas = {dn: float(np.clip(x[25 + i], 0, 1)) * DELTA_MAX for i, dn in enumerate(PRIMARY)}
    db = None
    if with_db:
        db = [float(max(0.0, x[31 + j])) for j in range(ARM)] + [0.0]
    return p, deltas, db


def run_cma(name, x0, stds, lower, upper, evaluate, popsize, max_gens, budget_s, seed=1):
    import cma
    es = cma.CMAEvolutionStrategy(list(x0), 1.0, {
        "bounds": [list(lower), list(upper)], "CMA_stds": list(stds), "popsize": popsize, "seed": seed,
        "verbose": -9, "tolfun": 1e-5, "tolx": 1e-5})
    t0 = time.perf_counter()
    best = (float("inf"), None)
    hist = []
    g = 0
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
        if g % 5 == 0 or g == 1:
            log(f"  [{name}] gen {g:4d} best {np.min(f):.4f} median {np.median(f):.4f} sigma {es.sigma:.3f} "
                f"overall-best {best[0]:.4f} t={time.perf_counter() - t0:.0f}s")
    log(f"  [{name}] stopped after {g} generations ({time.perf_counter() - t0:.0f} s): {es.stop() or 'budget'}; "
        f"best {best[0]:.4f}")
    return best, hist, {k: str(v) for k, v in (es.stop() or {}).items()}


# ================================================================================================
# Stage: fit1 (B_nvidia start, 25 log-multipliers + per-dataset delta), optional deadband
# ================================================================================================
def stage_fit1(with_db=False, budget_min=25.0, max_gens=200, start=None, offsets=None, tag="fit1"):
    prep = jload(os.path.join(CACHE, "prep.json"))
    before = jload(os.path.join(CACHE, "before.json"))
    base = base_pos_params("B_nvidia")
    items = items_for(prep, "fit")
    log(f"=== stage {tag}: CMA-ES from B_nvidia, deadband={'yes' if with_db else 'no'}, "
        f"{len(items)} TRAIN windows (W=4), budget {budget_min} min ===")
    if start is None:
        x0 = [0.0] * 25 + [before["shared_delta_s"][dn] / DELTA_MAX for dn in PRIMARY]
    else:
        x0 = list(start[:31])
    stds = [0.4] * 25 + [0.25] * 6
    if start is not None:
        stds = [0.15] * 25 + [0.1] * 6      # polish around a previous optimum
    lower = [-2.3] * 25 + [0.0] * 6
    upper = [2.3] * 25 + [1.0] * 6
    if with_db:
        x0 += [0.2] * ARM
        stds += [0.25] * ARM
        lower += [0.0] * ARM
        upper += [2.0] * ARM
        if start is not None:
            stds = [0.15] * 25 + [0.1] * 6 + [0.25] * ARM
    offs = offsets or {dn: prep["datasets"][dn]["fk_offsets_deg"] for dn in PRIMARY}

    def evaluate(X):
        pairs = []
        for x in X:
            p, deltas, db = decode_fit1(x, base, with_db)
            pairs.append((cfg_pos("B_nvidia", p, db), nuis_from(prep, offsets=offs, deltas=deltas)))
        res = run_many(pairs, items, chunk=8)
        return [objective_from_recs(r)[0] for r in res]

    f0 = evaluate([x0])[0]
    log(f"  [{tag}] start objective (TRAIN fit subset, OVERALL arm-mean RMSE) {f0:.4f}")
    popsize = 16
    best, hist, why = run_cma(tag, x0, stds, lower, upper, evaluate, popsize, max_gens, budget_min * 60)
    p, deltas, db = decode_fit1(best[1], base, with_db)
    out = {"x": best[1], "objective_train_fit_subset": best[0], "start_objective": f0, "history": hist,
           "stop": why, "params": p, "deltas_s": deltas, "deadband_deg": db, "base": base, "with_deadband": with_db,
           "offsets_deg": offs}
    jdump(out, os.path.join(CACHE, f"{tag}.json"))
    return out


# ================================================================================================
# Stage: offsets (per-dataset refinement with the fitted model; FK 2-sigma band constraint)
# ================================================================================================
def stage_offsets(budget_min=10.0):
    import cma
    prep = jload(os.path.join(CACHE, "prep.json"))
    f1 = jload(os.path.join(CACHE, "fit1.json"))
    log("=== stage offsets: per-dataset offsets (lift, elbow, wflex, roll) refined on TRAIN with the fit1 model ===")
    dsets = datasets()
    cfg = cfg_pos("B_nvidia", f1["params"], None)
    res_all = {}
    t_start = time.perf_counter()
    per_ds_budget = budget_min * 60 / len(PRIMARY)
    for dn in PRIMARY:
        p = prep["datasets"][dn]
        items = [(dn, ep, t0, 4) for ep, t0 in p["fit_windows"]]
        picks = pick_events(dsets[dn])
        usable = p["fk_usable"]
        se = max(p["fk_identity"].get("slope_se", 1.0), 0.005) if usable else None
        fk_off = np.asarray(p["fk_offsets_deg"])

        def nu_for(off4):
            off = np.zeros(6)
            off[OFFSET_JOINTS] = off4
            n = nuis_from(prep, offsets={dn: off.tolist()}, deltas=f1["deltas_s"])
            return off, n

        def evaluate(X):
            pairs = [(cfg, nu_for(x)[1]) for x in X]
            res = run_many(pairs, items, chunk=2)
            f = []
            for x, r in zip(X, res):
                a = aggregate(r, [dn])["per_ds"][dn]["arm"]
                pen = 0.0
                if usable:
                    off6 = nu_for(x)[0]
                    pen, _ = fk_band_penalty(picks, off6, se)
                f.append(a + 0.05 * pen)
            return f

        f_zero = evaluate([np.zeros(4)])[0]
        f_fk = evaluate([fk_off[OFFSET_JOINTS]])[0]
        es = cma.CMAEvolutionStrategy(list(fk_off[OFFSET_JOINTS]), 1.5, {
            "bounds": [[-5] * 4, [5] * 4], "popsize": 8, "seed": 2, "verbose": -9, "tolfun": 1e-4})
        t0 = time.perf_counter()
        best = (f_fk, list(fk_off[OFFSET_JOINTS]))
        g = 0
        while not es.stop() and time.perf_counter() - t0 < per_ds_budget and g < 40:
            X = es.ask()
            f = evaluate(X)
            es.tell(X, f)
            g += 1
            i = int(np.argmin(f))
            if f[i] < best[0]:
                best = (float(f[i]), list(map(float, X[i])))
        # profile: TRAIN-subset RMSE change for +-3 deg on each offset around the best
        prof = {}
        xb = np.array(best[1])
        Xp = []
        for k in range(4):
            for s in (-3.0, 3.0):
                x = xb.copy()
                x[k] = np.clip(x[k] + s, -5, 5)
                Xp.append(x)
        fp = evaluate(Xp)
        for k, jn in enumerate([JOINTS[i] for i in OFFSET_JOINTS]):
            prof[jn] = {"minus3": fp[2 * k] - best[0], "plus3": fp[2 * k + 1] - best[0]}
        off6 = np.zeros(6)
        off6[OFFSET_JOINTS] = best[1]
        fk_after = fk_stats(picks, off6) if len(picks) >= 5 else None
        res_all[dn] = {"offsets_deg": off6.tolist(), "objective": best[0], "objective_zero_offsets": f_zero,
                       "objective_fk_offsets": f_fk, "generations": g, "profile_pm3deg": prof,
                       "fk_usable": usable, "fk_after": fk_after}
        log(f"  {dn:16s} offsets {np.round(off6[1:5], 2).tolist()} deg  obj {best[0]:.4f} (zero {f_zero:.4f}, "
            f"FK-only {f_fk:.4f}), {g} gens; profile +-3deg: " +
            ", ".join(f"{j} {v['minus3']:+.3f}/{v['plus3']:+.3f}" for j, v in prof.items()))
    log(f"  offsets stage {time.perf_counter() - t_start:.0f} s")
    jdump(res_all, os.path.join(CACHE, "offsets.json"))


def fitted_offsets(prep):
    p = os.path.join(CACHE, "offsets.json")
    if os.path.exists(p):
        o = jload(p)
        return {dn: o[dn]["offsets_deg"] for dn in PRIMARY}
    return {dn: prep["datasets"][dn]["fk_offsets_deg"] for dn in PRIMARY}


# ================================================================================================
# Stage: fit2 (BAM M6 start: vin + per-joint friction scale + per-joint error_gain_ratio scale + delta)
# ================================================================================================
def decode_fit2(x, model, pset):
    vin = float(x[0])
    fric = [math.exp(v) for v in x[1:6]] + [1.0]
    egr = [math.exp(v) for v in x[6:11]] + [1.0]
    deltas = {dn: float(np.clip(x[11 + i], 0, 1)) * DELTA_MAX for i, dn in enumerate(PRIMARY)}
    return cfg_bam(model, pset, vin, fric, egr), deltas


def stage_fit2(budget_min=40.0, max_gens=150, model="m6"):
    prep = jload(os.path.join(CACHE, "prep.json"))
    before = jload(os.path.join(CACHE, "before.json"))
    offs = fitted_offsets(prep)
    items = items_for(prep, "fit_bam")
    log(f"=== stage fit2: BAM {model}, {len(items)} TRAIN windows, budget {budget_min} min ===")
    # 1) choose the servo parameter set + vin start on TRAIN (shared deltas)
    cands = [("7.4V", 5.0), ("7.4V", 6.0), ("7.4V", 7.4), ("12V", 12.0)]
    pairs = [(cfg_bam(model, s, v), nuis_from(prep, offsets=offs, deltas=before["shared_delta_s"])) for s, v in cands]
    res = run_many(pairs, items, chunk=4)
    scores = {f"{s}@{v}V": objective_from_recs(r)[0] for (s, v), r in zip(cands, res)}
    log(f"  [fit2] TRAIN-subset start candidates: {scores}")
    out = {"candidates_train_fit_subset": scores, "runs": {}}
    sets = sorted({s for s, _ in cands}, key=lambda s: min(v for k, v in scores.items() if k.startswith(s)))
    budgets = {sets[0]: 0.7 * budget_min, sets[1]: 0.3 * budget_min}
    for pset in sets:
        vin0 = min(((v, scores[f"{s}@{v}V"]) for s, v in cands if s == pset), key=lambda t: t[1])[0]
        vlo, vhi = (4.5, 8.4) if pset == "7.4V" else (10.0, 12.6)
        x0 = [vin0] + [0.0] * 10 + [before["shared_delta_s"][dn] / DELTA_MAX for dn in PRIMARY]
        stds = [0.5 if pset == "7.4V" else 0.4] + [0.3] * 10 + [0.25] * 6
        lower = [vlo] + [-1.6] * 10 + [0.0] * 6
        upper = [vhi] + [1.6] * 10 + [1.0] * 6

        def evaluate(X, pset=pset):
            pr = []
            for x in X:
                cfg, deltas = decode_fit2(x, model, pset)
                pr.append((cfg, nuis_from(prep, offsets=offs, deltas=deltas)))
            r = run_many(pr, items, chunk=3)
            return [objective_from_recs(rr)[0] for rr in r]

        tag = f"fit2_{model}_{pset}"
        best, hist, why = run_cma(tag, x0, stds, lower, upper, evaluate, 12, max_gens, budgets[pset] * 60)
        cfg, deltas = decode_fit2(best[1], model, pset)
        out["runs"][pset] = {"x": best[1], "objective_train_fit_subset": best[0], "history": hist, "stop": why,
                             "cfg": cfg, "deltas_s": deltas, "x0": x0}
    out["offsets_deg"] = offs
    jdump(out, os.path.join(CACHE, "fit2.json"))


# ================================================================================================
# Stage: final evaluation (TRAIN all windows W=4; TEST W=4 and W=1)  -- TEST evaluated only here
# ================================================================================================
def final_models(prep):
    before = jload(os.path.join(CACHE, "before.json"))
    shared = before["shared_delta_s"]
    fk_off = {dn: prep["datasets"][dn]["fk_offsets_deg"] for dn in PRIMARY}
    offs = fitted_offsets(prep)
    rows = {}
    for name, cfg in before_cfgs().items():
        rows[f"before:{name}"] = (cfg, nuis_from(prep, offsets=fk_off, deltas=shared))
    for name in before_cfgs():
        own = before["own_delta_s"][name]
        rows[f"before_owndelta:{name}"] = (before_cfgs()[name], nuis_from(prep, offsets=fk_off, deltas=own))
    for tag in ("fit1", "fit1_refit", "fit1db"):
        p = os.path.join(CACHE, f"{tag}.json")
        if os.path.exists(p):
            f = jload(p)
            o = f.get("offsets_deg", fk_off)
            rows[f"after:{tag}"] = (cfg_pos("B_nvidia", f["params"], f["deadband_deg"]),
                                    nuis_from(prep, offsets=o, deltas=f["deltas_s"]))
            if tag == "fit1":
                rows["after:fit1@shared_delta"] = (cfg_pos("B_nvidia", f["params"], None),
                                                   nuis_from(prep, offsets=o, deltas=shared))
    p = os.path.join(CACHE, "fit2.json")
    if os.path.exists(p):
        f = jload(p)
        for pset, r in f["runs"].items():
            rows[f"after:fit2_{pset}"] = (r["cfg"], nuis_from(prep, offsets=f["offsets_deg"], deltas=r["deltas_s"]))
    return rows


def stage_final(only_train=False):
    prep = jload(os.path.join(CACHE, "prep.json"))
    dsets = datasets()
    rows = final_models(prep)
    log(f"=== stage final: {len(rows)} model rows on TRAIN W=4" + ("" if only_train else ", TEST W=4, TEST W=1") + " ===")
    splits = [("train", 4)] + ([] if only_train else [("test", 4), ("test", 1)])
    path = os.path.join(CACHE, "final.json")
    final = jload(path) if os.path.exists(path) else {}
    for split, W in splits:
        items = items_for(prep, (split, W))
        todo = [k for k in rows if f"{split}{W}" not in final.get(k, {})]
        if not todo:
            continue
        t = time.perf_counter()
        res = run_many([rows[k] for k in todo], items, chunk=16 if W == 4 else 48)
        for k, recs in zip(todo, res):
            final.setdefault(k, {})[f"{split}{W}"] = aggregate(recs)
        log(f"  {split} W={W}: {len(todo)} rows x {len(items)} windows in {time.perf_counter() - t:.0f} s")
        # baselines with the same stops
        for kind in ("perfect_tracking", "best_pure_delay", "best_pure_delay_unclipped", "perfect_tracking_unclipped"):
            recs = []
            for dn in PRIMARY:
                nu = nuis_from(prep)[dn]
                d = prep["datasets"][dn]["best_pure_delay_frames"]["unclipped" if "unclipped" in kind else "clipped"]
                k2 = "perfect_tracking" if kind.startswith("perfect") else "delay"
                recs += baseline_recs(dsets[dn], [(ep, t0) for (n_, ep, t0, W_) in items if n_ == dn], W, nu, k2, d,
                                      clip="unclipped" not in kind)
            final.setdefault(f"baseline:{kind}", {})[f"{split}{W}"] = aggregate(recs)
        jdump(final, path)
    # share of frames at the detected stops / beyond CAD (TEST)
    if not only_train:
        share = {}
        for dn in PRIMARY:
            p = prep["datasets"][dn]
            lo, hi = np.asarray(p["lo_deg"]), np.asarray(p["hi_deg"])
            stop_lo = np.array([p["ranges"][j]["lo"]["stop"] for j in JOINTS[:ARM]])
            stop_hi = np.array([p["ranges"][j]["hi"]["stop"] for j in JOINTS[:ARM]])
            n_at, n_cad, n = 0, 0, 0
            for (n_, ep, t0, W_) in items_for(prep, ("test", 4)):
                if n_ != dn:
                    continue
                S = np.degrees(dsets[dn]["episodes"][ep]["S"][t0 + 1:t0 + 121, :ARM])
                at = ((S <= lo[:ARM] + 0.5) & stop_lo) | ((S >= hi[:ARM] - 0.5) & stop_hi)
                cad = (S < CAD_DEG[:ARM, 0] - 1.0) | (S > CAD_DEG[:ARM, 1] + 1.0)
                n_at += int(at.any(1).sum())
                n_cad += int(cad.any(1).sum())
                n += len(S)
            share[dn] = {"frames": n, "frac_at_detected_stop": n_at / max(n, 1), "frac_beyond_cad_1deg": n_cad / max(n, 1)}
        final["_limit_share_test_w4"] = share
        jdump(final, path)
    log("  final stage done")


# ================================================================================================
# Stage: spread (per-arm refits around the pooled fit -> domain-randomisation ranges)
# ================================================================================================
SPREAD_NAMES = ([f"kp_{JOINTS[j]}" for j in range(ARM)] + [f"frictionloss_{JOINTS[j]}" for j in range(ARM)]
                + ["kv_and_damping_all", "armature_all", "delta"])


def stage_spread(src="fit1", budget_min=6.0):
    """Per dataset (= per physical arm): re-fit per-joint kp and frictionloss multipliers, one shared
    kv+damping multiplier, one armature multiplier and delta, starting from the pooled fit, on that arm's
    TRAIN fit subset. The spread over the 6 arms is the evidence for the domain-randomisation ranges."""
    import cma
    prep = jload(os.path.join(CACHE, "prep.json"))
    f = jload(os.path.join(CACHE, f"{src}.json"))
    offs = f.get("offsets_deg") or fitted_offsets(prep)
    base = f["params"]
    db = f.get("deadband_deg")
    log(f"=== stage spread: per-arm refits around {src} ===")
    out = {}
    per_budget = budget_min * 60 / len(PRIMARY)
    for dn in PRIMARY:
        items = [(dn, ep, t0, 4) for ep, t0 in prep["datasets"][dn]["fit_windows"]]

        def dec(x):
            p = {k: list(base[k]) for k in base}
            for j in range(ARM):
                p["kp"][j] = base["kp"][j] * math.exp(x[j])
                p["frictionloss"][j] = base["frictionloss"][j] * math.exp(x[ARM + j])
                p["kv"][j] = base["kv"][j] * math.exp(x[10])
                p["damping"][j] = base["damping"][j] * math.exp(x[10])
                p["armature"][j] = base["armature"][j] * math.exp(x[11])
            return p, float(np.clip(x[12], 0, 1)) * DELTA_MAX

        def evaluate(X):
            pairs = []
            for x in X:
                p, dl = dec(x)
                pairs.append((cfg_pos("B_nvidia", p, db), nuis_from(prep, offsets=offs, deltas={dn: dl})))
            res = run_many(pairs, items, chunk=2)
            return [aggregate(r, [dn])["per_ds"][dn]["arm"] for r in res]

        x0 = [0.0] * 12 + [f["deltas_s"][dn] / DELTA_MAX]
        f_pooled = evaluate([x0])[0]
        es = cma.CMAEvolutionStrategy(x0, 1.0, {"bounds": [[-1.6] * 12 + [0.0], [1.6] * 12 + [1.0]],
                                                "CMA_stds": [0.3] * 12 + [0.2], "popsize": 10, "seed": 3,
                                                "verbose": -9, "tolfun": 1e-4})
        t0 = time.perf_counter()
        best = (f_pooled, x0)
        g = 0
        while not es.stop() and time.perf_counter() - t0 < per_budget and g < 60:
            X = es.ask()
            fx = evaluate(X)
            es.tell(X, fx)
            g += 1
            i = int(np.argmin(fx))
            if fx[i] < best[0]:
                best = (float(fx[i]), list(map(float, X[i])))
        mult = [math.exp(v) for v in best[1][:12]] + [float(np.clip(best[1][12], 0, 1)) * DELTA_MAX]
        out[dn] = {"multipliers_vs_pooled": dict(zip(SPREAD_NAMES, mult)), "train_fit_subset_arm_pooled": f_pooled,
                   "train_fit_subset_arm_per_arm": best[0], "generations": g}
        log(f"  {dn:16s} pooled {f_pooled:.4f} -> per-arm {best[0]:.4f} ({g} gens); kp x " +
            " ".join(f"{m:.2f}" for m in mult[:5]) + "; fric x " + " ".join(f"{m:.2f}" for m in mult[5:10]) +
            f"; kv/damp x {mult[10]:.2f}; arm x {mult[11]:.2f}; delta {1000 * mult[12]:.1f} ms")
    jdump({"source": src, "per_arm": out}, os.path.join(CACHE, "spread.json"))


# ================================================================================================
# CLI
# ================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "before", "fit1", "offsets", "fit1_refit", "fit1db", "fit2", "spread",
                                      "final_train", "final", "report", "all"])
    ap.add_argument("--budget", type=float, default=None, help="minutes")
    ap.add_argument("--gens", type=int, default=None)
    args = ap.parse_args()
    t = time.perf_counter()
    if args.stage in ("prep", "all"):
        stage_prep()
    if args.stage in ("before", "all"):
        stage_before()
    if args.stage in ("fit1", "all"):
        stage_fit1(False, budget_min=args.budget or 25.0, max_gens=args.gens or 200)
    if args.stage in ("offsets", "all"):
        stage_offsets(budget_min=args.budget or 10.0)
    if args.stage in ("fit1_refit", "all"):
        prep = jload(os.path.join(CACHE, "prep.json"))
        f1 = jload(os.path.join(CACHE, "fit1.json"))
        stage_fit1(False, budget_min=args.budget or 8.0, max_gens=args.gens or 60, start=f1["x"],
                   offsets=fitted_offsets(prep), tag="fit1_refit")
    if args.stage in ("fit1db", "all"):
        prep = jload(os.path.join(CACHE, "prep.json"))
        src = "fit1_refit" if os.path.exists(os.path.join(CACHE, "fit1_refit.json")) else "fit1"
        f1 = jload(os.path.join(CACHE, f"{src}.json"))
        stage_fit1(True, budget_min=args.budget or 15.0, max_gens=args.gens or 80, start=f1["x"],
                   offsets=fitted_offsets(prep), tag="fit1db")
    if args.stage in ("fit2", "all"):
        stage_fit2(budget_min=args.budget or 40.0, max_gens=args.gens or 150)
    if args.stage in ("spread", "all"):
        stage_spread(budget_min=args.budget or 6.0)
    if args.stage == "final_train":
        stage_final(only_train=True)
    if args.stage in ("final", "all"):
        stage_final()
    if args.stage in ("report", "all"):
        import fit_report
        fit_report.main()
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
    log(f"stage {args.stage} finished in {time.perf_counter() - t:.0f} s")


if __name__ == "__main__":
    main()

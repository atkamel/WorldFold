"""Cloth step 1: reproduce clothilde-sim's published accuracy on Coltraro's free-hanging mocap, then a held-out test
and the floors. Stages (results/ cache each one):

  protocol  paper's own (alpha, delta) on all 32 train recordings -> which metric variant is the paper's
            (pre-registered rule below) + harness check on the weight-free spatial std
  fit       per-recording in-sample sweep on the train session (paper sec. 5.4 protocol) -> reproduction check
  freeze    sha256 of the per-condition fitted params before the held-out session is scored
  final     held-out session scored ONCE with: rigid null, defaults, the paper's a-priori formula (coefficients fitted
            on the train files), our train-session fit; plus the paper's in-sample params as the best case

Sessions (see coltraro_data.parse_name): train = *_hanger files = the paper's repetition II (Table 4); held-out =
*_hands files = the paper's repetition I (Table 3). The released suffixes are swapped relative to the paper's labels.
Revision 2026-10-16 (after the first failed check, before any held-out scoring): floor at z = 0 removed (phantom),
session labels corrected, reproduction rule applied to the same-recordings paper table (same 25 % tolerances).
  floors    marker noise, real repeat (Self-collisions rep1-3), sim chaos (nudges), numerics (substeps, mesh, tol)
  report    results/cloth_table.md

Pre-registered (written before any scoring):
  * metric variant = the one of {lit_marker, lit_sim, awrms_cm} with the smallest median |log(ours/paper)| over
    repetition I run with the paper's own params; harness valid if median |log(std ours/paper)| <= log 1.25.
  * reproduction passes if our fitted repetition-I global mean (chosen variant) <= 0.64 and the A2 mean is within
    25% of 0.71 (0.53-0.89). On failure: find the protocol difference, no tuning toward the number.
  * ordering on repetition II (rms_mm): our fit < defaults < rigid null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import coltraro_data as CD  # noqa: E402

# simulator profile (env so spawned workers inherit it): "paper" = clothilde's solver with the paper's integrator
# settings (results/), "paper2022" = + the 2022 jointly solved FEM constraints (results/paper2022/)
PROFILE = os.environ.get("CLOTH_SYSID_PROFILE", "paper")
RESULTS = os.path.join(HERE, "results") if PROFILE == "paper" else os.path.join(HERE, "results", PROFILE)
NPROC = max(1, min(int(os.environ.get("CLOTH_SYSID_PROCS", "6")), (os.cpu_count() or 2)))
VARIANTS = ("lit_marker", "lit_sim", "awrms_cm")
FITTED = os.path.join(RESULTS, "fitted_rep1.json")
PREREG = os.path.join(RESULTS, "prereg.json")


def profile_sim():
    import replay
    return {"paper": replay.PAPER_SIM, "paper2022": replay.PAPER2022}[PROFILE]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(os.path.join(RESULTS, "log.txt"), "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def low_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def _job(job):
    import replay
    kind = job.get("kind", "sim")
    try:
        if kind == "rigid":
            return {**rigid_null(job["path"]), "tag": job.get("tag")}
        sim = dict(profile_sim(), **(job.get("sim") or {}))
        r = replay.run(job["path"], job["alpha"], job["delta"], sim=sim, nudge=job.get("nudge", 0.0),
                       seed=job.get("seed", 0))
        r["tag"] = job.get("tag")
        return r
    except Exception as e:  # keep the pool alive; failures are reported, never silently dropped
        return {"name": os.path.basename(job["path"])[:-4], "error": repr(e), "tag": job.get("tag"),
                "alpha": job.get("alpha"), "delta": job.get("delta")}


def run_many(jobs):
    with mp.get_context("spawn").Pool(NPROC, initializer=low_priority) as pool:
        return pool.map(_job, jobs, chunksize=1)


def train():
    """Fit session: *_hanger files (the paper's repetition II, Table 4; see coltraro_data.parse_name)."""
    return [r for r in CD.free_hanging() if r["mode"] == "hanger"]


def held():
    """Held-out session: *_hands files (the paper's repetition I, Table 3), scored once in `final`."""
    return [r for r in CD.free_hanging() if r["mode"] == "hands"]


def ref(o):
    return CD.PAPER[o["rep"]][o["key"]]


# ---------------------------------------------------------------- baselines
def rigid_null(path):
    """Initial shape carried rigidly by the two corners: translation of their midpoint + yaw/tilt of their line."""
    import replay
    t, P = replay.load(path)
    f0 = CD.first_full_frame(P)
    t, P = t[f0:] - t[f0], P[f0:]
    N = P.shape[1]
    c0, c1 = CD.corner_ids(N)
    A = np.stack([CD.fill_track(t, P[:, k])[0] for k in (c0, c1)], axis=1)
    X0 = P[0]
    m0, v0 = 0.5 * (A[0, 0] + A[0, 1]), A[0, 1] - A[0, 0]
    pred = np.empty_like(P)
    for n in range(len(t)):
        m, v = 0.5 * (A[n, 0] + A[n, 1]), A[n, 1] - A[n, 0]
        Rm = _rot_between(v0, v)
        pred[n] = (X0 - m0) @ Rm.T + m
    R_, C_ = CD.grid_shape(N)
    Tc = CD.coarse_quads(R_, C_)
    w = CD.lumped_areas(X0, Tc)
    out = replay.score(pred, P, w, w)
    out.update({"name": CD.parse_name(path)["name"], "key": CD.parse_name(path)["key"], "model": "rigid"})
    return out


def _rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def apriori(fabric, size, V, coef="I"):
    """Paper Table 6 / eq. 13: delta, alpha from density, size S (A3 1, A2 2) and velocity V (m^2/s^2)."""
    c = {"I": ((-0.0223, -0.0178, 0.0714, 0.7664), (0.2082, -0.1481, 1.1804, 1.7440)),
         "II": ((-0.0359, -0.0117, 0.0780, 0.7890), (0.2155, -0.1711, 1.4410, 1.9387))}[coef]
    rho, S = CD.RHO[fabric], 2.0 if size == "A2" else 1.0
    d = c[0][0] + c[0][1] * S + c[0][2] * V + c[0][3] * rho
    a = c[1][0] + c[1][1] * S + c[1][2] * V + c[1][3] * rho
    return float(np.clip(a, 1e-4, 4 * rho)), float(np.clip(d, 1e-4, rho))


def velocity_V(path):
    """Paper: average (time and all nodes) of the 50% highest squared velocities. Marker velocities (finite diff)."""
    import replay
    t, P = replay.load(path)
    v = np.diff(P, axis=0) / np.diff(t)[:, None, None]
    s = np.nansum(v * v, axis=2)[~np.isnan(v).any(axis=2)]
    s = np.sort(s)
    return float(s[len(s) // 2:].mean())


# ---------------------------------------------------------------- stages
def stage_protocol():
    recs = train()
    jobs = [{"path": r["path"], "alpha": ref(r)["alpha"], "delta": ref(r)["delta"],
             "tag": "paper_params"} for r in recs]
    log(f"protocol: {len(jobs)} train-session (paper repetition II) runs with the paper's own params, {NPROC} procs")
    out = run_many(jobs)
    bad = [o for o in out if "error" in o]
    if bad:
        log(f"protocol: {len(bad)} failures: {bad[:3]}")
    ok = [o for o in out if "error" not in o]
    score = {}
    for v in VARIANTS:
        lr = [abs(np.log(o[v] / ref(o)["err_cm"])) for o in ok]
        score[v] = {"median_abs_log_ratio": float(np.median(lr)),
                    "median_ratio": float(np.median([o[v] / ref(o)["err_cm"] for o in ok]))}
    std_lr = [abs(np.log(o["std_cm"] / ref(o)["std_cm"])) for o in ok]
    chosen = min(VARIANTS, key=lambda v: score[v]["median_abs_log_ratio"])
    res = {"runs": ok, "failures": bad, "variant_scores": score, "chosen_variant": chosen,
           "std_median_abs_log_ratio": float(np.median(std_lr)),
           "std_median_ratio": float(np.median([o["std_cm"] / ref(o)["std_cm"] for o in ok])),
           "harness_valid": bool(np.median(std_lr) <= np.log(1.25)),
           "wall_per_sim_s_median": float(np.median([o["wall_per_sim_s"] for o in ok]))}
    jdump(res, os.path.join(RESULTS, "protocol.json"))
    log(f"protocol: chosen {chosen} {score[chosen]}; std ratio median {res['std_median_ratio']:.3f} "
        f"harness_valid {res['harness_valid']}")
    return res


def stage_protocol26():
    """Same paper params under clothilde-sim's 2026 defaults (trapezoidal, 10 sub-steps, 0.75 %): shows how much
    the integrator choice alone moves the result (for the report; no decision depends on it)."""
    import replay
    var = jload(os.path.join(RESULTS, "protocol.json"))["chosen_variant"]
    jobs = [{"path": r["path"], "alpha": ref(r)["alpha"], "delta": ref(r)["delta"],
             "sim": replay.CLOTHILDE_2026, "tag": "clothilde2026"} for r in train()]
    out = [o for o in run_many(jobs) if "error" not in o]
    good = [o for o in out if not o["nan"] and o["missing_frac"] < 0.5]
    res = {"runs": out, "n_nan": len(out) - len(good),
           "ratio_chosen": float(np.median([o[var] / ref(o)["err_cm"] for o in good])),
           "std_median_ratio": float(np.median([o["std_cm"] / ref(o)["std_cm"] for o in good]))}
    jdump(res, os.path.join(RESULTS, "protocol_clothilde2026.json"))
    log(f"protocol26: ratio {res['ratio_chosen']:.3f} std ratio {res['std_median_ratio']:.3f} nan {res['n_nan']}")


def stage_insample2(coarse=8, fine=5):
    """Diagnostic upper bound after `final`: the same in-sample sweep on repetition II (never used for selection)."""
    pre = jload(PREREG)
    if not os.path.exists(os.path.join(RESULTS, "final.json")):
        raise RuntimeError("run `final` first: repetition II must be scored with frozen params before any fit on it")
    var = jload(FITTED)["variant"]
    recs = held()
    fc = (np.arange(coarse) + 0.5) / coarse
    jobs = [{"path": r["path"], "alpha": a, "delta": d, "tag": "coarse"}
            for r in recs for a, d in _grid(CD.RHO[r["fabric"]], fc, fc)]
    out = run_many(jobs)
    best = {}
    for o in out:
        if "error" not in o and (o["name"] not in best or o[var] < best[o["name"]][var]):
            best[o["name"]] = o
    step = 1.0 / coarse
    offs = (np.arange(fine) - (fine - 1) / 2) * step / (fine - 1) * 2
    jobs = []
    for r in recs:
        rho, b = CD.RHO[r["fabric"]], best[r["name"]]
        fa = np.unique(np.clip(b["alpha"] / (4 * rho) + offs, 1e-3, 1.0))
        fd = np.unique(np.clip(b["delta"] / rho + offs, 1e-3, 1.0))
        jobs += [{"path": r["path"], "alpha": a, "delta": d, "tag": "fine"} for a, d in _grid(rho, fa, fd)]
    for o in run_many(jobs):
        if "error" not in o and o[var] < best[o["name"]][var]:
            best[o["name"]] = o
    fin = jload(os.path.join(RESULTS, "final.json"))
    fin["runs"] += [dict(b, tag="ours_insample_II") for b in best.values()]
    fin["insample2_note"] = f"added after final (frozen {pre['sha256'][:16]}), diagnostic upper bound only"
    jdump(fin, os.path.join(RESULTS, "final.json"))
    log(f"insample2: mean {var} {np.mean([b[var] for b in best.values()]):.3f}")


def _grid(rho, fa, fd):
    return [(4 * rho * a, rho * d) for a in fa for d in fd]


def stage_final_paper():
    """Held-out yardstick without our own fit (2026-10-16 scope cut): clothilde with the paper's params on the 32
    held-out (*_hands) files. paper_fabric = one (alpha, delta) per fabric, the mean of the paper's train-session
    (Table 4) values, i.e. what a folding sim would carry to a new motion; apriori_II = the paper's own formula;
    paper_insample_held = the paper's per-recording fit on these files (best case). Nothing is fitted here."""
    recs = held()
    by_fab = {}
    for r in train():
        by_fab.setdefault(r["fabric"], []).append(ref(r))
    fab = {f: (float(np.mean([p["alpha"] for p in ps])), float(np.mean([p["delta"] for p in ps])))
           for f, ps in by_fab.items()}
    jobs = []
    for r in recs:
        rho = CD.RHO[r["fabric"]]
        a_pr, d_pr = apriori(r["fabric"], r["size"], velocity_V(r["path"]), "II")
        jobs += [{"path": r["path"], "kind": "rigid", "tag": "rigid"},
                 {"path": r["path"], "alpha": 2 * rho, "delta": rho, "tag": "defaults"},
                 {"path": r["path"], "alpha": fab[r["fabric"]][0], "delta": fab[r["fabric"]][1], "tag": "paper_fabric"},
                 {"path": r["path"], "alpha": a_pr, "delta": d_pr, "tag": "apriori_II"},
                 {"path": r["path"], "alpha": ref(r)["alpha"], "delta": ref(r)["delta"], "tag": "paper_insample_held"}]
    log(f"final_paper: {len(jobs)} runs on the held-out session, {NPROC} procs; per-fabric params {fab}")
    out = run_many(jobs)
    jdump({"fabric_params": fab, "runs": out}, os.path.join(RESULTS, "final_paper.json"))
    for tag in ("rigid", "defaults", "paper_fabric", "apriori_II", "paper_insample_held"):
        rs = [o for o in out if o.get("tag") == tag and "error" not in o]
        log(f"final_paper {tag}: rms_mm median {np.median([o['rms_mm'] for o in rs]):.1f} mean "
            f"{np.mean([o['rms_mm'] for o in rs]):.1f}, lit_marker mean {np.mean([o['lit_marker'] for o in rs]):.3f} "
            f"({len(rs)} ok)")
    return out


def stage_fit(coarse=8, fine=5):
    """Per-recording 2-D sweep on the train session (paper repetition II; paper: sweep over [0,rho] x [0,4rho])."""
    proto = jload(os.path.join(RESULTS, "protocol.json"))
    var = proto["chosen_variant"]
    recs = train()
    fc = (np.arange(coarse) + 0.5) / coarse
    jobs = []
    for r in recs:
        rho = CD.RHO[r["fabric"]]
        for a, d in _grid(rho, fc, fc):
            jobs.append({"path": r["path"], "alpha": a, "delta": d, "tag": "coarse"})
    log(f"fit: coarse {len(jobs)} runs ({coarse}x{coarse} per recording), metric {var}")
    out = run_many(jobs)
    jdump(out, os.path.join(RESULTS, "fit_coarse.json"))
    best = {}
    for o in out:
        if "error" in o:
            continue
        if o["name"] not in best or o[var] < best[o["name"]][var]:
            best[o["name"]] = o
    step = 1.0 / coarse
    jobs = []
    for r in recs:
        rho = CD.RHO[r["fabric"]]
        b = best[r["name"]]
        fa0, fd0 = b["alpha"] / (4 * rho), b["delta"] / rho
        offs = (np.arange(fine) - (fine - 1) / 2) * step / (fine - 1) * 2
        fa = np.clip(fa0 + offs, 1e-3, 1.0)
        fd = np.clip(fd0 + offs, 1e-3, 1.0)
        for a, d in _grid(rho, np.unique(fa), np.unique(fd)):
            jobs.append({"path": r["path"], "alpha": a, "delta": d, "tag": "fine"})
    log(f"fit: fine {len(jobs)} runs")
    out2 = run_many(jobs)
    jdump(out2, os.path.join(RESULTS, "fit_fine.json"))
    for o in out2:
        if "error" not in o and o[var] < best[o["name"]][var]:
            best[o["name"]] = o
    fitted = {}
    for r in recs:
        b = best[r["name"]]
        fitted[r["key"]] = {"alpha": b["alpha"], "delta": b["delta"], var: b[var], "rms_mm": b["rms_mm"],
                            "std_cm": b["std_cm"], "fabric": r["fabric"], "size": r["size"], "motion": r["motion"],
                            "speed": r["speed"], "paper": ref(r)}
    res = {"variant": var, "fitted": fitted, "summary": summarise(fitted, var)}
    jdump(res, FITTED)
    s, p = res["summary"], res["summary"]["paper"]
    log(f"fit: global {s['global']:.3f} (paper {p['global']:.3f}), A2 {s['A2']:.3f} ({p['A2']:.3f}), "
        f"A3 {s['A3']:.3f} ({p['A3']:.3f}); pass {s['pass']}")
    return res


GROUPS = ["polyester", "wool", "denim", "cotton", "A2", "A3", "shake", "twist", "slow", "fast", "global"]


def summarise(fitted, var):
    """Group means, ours vs the paper's in-sample errors on the SAME recordings. Pre-registered rule, applied to the
    same-recordings reference (Table 5 for repetition I, the equivalent means of Table 4 for repetition II):
    global <= 1.25 x paper global and A2 within +-25 % of the paper's A2 mean."""
    g = {k: [] for k in GROUPS}
    q = {k: [] for k in GROUPS}
    for v in fitted.values():
        for k in ("global", v["fabric"], v["size"], v["motion"], v["speed"]):
            g[k].append(v[var])
            q[k].append(v["paper"]["err_cm"])
    s = {k: float(np.mean(x)) for k, x in g.items() if x}
    s["paper"] = {k: float(np.mean(x)) for k, x in q.items() if x}
    s["pass"] = bool(s["global"] <= 1.25 * s["paper"]["global"] and abs(s["A2"] / s["paper"]["A2"] - 1) <= 0.25)
    return s


def stage_freeze():
    with open(FITTED, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    jdump({"fitted": os.path.basename(FITTED), "sha256": h, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "rule": "repetition II scored once with these per-condition params; ordering ours < defaults < rigid"},
          PREREG)
    log(f"frozen {FITTED} sha256 {h}")


def stage_final():
    pre = jload(PREREG)
    with open(FITTED, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    if h != pre["sha256"]:
        raise RuntimeError("fitted params changed after freeze")
    fit = jload(FITTED)
    recs = held()
    jobs = []
    for r in recs:
        rho = CD.RHO[r["fabric"]]
        f = fit["fitted"][r["key"]]
        a_pr, d_pr = apriori(r["fabric"], r["size"], velocity_V(r["path"]), "II")   # coefficients fitted on the train files
        p4 = ref(r)                               # paper in-sample params on these files
        jobs += [{"path": r["path"], "kind": "rigid", "tag": "rigid"},
                 {"path": r["path"], "alpha": 2 * rho, "delta": rho, "tag": "defaults"},
                 {"path": r["path"], "alpha": a_pr, "delta": d_pr, "tag": "apriori_II"},
                 {"path": r["path"], "alpha": f["alpha"], "delta": f["delta"], "tag": "ours_rep1"},
                 {"path": r["path"], "alpha": p4["alpha"], "delta": p4["delta"], "tag": "paper_insample_held"}]
    log(f"final: {len(jobs)} runs on repetition II (held-out), sha256 {h[:16]}")
    out = run_many(jobs)
    jdump({"sha256": h, "runs": out}, os.path.join(RESULTS, "final.json"))
    return out


def stage_floors():
    import replay
    out = {}
    # marker noise: still markers in the first 0.25 s before motion (std of each coordinate), per recording
    noise = []
    for r in CD.free_hanging():
        t, P = replay.load(r["path"])
        k = t <= 0.25
        sd = np.nanstd(P[k], axis=0)                  # (N, 3)
        noise.append(float(1000 * np.sqrt(np.nanmean(np.sum(sd ** 2, axis=1)))))
    out["marker_still_jitter_mm"] = {"median": float(np.median(noise)), "p90": float(np.percentile(noise, 90)),
                                     "note": "includes any real motion in the first 0.25 s, so an upper bound"}
    # real repeat: Self-collisions rep1..3, same fabric and sub-case, hand-released
    d = os.path.join(CD.ensure_data(), "Self-collisions")
    groups = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv"):
            base = f.rsplit("_rep", 1)[0]
            groups.setdefault(base, []).append(os.path.join(d, f))
    # Markers there are unlabelled and include 2 static rod markers, so takes are compared as point sets:
    # symmetric marker Chamfer (mean nearest-neighbour distance, both ways) after dropping static markers.
    rep_res = {}
    for base, fs in groups.items():
        tr = [_cloth_points(*CD.load_raw(f)) for f in fs]
        n = min(len(x) for x in tr)
        starts, ends = [], []
        for i in range(len(tr)):
            for j in range(i + 1, len(tr)):
                starts.append(_set_chamfer(tr[i][0], tr[j][0]))
                ends.append(_set_chamfer(tr[i][n - 1], tr[j][n - 1]))
        rep_res[base] = {"start_rms_mm": float(np.nanmean(starts)), "end_rms_mm": float(np.nanmean(ends)),
                         "takes": len(fs), "metric": "symmetric marker-set Chamfer, mm"}
    out["real_repeat_selfcollision"] = {"per_case": rep_res,
                                        "end_rms_mm_median": float(np.median([v["end_rms_mm"] for v in rep_res.values()])),
                                        "start_rms_mm_median": float(np.median([v["start_rms_mm"] for v in rep_res.values()]))}
    # sim-side floors on 4 repetition-I recordings (one per fabric, A2 fast shake), at the fitted params
    fit = jload(FITTED)["fitted"]
    pick = [r for r in train() if r["size"] == "A2" and r["motion"] == "shake" and r["speed"] == "fast"]
    jobs = []
    for r in pick:
        f = fit[r["key"]]
        base = {"path": r["path"], "alpha": f["alpha"], "delta": f["delta"]}
        for s in range(10):
            jobs.append({**base, "nudge": 1e-9, "seed": s, "tag": f"chaos{s}"})
        jobs.append({**base, "tag": "base"})
        jobs.append({**base, "sim": {"sub_steps": 20}, "tag": "sub20"})
        jobs.append({**base, "sim": {"sub_steps": 5}, "tag": "sub5"})
        jobs.append({**base, "sim": {"tol": 0.00375}, "tag": "tol_half"})
        jobs.append({**base, "sim": {"refine": 3}, "tag": "refine3"})
        jobs.append({**base, "sim": {"refine": 4}, "tag": "refine4"})
    log(f"floors: {len(jobs)} sim runs")
    res = run_many(jobs)
    out["sim_runs"] = res
    jdump(out, os.path.join(RESULTS, "floors.json"))
    return out


def _cloth_points(t, P):
    """Per frame, the cloth markers only: drop columns that never move (> 98 % frames within 2 mm of their median)."""
    med = np.nanmedian(P, axis=0)
    still = np.nanmean(np.linalg.norm(P - med[None], axis=2) < 0.002, axis=0) > 0.98
    Q = P[:, ~still, :]
    return [q[~np.isnan(q).any(axis=1)] for q in Q]


def _set_chamfer(a, b):
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    return float(1000 * 0.5 * (da.mean() + db.mean()))


def _frame_rms(a, b):
    d = a - b
    ok = ~np.isnan(d).any(axis=1)
    return float(1000 * np.sqrt(np.mean(np.sum(d[ok] ** 2, axis=1)))) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["protocol", "protocol26", "fit", "freeze", "final", "final_paper", "insample2", "floors",
                             "report", "all"])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    low_priority()
    stages = ["protocol", "fit", "freeze", "final", "floors", "report"] if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "report":
            import cloth_report
            cloth_report.write()
        else:
            globals()["stage_" + s]()


if __name__ == "__main__":
    main()

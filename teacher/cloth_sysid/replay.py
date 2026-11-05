"""Replay one Coltraro free-hanging recording in clothilde-sim with the two upper corners scripted along their
recorded paths (paper protocol: marker grid refined x2, kappa = 0, only alpha and delta free), and score it.

Metric variants are all computed from the same run (which one is the paper's is decided by stage `protocol`):
  lit_marker : 100*sqrt(sum_k w_k d_k^2), w = lumped areas of the marker mesh (m^2)        [paper eq. 10, literal]
  lit_sim    : same, w = clothilde's own lumped mass at the marker-coincident nodes (m^2)
  awrms      : area-weighted RMS distance, cm (marker-mesh weights normalised to sum 1)
  rms_mm     : plain RMS over present markers, mm  (our primary, unweighted)
  std_cm     : spatial std over present markers of |d| (paper eq. 12, weight-free) -> harness check
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor", "clothilde-sim", "python_code")
sys.path.insert(0, VENDOR)
sys.path.insert(0, HERE)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import coltraro_data as CD  # noqa: E402

# Paper protocol (Coltraro AMM 2022 sec. 7 + 9.1, AMM 2024, MMT 2025): implicit Euler, one step per frame with
# the recording linearly resampled to dt = 0.01 s, fast-projection tolerance 0.1 %, bending 0, inextensible.
# No floor: the free-hanging model has no contact term, and in this dataset z = 0 is NOT below the cloth (28 of 32 A2
# recordings go below it, to -7.8 cm), so clothilde's always-on floor at z = 0 would push the cloth onto a phantom plane.
PAPER_SIM = {"implicit_euler": True, "resample_hz": 100.0, "sub_steps": 1, "tol": 0.001, "kappa": 0.0,
             "kappa_bnd": 0.0, "str": 0.0, "shr": 0.0, "mu_f": 0.2, "mu_s": 0.35, "thck": 0.95, "refine": 2,
             "floor": False, "self_collisions": False}
# clothilde-sim 2026 defaults (repo commit 383987e): trapezoidal rule, 10 sub-steps per frame, tolerance 0.75 %
CLOTHILDE_2026 = dict(PAPER_SIM, implicit_euler=False, resample_hz=None, sub_steps=10, tol=0.0075)
# PAPER_SIM integrator + the 2022 paper's jointly solved FEM metric constraints (paper_solver.InextensibleCloth)
PAPER2022 = dict(PAPER_SIM, solver="paper2022")
SIM_DEFAULTS = PAPER_SIM
_CACHE = {}


def resample(t, P, hz):
    """Linear resampling to a uniform grid (paper: 'replace them through linear interpolation ... dt = 0.01').
    A resampled marker is NaN if either bracketing original sample is missing (no interpolation across gaps)."""
    tn = np.arange(0.0, t[-1] + 1e-9, 1.0 / hz)
    j = np.clip(np.searchsorted(t, tn, side="right") - 1, 0, len(t) - 2)
    w = ((tn - t[j]) / (t[j + 1] - t[j]))[:, None, None]
    Q = (1 - w) * P[j] + w * P[j + 1]
    return tn, Q


def load(path):
    if path not in _CACHE:
        t, P = CD.load_csv(path)
        _CACHE[path] = (t, P)
    return _CACHE[path]


def build(P0, R, C, refine):
    from implementation.utils import refine_rect_quad_mesh
    Tc = CD.coarse_quads(R, C)
    Tf, S = refine_rect_quad_mesh(Tc, refine, refine, num_vertices=R * C)
    return Tc, Tf, S, np.asarray(S @ P0)


def run(path, alpha, delta, sim=None, t_max=None, nudge=0.0, seed=0, keep_pred=False):
    """Simulate one recording; returns metrics dict (and predictions if keep_pred)."""
    from implementation.Cloth import Cloth
    o = dict(SIM_DEFAULTS, **(sim or {}))
    info = CD.parse_name(path)
    t, P = load(path)
    N = P.shape[1]
    R, C = CD.grid_shape(N)
    f0 = CD.first_full_frame(P)
    t, P = t[f0:] - t[f0], P[f0:]
    if t_max is not None:
        keep = t <= t_max + 1e-9
        t, P = t[keep], P[keep]
    corners = CD.corner_ids(N)
    gaps = {k: CD.fill_track(t, P[:, k])[1] for k in corners}
    Pc = P.copy()
    for k in corners:                                # driven corners: gap-filled before any resampling
        Pc[:, k] = CD.fill_track(t, P[:, k])[0]
    if o.get("resample_hz"):
        t_rs, P = resample(t, P, o["resample_hz"])
        Pc = resample(t, Pc, o["resample_hz"])[1]
        t = t_rs
    elif o.get("frame_dt"):                          # hypothesis test: every recorded frame stepped as frame_dt
        t = np.arange(len(t)) * o["frame_dt"]
    U = Pc[:, corners, :]                            # (T, 2, 3) driven paths
    Tc, Tf, S, X0 = build(P[0], R, C, o["refine"])
    if nudge > 0:
        X0 = X0 + nudge * np.random.default_rng(seed).standard_normal(X0.shape)
    dt = float(np.median(np.diff(t)))
    if o.get("solver") == "paper2022":
        from paper_solver import InextensibleCloth
        cl = InextensibleCloth(X0, Tf)
    else:
        cl = Cloth(X0, Tf)
    cl.setSimulatorParameters(dt=dt, sub_steps=o["sub_steps"], tol=o["tol"], rho=CD.RHO[info["fabric"]],
                              delta=delta, alpha=alpha, kappa=o["kappa"], kappa_bnd=o["kappa_bnd"],
                              str=o["str"], shr=o["shr"], mu_f=o["mu_f"], mu_s=o["mu_s"], thck=o["thck"],
                              max_mov=o.get("max_mov", 0.1))
    cl.implicitEuler = bool(o["implicit_euler"])
    if not o.get("self_collisions", True):           # paper free-hanging model (eq. 4) has no collision terms
        cl.selfCollisions = lambda phi, n_iter, max_iters=50: phi
    if not o.get("floor", True):
        cl.floorCollisions = lambda phi: phi
    ctrl = list(corners)                    # original vertices keep ids 0..N-1 in the refined mesh
    pred = np.empty((len(t), N, 3))
    pred[0] = cl.positions[:N]
    tic = time.perf_counter()
    for n in range(1, len(t)):
        cl.simulate(u=U[n].copy(), control=ctrl)
        pred[n] = cl.positions[:N]
    wall = time.perf_counter() - tic
    w_marker = CD.lumped_areas(P[0], Tc)
    w_sim = np.asarray(cl.M.diagonal())[:N]
    out = score(pred, P, w_marker, w_sim)
    out.update({"name": info["name"], "key": info["key"], "rep": info["rep"], "alpha": alpha, "delta": delta,
                "frames": int(len(t)), "dt": dt, "sim_s": float(t[-1]), "wall_s": wall,
                "wall_per_sim_s": wall / max(t[-1], 1e-9), "avg_iters": cl.total_iters / max(len(t) - 1, 1),
                "corner_gap_frames": gaps, "first_frame": f0, "area_marker_m2": float(w_marker.sum()),
                "area_sim_marker_nodes_m2": float(w_sim.sum()), "nan": bool(np.isnan(pred).any()),
                "proj_damped": int(getattr(cl, "proj_damped", 0)),
                "proj_unconverged": int(getattr(cl, "proj_unconverged", 0))})
    if keep_pred:
        out["pred"] = pred
        out["real"] = P
    return out


def score(pred, real, w_marker, w_sim):
    """Per-frame metrics averaged over frames 1..T-1 (frame 0 is the shared start state)."""
    d = pred[1:] - real[1:]                              # (T-1, N, 3) m
    present = ~np.isnan(d).any(axis=2)                   # missing markers dropped per frame (paper Remark 5.2)
    d2 = np.where(present, np.nansum(d * d, axis=2), 0.0)
    dist = np.sqrt(d2)
    wm = present * w_marker[None]
    ws = present * w_sim[None]
    lit_marker = 100.0 * np.sqrt((wm * d2).sum(1))
    lit_sim = 100.0 * np.sqrt((ws * d2).sum(1))
    awrms = 100.0 * np.sqrt((wm * d2).sum(1) / wm.sum(1))
    cnt = present.sum(1)
    rms_mm = 1000.0 * np.sqrt(d2.sum(1) / cnt)
    mean_d = dist.sum(1) / cnt
    std_cm = 100.0 * np.sqrt(np.maximum((np.where(present, dist, 0.0) ** 2).sum(1) / cnt - mean_d ** 2, 0.0))
    return {"lit_marker": float(lit_marker.mean()), "lit_sim": float(lit_sim.mean()), "awrms_cm": float(awrms.mean()),
            "rms_mm": float(rms_mm.mean()), "rms_mm_all": float(1000.0 * np.sqrt(d2.sum() / cnt.sum())),
            "std_cm": float(std_cm.mean()), "missing_frac": float(1.0 - present.mean())}


if __name__ == "__main__":
    import json
    name = sys.argv[1] if len(sys.argv) > 1 else "denim_A2_shake_slow_hanger"
    tmax = float(sys.argv[2]) if len(sys.argv) > 2 else None
    path = os.path.join(CD.ensure_data(), "Free-hanging", name + ".csv")
    info = CD.parse_name(path)
    ref = CD.PAPER[info["rep"]][info["key"]]
    r = run(path, ref["alpha"], ref["delta"], t_max=tmax)
    r["paper"] = ref
    print(json.dumps(r, indent=1, default=float))

"""Unit checks for the cloth harness (python cloth_tests.py). Writes results/cloth_tests.json.

Gating tests must PASS before any scoring. The cantilever check is informational: the paper protocol uses
kappa = 0, so it does not gate step 1; it reports how clothilde's bending constant maps to Peirce's bending length.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import coltraro_data as CD  # noqa: E402
import replay  # noqa: E402

REC = os.path.join(CD.ensure_data(), "Free-hanging", "denim_A2_shake_slow_hanger.csv")


def t_parse():
    t, P = CD.load_csv(REC)
    R, C = CD.grid_shape(P.shape[1])
    g = CD.check_grid(P[CD.first_full_frame(P)], R, C)
    dt = float(np.median(np.diff(t)))
    ok = (P.shape[1] == 20 and abs(dt - 1 / 120) < 1e-5 and g["rows_down"] and 0.13 < g["h_mean_m"] < 0.17
          and 0.11 < g["v_mean_m"] < 0.15 and g["h_cv"] < 0.15 and g["v_cv"] < 0.15)
    # every free-hanging recording: labels 1..N parse and rows go down on the first full frame
    allrows, worst = [], {"h_cv": 0.0, "v_cv": 0.0}
    for r in CD.free_hanging():
        tt, PP = CD.load_csv(r["path"])
        gg = CD.check_grid(PP[CD.first_full_frame(PP)], *CD.grid_shape(PP.shape[1]))
        allrows.append(gg["rows_down"] and gg["h_cv"] < 0.2 and gg["v_cv"] < 0.2)
        if gg["h_cv"] + gg["v_cv"] > worst["h_cv"] + worst["v_cv"]:
            worst = dict(gg, name=r["name"])
    return {"ok": bool(ok and all(allrows)), "dt": dt, "grid": g, "all_grids_ok": int(sum(allrows)),
            "n_recordings": len(allrows), "worst_grid": worst}


def t_mapping():
    t, P = CD.load_csv(REC)
    R, C = CD.grid_shape(P.shape[1])
    Tc, Tf, S, X = replay.build(P[0], R, C, 2)
    ident = np.allclose(S[:R * C].toarray(), np.eye(R * C))
    keep = np.allclose(X[:R * C], P[0])
    shape_ok = X.shape[0] == (2 * R - 1) * (2 * C - 1) and Tf.shape[0] == 4 * (R - 1) * (C - 1)
    mid = np.allclose(X[R * C:].min(0), X.min(0), atol=0.2)   # refined nodes lie within the sheet's bounds
    return {"ok": bool(ident and keep and shape_ok and mid), "n_fine": int(X.shape[0]), "quads": int(Tf.shape[0])}


def t_pins():
    out = {}
    for tag, sim in (("paper", replay.PAPER_SIM), ("clothilde2026", replay.CLOTHILDE_2026)):
        r = replay.run(REC, 0.416, 0.139, sim=sim, t_max=1.0, keep_pred=True)
        P = r["real"]                                   # the (resampled) recording the run was driven by
        out[tag] = max(float(np.nanmax(np.linalg.norm(r["pred"][:, k] - P[:, k], axis=1))) for k in CD.corner_ids(20))
        out[tag + "_dt"] = r["dt"]
        out[tag + "_nan"] = r["nan"]
    ok = (max(out["paper"], out["clothilde2026"]) < 1e-3 and abs(out["paper_dt"] - 0.01) < 1e-9
          and not out["paper_nan"] and not out["clothilde2026_nan"])
    return {"ok": bool(ok), "max_corner_dev_m": out}


def t_metric():
    t, P = CD.load_csv(REC)
    P = P[:50].copy()
    P[np.isnan(P)] = 0.0
    R, C = CD.grid_shape(20)
    w = CD.lumped_areas(P[0], CD.coarse_quads(R, C))
    z = replay.score(P.copy(), P, w, w)
    off = P.copy()
    off[..., 2] += 0.01                                   # every marker 1 cm too high
    s = replay.score(off, P, w, w)
    exp_lit = 100 * np.sqrt(w.sum() * 0.01 ** 2)
    ok = (max(z["rms_mm"], z["lit_marker"], z["awrms_cm"], z["std_cm"]) < 1e-12 and abs(s["rms_mm"] - 10) < 1e-9
          and abs(s["awrms_cm"] - 1.0) < 1e-9 and abs(s["lit_marker"] - exp_lit) < 1e-9 and s["std_cm"] < 1e-6)
    # a missing marker is dropped, not counted as zero error
    m = P.copy()
    m[1:, 3, :] = np.nan
    s2 = replay.score(off, m, w, w)
    ok2 = abs(s2["rms_mm"] - 10) < 1e-9 and s2["missing_frac"] > 0
    return {"ok": bool(ok and ok2), "lit_for_1cm": float(s["lit_marker"]), "area_m2": float(w.sum())}


def t_paper_solver():
    """2022 joint solver: C = 0 at rest and under a rigid motion, Jacobian = finite differences, projection converges,
    a 2 s two-corner hang keeps boundary lengths and area (inextensibility)."""
    from paper_solver import InextensibleCloth
    t, P = CD.load_csv(REC)
    R, C = CD.grid_shape(20)
    Tc, Tf, S, X = replay.build(P[0], R, C, 2)
    cl = InextensibleCloth(X, Tf)
    cl.setSimulatorParameters(dt=0.01, sub_steps=1, tol=0.001, rho=0.3046, delta=0.2, alpha=0.4, kappa=0.0,
                              kappa_bnd=0.0, str=0.0, shr=0.0)
    cl.implicitEuler = True
    c_rest = cl.constraints(X, np.zeros((0, 3)), [])[0]
    th = 0.7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    c_rigid = cl.constraints(X @ Rz.T + np.array([0.3, -0.2, 0.1]), np.zeros((0, 3)), [])[0]
    rng = np.random.default_rng(0)
    Y = X + 0.01 * rng.standard_normal(X.shape)
    c, G = cl.constraints(Y, np.zeros((0, 3)), [])[:2]
    dv = 1e-7 * rng.standard_normal(X.shape)
    c2 = cl.constraints(Y + dv, np.zeros((0, 3)), [])[0]
    fd = c2 - c
    lin = G @ dv.ravel(order="F")
    jac_rel = float(np.max(np.abs(fd - lin)) / np.max(np.abs(fd)))
    proj = cl.project(Y.ravel(order="F"), np.zeros((0, 3)), [])
    cp = cl.constraints(proj.reshape((-1, 3), order="F"), np.zeros((0, 3)), [])
    proj_err = max(float(np.max(np.abs(cp[2]) / cl.scale)), float(np.max(np.abs(cp[3]) / cl.l2)))
    area0 = CD.lumped_areas(X, Tf).sum()
    ctrl = CD.corner_ids(20)
    u = X[ctrl].copy()
    for _ in range(200):
        cl.simulate(u=u.copy(), control=ctrl)
    Z = cl.positions
    e = Z[cl.bnd_edges[:, 0]] - Z[cl.bnd_edges[:, 1]]
    edge_err = float(np.max(np.abs(np.sqrt(np.einsum("ij,ij->i", e, e)) / np.sqrt(cl.l2) - 1)))
    area_err = float(abs(CD.lumped_areas(Z, Tf).sum() / area0 - 1))
    ok = (np.max(np.abs(c_rest)) < 1e-12 and np.max(np.abs(c_rigid)) < 1e-10 and jac_rel < 1e-4
          and proj_err <= 1e-3 and edge_err < 2e-3 and area_err < 1e-2 and not np.isnan(Z).any())
    return {"ok": bool(ok), "c_rest": float(np.max(np.abs(c_rest))), "c_rigid": float(np.max(np.abs(c_rigid))),
            "jacobian_rel_err": jac_rel, "projection_rel_err": proj_err, "hang_edge_err": edge_err,
            "hang_area_err": area_err, "n_constraints": int(len(c)), "n_dof": int(3 * len(X)),
            "mean_proj_iters": float(np.mean(cl.proj_iters))}


def t_cantilever():
    """Informational: B implied by the droop vs clothilde's kappa, at three kappas (stable ratio = kappa acts as B)."""
    rows = [_cantilever(k) for k in (5e-5, 2e-4, 8e-4)]
    ratios = [r["ratio_B_over_kappa"] for r in rows]
    return {"ok": True, "informational": True, "rows": rows, "ratio_spread": float(np.nanmax(ratios) / np.nanmin(ratios))}


def _cantilever(kappa, rho=0.2, L=0.10, W=0.04, nx=21, ny=9, seconds=4.0):
    """Strip of length L clamped (first node column) horizontally; Peirce: c = l*(cos(th/2)/(8 tan th))^(1/3) with
    c = (B/(w g))^(1/3), valid for moderate droop angles."""
    from implementation.Cloth import Cloth
    from implementation.utils import createMesh
    X, T = createMesh([0, L, 0, W], nx, ny, lambda x, y: x, lambda x, y: y, lambda x, y: 0 * x + 0.5)
    cl = Cloth(X, T)
    # many pinned nodes make the exact (str = shr = 0) constraint system singular; use clothilde's small default
    # compliances here (this check is informational only)
    cl.setSimulatorParameters(dt=1 / 120, sub_steps=10, rho=rho, delta=rho, alpha=3 * rho, kappa=kappa,
                              kappa_bnd=0.0, str=0.01e-4, shr=0.1e-4)
    xs = X[:, 0]
    ctrl = [int(i) for i in np.nonzero(xs <= xs.min() + (L / (nx - 1)) * 1.01)[0]]
    u = X[ctrl].copy()
    for _ in range(int(seconds * 120)):
        cl.simulate(u=u.copy(), control=ctrl)
    p = cl.positions
    clamp_x = xs[ctrl].max()
    tip = p[np.argmax(xs)]
    edge = np.array([clamp_x, tip[1], X[0, 2]])
    l = L - clamp_x
    th = float(np.arctan2(edge[2] - tip[2], tip[0] - edge[0]))
    c_peirce = l * (np.cos(th / 2) / (8 * np.tan(th))) ** (1 / 3) if th > 1e-6 else float("nan")
    B_implied = rho * 9.8 * c_peirce ** 3
    return {"ok": True, "informational": True, "droop_deg": float(np.degrees(th)), "overhang_m": float(l),
            "c_peirce_m": float(c_peirce), "B_implied_Nm": float(B_implied), "kappa": kappa,
            "ratio_B_over_kappa": float(B_implied / kappa)}


if __name__ == "__main__":
    res = {}
    for name, fn in (("parse", t_parse), ("mapping", t_mapping), ("pins", t_pins), ("metric", t_metric), ("paper_solver", t_paper_solver),
                     ("cantilever_info", t_cantilever)):
        res[name] = fn()
        tag = "INFO" if res[name].get("informational") else ("PASS" if res[name]["ok"] else "FAIL")
        print(f"{name:16s} {tag}  {json.dumps(res[name], default=float)[:300]}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "cloth_tests.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    sys.exit(0 if all(r["ok"] for r in res.values()) else 1)

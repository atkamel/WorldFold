"""Cloth step 2: Newton (VBD) on Coltraro's free-hanging mocap, same protocol and metric as replay.py, so Newton's
error reads directly against clothilde's (which reproduces the paper, results/paper2022/protocol.json).

Protocol (as replay.run): the two upper corners follow their recorded paths (no gripper, no contact), marker grid
refined x2, recording resampled to 100 Hz, frame 0 = the recorded hanging shape (taken as the strain-free rest),
score = replay.score (lit_marker = the paper's metric, rms_mm = plain marker RMS).

Batching: one Newton model holds K independent copies (any mix of recordings and parameter sets, no contacts between
them), so one GPU run scores a whole CMA generation. Copies of shorter recordings hold their last corner pose.

Parameters per copy (CMA works on x in [0, 1]^7, see decode):
  tri_ke, tri_ka  membrane (stable neo-Hookean mu, lambda; N/m, mesh-independent)
  tri_kd          membrane strain-rate damping, as the time constant tri_kd / tri_ke
  edge_ke, edge_kd bending (Newton hinge: per-hinge stiffness = edge_ke * rest edge length, so moving to a finer mesh
                  scales edge_ke ~ 1 / edge length); damping as the time constant edge_kd / edge_ke
  alpha, delta    clothilde's air model, unchanged: inertia rho*A, drag -alpha*A*v, gravity on delta*A, i.e. per
                  particle f = m*g*(1 - delta/rho)*z - (alpha/rho)*m*v (VBD ignores Newton's tri_drag/tri_lift)

Two VBD details handled here: scripted particles have mass 0 and are placed at the end-of-substep target before the
step, and a hook restores their start-of-substep position in VBD's previous-position buffer, so element damping and
the velocity update see their real motion.

Stages (results/newton/):
  test      local checks (CPU ok)
  bench     copies*sim-s per wall-s for K copies, plus a numerics check (substeps/iterations x2)
  fit       lockstep CMA, one ES per fabric, train session (*_hanger), objective = mean lit_marker
  freeze    sha256 of newton_fitted.json
  final     held-out session (*_hands) scored once: newton_defaults (grip_bench values), newton_fitted
  ladder    strip-down rungs, each refit briefly, held-out scored

Gate 1 (held-out, lit_marker mean over the 32 *_hands recordings), against clothilde with the paper's per-fabric
params on the same recordings (results/paper2022/final_paper.json: clothilde 0.778, rigid null 1.183):
  (a) newton_fitted <= 1.5 x clothilde = 1.167      (set in the plan)
  (b) newton_fitted <= 0.98, i.e. at least half of clothilde's gain over the rigid null
      (added 2026-10-16 before any Newton fit was scored: (a) alone is about the rigid null's error)
Gate 2 (ladder): fastest rung whose held-out lit_marker stays within 25 % of the full rung.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor", "clothilde-sim", "python_code"))

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton  # noqa: E402
import coltraro_data as CD  # noqa: E402
import replay  # noqa: E402  (load / resample / build / score only; never imports clothilde's solver)

RESULTS = os.path.join(HERE, "results", "newton")
G = 9.8                                              # clothilde's gravity
FULL = {"sub_steps": 10, "iterations": 10, "refine": 2, "hz": 100.0}
NAMES = ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd", "alpha", "delta")
GRIP_BENCH = {"tri_ke": 400.0, "tri_ka": 400.0, "tri_kd": 1e-3, "edge_ke": 2e-3, "edge_kd": 5e-6}   # Newton as we had it
FABRICS = ("polyester", "cotton", "denim", "wool")
NAN_PENALTY = 50.0                                   # lit_marker of a blown-up copy (rigid null is ~2-4)


def log(msg):
    os.makedirs(RESULTS, exist_ok=True)
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(os.path.join(RESULTS, "log.txt"), "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------ parameters
# log10 range of edge_ke on the fit mesh (A2 refine 2: B = edge_ke * 0.0198 m). "wide" = the first fit;
# "physical" = published woven bending rigidity 2e-6 .. 2e-4 N*m (KES B ~0.02-2 gf*cm^2/cm), added 2026-10-16 after the
# wide fit put cotton at the range's top (B ~1.7e-3 N*m, ~100x measured cotton): free hanging barely constrains bending
EDGE_RANGES = {"wide": (-6.0, -1.0), "physical": (-4.0, -2.0)}
EDGE_RANGE = EDGE_RANGES["wide"]


def decode(x, rho):
    """x in [0,1]^7 -> physical parameters (log ranges for stiffness and damping time constants)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    lo, hi = EDGE_RANGE
    tri_ke = 10 ** (2 + 4 * x[0])                     # 1e2 .. 1e6 N/m
    tri_ka = tri_ke * 10 ** (-2 + 4 * x[1])           # lambda / mu 0.01 .. 100
    tri_kd = tri_ke * 10 ** (-6 + 5 * x[2])           # time constant 1e-6 .. 0.1 s
    edge_ke = 10 ** (lo + (hi - lo) * x[3])           # see EDGE_RANGES
    edge_kd = edge_ke * 10 ** (-5 + 5 * x[4])         # time constant 1e-5 .. 1 s
    return {"tri_ke": tri_ke, "tri_ka": tri_ka, "tri_kd": tri_kd, "edge_ke": edge_ke, "edge_kd": edge_kd,
            "alpha": 4 * rho * x[5], "delta": rho * x[6]}


def encode(p, rho):
    lo, hi = EDGE_RANGE
    return [(math.log10(p["tri_ke"]) - 2) / 4, (math.log10(p["tri_ka"] / p["tri_ke"]) + 2) / 4,
            (math.log10(p["tri_kd"] / p["tri_ke"]) + 6) / 5, (math.log10(p["edge_ke"]) - lo) / (hi - lo),
            (math.log10(p["edge_kd"] / p["edge_ke"]) + 5) / 5, p["alpha"] / (4 * rho), p["delta"] / rho]


def paper_air(fabric):
    """Mean of the paper's train-session (Table 4) alpha, delta for this fabric (clothilde's carry-over params)."""
    ps = [v for k, v in CD.PAPER["II"].items() if k.startswith(fabric + "_")]
    return float(np.mean([p["alpha"] for p in ps])), float(np.mean([p["delta"] for p in ps]))


def start_params(fabric):
    a, d = paper_air(fabric)
    return {"tri_ke": 1e4, "tri_ka": 1e4, "tri_kd": 10.0, "edge_ke": 2e-4, "edge_kd": 2e-6, "alpha": a, "delta": d}


# ------------------------------------------------------------------------------------------------ recordings
def quads_to_tris(T):
    """Split each quad along alternating diagonals (checkerboard via BFS over shared edges), so the triangulation
    has no preferred shear direction."""
    T = np.asarray(T, int)
    edge_q = {}
    for f, q in enumerate(T):
        for k in range(4):
            edge_q.setdefault(tuple(sorted((q[k], q[(k + 1) % 4]))), []).append(f)
    par = -np.ones(len(T), int)
    for s in range(len(T)):
        if par[s] >= 0:
            continue
        par[s] = 0
        dq = deque([s])
        while dq:
            f = dq.popleft()
            q = T[f]
            for k in range(4):
                for g in edge_q[tuple(sorted((q[k], q[(k + 1) % 4])))]:
                    if par[g] < 0:
                        par[g] = 1 - par[f]
                        dq.append(g)
    tris = []
    for f, (a, b, c, d) in enumerate(T):
        tris += [(a, b, c), (a, c, d)] if par[f] == 0 else [(a, b, d), (b, c, d)]
    return np.array(tris, np.int32)


_PREP = {}


def prepare(path, refine=2, hz=100.0):
    """replay.run's preprocessing, unchanged: first full frame, corners gap-filled, linear resampling, refined mesh."""
    key = (path, refine, hz)
    if key in _PREP:
        return _PREP[key]
    info = CD.parse_name(path)
    t, P = replay.load(path)
    N = P.shape[1]
    R, C = CD.grid_shape(N)
    f0 = CD.first_full_frame(P)
    t, P = t[f0:] - t[f0], P[f0:]
    corners = CD.corner_ids(N)
    Pc = P.copy()
    for k in corners:
        Pc[:, k] = CD.fill_track(t, P[:, k])[0]
    t_rs, P = replay.resample(t, P, hz)
    Pc = replay.resample(t, Pc, hz)[1]
    Tc, Tf, S, X0 = replay.build(P[0], R, C, refine)
    r = {"info": info, "name": info["name"], "key": info["key"], "fabric": info["fabric"], "rho": CD.RHO[info["fabric"]],
         "t": t_rs, "P": P, "U": Pc[:, corners, :], "X0": np.asarray(X0, float), "tris": quads_to_tris(Tf),
         "corners": list(corners), "N": N, "w": CD.lumped_areas(P[0], Tc)}
    _PREP[key] = r
    return r


def train():
    return [r for r in CD.free_hanging() if r["mode"] == "hanger"]


def held():
    return [r for r in CD.free_hanging() if r["mode"] == "hands"]


# ------------------------------------------------------------------------------------------------ kernels
@wp.kernel
def k_drive(frame: wp.array[int], sub: wp.array[int], n_sub: int, traj: wp.array2d[wp.vec3],
            drv_idx: wp.array[int], q: wp.array[wp.vec3], drv_prev: wp.array[wp.vec3]):
    """Scripted particles: end-of-substep target into q, start-of-substep position kept for the hook."""
    j = wp.tid()
    f = frame[0]
    a = traj[f - 1, j]
    b = traj[f, j]
    s = float(sub[0])
    drv_prev[j] = a + (b - a) * (s / float(n_sub))
    q[drv_idx[j]] = a + (b - a) * ((s + 1.0) / float(n_sub))


@wp.kernel
def k_air(qd: wp.array[wp.vec3], mass: wp.array[float], buoy: wp.array[float], drag: wp.array[float], g: float,
          f: wp.array[wp.vec3]):
    """clothilde's air model as an external force: part of gravity cancelled (delta < rho) plus linear drag."""
    i = wp.tid()
    m = mass[i]
    if m == 0.0:
        f[i] = wp.vec3(0.0)
        return
    f[i] = m * (wp.vec3(0.0, 0.0, buoy[i] * g) - drag[i] * qd[i])


@wp.kernel
def k_fix_prev(drv_idx: wp.array[int], drv_prev: wp.array[wp.vec3], q_prev: wp.array[wp.vec3]):
    j = wp.tid()
    q_prev[drv_idx[j]] = drv_prev[j]


@wp.kernel
def k_next_sub(sub: wp.array[int]):
    sub[0] = sub[0] + 1


@wp.kernel
def k_record(frame: wp.array[int], sub: wp.array[int], q: wp.array[wp.vec3], mk_idx: wp.array[int],
             out: wp.array2d[wp.vec3]):
    j = wp.tid()
    out[frame[0], j] = q[mk_idx[j]]
    if j == 0:
        sub[0] = 0


@wp.kernel
def k_next_frame(frame: wp.array[int]):
    frame[0] = frame[0] + 1


class DrivenVBD(newton.solvers.SolverVBD):
    """SolverVBD whose previous-position buffer holds the scripted particles' start-of-substep positions."""

    drv = None

    def _initialize_particles(self, state_in, state_out, dt):
        super()._initialize_particles(state_in, state_out, dt)
        if self.drv is not None:
            idx, prev = self.drv
            wp.launch(k_fix_prev, dim=len(idx), inputs=[idx, prev, self.particle_q_prev], device=self.device)


# ------------------------------------------------------------------------------------------------ batch
class Batch:
    """K copies, copy k = recording recs[k]. Parameters are set per copy with set_params (no rebuild)."""

    def __init__(self, recs, settings=None, device=None, graph=True):
        self.s = dict(FULL, **(settings or {}))
        self.recs = recs
        self.device = wp.get_device(device)
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
        self.spans, drv, mk = [], [], []
        for r in recs:
            p0, t0, e0 = len(b.particle_q), len(b.tri_indices), len(b.edge_indices)
            b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                             vertices=[wp.vec3(*v) for v in r["X0"]], indices=r["tris"].reshape(-1).tolist(),
                             density=r["rho"], tri_ke=1e4, tri_ka=1e4, tri_kd=1.0, edge_ke=1e-4, edge_kd=1e-6,
                             particle_radius=0.005)
            for c in r["corners"]:
                b.particle_mass[p0 + c] = 0.0
                drv.append(p0 + c)
            mk += list(range(p0, p0 + r["N"]))           # markers = the original vertices (kept first by refinement)
            self.spans.append((p0, len(b.particle_q), t0, len(b.tri_indices), e0, len(b.edge_indices)))
        b.color(include_bending=True)
        self.model = b.finalize(device=self.device)
        self.model.edge_rest_angle.zero_()               # cloth is flat at rest (no shape memory)
        self.T = max(len(r["t"]) for r in recs)
        traj = np.zeros((self.T, len(drv), 3), np.float32)
        j = 0
        for r in recs:
            U = r["U"]
            Up = np.concatenate([U, np.repeat(U[-1:], self.T - len(U), axis=0)], axis=0)
            traj[:, j:j + 2] = Up
            j += 2
        d = self.device
        self.traj = wp.array(traj, dtype=wp.vec3, device=d)
        self.drv_idx = wp.array(np.array(drv, np.int32), dtype=int, device=d)
        self.drv_prev = wp.zeros(len(drv), dtype=wp.vec3, device=d)
        self.mk_idx = wp.array(np.array(mk, np.int32), dtype=int, device=d)
        self.out = wp.zeros((self.T, len(mk)), dtype=wp.vec3, device=d)
        self.frame = wp.zeros(1, dtype=int, device=d)
        self.sub = wp.zeros(1, dtype=int, device=d)
        n = self.model.particle_count
        self.buoy = wp.zeros(n, dtype=float, device=d)
        self.drag = wp.zeros(n, dtype=float, device=d)
        self.solver = DrivenVBD(self.model, iterations=self.s["iterations"], particle_enable_self_contact=False,
                                rigid_compliant_alm=True)
        self.solver.drv = (self.drv_idx, self.drv_prev)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.dt = 1.0 / self.s["hz"] / self.s["sub_steps"]
        self.graph = None
        self.use_graph = graph and self.device.is_cuda

    def set_params(self, params):
        tm = self.model.tri_materials.numpy()
        eb = self.model.edge_bending_properties.numpy()
        buoy = np.zeros(self.model.particle_count, np.float32)
        drag = np.zeros(self.model.particle_count, np.float32)
        for (p0, p1, t0, t1, e0, e1), r, p in zip(self.spans, self.recs, params):
            tm[t0:t1, 0], tm[t0:t1, 1], tm[t0:t1, 2] = p["tri_ke"], p["tri_ka"], p["tri_kd"]
            eb[e0:e1, 0], eb[e0:e1, 1] = p["edge_ke"], p["edge_kd"]
            buoy[p0:p1] = 1.0 - p["delta"] / r["rho"]
            drag[p0:p1] = p["alpha"] / r["rho"]
        self.model.tri_materials.assign(tm)
        self.model.edge_bending_properties.assign(eb)
        self.buoy.assign(buoy)
        self.drag.assign(drag)

    def _frame(self):
        for _ in range(self.s["sub_steps"]):
            wp.launch(k_drive, dim=len(self.drv_idx), inputs=[self.frame, self.sub, self.s["sub_steps"], self.traj,
                                                              self.drv_idx, self.s0.particle_q, self.drv_prev])
            wp.launch(k_air, dim=self.model.particle_count, inputs=[self.s0.particle_qd, self.model.particle_mass,
                                                                    self.buoy, self.drag, G, self.s0.particle_f])
            self.solver.step(self.s0, self.s1, None, None, self.dt)
            wp.copy(self.s0.particle_q, self.s1.particle_q)
            wp.copy(self.s0.particle_qd, self.s1.particle_qd)
            wp.launch(k_next_sub, dim=1, inputs=[self.sub])
        wp.launch(k_record, dim=len(self.mk_idx), inputs=[self.frame, self.sub, self.s0.particle_q, self.mk_idx,
                                                          self.out])
        wp.launch(k_next_frame, dim=1, inputs=[self.frame])

    def run(self, params):
        """Simulate all copies with `params` (one dict per copy); returns per-copy score dicts."""
        self.set_params(params)
        for st in (self.s0, self.s1):
            st.particle_q.assign(self.model.particle_q)
            st.particle_qd.zero_()
            st.particle_f.zero_()
        self.out.zero_()
        self.frame.zero_()
        wp.launch(k_record, dim=len(self.mk_idx), inputs=[self.frame, self.sub, self.s0.particle_q, self.mk_idx,
                                                          self.out])                       # frame 0
        self.frame.assign(np.array([1], np.int32))
        self.sub.zero_()
        tic = time.perf_counter()
        if self.use_graph:
            if self.graph is None:
                with wp.ScopedCapture(device=self.device) as cap:
                    self._frame()
                self.graph = cap.graph
                self.frame.assign(np.array([1], np.int32))      # capture does not execute; keep counters at 1 / 0
                self.sub.zero_()
            for _ in range(1, self.T):
                wp.capture_launch(self.graph)
        else:
            for _ in range(1, self.T):
                self._frame()
        wp.synchronize_device(self.device)
        wall = time.perf_counter() - tic
        out = self.out.numpy().astype(float)
        res, j = [], 0
        for r in self.recs:
            n = len(r["t"])
            pred = out[:n, j:j + r["N"]]
            j += r["N"]
            bad = not np.isfinite(pred).all() or np.abs(pred).max() > 100.0
            if bad:
                s = {"lit_marker": NAN_PENALTY, "rms_mm": 1000.0 * NAN_PENALTY, "std_cm": float("nan"),
                     "awrms_cm": float("nan"), "lit_sim": float("nan"), "rms_mm_all": float("nan"),
                     "missing_frac": float("nan")}
            else:
                s = replay.score(pred, r["P"], r["w"], r["w"])
            s.update({"name": r["name"], "key": r["key"], "nan": bool(bad), "frames": n})
            res.append(s)
        self.last_wall = wall
        self.last_pred = out
        return res


# ------------------------------------------------------------------------------------------------ stages
def stage_test():
    """Local checks: scripted corners on their paths; still hang (corners held 6 s) matches clothilde's
    (hang_ref_clothilde.py; the real frame-0 shape is not a pin equilibrium: hands clamp the corner angle, so both
    sims swing the bottom ~5 cm under the corners); copies do not interact; extreme parameters do not crash."""
    res = {}
    path = [r for r in train() if r["fabric"] == "denim" and r["size"] == "A2" and r["motion"] == "shake"][0]["path"]
    r = prepare(path)
    p = start_params("denim")
    short = dict(r, t=r["t"][:201], P=r["P"][:201], U=r["U"][:201])                 # 2 s
    b = Batch([short])
    out = b.run([p])[0]
    pred = b.last_pred[:201, :r["N"]]
    err = float(np.abs(pred[1:, r["corners"]] - short["U"][1:]).max())
    res["corner_tracking_max_m"] = err
    res["corner_tracking_pass"] = err < 1e-5
    res["short_run"] = {k: out[k] for k in ("lit_marker", "rms_mm", "nan")}
    still = dict(short, U=np.repeat(short["U"][:1], 601, axis=0), t=np.arange(601) / 100.0,
                 P=np.repeat(short["P"][:1], 601, axis=0))                                # 6 s hang, corners still
    b2 = Batch([still])
    b2.run([dict(p, tri_ke=1e5, tri_ka=1e5)])
    ref = np.load(os.path.join(RESULTS, "hang_ref_clothilde.npy"))                     # frames 50, 100, 300, 600
    dif = [float(np.linalg.norm(b2.last_pred[f, :r["N"]] - ref[i], axis=1).max() * 1000)
           for i, f in enumerate((50, 100, 300, 600))]
    res["still_hang_vs_clothilde_max_mm"] = dif
    d600 = np.linalg.norm(b2.last_pred[600, :r["N"]] - ref[3], axis=1) * 1000
    res["still_hang_vs_clothilde_rms_mm"] = float(np.sqrt((d600 ** 2).mean()))
    # Judged on RMS: Newton's triangle membrane and clothilde's quad metric constraints are different
    # discretizations; they settle ~5 mm RMS / ~12 mm max apart (9 mm max even at 475 Newton vertices). A harness bug
    # (gravity, mass, pinning) shows up as centimetres.
    res["still_hang_pass"] = res["still_hang_vs_clothilde_rms_mm"] < 6.0
    res["still_hang_drift_from_real_frame0_mm"] = float(np.linalg.norm(b2.last_pred[600, :r["N"]] - r["P"][0], axis=1).max() * 1000)
    # copies must not interact: copy 0 identical when the other copies' parameters change
    b4 = Batch([short] * 4)
    oa = b4.run([p, dict(p, tri_ke=3e3), p, dict(p, alpha=0.0)])
    ob = b4.run([p, dict(p, tri_ke=1e5, edge_ke=1e-2), p, dict(p, delta=0.0)])
    res["batch_independence_diff_mm"] = float(abs(oa[0]["rms_mm"] - ob[0]["rms_mm"]))
    res["batch_independence_pass"] = res["batch_independence_diff_mm"] < 1e-6
    # info: VBD updates colour groups in sequence, so the colouring of a bigger batch changes the update order
    res["info_coloring_order_effect_mm"] = float(abs(oa[0]["rms_mm"] - out["rms_mm"]))
    lo, hi = decode(np.zeros(7), r["rho"]), decode(np.ones(7), r["rho"])
    ox = Batch([short] * 2).run([lo, hi])
    res["extremes"] = [{"rms_mm": o["rms_mm"], "nan": o["nan"]} for o in ox]
    res["extremes_pass"] = True                                                       # reaching here = no crash
    res["all_pass"] = all(v for k, v in res.items() if k.endswith("_pass"))
    jdump(res, os.path.join(RESULTS, "tests.json"))
    log(f"test: {json.dumps(res)}")
    return res


def stage_bench(ks=(16, 256, 1024)):
    """Throughput (copies * sim-s / wall-s) and the numerics check (2x substeps, 2x iterations) on 4 recordings."""
    recs = [prepare(r["path"]) for r in train() if r["size"] == "A2" and r["motion"] == "shake" and r["speed"] == "fast"]
    res = {"device": str(wp.get_device()), "throughput": []}
    for k in ks:
        sel = [recs[i % len(recs)] for i in range(k)]
        sel = [dict(r, t=r["t"][:1001], P=r["P"][:1001], U=r["U"][:1001]) for r in sel]       # 10 s each
        b = Batch(sel)
        params = [start_params(r["fabric"]) for r in sel]
        b.run(params)                                                                      # warm-up + capture
        b.run(params)
        res["throughput"].append({"copies": k, "wall_s": b.last_wall, "sim_s": 10.0,
                                  "copy_sim_s_per_wall_s": k * 10.0 / b.last_wall})
        log(f"bench: K={k} wall {b.last_wall:.2f} s for 10 s -> {k * 10.0 / b.last_wall:.1f} copy-sim-s per wall-s")
    num = {}
    for tag, st in (("full", {}), ("sub20", {"sub_steps": 20}), ("it20", {"iterations": 20})):
        b = Batch(recs, st)
        o = b.run([start_params(r["fabric"]) for r in recs])
        num[tag] = [x["lit_marker"] for x in o]
    res["numerics_lit_marker"] = num
    res["numerics_max_rel_change"] = float(max(abs(a / b_ - 1) for t in ("sub20", "it20")
                                               for a, b_ in zip(num[t], num["full"])))
    jdump(res, os.path.join(RESULTS, "bench.json"))
    log(f"bench numerics: {num}, max rel change {res['numerics_max_rel_change']:.3f}")
    return res


def _lockstep_cma(tag, fabrics, recs_by_fab, x0_by_fab, settings, popsize, max_gens, sigma0, seed=1, resume=True):
    """One CMA-ES per fabric, asked and told in lockstep so each generation is one batched run of
    (fabric x candidate x its recordings). Checkpointed every generation."""
    import cma
    os.makedirs(RESULTS, exist_ok=True)
    ck = os.path.join(RESULTS, f"{tag}_es.pkl")
    sig = hashlib.sha256(json.dumps([sorted(fabrics), {f: [r["name"] for r in recs_by_fab[f]] for f in fabrics},
                                     dict(FULL, **(settings or {})), popsize, sigma0, seed,
                                     {f: list(map(float, x0_by_fab[f])) for f in fabrics}]).encode()).hexdigest()
    st = None
    if resume and os.path.exists(ck):
        with open(ck, "rb") as fh:
            st = pickle.load(fh)
        if st.get("sig") != sig:
            log(f"[{tag}] checkpoint is for a different setup: starting fresh")
            st = None
    if st is not None:
        es, best, hist, g = st["es"], st["best"], st["hist"], st["g"]
        log(f"[{tag}] resumed at gen {g}")
    else:
        es = {f: cma.CMAEvolutionStrategy(list(x0_by_fab[f]), sigma0, {
            "bounds": [[0.0] * 7, [1.0] * 7], "popsize": popsize, "seed": seed, "verbose": -9,
            "tolfun": 1e-5, "tolx": 1e-5}) for f in fabrics}
        best, hist, g = {f: (float("inf"), list(x0_by_fab[f])) for f in fabrics}, [], 0
    layout = [(f, c, r) for f in fabrics for c in range(popsize) for r in recs_by_fab[f]]
    batch = Batch([r for _, _, r in layout], settings)
    t0 = time.perf_counter()
    while g < max_gens and not all(es[f].stop() for f in fabrics):
        X = {f: es[f].ask() for f in fabrics}
        params = [decode(X[f][c], r["rho"]) for f, c, r in layout]
        out = batch.run(params)
        F = {f: np.zeros(popsize) for f in fabrics}
        for (f, c, _), o in zip(layout, out):
            F[f][c] += o["lit_marker"] / len(recs_by_fab[f])
        for f in fabrics:
            es[f].tell(X[f], list(F[f]))
            i = int(np.argmin(F[f]))
            if F[f][i] < best[f][0]:
                best[f] = (float(F[f][i]), list(map(float, X[f][i])))
        g += 1
        hist.append({"gen": g, "wall": batch.last_wall,
                     **{f: {"best": float(F[f].min()), "median": float(np.median(F[f])), "sigma": float(es[f].sigma),
                            "nan": int(sum(o["nan"] for (ff, _, _), o in zip(layout, out) if ff == f))}
                        for f in fabrics}})
        with open(ck + ".tmp", "wb") as fh:
            pickle.dump({"es": es, "best": best, "hist": hist, "g": g, "sig": sig}, fh)
        os.replace(ck + ".tmp", ck)
        log(f"[{tag}] gen {g} ({batch.last_wall:.0f} s): " +
            ", ".join(f"{f} {F[f].min():.3f}/{best[f][0]:.3f}" for f in fabrics))
    log(f"[{tag}] done after {g} generations, {time.perf_counter() - t0:.0f} s")
    return best, hist


def stage_fit(popsize=16, max_gens=30, bend="wide"):
    """bend = "wide" -> newton_fitted.json; "physical" -> newton_fitted_physbend.json (edge_ke limited, see EDGE_RANGES)."""
    global EDGE_RANGE
    EDGE_RANGE = EDGE_RANGES[bend]
    suffix = "" if bend == "wide" else "_physbend"
    recs = {f: [prepare(r["path"]) for r in train() if r["fabric"] == f] for f in FABRICS}
    x0 = {f: encode(start_params(f), CD.RHO[f]) for f in FABRICS}
    best, hist = _lockstep_cma("fit" + suffix, FABRICS, recs, x0, {}, popsize, max_gens, 0.15)
    fitted = {f: {"x": best[f][1], "train_lit_marker": best[f][0], **decode(best[f][1], CD.RHO[f])} for f in FABRICS}
    jdump({"settings": FULL, "edge_range_log10": EDGE_RANGE, "fitted": fitted, "hist": hist},
          os.path.join(RESULTS, f"newton_fitted{suffix}.json"))
    return fitted


def stage_freeze(name="newton_fitted.json", pre="prereg.json"):
    path = os.path.join(RESULTS, name)
    with open(path, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    jdump({"file": name, "sha256": h, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(RESULTS, pre))
    log(f"frozen {name} sha256 {h}")


def stage_final_physbend():
    """Second declared model (rule set before its fit): the hand-off uses it if held-out lit_marker passes gate 1b."""
    pre = jload(os.path.join(RESULTS, "prereg_physbend.json"))
    path = os.path.join(RESULTS, "newton_fitted_physbend.json")
    with open(path, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    if h != pre["sha256"]:
        raise RuntimeError("newton_fitted_physbend.json changed after freeze")
    fit = jload(path)["fitted"]
    out, wall = _held_scores(fit, {}, tags=("newton_fitted",))
    for o in out:
        o["tag"] = "newton_physbend"
    jdump({"sha256": h, "runs": out, "wall_s": wall}, os.path.join(RESULTS, "final_physbend.json"))
    b, layout = _held_scores.last
    j, preds = 0, {}
    for (_, r) in layout:
        preds[f"newton_physbend|{r['name']}"] = b.last_pred[:len(r["t"]), j:j + r["N"]].astype(np.float32)
        j += r["N"]
    np.savez_compressed(os.path.join(RESULTS, "final_physbend_pred.npz"), **preds)
    log(f"final_physbend: lit_marker mean {np.mean([o['lit_marker'] for o in out]):.3f}, rms_mm median "
        f"{np.median([o['rms_mm'] for o in out]):.1f}, nan {sum(o['nan'] for o in out)}")
    return out


def _held_scores(fitted, settings, tags=("newton_defaults", "newton_fitted")):
    recs = [prepare(r["path"], refine=dict(FULL, **settings)["refine"]) for r in held()]
    layout, params = [], []
    for r in recs:
        f = r["fabric"]
        for tag in tags:
            if tag == "newton_defaults":
                p = dict(GRIP_BENCH, alpha=0.0, delta=r["rho"])
            else:
                p = {k: fitted[f][k] for k in NAMES}
            layout.append((tag, r))
            params.append(p)
    b = Batch([r for _, r in layout], settings)
    out = b.run(params)
    for (tag, r), o in zip(layout, out):
        o.update({"tag": tag, "fabric": r["fabric"]})
    _held_scores.last = (b, layout)
    return out, b.last_wall


def stage_final():
    pre = jload(os.path.join(RESULTS, "prereg.json"))
    path = os.path.join(RESULTS, "newton_fitted.json")
    with open(path, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    if h != pre["sha256"]:
        raise RuntimeError("newton_fitted.json changed after freeze")
    fit = jload(path)["fitted"]
    out, wall = _held_scores(fit, {})
    jdump({"sha256": h, "runs": out, "wall_s": wall}, os.path.join(RESULTS, "final.json"))
    b, layout = _held_scores.last                      # predictions for the independent recheck (verify_held.py)
    j, preds = 0, {}
    for (tag, r) in layout:
        preds[f"{tag}|{r['name']}"] = b.last_pred[:len(r["t"]), j:j + r["N"]].astype(np.float32)
        j += r["N"]
    np.savez_compressed(os.path.join(RESULTS, "final_pred.npz"), **preds)
    for tag in ("newton_defaults", "newton_fitted"):
        rs = [o for o in out if o["tag"] == tag]
        log(f"final {tag}: lit_marker mean {np.mean([o['lit_marker'] for o in rs]):.3f}, rms_mm median "
            f"{np.median([o['rms_mm'] for o in rs]):.1f}, nan {sum(o['nan'] for o in rs)}")
    return out


RUNGS = [("full", {}), ("sub20", {"sub_steps": 20}), ("it5", {"iterations": 5}), ("sub5", {"sub_steps": 5}),
         ("sub5_it5", {"sub_steps": 5, "iterations": 5}),
         ("sub2_it5", {"sub_steps": 2, "iterations": 5}), ("sub5_it2", {"sub_steps": 5, "iterations": 2}),
         ("refine1_sub5_it5", {"refine": 1, "sub_steps": 5, "iterations": 5})]


def stage_ladder(gens=8, popsize=12):
    """Each rung: short refit from the fitted point on the train session, then the held-out score and the throughput
    at the held-out batch size. Fitted parameters absorb each setting's numerical damping, hence the refit."""
    fit = jload(os.path.join(RESULTS, "newton_fitted.json"))["fitted"]
    rows = []
    for name, st in RUNGS:
        s = dict(FULL, **st)
        recs = {f: [prepare(r["path"], refine=s["refine"]) for r in train() if r["fabric"] == f] for f in FABRICS}
        x0 = {f: fit[f]["x"] for f in FABRICS}
        if name == "full":
            refit = {f: dict(fit[f]) for f in FABRICS}
        else:
            best, _ = _lockstep_cma(f"ladder_{name}", FABRICS, recs, x0, st, popsize, gens, 0.05)
            refit = {f: {"x": best[f][1], **decode(best[f][1], CD.RHO[f])} for f in FABRICS}
        out, _ = _held_scores(refit, st, tags=("newton_fitted",))
        b, layout = _held_scores.last                   # time a second run: the first includes graph capture
        b.run([{k: refit[r["fabric"]][k] for k in NAMES} for _, r in layout])
        wall = b.last_wall
        held_s = sum(o["frames"] - 1 for o in out) / s["hz"]
        row = {"rung": name, "settings": s, "params": refit, "held_lit_marker_mean": float(np.mean([o["lit_marker"] for o in out])),
               "held_rms_mm_median": float(np.median([o["rms_mm"] for o in out])), "nan": int(sum(o["nan"] for o in out)),
               "wall_s": wall, "copy_sim_s_per_wall_s": held_s / wall}
        rows.append(row)
        jdump(rows, os.path.join(RESULTS, "ladder.json"))
        log(f"ladder {name}: held lit_marker {row['held_lit_marker_mean']:.3f}, rms {row['held_rms_mm_median']:.1f} mm, "
            f"{row['copy_sim_s_per_wall_s']:.0f} copy-sim-s/s")
    return rows


def mesh_bending_geometry(recs):
    """Mean over interior hinges of (rest length L, area of the two adjacent triangles A) on the fit mesh."""
    Ls, As = [], []
    for r in recs:
        X, T = r["X0"], r["tris"]
        area = 0.5 * np.linalg.norm(np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), axis=1)
        e2t = {}
        for t, (a, b, c) in enumerate(T):
            for u, v in ((a, b), (b, c), (c, a)):
                e2t.setdefault((min(u, v), max(u, v)), []).append(t)
        for (u, v), ts in e2t.items():
            if len(ts) == 2:
                Ls.append(np.linalg.norm(X[u] - X[v]))
                As.append(area[ts[0]] + area[ts[1]])
    return float(np.mean(Ls)), float(np.mean(As))


# RGBench (AAAI 2026) Table 3, lab-measured per class (Style3D units): stretch g/s^2, bending g*m^2/s^2/rad, g/m^2
RGB_COTTON = {"stretch": 1.0e6, "bending": 2800.0, "density": 170.0}
RGB_KNIT = {"stretch": 2.5e4, "bending": 200.0, "density": 190.0}
RGB_TSHIRT = {"stretch": 2.3e5, "bending": 1000.0, "density": 220.0}     # configs/cloth_params/green_tshirt.yaml (MIT)


def pick_rung():
    """Gate 2: the fastest ladder rung (RL-size throughput from stage speed when measured) whose held-out lit_marker
    stays within 25 % of the full rung's. Solver rungs only (2026-10-17, after the speed run): the mesh is set by the
    folding task, not by this sheet; the refine-1 rung is the 4x5 marker grid (14 cm cells), which cannot fold."""
    lad = jload(os.path.join(RESULTS, "ladder.json")) if os.path.exists(os.path.join(RESULTS, "ladder.json")) else []
    lad_all, lad = lad, [r for r in lad if r["settings"]["refine"] == FULL["refine"]]
    full = next((r for r in lad if r["rung"] == "full"), None)
    spd = {r["rung"]: r for r in jload(os.path.join(RESULTS, "speed.json"))} if os.path.exists(
        os.path.join(RESULTS, "speed.json")) else {}
    for r in lad:
        if r["rung"] in spd:
            r["copy_sim_s_per_wall_s_1024"] = spd[r["rung"]]["copy_sim_s_per_wall_s"]
    key = "copy_sim_s_per_wall_s_1024" if lad and all("copy_sim_s_per_wall_s_1024" in r for r in lad) \
        else "copy_sim_s_per_wall_s"
    ok = [r for r in lad if full and r["nan"] == 0 and r["held_lit_marker_mean"] <= 1.25 * full["held_lit_marker_mean"]]
    for r in lad_all:
        if r["rung"] in spd:
            r["copy_sim_s_per_wall_s_1024"] = spd[r["rung"]]["copy_sim_s_per_wall_s"]
    return (max(ok, key=lambda r: r[key]) if ok else None), lad_all


def stage_strip_physbend(gens=8, popsize=12):
    """The physical-bending model refit at the rung gate 2 picked (on the wide ladder), then held-out scored."""
    global EDGE_RANGE
    EDGE_RANGE = EDGE_RANGES["physical"]
    rung, _ = pick_rung()
    fit = jload(os.path.join(RESULTS, "newton_fitted_physbend.json"))["fitted"]
    st = {k: v for k, v in rung["settings"].items() if FULL.get(k) != v}
    s = dict(FULL, **st)
    recs = {f: [prepare(r["path"], refine=s["refine"]) for r in train() if r["fabric"] == f] for f in FABRICS}
    best, _ = _lockstep_cma(f"strip_physbend_{rung['rung']}", FABRICS, recs, {f: fit[f]["x"] for f in FABRICS}, st,
                            popsize, gens, 0.05)
    refit = {f: {"x": best[f][1], **decode(best[f][1], CD.RHO[f])} for f in FABRICS}
    out, _ = _held_scores(refit, st, tags=("newton_fitted",))
    row = {"rung": rung["rung"], "settings": s, "params": refit,
           "held_lit_marker_mean": float(np.mean([o["lit_marker"] for o in out])),
           "held_rms_mm_median": float(np.median([o["rms_mm"] for o in out])), "nan": int(sum(o["nan"] for o in out)),
           "copy_sim_s_per_wall_s": rung.get("copy_sim_s_per_wall_s_1024", rung["copy_sim_s_per_wall_s"])}
    jdump(row, os.path.join(RESULTS, "strip_physbend.json"))
    log(f"strip_physbend {rung['rung']}: held lit_marker {row['held_lit_marker_mean']:.3f}, rms {row['held_rms_mm_median']:.1f}")
    return row


def stage_handoff():
    """cloth_fitted_rl.json for the RL env + the Newton section of cloth_table.md, from fit / final / ladder."""
    fit = jload(os.path.join(RESULTS, "newton_fitted.json"))
    fin = jload(os.path.join(RESULTS, "final.json"))
    lad = jload(os.path.join(RESULTS, "ladder.json")) if os.path.exists(os.path.join(RESULTS, "ladder.json")) else []
    clp = os.path.join(HERE, "results", "paper2022", "final_paper.json")
    cl = jload(clp)["runs"] if os.path.exists(clp) else []

    def mean_tag(runs, tag, k="lit_marker"):
        v = [o[k] for o in runs if o.get("tag") == tag and "error" not in o]
        return float(np.mean(v)) if v else None

    def med_tag(runs, tag, k="rms_mm"):
        v = [o[k] for o in runs if o.get("tag") == tag and "error" not in o]
        return float(np.median(v)) if v else None

    ref_cl, ref_rigid = mean_tag(cl, "paper_fabric"), mean_tag(cl, "rigid")
    nf = mean_tag(fin["runs"], "newton_fitted")
    gate1a = nf <= 1.5 * ref_cl
    gate1b = nf <= ref_rigid - 0.5 * (ref_rigid - ref_cl)
    strip, lad = pick_rung()
    # model choice, rule set before the physical-bending fit: use it if its held-out score passes gate 1b
    pb_path = os.path.join(RESULTS, "final_physbend.json")
    pb = jload(pb_path)["runs"] if os.path.exists(pb_path) else []
    pb_mean = float(np.mean([o["lit_marker"] for o in pb])) if pb else None
    use_pb = pb_mean is not None and pb_mean <= ref_rigid - 0.5 * (ref_rigid - ref_cl)
    if use_pb:
        fit = jload(os.path.join(RESULTS, "newton_fitted_physbend.json"))
        sp = os.path.join(RESULTS, "strip_physbend.json")
        strip = jload(sp) if os.path.exists(sp) else None
    L, A = mesh_bending_geometry([prepare(r["path"]) for r in train() if r["size"] == "A2"])

    def card(p, rho, tag):
        B = p["edge_ke"] * A / (3 * L)                   # Newton hinge k = edge_ke*L  <->  3*B*L^2/A_pair
        return {"density_kg_m2": rho, "tri_ke": p["tri_ke"], "tri_ka": p["tri_ka"], "tri_kd": p["tri_kd"],
                "edge_ke_fit_mesh": p["edge_ke"], "edge_kd_fit_mesh": p["edge_kd"],
                "bending_B_equiv_Nm": B, "bending_tau_s": p["edge_kd"] / p["edge_ke"],
                "air_drag_rate_per_s": p["alpha"] / rho, "air_gravity_scale": p["delta"] / rho, "tag": tag}

    fabrics = {f: card(fit["fitted"][f], CD.RHO[f], "fitted on Coltraro mocap (woven, ~1 mm markers)") for f in FABRICS}
    stripped = None
    if strip:
        L_fit, A_fit = L, A
        L, A = mesh_bending_geometry([prepare(r["path"], refine=strip["settings"]["refine"]) for r in train()
                                      if r["size"] == "A2"])          # the stripped card's own mesh
        stripped = {f: card(strip["params"][f], CD.RHO[f], f"refit at rung {strip['rung']}") for f in FABRICS}
        stripped["mesh_mean_L_m"], stripped["mesh_mean_A_pair_m2"] = L, A
        L, A = L_fit, A_fit
    c = fit["fitted"]["cotton"]
    s_lo, s_hi = RGB_KNIT["stretch"] / RGB_COTTON["stretch"], RGB_TSHIRT["stretch"] / RGB_COTTON["stretch"]
    b_lo, b_hi = RGB_KNIT["bending"] / RGB_COTTON["bending"], RGB_TSHIRT["bending"] / RGB_COTTON["bending"]
    gm = lambda lo, hi: math.sqrt(lo * hi)
    knit = card({"tri_ke": c["tri_ke"] * gm(s_lo, s_hi), "tri_ka": c["tri_ka"] * gm(s_lo, s_hi),
                 "tri_kd": c["tri_kd"] * gm(s_lo, s_hi), "edge_ke": c["edge_ke"] * gm(b_lo, b_hi),
                 "edge_kd": c["edge_kd"] * gm(b_lo, b_hi), "alpha": c["alpha"] / CD.RHO["cotton"] * 0.19,
                 "delta": c["delta"] / CD.RHO["cotton"] * 0.19}, 0.19,
                "prior: fitted cotton x RGBench lab-measured knit/cotton ratios (geometric mean of knit class and "
                "their T-shirt); not validated on knit motion (none exists online)")
    knit["randomize"] = {"stretch_x_cotton": [s_lo, s_hi], "bending_x_cotton": [b_lo, b_hi],
                         "density_kg_m2": [0.161, 0.220]}
    out = {
        "what": "Newton 1.6.1 SolverVBD cloth parameters for the RL env, fitted to real cloth (Coltraro IJRR 2025 "
                "free-hanging mocap, two corners scripted), plus a knit T-shirt prior. Woven numbers are validated "
                "on held-out recordings; knit numbers are a prior.",
        "solver": {"full": fit["settings"], "stripped": strip["settings"] if strip else None,
                   "particle_enable_self_contact": "True in the RL env (off in the fit: free hanging)",
                   "edge_rest_angle": 0.0},
        "air_model": "clothilde's air model as an external force per free particle (newton_cloth.k_air): "
                     "f = m*g*(1 - air_gravity_scale)*z_up - air_drag_rate_per_s*m*v, so gravity acts as "
                     "air_gravity_scale*g plus linear drag; VBD has no air drag of its own",
        "mesh_scaling": {"tri_ke, tri_ka, tri_kd": "mesh-independent (per area)",
                         "edge_ke": "per hinge: edge_ke = 3*B*L/A_pair (L rest length, A_pair area of its two "
                                    "triangles); edge_kd = bending_tau_s * edge_ke",
                         "fit_mesh_mean_L_m": L, "fit_mesh_mean_A_pair_m2": A},
        "fabrics_full": fabrics, "fabrics_stripped": stripped, "knit_tshirt_prior": knit,
        "friction": {"cloth_cloth": {"nominal": 0.6, "range": [0.4, 0.8]},
                     "cloth_table": {"nominal": 0.5, "range": [0.25, 0.8]},
                     "source": "measured: Coltraro 0.48-0.80, Cloth-Var 0.40-0.68; RGBench T-shirt 0.6 cloth, "
                               "0.266 ground (tuned). VBD mixes pair friction as sqrt(mu_cloth*mu_shape)",
                     "tag": "measured ranges, not fitted (contact recordings cut from this step)"},
        "validation": {"held_out_recordings": 32, "metric": "lit_marker (Coltraro eq. 10, cm) mean; rms_mm median",
                       "rigid_null": {"lit_marker": ref_rigid, "rms_mm": med_tag(cl, "rigid")},
                       "clothilde_paper_per_fabric": {"lit_marker": ref_cl, "rms_mm": med_tag(cl, "paper_fabric")},
                       "newton_defaults": {"lit_marker": mean_tag(fin["runs"], "newton_defaults"),
                                           "rms_mm": med_tag(fin["runs"], "newton_defaults")},
                       "newton_fitted": {"lit_marker": nf, "rms_mm": med_tag(fin["runs"], "newton_fitted")},
                       "newton_physbend": ({"lit_marker": pb_mean, "rms_mm": med_tag(pb, "newton_physbend")}
                                           if pb else None),
                       "handoff_model": "physbend (bending limited to published woven range)" if use_pb else
                                        "wide (bending unconstrained; cotton/denim above published range)",
                       "newton_stripped": ({"lit_marker": strip["held_lit_marker_mean"],
                                            "rms_mm": strip["held_rms_mm_median"],
                                            "copy_sim_s_per_wall_s_L4": strip["copy_sim_s_per_wall_s"]} if strip else None),
                       "gate1a_le_1p5x_clothilde": bool(gate1a), "gate1b_half_of_clothilde_gain": bool(gate1b),
                       "frozen_sha256": fin["sha256"]},
        "caveats": ["air model is the paper's isotropic one: gravity acts as air_gravity_scale*g (0.4-0.7) in every "
                    "direction of motion. Equivalent to real weight + added mass for hanging/flinging, but on a table "
                    "the friction load is also scaled, so in-plane sliding stops ~1/air_gravity_scale too late. For "
                    "table phases use full gravity (air terms only matter in flings/drops) or refit a normal-direction "
                    "air model (one ~20 min L4 fit)",
                    "membrane stiffness is VBD-effective, not physical: denim fits soft (tri_ke ~0.3-0.9e3 N/m) while "
                    "polyester fits ~5e5; check stretch under a grasp in the interaction step",
                    "corners are point pins in the fit; real hands/hangers clamp the corner angle (both sims share this)",
                    "fitted stiffness is VBD-effective at the stated iterations/substeps: refit if those change",
                    "free-hanging motion only; bending and friction are not identified by contact recordings",
                    "Coltraro data is research-only (CC BY-NC-SA); these numbers are for our research env"]}
    jdump(out, os.path.join(RESULTS, "cloth_fitted_rl.json"))
    L_ = ["# Cloth step 2: Newton VBD fitted to real cloth", "",
          "Held-out session (32 `*_hands` recordings, scored once after sha256 freeze "
          f"`{fin['sha256'][:16]}`), two corners scripted along their recorded paths.", "",
          "| Model | paper metric (lit_marker, mean) | marker RMS mm (median) |", "|---|---|---|"]
    v = out["validation"]
    for k, lab in (("rigid_null", "rigid sheet carried by the corners"), ("newton_defaults", "Newton as we had it"),
                   ("clothilde_paper_per_fabric", "clothilde, paper's per-fabric params"),
                   ("newton_fitted", "Newton fitted, bending free (full)"),
                   ("newton_physbend", "Newton fitted, bending in published woven range (full)"),
                   ("newton_stripped", "Newton hand-off model, stripped")):
        if v.get(k):
            L_.append(f"| {lab} | {v[k]['lit_marker']:.3f} | {v[k]['rms_mm']:.1f} |")
    L_ += ["", f"- Gate 1a (<= 1.5x clothilde): **{'PASS' if gate1a else 'FAIL'}**; gate 1b (at least half of "
           f"clothilde's gain over rigid): **{'PASS' if gate1b else 'FAIL'}**.", f"- Hand-off model: {v['handoff_model']}."]
    if lad:
        L_ += ["", "| Rung | settings | held lit_marker | RMS mm | copy-sim-s per wall-s, L4: 32 copies / 1024 copies | NaN |",
               "|---|---|---|---|---|---|"]
        for r in lad:
            s = r["settings"]
            big = r.get("copy_sim_s_per_wall_s_1024")
            L_.append(f"| {r['rung']} | sub {s['sub_steps']}, it {s['iterations']}, refine {s['refine']} | "
                      f"{r['held_lit_marker_mean']:.3f} | {r['held_rms_mm_median']:.1f} | {r['copy_sim_s_per_wall_s']:.0f} / "
                      f"{big:.0f} | {r['nan']} |" if big else
                      f"| {r['rung']} | sub {s['sub_steps']}, it {s['iterations']}, refine {s['refine']} | "
                      f"{r['held_lit_marker_mean']:.3f} | {r['held_rms_mm_median']:.1f} | {r['copy_sim_s_per_wall_s']:.0f} / – | {r['nan']} |")
        L_.append("")
        L_.append(f"Gate 2 (fastest rung within 25 % of full): **{strip['rung'] if strip else 'none'}**.")
    with open(os.path.join(RESULTS, "cloth_table.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L_) + "\n")
    print("\n".join(L_))
    return out


def stage_speed(k=1024):
    """RL-size throughput per ladder rung: K copies of the 4 A2 fast-shake recordings (10 s), timed on a second run."""
    recs = [prepare(r["path"]) for r in train() if r["size"] == "A2" and r["motion"] == "shake" and r["speed"] == "fast"]
    rows = []
    for name, st in RUNGS:
        s = dict(FULL, **st)
        rr = [prepare(r["path"], refine=s["refine"]) for r in train()
              if r["size"] == "A2" and r["motion"] == "shake" and r["speed"] == "fast"] if s["refine"] != 2 else recs
        sel = [dict(rr[i % len(rr)], t=rr[i % len(rr)]["t"][:1001], P=rr[i % len(rr)]["P"][:1001],
                    U=rr[i % len(rr)]["U"][:1001]) for i in range(k)]
        b = Batch(sel, st)
        params = [start_params(r["fabric"]) for r in sel]
        b.run(params)
        b.run(params)
        rows.append({"rung": name, "settings": s, "copies": k, "wall_s": b.last_wall,
                     "copy_sim_s_per_wall_s": k * 10.0 / b.last_wall, "sim_steps_per_s": k * 1000 / b.last_wall})
        jdump(rows, os.path.join(RESULTS, "speed.json"))
        log(f"speed {name}: {k} copies x 10 s in {b.last_wall:.1f} s -> {k * 10.0 / b.last_wall:.0f} copy-sim-s/s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["test", "bench", "fit", "freeze", "final", "ladder", "handoff",
                                                       "fit_physbend", "freeze_physbend", "final_physbend", "speed",
                                                       "strip_physbend"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--gens", type=int, default=30)
    a = ap.parse_args()
    if a.device:
        wp.set_device(a.device)
    os.makedirs(RESULTS, exist_ok=True)
    if a.stage == "fit":
        stage_fit(max_gens=a.gens)
    elif a.stage == "fit_physbend":
        stage_fit(max_gens=a.gens, bend="physical")
    elif a.stage == "freeze_physbend":
        stage_freeze("newton_fitted_physbend.json", "prereg_physbend.json")
    else:
        globals()["stage_" + a.stage]()


if __name__ == "__main__":
    main()

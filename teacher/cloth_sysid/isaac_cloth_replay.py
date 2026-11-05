"""Isaac particle cloth (LeHome's settings, v2 cloth values) on held-out Coltraro free-hanging recordings: the two upper
corners follow their recorded paths, the sheet (~5 mm particle spacing, like LeHome's garments) hangs and swings, and
the markers are compared with the real ones (same score as replay.score: rms_mm over present markers, frames 1..T-1).

Cloth values (as oracle/isaac_physics.py v2, per fabric): particle mass = areal density x area / particles; LeHome's
stretch / bend / shear stiffness and spring damping scaled by (that mass / LeHome's 1e-2 kg); gravity x1; no stickiness.
Corners: heavy particles whose positions and velocities are overwritten every physics step (Isaac has no pins).
LeHome's particle-system settings and dt 1/90 (GPU pipeline: Isaac's cloth view needs it); no floor (the recordings' z = 0 is not a floor).

    .venv/bin/python isaac_cloth_replay.py --data <dir of prep_isaac_cloth.py npz> --out <json>
"""
import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dt", type=float, default=1.0 / 90.0)
ap.add_argument("--stretch_mult", type=float, nargs="+", default=[1.0])   # sweep: x v2 stretch stiffness
ap.add_argument("--bend_mult", type=float, nargs="+", default=[1.0])      # sweep: x v2 bend + shear stiffness
ap.add_argument("--pattern", default="*.npz")
ap.add_argument("--iters", type=int, default=32)                         # particle-system position iterations
ap.add_argument("--damping", type=float, nargs="+", default=[0.05])      # particle-material velocity damping
ap.add_argument("--variant", default="v2", choices=["v2", "orig"])   # orig = LeHome as shipped (1e-2 kg/pt, gravity x2)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaacsim.core.api.materials.particle_material import ParticleMaterial  # noqa: E402
from isaacsim.core.prims import ClothPrim, SingleClothPrim, SingleParticleSystem  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
from pxr import Gf, UsdGeom, Vt  # noqa: E402

LEHOME = {"stretch_stiffness": 1e8, "bend_stiffness": 5000.0, "shear_stiffness": 5000.0, "spring_damping": 10.0}
SPACING_X = 2.0


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device="cuda:0", use_fabric=False,
                                                              gravity=(0.0, 0.0, -9.8)))
    stage = get_current_stage()
    psys = SingleParticleSystem(prim_path="/World/ps", particle_system_enabled=True, enable_ccd=True,
                                solver_position_iteration_count=args.iters, global_self_collision_enabled=True,
                                non_particle_collision_enabled=True, contact_offset=0.008, rest_offset=0.004,
                                particle_contact_offset=0.005, fluid_rest_offset=0.0025, solid_rest_offset=0.0025,
                                max_velocity=5.0)
    recs, cloths = [], []
    import itertools
    items = list(itertools.product(sorted(glob.glob(os.path.join(args.data, args.pattern))), args.stretch_mult, args.bend_mult,
                                   args.damping))
    for k, (path, sm, bm, dmp) in enumerate(items):
        d = {key: v for key, v in np.load(path, allow_pickle=True).items()}
        off = np.array([SPACING_X * k, 0.0, 0.0])
        X0, tris = d["X0"] + off, d["tris"].astype(int)
        area = 0.5 * np.linalg.norm(np.cross(X0[tris[:, 1]] - X0[tris[:, 0]], X0[tris[:, 2]] - X0[tris[:, 0]]), axis=1).sum()
        m_pt = float(d["rho"]) * area / len(X0) if args.variant == "v2" else 1e-2
        r = m_pt / 1e-2
        gs, adh, pfs = (1.0, 0.0, 1.2) if args.variant == "v2" else (2.0, 0.1, 0.6)
        mpath = f"/World/cloth_{k}"
        mesh = UsdGeom.Mesh.Define(stage, mpath)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in X0]))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(tris)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(tris.reshape(-1).tolist()))
        mat = ParticleMaterial(prim_path=f"/World/mat_{k}", adhesion=adh, friction=0.5, particle_friction_scale=pfs,
                               damping=dmp, drag=0.0, lift=0.0, gravity_scale=gs)
        SingleClothPrim(prim_path=mpath, particle_system=psys, particle_material=mat, particle_mass=m_pt,
                        self_collision=True, self_collision_filter=True,
                        stretch_stiffness=LEHOME["stretch_stiffness"] * r * sm, bend_stiffness=LEHOME["bend_stiffness"] * r * bm,
                        shear_stiffness=LEHOME["shear_stiffness"] * r * bm, spring_damping=LEHOME["spring_damping"] * r)
        recs.append(dict(name=f"{d['name']}|s{sm:g}|b{bm:g}|d{dmp:g}", combo=f"s{sm:g}|b{bm:g}|d{dmp:g}", fabric=str(d["fabric"]), off=off, U=d["U"] + off, P=d["P"] + off,
                         t=d["t"], N=int(d["N"]), corners=d["corners"].astype(int), n=len(X0), m_pt=m_pt))
        print("CLOTH", recs[-1]["name"], "particles", len(X0), "mass/pt %.2e" % m_pt, "stiff x %.2e" % r, flush=True)
    view = ClothPrim("/World/cloth_.*")
    sim.reset()
    view.initialize()
    nmax = view.max_particles_per_cloth
    masses = view.get_particle_masses().clone() if hasattr(view, "get_particle_masses") else None
    if masses is not None:
        for i, rc in enumerate(recs):
            masses[i, rc["corners"]] = 1e3                    # corners: heavy, then placed every step
        # ClothPrim.set_particle_masses calls a missing get_masses (Isaac 5.1): use the physics view directly
        view._physics_view.set_masses(masses, torch.arange(view.count, device=masses.device))
    T = max(len(rc["t"]) for rc in recs)
    steps = int(np.ceil((T - 1) * 0.01 / args.dt))
    pred = [np.full((len(rc["t"]), rc["N"], 3), np.nan) for rc in recs]
    for i, rc in enumerate(recs):
        pred[i][0] = rc["P"][0]
    for s in range(1, steps + 1):
        tt = s * args.dt
        pos = view.get_world_positions().clone()
        vel = torch.zeros_like(pos)
        for i, rc in enumerate(recs):
            f = min(tt / 0.01, len(rc["t"]) - 1.0001)
            j = int(f)
            a = f - j
            u = (1 - a) * rc["U"][j] + a * rc["U"][j + 1]
            v = (rc["U"][j + 1] - rc["U"][j]) / 0.01
            pos[i, rc["corners"]] = torch.tensor(u, dtype=pos.dtype, device=pos.device)
            vel[i, rc["corners"]] = torch.tensor(v, dtype=pos.dtype, device=pos.device)
        view.set_world_positions(pos)
        cur_v = view.get_velocities().clone()
        for i, rc in enumerate(recs):
            cur_v[i, rc["corners"]] = vel[i, rc["corners"]]
        view.set_velocities(cur_v)
        sim.step(render=False)
        now = view.get_world_positions().cpu().numpy()
        for i, rc in enumerate(recs):
            fr = int(round(tt / 0.01))
            if 1 <= fr < len(rc["t"]) and abs(fr * 0.01 - tt) < args.dt / 2:
                pred[i][fr] = now[i, :rc["N"]]
        if s % 300 == 0:
            print("STEP", s, "/", steps, flush=True)
    out = {}
    for i, rc in enumerate(recs):
        P, Q = rc["P"], pred[i]
        for fr in range(1, len(rc["t"])):                    # fill frames without an exact physics sample
            if np.isnan(Q[fr]).all():
                Q[fr] = Q[fr - 1]
        d = Q[1:] - P[1:]
        ok = ~np.isnan(d).any(axis=2)
        rms = 1000 * np.sqrt(np.where(ok, (d ** 2).sum(2), 0).sum(1) / np.maximum(ok.sum(1), 1))
        mv = P[1:] - P[:1]
        okm = ~np.isnan(mv).any(axis=2)
        amp = 1000 * np.sqrt(np.where(okm, (mv ** 2).sum(2), 0).sum() / okm.sum())
        finite = bool(np.isfinite(Q).all())
        cerr = 1000 * float(np.nanmean(np.linalg.norm(Q[1:, rc["corners"]] - rc["U"][1:len(Q)], axis=2)))
        dz = 1000 * float(np.nanmean(Q[1:, :, 2] - P[1:, :, 2]))
        span_sim = 1000 * float(np.nanmean(np.linalg.norm(Q[1:, 0] - Q[1:, -1], axis=1)))
        span_real = 1000 * float(np.nanmean(np.linalg.norm(P[1:, 0] - P[1:, -1], axis=1)))
        out[rc["name"]] = {"fabric": rc["fabric"], "rms_mm": round(float(np.nanmean(rms)), 2), "motion_mm": round(float(amp), 2),
                           "pct": round(float(100 * np.nanmean(rms) / amp), 1), "finite": finite, "corner_err_mm": round(cerr, 1),
                           "mean_dz_mm": round(dz, 1), "diag_span_sim_mm": round(span_sim, 1), "diag_span_real_mm": round(span_real, 1)}
        print("RESULT", rc["name"], json.dumps(out[rc["name"]]), flush=True)
    summ = {}
    for c in sorted({k.split("|", 1)[1] for k in out}):
        vs = [v for k, v in out.items() if k.split("|", 1)[1] == c]
        rm, mo = float(np.mean([v["rms_mm"] for v in vs])), float(np.mean([v["motion_mm"] for v in vs]))
        summ[c] = {"rms_mm": round(rm, 2), "pct": round(100 * rm / mo, 1),
                   "stretch_pct": round(float(np.mean([100 * (v["diag_span_sim_mm"] / v["diag_span_real_mm"] - 1) for v in vs])), 1),
                   "mean_dz_mm": round(float(np.mean([v["mean_dz_mm"] for v in vs])), 1), "finite": all(v["finite"] for v in vs)}
        print("COMBO", c, json.dumps(summ[c]), flush=True)
    summ["reference"] = {"newton_handoff_pct_all32": 17, "clothilde_pct_all32": 19, "rigid_sheet_pct_all32": 25}
    print("SUMMARY", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "per_recording": out}, open(args.out, "w"), indent=1)


main()
sys.stdout.flush()
os._exit(0)

"""Phase 0 of docs/warp_port.md: can MuJoCo Warp load and step the fold model?

Compiles the stock model on mujoco 3.13, uploads it with the batched fields the
port needs, steps a few worlds on whatever device Warp picks (CPU on the Mac),
and compares the cloth vertices against the CPU simulator stepped the same way.

    .venv-warp/bin/python scripts/warp_probe.py [--substeps 100] [--nworld 2]
"""
import argparse
from pathlib import Path
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mujuco.sim_main import ARM_TIMESTEP, compile_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substeps", type=int, default=100)
    ap.add_argument("--nworld", type=int, default=2)
    ap.add_argument("--nconmax", type=int, default=1024, help="contacts per world; Warp default (512 total) overflows once the cloth lands")
    ap.add_argument("--njmax", type=int, default=4096, help="constraint rows per world")
    args = ap.parse_args()

    mjm = compile_model(ARM_TIMESTEP)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    print(f"mujoco {mujoco.__version__}, mujoco_warp {mjw.__version__}, warp {wp.__version__}")
    print(f"nv={mjm.nv} nq={mjm.nq} nbody={mjm.nbody} ngeom={mjm.ngeom} neq={mjm.neq} nflex={mjm.nflex} "
          f"nflexvert={mjm.nflexvert} nflexedge={mjm.nflexedge} nu={mjm.nu}")
    print(f"integrator={mujoco.mjtIntegrator(mjm.opt.integrator).name} solver={mujoco.mjtSolver(mjm.opt.solver).name} "
          f"cone={mujoco.mjtCone(mjm.opt.cone).name} jacobian={mujoco.mjtJacobian(mjm.opt.jacobian).name} "
          f"timestep={mjm.opt.timestep}")

    wp.init()
    print(f"warp device: {wp.get_device()}")
    batched = {"eq_data": args.nworld, "body_mass": args.nworld, "geom_friction": args.nworld, "dof_damping": args.nworld}
    t0 = time.perf_counter()
    m = mjw.put_model(mjm, batch_sizes=batched)
    m.opt.warn_overflow = True
    d = mjw.put_data(mjm, mjd, nworld=args.nworld, nconmax=args.nconmax, njmax=args.njmax)
    print(f"put_model + put_data: {time.perf_counter() - t0:.1f}s  (batched: {sorted(batched)})")
    from mujoco_warp._src.io import is_sparse
    print(f"is_sparse={is_sparse(mjm)} naconmax={d.naconmax} njmax={d.njmax}")

    ref = mujoco.MjData(mjm)
    verts0 = d.flexvert_xpos.numpy()[0].copy()
    mjw.step(m, d)                      # warm-up: kernel compile, not timed
    mujoco.mj_step(mjm, ref)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.substeps - 1):
        mjw.step(m, d)
    wp.synchronize()
    warp_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(args.substeps - 1):
        mujoco.mj_step(mjm, ref)
    cpu_time = time.perf_counter() - t0

    verts = d.flexvert_xpos.numpy()
    diff_worlds = np.abs(verts[0] - verts[-1]).max()
    err = np.linalg.norm(verts[0] - ref.flexvert_xpos, axis=1)
    moved = np.linalg.norm(verts[0] - verts0, axis=1)
    print(f"{args.substeps} substeps ({args.substeps * ARM_TIMESTEP * 1e3:.0f} ms of sim): warp {warp_time:.2f}s for {args.nworld} worlds "
          f"({args.nworld * (args.substeps - 1) / warp_time:.0f} world-substeps/s on {wp.get_device()}), cpu {cpu_time:.2f}s "
          f"({(args.substeps - 1) / cpu_time:.0f}/s)")
    print(f"nefc={d.nefc.numpy().tolist()} nacon={int(d.nacon.numpy()[0])} overflow={d.overflow.numpy().tolist()}")
    print(f"cloth moved (warp): mean {moved.mean() * 1e3:.3f} mm, max {moved.max() * 1e3:.3f} mm")
    print(f"warp vs cpu vertex error: mean {err.mean() * 1e6:.2f} um, max {err.max() * 1e6:.2f} um; "
          f"world 0 vs world {args.nworld - 1}: {diff_worlds * 1e6:.2f} um")
    z = verts[0][:, 2]
    print(f"cloth z (warp): min {z.min():.4f} max {z.max():.4f}; cpu min {ref.flexvert_xpos[:, 2].min():.4f}; "
          f"table top + radius = {0.42 + 0.01:.4f}")
    print(f"qacc max: warp {np.abs(d.qacc.numpy()[0]).max():.3g}, cpu {np.abs(ref.qacc).max():.3g}")
    contact_breakdown(mjm, ref, d)



def contact_breakdown(mjm, ref, d):
    """Who touches what after the settle, both sims (appended by the phase 0 probe)."""
    name = lambda g: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{g}"
    print(f"cpu ncon={ref.ncon}")
    pairs = {}
    for i in range(ref.ncon):
        c = ref.contact[i]
        side = lambda k: (name(c.geom[k]) if c.geom[k] >= 0
                          else f"flex{c.flex[k]}" + ("v" if c.vert[k] >= 0 else "") + ("e" if c.elem[k] >= 0 else ""))
        pairs.setdefault((side(0), side(1)), []).append(c.dist)
    for k, v in pairs.items():
        print(f"  cpu  {k}: n={len(v)} dist mean {np.mean(v) * 1e3:.2f} mm min {np.min(v) * 1e3:.2f} mm")
    n = int(d.nacon.numpy()[0])
    wid = d.contact.worldid.numpy()[:n]
    sel = wid == 0
    geom = d.contact.geom.numpy()[:n][sel]
    flex = d.contact.flex.numpy()[:n][sel]
    vert = d.contact.vert.numpy()[:n][sel]
    elem = d.contact.elem.numpy()[:n][sel]
    dist = d.contact.dist.numpy()[:n][sel]
    dim = d.contact.dim.numpy()[:n][sel]
    print(f"warp nacon (world 0)={sel.sum()}  condim counts={dict(zip(*np.unique(dim, return_counts=True)))}")
    keys = {}
    for g, f, v, e, dd in zip(geom, flex, vert, elem, dist):
        side = lambda i: (name(g[i]) if g[i] >= 0 else f"flex{f[i]}" + ("v" if v[i] >= 0 else "") + ("e" if e[i] >= 0 else ""))
        keys.setdefault((side(0), side(1)), []).append(dd)
    for k, v in keys.items():
        print(f"  warp {k}: n={len(v)} dist mean {np.mean(v) * 1e3:.2f} mm min {np.min(v) * 1e3:.2f} mm")
    nv_touch = len(np.unique(vert[vert >= 0]))
    print(f"warp distinct cloth vertices in contact: {nv_touch}; distinct elements: {len(np.unique(elem[elem >= 0]))}")


if __name__ == "__main__":
    main()

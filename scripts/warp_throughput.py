"""Phase 2 check of docs/warp_port.md: substeps per second against the CPU figures.

Sweeps solver (Newton, the env default; CG, what Warp's cloth benchmark uses),
timestep and world count. Each configuration resets once and times control steps.

Resets (with the 2,000-substep settle) then times control steps of 100
substeps each, with zero actions, for each world count. The go/no-go in the
plan is 35,000 world-substeps/s on a WATcloud RTX 3090 shard (5x the 12-worker
Mac: 7,000/s). CPU MuJoCo does 1,190 substeps/s per core on the M4.

    python scripts/warp_throughput.py --nworld 256 1024 4096 --control-steps 20 --solver newton cg
"""
import argparse
from pathlib import Path
import sys
import time

import mujoco_warp as mjw
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mujuco"))
from warp_sim import WarpClothSim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--control-steps", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--solver", nargs="+", default=["newton"], choices=["newton", "cg"])
    ap.add_argument("--timestep", type=float, nargs="+", default=[0.0005])
    ap.add_argument("--nconmax", type=int, default=1024, help="contacts per world (398 measured flat on the table; 768 overflowed at 1,024 worlds)")
    ap.add_argument("--nccdmax", type=int, default=64, help="convex-convex contacts per world (arm geoms only)")
    ap.add_argument("--no-graph", action="store_true", help="step eagerly instead of replaying CUDA graphs (about half the throughput)")
    ap.add_argument("--njmax", type=int, default=2560, help="constraint rows per world (1,924 measured flat); memory scales with nworld x njmax")
    args = ap.parse_args()
    for solver, timestep, n in ((s, t, n) for s in args.solver for t in args.timestep for n in args.nworld):
        print(f"--- solver={solver} timestep={timestep * 1e3:g} ms", flush=True)
        sim = WarpClothSim(nworld=n, device=args.device, solver=solver, timestep=timestep, nconmax=args.nconmax, njmax=args.njmax,
                           nccdmax=args.nccdmax, use_graph=not args.no_graph)
        if sim.device.is_cuda:
            print(f"after put_data: free {sim.device.free_memory / 2**30:.2f} of {sim.device.total_memory / 2**30:.1f} GiB, "
                  f"mempool={wp.is_mempool_enabled(sim.device)}", flush=True)
        t0 = time.perf_counter()
        try:
            sim.reset()
        except RuntimeError as e:
            if sim.device.is_cuda:
                print(f"reset failed: {str(e)[:160]}; free {sim.device.free_memory / 2**30:.2f} GiB, "
                      f"pool current {wp.get_mempool_used_mem_current(sim.device) / 2**30:.2f} GiB, "
                      f"pool high {wp.get_mempool_used_mem_high(sim.device) / 2**30:.2f} GiB", flush=True)
            raise
        wp.synchronize()
        settle = time.perf_counter() - t0
        action = torch.zeros(n, 14, device=sim.tdev)
        sim.step(action)                       # warm-up: graph capture for the 100-substep loop
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.control_steps):
            sim.step(action)
        wp.synchronize()
        dt = time.perf_counter() - t0
        substeps = n * args.control_steps * sim.n_substeps
        mem = (f", warp mem {wp.get_mempool_used_mem_high(sim.device) / 2**30:.1f} GiB, free {sim.device.free_memory / 2**30:.2f} GiB, "
               f"graph={sim.use_graph}") if sim.device.is_cuda else ""
        bits = 0
        for v in sim.overflow.cpu().numpy().tolist():
            bits |= int(v)
        flags = [f.name for f in mjw.OverflowType if f.value and (bits & f.value) == f.value and f.name != "ALL"]
        print(f"nworld={n:5d}  settle {settle:6.1f}s  {substeps / dt:9.0f} world-substeps/s  "
              f"{n * args.control_steps / dt:7.1f} control-steps/s  nefc={int(sim.d.nefc.numpy().max())} "
              f"overflow={'|'.join(flags) or 0}{mem} ({sim.device})", flush=True)
        del sim


if __name__ == "__main__":
    main()

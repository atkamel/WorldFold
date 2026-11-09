"""Drape probe for the cloth stiffness knobs (batch S, user 2026-11-08): the sheet is laid flat with OVERHANG of it past
the table's far edge and left to droop, then how far its free edge hangs below its height on the table. The same test
with a ruler on the real cloth gives the number a sim cloth has to match. Each config also reads back what PhysX was
given: the authored bend / shear / stretch stiffness, the cooked springs, and the particle count and mass.

  isaac/watcloud.sbatch isaac/tools/drape_probe.py --out outputs/isaac/drape 'base|' 'bend500k|bend_stiffness=500000'

One Isaac process per config (Kit runs one scene per process), each appending a row to <out>/drape.jsonl.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mujoco_sim.cloth_params import CLOTH_COUNT as N   # noqa: E402

OVERHANG = 0.08          # m of cloth past the table edge (ruler test: lay it flat 8 cm over the edge); --overhang
DROOP_STEPS = 80         # control steps (4 s) after the reset's settle, before measuring
STILL_MM = 0.5           # the free edge moved less than this over the last 10 steps: settled
CHILD_S = 900            # a child longer than this is killed
ON_TABLE = np.array([ix * N + iy for ix in range(N) for iy in range(6)])   # grid rows well inside the table edge


def row(iy):
    return np.array([ix * N + iy for ix in range(N)])


def drape_metrics(cloth, edge_y):
    """From the 11x11 grid (free edge iy = 10, table edge at y = edge_y): the free edge's droop below the sheet's height
    on the table and its reach past the edge (cm), and each row's (past the edge, below the table height) in cm."""
    c = np.asarray(cloth, float)
    z0 = float(np.median(c[ON_TABLE, 2]))
    prof = {int(iy): [round(100 * float(c[row(iy), 1].mean() - edge_y), 2), round(100 * float(z0 - c[row(iy), 2].mean()), 2)]
            for iy in range(6, N)}
    return {"droop_cm": prof[N - 1][1], "reach_cm": prof[N - 1][0], "profile": prof}


def _springs(k):
    k = np.asarray(k.detach().cpu().numpy() if hasattr(k, "detach") else k, float).ravel()
    vals, counts = np.unique(k, return_counts=True)
    return {"n": int(k.size), "values": {f"{v:g}": int(n) for v, n in zip(vals[:8], counts[:8])}}


def readback(env):
    view = env._cp.cloth._cloth_prim_view
    out = {"particles": int(len(env._particles))}
    reads = {"mass_g": lambda: 1000 * float(view._physics_view.get_masses().sum()),
             "bend": lambda: float(view.get_cloths_bend_stiffnesses()[0]),
             "shear": lambda: float(view.get_cloths_shear_stiffnesses()[0]),
             "stretch": lambda: float(view.get_cloths_stretch_stiffnesses()[0]),
             "springs": lambda: _springs(view.get_stretch_stiffnesses())}
    for key, read in reads.items():
        try:
            out[key] = read()
        except Exception as e:     # these getters have not run on this pipeline yet: a failed read keeps the drape
            out[key] = f"error {type(e).__name__}: {e}"
    return out


def probe(name, knobs, overhang=OVERHANG):
    from isaac.sim.isaac_env import TABLE_SIZE, IsaacClothFoldEnv
    from isaac.tools.grasp_bench import parse_kv
    env = IsaacClothFoldEnv(profile="friction", grasp_knobs=parse_kv(knobs) or None)
    edge_y = TABLE_SIZE / 2                     # copy 0's table is centred on the origin
    env.reset(seed=0)
    dy = edge_y + overhang - float(env.cloth_positions()[row(N - 1), 1].mean())
    env.reset(seed=0, options={"cloth_pose": [0.0, dy]})
    tip = []
    for _ in range(DROOP_STEPS):
        env._advance()
        tip.append(float(env.cloth_positions()[row(N - 1), 2].mean()))
    out = {"name": name, "knobs": knobs, "overhang_cm": 100 * overhang,
           "settled": abs(tip[-1] - tip[-11]) * 1000 < STILL_MM, **drape_metrics(env.cloth_positions(), edge_y)}
    return out | readback(env)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", nargs=2, metavar=("NAME", "KNOBS"))
    ap.add_argument("--overhang", type=float, default=OVERHANG, help="m past the table edge")
    ap.add_argument("configs", nargs="*", help="name|GRASP_KNOBS")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.child:
        result = probe(*args.child, overhang=args.overhang)
        with open(out / "drape.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        print("DRAPE", json.dumps(result), flush=True)
        os._exit(0)                             # Kit can hang on shutdown
    for cfg in args.configs:
        name, knobs = (cfg.split("|") + [""])[:2]
        with open(out / f"{name}.log", "w") as log:
            try:
                rc = subprocess.run([sys.executable, "-u", __file__, "--out", args.out, "--overhang", str(args.overhang),
                                     "--child", name, knobs],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=CHILD_S).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        print("DRAPE done", name, "exit", rc, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

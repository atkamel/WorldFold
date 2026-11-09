"""Cantilever probe for the cloth stiffness knobs (batch S / G, 2026-11-09): the sheet is laid flat with a short flap
past the table's far edge and left to droop, then the flap's chord angle from the edge to its free edge and the Peirce
bending length c = L (cos(t/2) / (8 tan t))^(1/3) (ASTM D1388's cantilever relation). The same test with a ruler on
the real cloth gives c to match. The table is a solid block, so a flap that reaches its side face lies on it and
measures nothing: a reading counts only while the free edge hangs at least CLEAR clear of the face. Run it with
adhesion=0 in the knobs, or the flap sticks to the face. Each config also reads back what PhysX was given: the
authored bend / shear / stretch stiffness, the cooked springs, and the particle count and mass.

  isaac/watcloud.sbatch isaac/tools/drape_probe.py --overhang 0.01,0.02,0.04 --out outputs/isaac/drape \\
      'base|adhesion=0' 'bend500k|adhesion=0,bend_stiffness=500000'

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

OVERHANGS = (0.01, 0.02, 0.03, 0.04, 0.06)   # m of flap past the table edge; --overhang
CLEAR = 0.006            # m the free edge must hang clear of the table's side face (rest offset 4 mm + margin)
DROOP_STEPS = 80         # control steps (4 s) after the reset's settle, before measuring
STILL_MM = 0.5           # the free edge moved less than this over the last 10 steps: settled
CHILD_S = 900            # a child longer than this is killed


def flap_metrics(particles, n, edge_y, overhang):
    """The full-resolution n x n sheet (index ix * n + iy, free edge iy = n - 1) with `overhang` m past the table edge at
    y = edge_y: the free edge's reach past the edge and drop below the sheet's height on the table (cm), the chord angle
    (deg), whether it hangs clear of the face, and the Peirce bending length (cm) when it does."""
    p = np.asarray(particles, float).reshape(n, n, 3)
    z0 = float(np.median(p[:, : n // 2, 2]))
    reach, drop = float(p[:, -1, 1].mean() - edge_y), float(z0 - p[:, -1, 2].mean())
    theta = float(np.arctan2(drop, reach))
    clear = reach >= CLEAR and 0.0 < theta < np.pi / 2
    c = overhang * (np.cos(theta / 2) / (8 * np.tan(theta))) ** (1 / 3) if clear else None
    return {"overhang_cm": round(100 * overhang, 2), "reach_cm": round(100 * reach, 2), "drop_cm": round(100 * drop, 2),
            "angle_deg": round(float(np.degrees(theta)), 1), "clear": bool(clear),
            "bending_length_cm": None if c is None else round(100 * float(c), 2)}


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


def probe(name, knobs, overhangs=OVERHANGS):
    from isaac.sim.isaac_env import CLOTH_SUBDIV, TABLE_SIZE, IsaacClothFoldEnv
    from isaac.tools.grasp_bench import parse_kv
    env = IsaacClothFoldEnv(profile="friction", grasp_knobs=parse_kv(knobs) or None)
    n = (N - 1) * int(env.grasp_knobs.get("cloth_subdiv") or CLOTH_SUBDIV) + 1
    free = np.arange(n) * n + (n - 1)
    edge_y = TABLE_SIZE / 2                     # copy 0's table is centred on the origin
    env.reset(seed=0)
    y_free = float(env._particle_positions()[free, 1].mean())
    flaps = []
    for overhang in overhangs:
        env.reset(seed=0, options={"cloth_pose": [0.0, edge_y + overhang - y_free]})
        tip = []
        for _ in range(DROOP_STEPS):
            env._advance()
            tip.append(float(env._particle_positions()[free, 2].mean()))
        flaps.append(flap_metrics(env._particle_positions(), n, edge_y, overhang)
                     | {"settled": abs(tip[-1] - tip[-11]) * 1000 < STILL_MM})
    return {"name": name, "knobs": knobs, "flaps": flaps} | readback(env)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", nargs=2, metavar=("NAME", "KNOBS"))
    ap.add_argument("--overhang", default=",".join(map(str, OVERHANGS)), help="m past the table edge, comma-separated")
    ap.add_argument("configs", nargs="*", help="name|GRASP_KNOBS")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.child:
        result = probe(*args.child, overhangs=[float(v) for v in args.overhang.split(",")])
        with open(out / "drape.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        print("DRAPE", json.dumps(result), flush=True)
        os._exit(0)                             # Kit can hang on shutdown
    for cfg in args.configs:
        name, knobs = (cfg.split("|") + [""])[:2]
        with open(out / f"{name}.log", "w") as log:
            try:
                rc = subprocess.run([sys.executable, "-u", __file__, "--out", args.out, "--overhang", args.overhang,
                                     "--child", name, knobs],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=CHILD_S).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        print("DRAPE done", name, "exit", rc, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

"""QD (2026-10-16): where does a stage-1 grip slip? Reads recovery_replay --trace-dir files and prints, per arm and per
stage-1 grasp episode, the phase at which `g` dropped, the site height vs the carried point at the close, and the
point's travel while held.   .venv/Scripts/python.exe isaac/slip_summary.py outputs/.../traces_dir [--stage 1]
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--stage", type=int, default=1)
    a = ap.parse_args()
    for f in sorted(glob.glob(os.path.join(a.trace_dir, "*.json"))):
        d = json.load(open(f))
        print(f"== seed {d['seed']}")
        for pr, tr in d["arms"].items():
            st, g, ph = np.array(tr["stage"]), np.array(tr["g"]), tr["phase"]
            site, corner = np.array(tr["site"]), np.array(tr["corner"])
            idx = np.where(st == a.stage)[0]
            if not len(idx):
                print(f"  {pr}: never in stage {a.stage}"); continue
            t0 = idx[0]
            held = np.where((st == a.stage) & (g == 1))[0]
            if not len(held):
                print(f"  {pr}: stage from t{t0}, never held (phases {sorted(set(ph[t0:]))})"); continue
            # grip episodes: runs of g == 1
            runs, s = [], held[0]
            for i, j in zip(held[:-1], held[1:]):
                if j != i + 1:
                    runs.append((s, i)); s = j
            runs.append((s, held[-1]))
            for s, e in runs:
                dz = site[s, 2] - corner[s, 2]
                travel = np.linalg.norm(corner[e, :2] - corner[s, :2])
                after = ph[e + 1] if e + 1 < len(ph) else "end"
                print(f"  {pr}: held t{s}-t{e} ({e - s + 1} steps) close in '{ph[s]}', site-point dz {100 * dz:.1f} cm, "
                      f"point z {100 * (corner[s, 2] - 0.42):.1f} cm, lost in '{ph[e]}'->'{after}', point moved "
                      f"{100 * travel:.1f} cm, dist at loss {100 * tr['dist'][e]:.1f} cm")


if __name__ == "__main__":
    main()

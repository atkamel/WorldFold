"""QD (2026-10-16, user: the first fold's release is violent and jerky): how much does each release move the cloth it
lets go of? Reads recovery_replay --trace-dir files. Per arm and release (the first step of 'release' in a stage), it
prints how far the carried point moves (xy) from the release start to the end of the retreat, its peak per-step speed,
and the gripper's peak per-step speed in the retreat. Also the stage-1 grip runs (as isaac/slip_summary.py).
    .venv/Scripts/python.exe isaac/release_summary.py outputs/.../traces_dir [more dirs]
"""
import glob
import json
import os
import sys

import numpy as np

DT = 0.05     # control step (s)


def releases(tr):
    ph, st = tr["phase"], np.array(tr["stage"])
    out = []
    for t in range(1, len(ph)):
        if ph[t] == "release" and ph[t - 1] != "release":
            e = t
            while e + 1 < len(ph) and ph[e + 1] in ("release", "retreat") and st[e + 1] == st[t]:
                e += 1
            out.append((int(st[t]), t, e))
    return out


def summarise(trace_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(trace_dir, "*.json"))):
        d = json.load(open(f))
        for pr, tr in d["arms"].items():
            c, site = np.array(tr["corner"]), np.array(tr["site"])
            for stage, s, e in releases(tr):
                move = float(np.linalg.norm(c[e, :2] - c[s, :2]))
                v = np.linalg.norm(np.diff(c[s:e + 1, :2], axis=0), axis=1) / DT if e > s else np.zeros(1)
                vg = np.linalg.norm(np.diff(site[s:e + 1], axis=0), axis=1) / DT if e > s else np.zeros(1)
                rows.append((d["seed"], pr, stage, s, e - s + 1, move, float(v.max()), float(vg.max())))
    print(f"== {trace_dir}")
    print("seed    arm     stage  t     steps  point moved (cm)  peak point speed (cm/s)  peak gripper speed (cm/s)")
    for r in rows:
        print(f"{r[0]}  {r[1]:6s}  {r[2]}      {r[3]:4d}  {r[4]:4d}   {100 * r[5]:8.1f}          {100 * r[6]:8.1f}"
              f"                 {100 * r[7]:8.1f}")
    for stage in sorted({r[2] for r in rows}):
        sub = [r for r in rows if r[2] == stage]
        print(f"stage {stage} mean: moved {100 * np.mean([r[5] for r in sub]):.1f} cm, peak point speed "
              f"{100 * np.mean([r[6] for r in sub]):.1f} cm/s, peak gripper speed {100 * np.mean([r[7] for r in sub]):.1f}"
              f" cm/s, release+retreat {np.mean([r[4] for r in sub]):.0f} steps (n = {len(sub)})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarise(p)

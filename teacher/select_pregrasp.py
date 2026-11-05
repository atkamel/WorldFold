"""Pick the pre-grasp snapshots for the grasp-on-demand test from a harvest run (run_teacher_grasp.sh with
TEACHER_SNAPSHOT_EVERY): for every logged grasp attempt (grasps/*.jsonl, not a parked close), the snapshot of the same
episode taken BEFORE the jaw started closing, as close as possible to `before` steps before the close.

    python select_pregrasp.py <harvest run dir> <out dir> [--before 10] [--window 8 20] [--per_garment 3]

The logger's t_close is the step its pinched set was taken, SETTLE_STEPS (6) after the jaw began closing, so the
window [t_close - 20, t_close - 8] keeps the jaw open. Output: <out>/<garment>/*.npz (copied), and a manifest.
"""
import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--before", type=int, default=10)
ap.add_argument("--window", type=int, nargs=2, default=[8, 20])
ap.add_argument("--per_garment", type=int, default=3)
args = ap.parse_args()

snaps = defaultdict(dict)                      # (garment, tag) -> {t: path}
for f in glob.glob(os.path.join(args.run, "snapshots", "*", "*.npz")):
    m = re.match(r"(.+)_(\d+)_t(\d+)\.npz$", os.path.basename(f))
    if m:
        snaps[(m.group(1), m.group(2))][int(m.group(3))] = f
chosen, manifest = defaultdict(list), []
for jl in sorted(glob.glob(os.path.join(args.run, "grasps", "*.jsonl"))):
    m = re.match(r"(.+)_(\d+)\.jsonl$", os.path.basename(jl))
    if not m:
        continue
    garment, tag = m.group(1), m.group(2)
    recs = [json.loads(l) for l in open(jl) if l.strip()]
    for r in recs:
        if r["outcome"] == "empty" and r.get("end") == "episode_end":
            continue                            # parked close
        tc = r["t_close"]
        cands = [(abs(tc - t - args.before), t, pth) for t, pth in snaps.get((garment, tag), {}).items()
                 if args.window[0] <= tc - t <= args.window[1]]
        if not cands:
            continue
        _, t, pth = min(cands)
        if any(c["t"] == t for c in chosen[garment]):
            continue
        chosen[garment].append(dict(t=t, path=pth, t_close=tc, side=r["side"], outcome=r["outcome"],
                                    max_lift_cm=r.get("max_lift_cm"), layers=r.get("layers")))
total = 0
for garment, lst in sorted(chosen.items()):
    lst.sort(key=lambda c: c["t"])
    keep = lst[:args.per_garment]
    os.makedirs(os.path.join(args.out, garment), exist_ok=True)
    for c in keep:
        dst = os.path.join(args.out, garment, os.path.basename(c["path"]))
        shutil.copy(c["path"], dst)
        manifest.append(dict(garment=garment, **{k: v for k, v in c.items() if k != "path"}, file=dst))
        total += 1
    print(f"{garment}: {len(keep)} pre-grasp states (of {len(lst)} attempts with a snapshot in the window)")
json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
print(f"SELECTED {total} pre-grasp states for {len(chosen)} garments -> {args.out}")

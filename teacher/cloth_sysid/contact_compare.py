"""Newton vs ZOZO on the same contact scenes (drag on table; two stacked layers). Both grids are row-major, x fastest,
frames at 100 Hz from t = 0 (the pin starts moving at 0.5 s). Compared as displacement fields (x(t) - x(0)) in mm.

  python contact_compare.py <tag>      # tag of the newton_contact drag/layers run, e.g. _noself_fs
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "contact")


def field_err(xn, xz):
    """Per-frame mean and max vertex distance (mm) between two displacement fields, over the common frames."""
    T = min(len(xn), len(xz))
    dn, dz = xn[:T] - xn[:1], xz[:T] - xz[:1]
    d = np.linalg.norm(dn - dz, axis=2) * 1000
    return d.mean(1), d.max(1), dn, dz


def summary(tag, xn, xz):
    mean, mx, dn, dz = field_err(xn, xz)
    return {"scene": tag, "mean_mm_final": round(float(mean[-1]), 2), "mean_mm_over_time": round(float(mean.mean()), 2),
            "max_mm_final": round(float(mx[-1]), 2),
            "newton_centroid_dx_mm": round(float(dn[-1, :, 0].mean() * 1000), 2),
            "zozo_centroid_dx_mm": round(float(dz[-1, :, 0].mean() * 1000), 2),
            "newton_far_corner_mm": [round(float(v) * 1000, 1) for v in dn[-1, -1, :2]],
            "zozo_far_corner_mm": [round(float(v) * 1000, 1) for v in dz[-1, -1, :2]]}


def main(tag):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, f"drag{tag}_*.npz"))):
        n = np.load(f)
        x, spans = n["x"], n["spans"]
        setting = os.path.basename(f)[len(f"drag{tag}_"):-4]
        for (p0, p1), mu in zip(spans, (0.3, 0.5, 0.7)):
            z = np.load(os.path.join(RES, "zozo", f"scene_D_mu{mu}.npz"))["x"]
            out.append(dict(summary(f"drag mu_table {mu}", x[:, p0:p1], z), setting=setting))
    for f in sorted(glob.glob(os.path.join(RES, f"layers{tag}_*.npz"))):
        n = np.load(f)
        x, spans = n["x"], n["spans"]
        setting = os.path.basename(f)[len(f"layers{tag}_"):-4]
        z = np.load(os.path.join(RES, "zozo", "scene_F.npz"))
        zx = z["x"]
        nb = spans[0][1] - spans[0][0]
        for (p0, p1), name, zs in zip(spans, ("bottom", "top"), (slice(0, nb), slice(nb, 2 * nb))):
            out.append(dict(summary(f"layers {name}", x[:, p0:p1], zx[:, zs]), setting=setting))
    for r in out:
        print(r)
    with open(os.path.join(RES, f"compare{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")

"""Top and side views of the shirt settling (cloth_<label>.npz from oracle_fold.cloth_test), one row per variant.

    python settle_views.py out.png cloth_a.npz cloth_b.npz ...
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, files = sys.argv[1], sys.argv[2:]
TIMES = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)          # s (frames every 0.5 s)
fig, ax = plt.subplots(2 * len(files), len(TIMES), figsize=(3 * len(TIMES), 5.2 * len(files)), squeeze=False)
for r, f in enumerate(files):
    d = np.load(f)
    fr = d["frames"].astype(float) * float(d["scale"]) + d["origin"]
    z0 = np.percentile(fr[-1][:, 2], 1)
    lab = os.path.basename(f)[6:-4]
    for c, t in enumerate(TIMES):
        k = min(int(round(t / 0.5)), len(fr) - 1)
        v = fr[k]
        a = ax[2 * r, c]; a.scatter(v[:, 0], v[:, 1], s=0.2, c=v[:, 2] - z0, cmap="viridis", vmin=0, vmax=0.05)
        a.set_aspect("equal"); a.set_xlim(-0.45, 0.45); a.set_ylim(-0.45, 0.45)
        sp = np.linalg.norm(fr[k] - fr[max(k - 2, 0)], axis=1).mean() * 1000 / max(1e-9, (k - max(k - 2, 0)) * 0.5)
        a.set_title(f"{lab} t={k * 0.5:.1f}s top\nmean speed {sp:.0f} mm/s", fontsize=8)
        b = ax[2 * r + 1, c]; b.scatter(v[:, 0], v[:, 2] - z0, s=0.2)
        b.set_xlim(-0.45, 0.45); b.set_ylim(-0.03, 0.15); b.set_title("side (x vs height)", fontsize=8)
plt.tight_layout(); plt.savefig(out, dpi=80)
print("saved", out)

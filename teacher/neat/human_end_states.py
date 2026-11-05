"""Question 1, step 1: the end state of every shirt Roy folded in the MuJoCo game (sticky cloth, so only the END state
counts), scored with his crumple-proof metric (mujoco_live/neat_metric.py) and drawn on one contact sheet for him to
mark which ones are neat.

    python human_end_states.py <mujoco_live dir> <out dir>

Per shirt (episode) of every session log in <mujoco_live>/demos: the last saved cloth shape (cloth states are logged
1x per second and at each drop). Duplicate shirts (identical end shapes, e.g. replayed or automated sessions) are kept
once. Writes end_states.json, end_states.npz (vertices) and sheet_*.png (top views, coloured by height).
"""
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

ML = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
sys.path.insert(0, ML)
from shirt import shirt_mesh  # noqa: E402
from neat_metric import fold_score  # noqa: E402

os.makedirs(OUT, exist_ok=True)
pts, tris = shirt_mesh(0.01)
H = 0.004
flat = np.c_[pts, np.full(len(pts), H / 2)]
FLAT_AREA = fold_score(flat, tris, 1.0, H)["area_ratio"]

rows, shapes, seen = [], [], set()
for f in sorted(glob.glob(os.path.join(ML, "demos", "*.jsonl"))):
    name = os.path.basename(f)[:-6]
    eps = defaultdict(list)
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "cloth_cm" in r:
            eps[r.get("episode", 0)].append(r)
    for e, rs in eps.items():
        c = np.asarray(rs[-1]["cloth_cm"], np.float64) / 100
        if c.shape != pts.shape[:1] + (3,):
            continue
        h = hashlib.md5(np.round(c, 4).tobytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        moved = float(np.linalg.norm(c[:, :2] - pts, axis=1).mean())
        if moved < 0.02:                       # never really touched: skip
            continue
        s = fold_score(c, tris, FLAT_AREA, H)
        rows.append(dict(id=len(rows), session=name, shirt=int(e) if e is not None else 0, seconds=rs[-1].get("t"),
                         n_states=len(rs), kind=("claude" if "claude" in name else "remote" if "remote" in name
                                                  else "server" if "server" in name else "human"), **s))
        shapes.append(c.astype(np.float32))
json.dump(rows, open(os.path.join(OUT, "end_states.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, "end_states.npz"), *shapes)
print(f"{len(rows)} distinct end states")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
order = sorted(range(len(rows)), key=lambda i: -rows[i]["score"])
PER = 30
for p in range(0, len(order), PER):
    ids = order[p:p + PER]
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, ids):
        c, r = shapes[i], rows[i]
        ax.tripcolor(c[:, 0], c[:, 1], tris, c[:, 2] * 1000, cmap="viridis", vmin=0, vmax=40, shading="gouraud")
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.25, 0.35); ax.set_aspect("equal")
        ax.set_title(f"#{r['id']} {r['kind']} {r['session'][4:13]} s{r['shirt']}\nscore {r['score']:.0f}  area "
                     f"{100 * r['area_ratio']:.0f}%  rect {r['rect']:.2f}  flat {r['flat']:.2f}", fontsize=8)
    fig.suptitle("End states of Roy's MuJoCo folds, best score first (colour = height, 0-40 mm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"sheet_{p // PER + 1}.png"), dpi=70)
    plt.close(fig)
print("sheets:", (len(order) + PER - 1) // PER)

"""Question 1, step 2: the end state of each of the teacher's 20 harvest folds (2026-10-19, its own physics), scored with
Roy's crumple-proof metric (mujoco_live/neat_metric.py) exactly as his MuJoCo folds were (human_end_states.py).

    python teacher_end_states.py <harvest run dir> <garments_flat dir> <mujoco_live dir> <out dir>

Snapshots (grasp_test_patch.py, every 5 policy steps) hold the cloth's local points and the garment prim's world pose;
faces and the local->world scale come from the flattened-model files, which share each garment's mesh. Start = first
snapshot (the flat reference for the area ratio), end = last snapshot (the episode ends on LeHome's success or at the
step limit).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

RUN, FLAT, ML, OUT = (os.path.abspath(a) for a in sys.argv[1:5])
sys.path.insert(0, ML)
from neat_metric import fold_score  # noqa: E402

H = 0.005          # cloth layer spacing in LeHome (flatten check: median layer gap 5.0 mm)
os.makedirs(OUT, exist_ok=True)


def quat_rot(q):
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world(d, scale):
    return (np.asarray(d["garment_points"], np.float64) * scale) @ quat_rot(d["garment_ori"]).T + d["garment_pos"]


rows, shapes, faces_of = [], [], {}
for gdir in sorted(glob.glob(os.path.join(RUN, "snapshots", "*"))):
    g = os.path.basename(gdir)
    fl = np.load(os.path.join(FLAT, f"flat_{g}.npz"))
    L, W = fl["local_points"], fl["world_points"]
    scale = np.linalg.norm(W - W.mean(0), axis=1).mean() / np.linalg.norm(L - L.mean(0), axis=1).mean()
    tris = fl["faces"].astype(np.int64)
    eps = defaultdict(list)
    for f in glob.glob(os.path.join(gdir, "*.npz")):
        eps[os.path.basename(f).split("_t")[0]].append(f)
    for tag, fs in eps.items():
        fs.sort(key=lambda f: int(f.rsplit("_t", 1)[1][:4]))
        d0, d1 = np.load(fs[0]), np.load(fs[-1])
        x0, x1 = world(d0, scale), world(d1, scale)
        table = float(np.percentile(x0[:, 2], 1)) - H / 2
        x0[:, 2] -= table
        x1[:, 2] -= table
        flat_area = fold_score(x0, tris, 1.0, H)["area_ratio"]
        s = fold_score(x1, tris, flat_area, H)
        s0 = fold_score(x0, tris, flat_area, H)
        rows.append(dict(id=len(rows), garment=g, steps=int(d1["t"]), start_spread_mm=round(float(np.ptp(x0[:, 2]) * 1000), 1),
                         start_score=s0["score"], **s))
        shapes.append(x1.astype(np.float32))
        faces_of[len(rows) - 1] = tris
json.dump(rows, open(os.path.join(OUT, "end_states.json"), "w"), indent=1)
print(f"{len(rows)} teacher folds")
for r in sorted(rows, key=lambda r: -r["score"]):
    print(f"  {r['garment']:18s} steps {r['steps']:3d}  score {r['score']:5.1f}  area {100 * r['area_ratio']:3.0f}%  "
          f"rect {r['rect']:.2f}  flat {r['flat']:.2f}  (start spread {r['start_spread_mm']} mm)")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
order = sorted(range(len(rows)), key=lambda i: -rows[i]["score"])
fig, axes = plt.subplots(4, 5, figsize=(16, 13))
for ax in axes.flat:
    ax.axis("off")
for ax, i in zip(axes.flat, order):
    c, r = shapes[i], rows[i]
    ctr = c[:, :2].mean(0)
    ax.tripcolor(c[:, 0] - ctr[0], c[:, 1] - ctr[1], faces_of[i], c[:, 2] * 1000, cmap="viridis", vmin=0, vmax=40,
                 shading="gouraud")
    ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.3, 0.3); ax.set_aspect("equal")
    ax.set_title(f"{r['garment']}\nscore {r['score']:.0f}  area {100 * r['area_ratio']:.0f}%  rect {r['rect']:.2f}  "
                 f"flat {r['flat']:.2f}", fontsize=8)
fig.suptitle("End states of the teacher's 20 folds (own physics), best score first (colour = height, 0-40 mm)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sheet_teacher.png"), dpi=70)

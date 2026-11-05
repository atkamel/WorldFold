"""Watchable fold videos from saved cloth shapes, for Roy to judge neatness by eye.

    python fold_videos.py teacher <harvest run dir> <garments_flat dir> <out.mp4>
    python fold_videos.py human <mujoco_live dir> <out.mp4> [max shirts]

A grid of folds playing together (all start at once, each holds its final shape at the end), each tile a shaded 3D
view of the cloth from the front-above, so wrinkles and lumps read the way they look on a table. Tiles are numbered
and labelled with the garment; the same numbers are in the printed list.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE = 300
LIGHT = np.array([0.3, -0.5, 0.8]); LIGHT /= np.linalg.norm(LIGHT)
COLORS = [(214, 96, 77), (67, 147, 195), (120, 170, 90), (200, 160, 60), (150, 110, 190)]


def view(x):
    """Oblique camera: 50 deg down from the front (the robots' side, -y)."""
    a = np.radians(50)
    R = np.array([[1, 0, 0], [0, np.sin(a), np.cos(a)], [0, -np.cos(a), np.sin(a)]])  # rows: screen x, screen up, depth
    return x @ R.T


def render(x, tris, color, scale, center):
    img = Image.new("RGB", (TILE, TILE), (245, 243, 238))
    d = ImageDraw.Draw(img)
    v = view(x - center)
    tri = v[tris]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n = n / nn
    wn = np.cross(x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]])
    wn = wn / (np.linalg.norm(wn, axis=1, keepdims=True) + 1e-12)
    shade = 0.35 + 0.65 * np.abs(wn @ LIGHT)                 # two-sided Lambert
    order = np.argsort(tri[:, :, 2].mean(1))[::-1]           # far to near (depth = 3rd row)
    px = tri[:, :, :2] * scale
    px[:, :, 0] += TILE / 2
    px[:, :, 1] = TILE / 2 - px[:, :, 1]
    col = np.asarray(color, float)
    for i in order:
        c = tuple(int(v_) for v_ in col * shade[i])
        d.polygon([tuple(p) for p in px[i]], fill=c)
    return img


def label(img, text):
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        f = ImageFont.load_default()
    d.rectangle([0, 0, TILE, 22], fill=(30, 30, 30))
    d.text((5, 3), text, fill=(255, 255, 255), font=f)
    return img


def grid_video(folds, out, cols=5, fps=8):
    """folds: list of dicts {name, frames: [x (N,3)], tris}. Writes an mp4 (imageio)."""
    import cv2
    n = len(folds)
    rows = (n + cols - 1) // cols
    T = max(len(f["frames"]) for f in folds) + 3 * fps        # hold the end 3 s
    w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cols * TILE, rows * TILE))
    for t in range(T):
        canvas = Image.new("RGB", (cols * TILE, rows * TILE), (255, 255, 255))
        for k, f in enumerate(folds):
            x = f["frames"][min(t, len(f["frames"]) - 1)]
            img = render(x, f["tris"], COLORS[k % len(COLORS)], f["scale"], f["center"])
            done = t >= len(f["frames"]) - 1
            label(img, f"{k + 1}. {f['name']}" + ("   (final)" if done else ""))
            canvas.paste(img, ((k % cols) * TILE, (k // cols) * TILE))
        w.write(np.asarray(canvas)[..., ::-1].copy())
        if t % 10 == 0:
            print(f"  frame {t}/{T}", flush=True)
    w.release()


def quat_rot(q):
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world(d, scale):
    """A snapshot's cloth in the world (same as teacher_end_states.world)."""
    return (np.asarray(d["garment_points"], np.float64) * scale) @ quat_rot(d["garment_ori"]).T + d["garment_pos"]


def teacher_folds(run, flat_dir):
    folds = []
    for gdir in sorted(glob.glob(os.path.join(run, "snapshots", "*"))):
        g = os.path.basename(gdir)
        fl = np.load(os.path.join(flat_dir, f"flat_{g}.npz"))
        L, W = fl["local_points"], fl["world_points"]
        scale = np.linalg.norm(W - W.mean(0), axis=1).mean() / np.linalg.norm(L - L.mean(0), axis=1).mean()
        fs = sorted(glob.glob(os.path.join(gdir, "*.npz")), key=lambda f: int(f.rsplit("_t", 1)[1][:4]))
        frames = [world(np.load(f), scale) for f in fs]
        z0 = np.percentile(frames[0][:, 2], 1)
        frames = [x - [0, 0, z0] for x in frames]
        c = frames[0].mean(0)
        span = np.ptp(frames[0][:, :2], axis=0).max()
        folds.append(dict(name=g.replace("_Seen_", " #"), frames=frames, tris=fl["faces"].astype(np.int64),
                          center=c, scale=0.8 * TILE / span))
    return folds


def human_folds(ml, max_shirts):
    sys.path.insert(0, ml)
    from shirt import shirt_mesh
    pts, tris = shirt_mesh(0.01)
    rows = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "neat", "human", "end_states.json")))
    rows = [r for r in rows if r["kind"] == "human" and r["area_ratio"] < 0.99]
    rows.sort(key=lambda r: -r["score"])
    folds = []
    for r in rows[:max_shirts]:
        frames = []
        for line in open(os.path.join(ml, "demos", r["session"] + ".jsonl")):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "cloth_cm" in d and d.get("episode", 0) == r["shirt"]:
                frames.append(np.asarray(d["cloth_cm"], float) / 100)
        if len(frames) > 60:                                    # long sessions: keep 60 evenly spaced shapes
            frames = [frames[i] for i in np.linspace(0, len(frames) - 1, 60).astype(int)]
        x0 = np.c_[pts, np.zeros(len(pts))]
        frames = [x0] + frames
        span = np.ptp(pts, axis=0).max()
        folds.append(dict(name=f"Roy {r['session'][4:8]}-{r['session'][9:13]} shirt {r['shirt']}", frames=frames,
                          tris=np.asarray(tris, np.int64), center=x0.mean(0), scale=0.8 * TILE / span))
    return folds


if __name__ == "__main__":
    if sys.argv[1] == "teacher":
        folds = teacher_folds(sys.argv[2], sys.argv[3])
        out = sys.argv[4]
    else:
        folds = human_folds(sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 20)
        out = sys.argv[3]
    for k, f in enumerate(folds):
        print(k + 1, f["name"], len(f["frames"]), "shapes")
    grid_video(folds, out)
    print("wrote", out)

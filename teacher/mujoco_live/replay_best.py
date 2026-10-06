"""Storyboard + replay of a recorded fold, drawn from the logged cloth states (exact, nothing re-simulated).
Usage: python replay_best.py demos/<session>.jsonl [shirt_number]  (default: the shirt with the best area)
Writes best_fold_storyboard.png and best_fold_replay.mp4 next to this script."""
import os, sys, json
from collections import defaultdict
import numpy as np
import mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
from world import World, ARMS

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(sys.argv[1])]
eps = defaultdict(list)
for r in recs:
    eps[r.get("episode", 0)].append(r)
if len(sys.argv) > 2:
    ep = int(sys.argv[2])
else:
    ep = min((min(r["score"]["area_cm2"] for r in rs if r.get("event") == "drop"), e)
             for e, rs in eps.items() if any(r.get("event") == "drop" for r in rs))[1]
rs = eps[ep]
events = [r for r in rs if r.get("event") in ("grab", "drop")]

# pair each successful grab with that arm's next drop; group pairs that start within 1.5 s = two-hand moves
pairs = []
for i, r in enumerate(events):
    if r["event"] == "grab" and r.get("n", 0) > 0:
        d = next((e for e in events[i + 1:] if e["event"] == "drop" and e["arm"] == r["arm"]), None)
        if d:
            pairs.append((r, d))
moves = []
for p in pairs:
    if moves and p[0]["tick"] - moves[-1][0][0]["tick"] < 90 and p[0]["arm"] not in [q[0]["arm"] for q in moves[-1]]:
        moves[-1].append(p)
    else:
        moves.append([p])

w = World()
FONT = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 15)
BOLD = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 17)
RGB = {"L": (1.0, 0.45, 0.1), "R": (0.2, 0.75, 0.3)}
t0 = events[0]["tick"]


def set_cloth(cloth_cm):
    w.cloth.x[:] = np.array(cloth_cm) / 100
    w.sync_shirt()
    w.refresh()


def arrows(scn, segs):
    eye = np.eye(3).ravel()
    for arm, a, b in segs:
        a, b = np.array(a) / 100, np.array(b) / 100
        a[2] = b[2] = 0.035
        col = np.array(list(RGB[arm]) + [0.95], np.float32)
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye, col)
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, a, b)
        scn.ngeom += 1
        for p, r in ((a, 0.008), (b, 0.012)):
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), p, eye, col)
            scn.ngeom += 1


# ---------------- storyboard (top-down, robots hidden, arrows = where each hand grabbed -> dropped)
PW, PH = 400, 300
r = mujoco.Renderer(w.m, PH, PW)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0.19, 0]; cam.distance, cam.elevation, cam.azimuth = 0.62, -89.5, 90
opt = mujoco.MjvOption(); opt.geomgroup[2] = 0
panels = []


def panel(cloth_cm, segs, title, sub):
    set_cloth(cloth_cm)
    r.update_scene(w.d, cam, opt)
    arrows(r.scene, segs)
    img = Image.fromarray(r.render())
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, 0, PW, 46), fill=(0, 0, 0, 160))
    d.text((8, 4), title, font=BOLD, fill=(255, 255, 255))
    d.text((8, 25), sub, font=FONT, fill=(230, 230, 230))
    panels.append(img)


panel(events[0]["cloth_cm"], [], "start: flat shirt", "100% of flat")
for k, mv in enumerate(moves, 1):
    segs, parts = [], []
    for g, d in mv:
        segs.append((g["arm"], g["tip_cm"], d["tip_cm"]))
        parts.append(f"{g['arm']} ({g['tip_cm'][0]:.0f},{g['tip_cm'][1]:.0f})->({d['tip_cm'][0]:.0f},{d['tip_cm'][1]:.0f})")
    last = max((d for g, d in mv), key=lambda e: e["tick"])
    s = last["score"]
    t = (mv[0][0]["tick"] - t0) / 60
    hands = "both hands" if len(mv) == 2 else ("left" if mv[0][0]["arm"] == "L" else "right") + " hand"
    panel(last["cloth_cm"], segs, f"{k}. {hands}  ->  {100 * s['area_vs_flat']:.0f}% of flat",
          f"t={t:.0f}s  " + "  ".join(parts))
cols = 4
rows = (len(panels) + cols - 1) // cols
sheet = Image.new("RGB", (cols * PW, rows * PH), (20, 20, 20))
for i, p in enumerate(panels):
    sheet.paste(p, ((i % cols) * PW, (i // cols) * PH))
out_png = os.path.join(HERE, "best_fold_storyboard.png")
sheet.save(out_png)
print(f"shirt {ep}: {len(moves)} moves ({len(pairs)} grabs that caught cloth) -> {out_png}")
for k, mv in enumerate(moves, 1):
    last = max((d for g, d in mv), key=lambda e: e["tick"])
    print(f"  {k:2d}. t={(mv[0][0]['tick'] - t0) / 60:6.1f}s  " + "  ".join(
        f"{g['arm']} {g['mode']} n={g['n']} ({g['tip_cm'][0]:.1f},{g['tip_cm'][1]:.1f}) -> ({d['tip_cm'][0]:.1f},{d['tip_cm'][1]:.1f})" for g, d in mv)
        + f"   => {100 * last['score']['area_vs_flat']:.0f}%  box {last['score']['rect_cm']}  rect {last['score']['rectangularity']}")

# ---------------- replay video (first person, arms posed from the log; cloth from recorded snapshots)
snaps = sorted([(e["tick"], e["cloth_cm"]) for e in rs if "cloth_cm" in e], key=lambda s: s[0])
frames = sorted([f for f in rs if f.get("frame")], key=lambda f: f["tick"])
ev_ticks = [e["tick"] for e in events]
busy = lambda t: any(abs(t - e) < 180 for e in ev_ticks)   # skip long idle stretches
W, H = 960, 560
rv = mujoco.Renderer(w.m, H, W)
cv = mujoco.MjvCamera(); cv.lookat[:] = [0, 0.20, 0]; cv.distance, cv.elevation, cv.azimuth = 0.46, -49, 90
writer = imageio.get_writer(os.path.join(HERE, "best_fold_replay.mp4"), fps=10, quality=7, macro_block_size=8)
fi, si = 0, 0
nvid = 0
for f in frames:
    if f["tick"] % 6 or not busy(f["tick"]):  # 10 frames per second of play, only around the action
        continue
    while si + 1 < len(snaps) and snaps[si + 1][0] <= f["tick"]:
        si += 1
    set_cloth(snaps[si][1])
    for a in ARMS:
        q = f["q"][a]
        for k in range(4):
            w.d.qpos[w.qadr[a][k]] = q[k]
        w.d.qpos[w.qadr[a][5]] = 0.0 if f["grip"][a] else 1.2
    w.refresh()
    w.m.vis.global_.fovy = 65
    rv.update_scene(w.d, cv)
    img = Image.fromarray(rv.render())
    d = ImageDraw.Draw(img, "RGBA")
    area = next((e["score"]["area_vs_flat"] for e in reversed(events) if e["tick"] <= f["tick"] and e["event"] == "drop"), 1.0)
    d.rectangle((6, 6, 560, 58), fill=(0, 0, 0, 150))
    d.text((14, 10), f"Roy's best fold (shirt {ep})   t={(f['tick'] - t0) / 60:5.1f}s   area {100 * area:.0f}% of flat", font=BOLD, fill=(255, 255, 255))
    d.text((14, 34), "replay from the recorded log: cloth shape 1x per second + at every grab/drop", font=FONT, fill=(220, 220, 220))
    writer.append_data(np.asarray(img))
    nvid += 1
writer.close()
print(f"replay video: {nvid} frames -> {os.path.join(HERE, 'best_fold_replay.mp4')}")

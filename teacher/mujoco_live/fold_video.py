"""Claude plays the game with the same action space as the human (two virtual mice: moves, button
presses, tandem key) and records a first-person video + the same demo log a human session makes.
Usage: python fold_video.py out.mp4"""
import sys, os, time
import numpy as np
import mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
from game import Game, ARMS, VIEWS, ARM_RGB

out = sys.argv[1]
g = Game(log=True)
os.replace(g.log_path, g.log_path) if os.path.exists(g.log_path) else None
g.log_path = g.log_path.replace(".jsonl", "_claude.jsonl")
M = {"L": 101, "R": 202}
W, H = 960, 560
r = mujoco.Renderer(g.w.m, H, W)
mini = mujoco.Renderer(g.w.m, 170, 240)
cam = mujoco.MjvCamera()
la, dist, el, az, fov = VIEWS[0]
cam.lookat[:] = la; cam.distance, cam.elevation, cam.azimuth = dist, el, az
mcam = mujoco.MjvCamera(); mcam.lookat[:] = [0, 0.18, 0]; mcam.distance, mcam.elevation, mcam.azimuth = 0.62, -89.5, 90
mopt = mujoco.MjvOption(); mopt.geomgroup[2] = 0
font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 18)
big = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 22)
writer = imageio.get_writer(out, fps=30, quality=7, macro_block_size=8)
label = ""
nframe = 0


def markers(scn):
    eye = np.eye(3).ravel()
    for a in ARMS:
        c = g.cmd[a]
        top = g.w.surface_under(c[:2])
        rgb = list(ARM_RGB[a]) if not g.w.held[a] else [0.9, 0.1, 0.1]
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.016, 0.0006, 0]),
                            np.array([c[0], c[1], top + 0.001]), eye, np.array(rgb + [0.5], np.float32))
        scn.ngeom += 1


def frame():
    global nframe
    nframe += 1
    g.w.m.vis.global_.fovy = fov
    r.update_scene(g.w.d, cam); markers(r.scene)
    img = Image.fromarray(r.render())
    g.w.m.vis.global_.fovy = 45
    mini.update_scene(g.w.d, mcam, mopt); markers(mini.scene)
    img.paste(Image.fromarray(mini.render()), (W - 250, H - 180))
    d = ImageDraw.Draw(img, "RGBA")
    s = g.score
    d.rectangle((6, 6, 470, 92), fill=(0, 0, 0, 150))
    d.text((14, 10), f"AREA {100 * s.get('area_vs_flat', 1):.0f}% of flat   box {s['rect_cm'][0]} x {s['rect_cm'][1]} cm", font=big, fill=(255, 255, 255))
    d.text((14, 40), f"rectangular {s['rectangularity']:.2f}   square {s['squareness']:.2f}", font=font, fill=(230, 230, 230))
    d.text((14, 64), f"Claude playing with 2 virtual mice   tandem: {['off', 'parallel', 'mirror'][g.tandem]}", font=font, fill=(200, 220, 255))
    d.rectangle((6, H - 40, 20 + 11 * len(label), H - 8), fill=(0, 0, 0, 150))
    d.text((14, H - 36), label, font=font, fill=(255, 230, 120))
    writer.append_data(np.asarray(img))


def tick(mice=None):
    g.tick(mice or {}, 1 / 60)
    if g.tick_n % 2 == 0:
        frame()


def wait(sec):
    for _ in range(int(sec * 60)):
        tick()


def goto(arm, xy, speed_cm=0.9):
    """Move that arm's mouse until its aim point reaches xy (cm). speed: cursor cm per tick."""
    h = M[arm]
    for _ in range(600):
        d = np.array(xy) - g.cursor[arm][:2] * 100
        if np.linalg.norm(d) < 0.1:
            break
        st = d / max(1.0, np.linalg.norm(d) / speed_cm)
        gn = g.gain[h]
        tick({h: (st[0] / 100 / gn, -st[1] / 100 / gn, 0, False, False, [])})
    for _ in range(600):  # let the arm catch up
        if np.linalg.norm(g.cmd[arm][:2] * 100 - xy) < 0.3:
            break
        tick()
    wait(0.2)


def press(arm, button="left"):
    tick({M[arm]: (0, 0, 0, button == "left", button == "right", [f"{button}_down"])})
    for _ in range(90):
        tick()
        if all(g.phase[a] != "down" for a in ARMS):
            break
    wait(0.2)


def release(arm, button="left"):
    tick({M[arm]: (0, 0, 0, False, False, [f"{button}_up"])})
    for _ in range(120):
        tick()
        if all(g.phase[a] in ("hover", "rising") for a in ARMS) and not any(g.w.held.values()):
            break
    wait(0.5)


# assign the two virtual mice
tick({M["L"]: (0, 0, 0, True, False, ["left_down"])}); tick({M["R"]: (0, 0, 0, True, False, ["left_down"])})
label = "the shirt, flat"; wait(1.0)

label = "1) both sleeves at once: mirror tandem, one mouse drives both arms"
g.key("T"); g.key("T")
goto("L", [-19, 26]); press("L"); goto("L", [-1.5, 26]); release("L")
g.key("T")  # off

label = "2) bottom-up: both hands on the hem, parallel tandem"
goto("L", [-8, 8.7]); goto("R", [8, 8.7])
g.key("T")  # parallel
press("L"); goto("L", [-8, 28]); release("L")

y = g.w.cloth.x[:, 1] * 100
far, near = y.max(), y.min()
label = f"3) fold again: wheel-click grabs the WHOLE stack at the far edge (y={far:.0f}) -> crease (y={near:.0f})"
goto("L", [-7, far - 1.2]); press("L", "middle"); goto("L", [-7, near + 1.0]); release("L", "middle")
g.key("T"); g.key("T")  # off
label = f"done: {100 * g.score['area_vs_flat']:.0f}% of flat area, box {g.score['rect_cm'][0]} x {g.score['rect_cm'][1]} cm"
wait(2.5)
writer.close()
print("final score", g.score)
print("video frames", nframe, "->", out)
print("demo log", g.log_path)

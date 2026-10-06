"""Render the cloth after scripted folds (virtual grippers, no arms) to look at it.
Saves a strip of images: flat | sleeve folded | both sleeves + bottom-up | 3 s later."""
import sys, numpy as np, mujoco
from PIL import Image
from world import World

out = sys.argv[1]
w = World()
r = mujoco.Renderer(w.m, 360, 480)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0.19, 0]; cam.distance = 0.42; cam.elevation = -40; cam.azimuth = 60
c = w.cloth
FPS = 60
grips = np.array([[-0.5, 0, 0.1], [0.5, 0, 0.1]])
far = np.array([[-0.5, 0, 0.1], [0.5, 0, 0.1]])
shots = []


opt = mujoco.MjvOption()
opt.geomgroup[2] = 0  # hide the robot meshes
cam2 = mujoco.MjvCamera(); cam2.lookat[:] = [0, 0.19, 0]; cam2.distance = 0.55; cam2.elevation = -89; cam2.azimuth = 90
cam.lookat[:] = [0, 0.19, 0]; cam.distance = 0.40; cam.elevation = -25; cam.azimuth = 90


def shot():
    w.sync_shirt(); w.refresh()
    r.update_scene(w.d, cam, scene_option=opt); a = r.render()
    r.update_scene(w.d, cam2, scene_option=opt); b = r.render()
    shots.append(np.concatenate([a, b], 0))


def move(goals, speed=0.15):
    starts = grips.copy()
    n = max(1, int(max(np.linalg.norm(goals[g] - starts[g]) for g in goals) / speed * FPS))
    for s in range(n):
        for g in goals:
            grips[g] = starts[g] + (goals[g] - starts[g]) * (s + 1) / n
        c.step(1 / FPS, grips)


def fold(pairs, lift=0.05):
    """pairs: {gid: (pick_xy, place_xy)} done together."""
    move({g: np.r_[p, 0.03] for g, (p, q) in pairs.items()}, 0.5)
    move({g: np.r_[p, 0.004] for g, (p, q) in pairs.items()})
    for g, (p, q) in pairs.items():
        c.grab(g, grips[g])
    move({g: np.r_[(np.array(p) + q) / 2, lift] for g, (p, q) in pairs.items()})
    move({g: np.r_[q, 0.02] for g, (p, q) in pairs.items()})
    move({g: np.r_[q, 0.012] for g, (p, q) in pairs.items()}, 0.05)
    for g in pairs:
        c.release(g)
    move({g: np.r_[q, 0.06] for g, (p, q) in pairs.items()}, 0.05)
    for _ in range(30):
        c.step(1 / FPS, grips)


shot()
fold({0: ((-0.195, 0.26), (-0.01, 0.26))})
shot()
fold({1: ((0.195, 0.26), (0.01, 0.26))})
fold({0: ((-0.06, 0.085), (-0.06, 0.28)), 1: ((0.06, 0.085), (0.06, 0.28))}, lift=0.07)
shot()
for _ in range(180):
    c.step(1 / FPS, grips)
shot()
print("score", w.score())
Image.fromarray(np.concatenate(shots, 1)).save(out)

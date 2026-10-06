"""Builds the local MuJoCo folding scene: table, flat T-shirt cloth, two SO101 arms."""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = os.path.join(HERE, "..", "teacher", "vendor", "so101_nexus", "assets", "SO101", "so101_new_calib.xml")

H = 0.02  # cloth lattice spacing (m)
# T-shirt as union of rectangles (x0, x1, y0, y1); hem near the robots (small y), collar far.
BODY = (-0.10, 0.10, 0.08, 0.30)
SLEEVE_L = (-0.20, -0.10, 0.22, 0.30)
SLEEVE_R = (0.10, 0.20, 0.22, 0.30)
ARM_BASE = {"L": (-0.17, -0.02), "R": (0.17, -0.02)}


def _inside(x, y):
    e = 1e-6
    return any(r[0] - e <= x <= r[1] + e and r[2] - e <= y <= r[3] + e for r in (BODY, SLEEVE_L, SLEEVE_R))


def shirt_mesh():
    xs = np.round(np.arange(-0.20, 0.20 + 1e-9, H), 4)
    ys = np.round(np.arange(0.08, 0.30 + 1e-9, H), 4)
    idx, pts = {}, []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if _inside(x, y):
                idx[(i, j)] = len(pts)
                pts.append((x, y))
    tris = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if all(k in idx for k in c) and _inside((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2):
                a, b, cc, d = (idx[k] for k in c)
                tris += [(a, b, cc), (a, cc, d)]
    return np.array(pts), np.array(tris)


def key_vertices(pts):
    """Named landmarks -> vertex index (nearest lattice point)."""
    named = {
        "hem_L": (-0.10, 0.08), "hem_R": (0.10, 0.08),
        "armpit_L": (-0.10, 0.22), "armpit_R": (0.10, 0.22),
        "shoulder_L": (-0.10, 0.30), "shoulder_R": (0.10, 0.30),
        "sleeve_L_lo": (-0.20, 0.22), "sleeve_L_hi": (-0.20, 0.30),
        "sleeve_R_lo": (0.20, 0.22), "sleeve_R_hi": (0.20, 0.30),
        "collar_mid": (0.0, 0.30), "hem_mid": (0.0, 0.08),
    }
    return {k: int(np.argmin(np.linalg.norm(pts - np.array(v), axis=1))) for k, v in named.items()}


def build():
    pts, tris = shirt_mesh()
    z0 = 0.004
    point = " ".join(f"{x:.4f} {y:.4f} {z0}" for x, y in pts)
    element = " ".join(f"{a} {b} {c}" for a, b, c in tris)
    xml = f"""
<mujoco model="fold_scene">
  <option timestep="0.002" integrator="discrete" cone="elliptic"/>
  <visual><global offwidth="1280" offheight="960"/><headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.80 0.76" rgb2="0.78 0.76 0.72" width="512" height="512"/>
    <material name="table" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0.2 1.2" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="table" type="plane" size="0.6 0.6 0.01" material="table" friction="0.8 0.01 0.001" contype="1" conaffinity="1"/>
    <camera name="top" pos="0 0.17 0.75" xyaxes="1 0 0 0 1 0" fovy="50"/>
    <flexcomp name="shirt" type="direct" dim="2" radius="0.002" mass="0.05" rgba="0.2 0.45 0.85 1"
              point="{point}" element="{element}">
      <edge equality="true" damping="0.002"/>
      <elasticity young="3e4" poisson="0.2" thickness="0.002" damping="0.0005" elastic2d="bend"/>
      <contact condim="3" friction="0.6 0.01 0.001" selfcollide="auto" internal="false"
               contype="2" conaffinity="1" solref="0.004 1"/>
    </flexcomp>
  </worldbody>
</mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    for side, (bx, by) in ARM_BASE.items():
        robot = mujoco.MjSpec.from_file(ROBOT)
        # arm geoms never touch the cloth (grasp is a pin, see server.py); keep them out of contact
        for g in robot.geoms:
            g.contype = 0
            g.conaffinity = 0
        # robot's base faces +x; rotate 90 deg so it faces the shirt (+y)
        frame = spec.worldbody.add_frame(pos=[bx, by, 0.0], quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        frame.attach_body(robot.body("base"), f"{side}_", "")
    model = spec.compile()
    return model, pts, tris


if __name__ == "__main__":
    m, pts, tris = build()
    print("compiled: nbody", m.nbody, "nflexvert", m.nflexvert, "nu", m.nu, "tris", len(tris))
    print([m.actuator(i).name for i in range(m.nu)])

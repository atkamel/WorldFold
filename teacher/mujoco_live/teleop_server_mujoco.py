"""Teleop server on the local MuJoCo world: tests the whole laptop<->server bridge for free.
The Isaac version runs the same controller + server on Modal (WorldFold/teacher/oracle/teleop/isaac_world.py).
    python teleop_server_mujoco.py [port]      then:   python teleop_client.py 127.0.0.1 [port]"""
import os, sys, time
import numpy as np
import mujoco
from world import World, ARMS, THICK

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
from teleop.controller import Game  # noqa: E402
from teleop.server import TeleopServer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FW, FH = 960, 600
VIEWS = [([0, 0.20, 0.0], 0.46, -49, 90, 65), ([0, 0.19, 0.0], 0.60, -70, 90, 45), ([0, 0.18, 0.0], 0.62, -89.5, 90, 45)]
ARM_RGB = {"L": (1.0, 0.45, 0.1), "R": (0.2, 0.75, 0.3)}


def main(port=7777):
    w = World()
    g = Game(w)
    r = mujoco.Renderer(w.m, FH, FW)
    cam = mujoco.MjvCamera()
    view = {"i": 0}

    def on_key(k):
        if k == "V":
            view["i"] = (view["i"] + 1) % len(VIEWS)
            g.msg = ["first person", "over the table", "top-down"][view["i"]]
            return True
        return False

    def render(game):
        la, dist, el, az, fov = VIEWS[view["i"]]
        cam.lookat[:] = la
        cam.distance, cam.elevation, cam.azimuth = dist, el, az
        w.m.vis.global_.fovy = fov
        r.update_scene(w.d, cam)
        scn, eye = r.scene, np.eye(3).ravel()
        for a in ARMS:
            c = game.cmd[a]
            top = w.surface_under(c[:2])
            rgb = list(ARM_RGB[a]) if not w.held[a] else [0.9, 0.1, 0.1]
            if game.reach_flash[a] > 0:
                rgb = [0.6, 0.6, 0.6]
            h = max(c[2] - top, 0.001)
            for typ, size, pos, alpha in [(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.016, 0.0006, 0], [c[0], c[1], top + 0.001], 0.5),
                                          (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.0012, h / 2, 0], [c[0], c[1], top + h / 2], 0.6)]:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], typ, np.array(size, float), np.array(pos, float), eye,
                                    np.array(rgb + [alpha], np.float32))
                scn.ngeom += 1
            yaw = w.jaw_yaw(a)
            u = np.array([np.cos(yaw), np.sin(yaw), 0.0]) * 0.022
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye,
                                np.array(rgb + [0.9], np.float32))
            mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025,
                                 np.array([c[0], c[1], top + 0.0015]) - u, np.array([c[0], c[1], top + 0.0015]) + u)
            scn.ngeom += 1
        return r.render()

    flat_area = w.area0 / 1e4
    hello = dict(backend="mujoco", dt=1 / 60, tris=w.cloth.tris.tolist(), thickness=THICK, flat_area_m2=flat_area,
                 frame_size=[FW, FH])
    os.makedirs(os.path.join(HERE, "demos"), exist_ok=True)
    log_path = os.path.join(HERE, "demos", time.strftime("%Y%m%d_%H%M%S") + "_server.jsonl")
    TeleopServer(g, render, hello, port=port, dt=1 / 60, frame_every=2, log_path=log_path, on_key=on_key,
                 idle_s=600, max_s=7200).serve()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7777)

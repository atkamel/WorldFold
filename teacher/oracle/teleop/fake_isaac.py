"""A fake LeHome env for dry-running the Isaac teleop backend on a laptop (no Isaac, no GPU, no cost).

Not physics: joints follow commands exactly, the shirt is a flat T-shirt on the Isaac table (z 0.52), cloth
points near a closed gripper stick to it and drop flat when it opens. Enough to exercise everything around
the simulator: Isaac frames, SO-101 kinematics, the scripted grasp/release moves, undo, the stream.
    python -m teleop.fake_isaac PORT [--state]   (from teacher/oracle)   then the laptop client or test_bridge.py
"""
import copy
import numpy as np
import fold_plans as fp
from so101_kin import HOME

TABLE_Z = 0.52


def tshirt(h=0.01):
    rects = ((-0.10, 0.10, -0.11, 0.11), (-0.20, -0.10, 0.03, 0.11), (0.10, 0.20, 0.03, 0.11))
    inside = lambda x, y: any(r[0] - 1e-6 <= x <= r[1] + 1e-6 and r[2] - 1e-6 <= y <= r[3] + 1e-6 for r in rects)
    xs = np.round(np.arange(-0.20, 0.20 + 1e-9, h), 4); ys = np.round(np.arange(-0.11, 0.11 + 1e-9, h), 4)
    idx, pts = {}, []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if inside(x, y):
                idx[(i, j)] = len(pts); pts.append((x, y))
    tris = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if all(k in idx for k in c) and inside((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2):
                a, b, cc, d = (idx[k] for k in c)
                tris += [(a, b, cc), (a, cc, d)]
    return np.c_[np.array(pts), np.full(len(pts), TABLE_Z + 0.002)], np.array(tris)


class FakeEnv:
    device = "cpu"
    step_dt = 1 / 30

    def __init__(self):
        self.verts0, self.faces = tshirt()
        self.reset()

    def reset(self):
        self.q = np.r_[HOME["left"], 0.55, HOME["right"], 0.55].astype(float)
        self.verts = self.verts0.copy()
        self.stuck = {"left": None, "right": None}

    def step(self, action):
        self.q = np.asarray(action.cpu().numpy() if hasattr(action, "cpu") else action, float).ravel()
        for side, sl, gi in (("left", slice(0, 5), 5), ("right", slice(6, 11), 11)):
            tip = fp.K.tip(self.q[sl], side)[0]
            closed = self.q[gi] < 0.0
            if closed and self.stuck[side] is None:
                near = np.where(np.linalg.norm(self.verts - tip, axis=1) < 0.02)[0]
                self.stuck[side] = (near, self.verts[near] - tip) if len(near) else None
            if not closed and self.stuck[side] is not None:
                ids, _ = self.stuck[side]
                self.verts[ids, 2] = TABLE_Z + 0.002 + 0.004 * (self.verts[ids, 2] > TABLE_Z + 0.01)
                self.stuck[side] = None
            if self.stuck[side] is not None:
                ids, off = self.stuck[side]
                self.verts[ids] = tip + off

    def _get_observations(self):
        return {"observation.state": self.q.copy(), "observation.images.top_rgb": self._image(), "check_status": []}

    def _image(self, W=640, H=480):
        im = np.full((H, W, 3), 200, np.uint8)
        # simple top view with the robots at the TOP (like the real raw top camera); world x->u, y->v flipped
        s = 900.0
        uv = lambda p: (int(W / 2 - p[0] * s), int(H / 2 + p[1] * s))
        for x, y, z in self.verts:
            u, v = uv((x, y))
            if 0 <= u < W and 0 <= v < H:
                im[max(0, v - 2):v + 2, max(0, u - 2):u + 2] = (60, 110, 210) if z < TABLE_Z + 0.01 else (150, 190, 255)
        for side, sl in (("left", slice(0, 5)), ("right", slice(6, 11))):
            u, v = uv(fp.K.tip(self.q[sl], side)[0])
            if 0 <= u < W and 0 <= v < H:
                im[max(0, v - 4):v + 4, max(0, u - 4):u + 4] = (240, 120, 30) if side == "left" else (60, 190, 80)
        return im


def helpers():
    def between_pads(q5, side, verts):
        return int((np.linalg.norm(verts - fp.K.tip(q5, side)[0], axis=1) < 0.02).sum())
    return dict(cloth_verts=lambda env: env.verts.copy(), cloth_faces=lambda env: env.faces, between_pads=between_pads,
                capture=lambda env: copy.deepcopy((env.q, env.verts, env.stuck)),
                restore=lambda env, s: setattr_all(env, copy.deepcopy(s)),
                set_cloth=lambda env, t: setattr(env, "cloth_overrides", t.cloth_overrides(len(env.verts0))))


def setattr_all(env, s):
    env.q, env.verts, env.stuck = s


if __name__ == "__main__":
    import sys, time
    from teleop.isaac_world import IsaacWorld
    from teleop.controller import Game
    from teleop.server import TeleopServer
    env = FakeEnv()
    w = IsaacWorld(env, helpers(), settle_ticks=5)
    print("aim-point IK:", w.ik_backend, flush=True)
    g = Game(w)
    state_view = "--state" in sys.argv          # the laptop draws (Polyscope client) instead of receiving JPEGs
    hello = dict(backend="fake-isaac", dt=w.dt, tris=w.faces.tolist(), thickness=0.004,
                 flat_cloth_cm=np.round(w.flat_local * 100, 2).tolist(), frame_size=[640, 480])
    if state_view:
        hello.update(w.state_hello())
    TeleopServer(g, w.render, hello, port=int(sys.argv[1]) if len(sys.argv) > 1 else 7777, dt=w.dt, frame_every=1,
                 idle_s=120, max_s=600, state=w.state_bytes if state_view else None).serve()

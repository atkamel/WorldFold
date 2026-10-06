"""Isaac (LeHome) backend for the teleop controller: real SO-101s, real rigid jaws, PhysX particle cloth.

Coordinates: the controller works in a 'table frame' (metres, table surface z = 0, shirt centre at (0, 0.19)
like the MuJoCo game); this class converts to Isaac world coordinates (shirt centre (0, 0), table ~0.52).

Grab  = the tested gather grasp (oracle runs + live sessions, 8/8 held with flat pads + fixed physics): slide 3 cm
        across the aim point with the MOVING finger leading, pressing 1.2 cm into the table, then close.
Release = the measured rule: open, back the fixed finger off 2 cm, lift slowly (0.7 mm per tick).
Both run as short scripted moves; while one runs, busy(arm) is True and the controller holds that arm still.

Needs helpers from oracle_fold (passed in, to avoid importing the script module): cloth_verts(env),
cloth_faces(env), between_pads(q5, side, verts), capture(env) -> state, restore(env, state),
set_cloth(env, tuning) (rebuilds the garment with the tuning's cloth values, no simulator restart).
The grasp/release numbers and the cloth physics are live-tunable (tuning.py); the tested values are the defaults.
"""
import os
import numpy as np
import fold_plans as fp
from so101_kin import HOME, GRIP_OPEN, GRIP_CLOSED
from .tuning import SPECS

SIDE = {"L": "left", "R": "right"}
NSIDE = {"L": 0, "R": 1}  # the native IK's arm index
SHIRT_Y = 0.19            # where the controller expects the shirt centre (table frame)


class _ClothView:
    def __init__(self, w):
        self.w = w

    @property
    def x(self):
        return self.w.verts_local()


class IsaacWorld:
    TUNE_GRIP = True          # the controller offers the grasp/release settings in the tuning panel
    CLOTH_LOG_S = None        # no full-cloth record every simulated second (a ~24 ms stall); grab/drop records keep it

    def __init__(self, env, helpers, settle_ticks=150):
        import torch
        self.env, self.h, self.torch = env, helpers, torch
        self.K = fp.K
        self.nik = self._load_native_ik()
        self.settle_ticks = settle_ticks
        self.xlim, self.ylim = (-0.32, 0.32), (-0.08, 0.44)
        self.cloth = _ClothView(self)
        self.k = max(1, int(os.environ.get("ORACLE_STEPS_PER_TICK", "1")))   # physics steps per control tick
        self.dt = float(getattr(env, "step_dt", 1 / 30)) * self.k
        self.flip = True                 # LeHome's top camera image is upside down for us; the FPV camera is not
        self.reset()

    # ---------------- frames
    def to_isaac(self, p):
        p = np.asarray(p, float)
        return p + self.O[:len(p)]

    def to_local(self, p):
        p = np.asarray(p, float)
        return p - self.O[:p.shape[-1]]

    # ---------------- lifecycle
    def _obs_state(self):
        s = np.asarray(self.obs["observation.state"], dtype=float).ravel()
        return {"L": s[0:5].copy(), "R": s[6:11].copy()}, {"L": float(s[5]), "R": float(s[11])}

    def _send(self, q5, g):
        import time
        t0 = time.perf_counter()
        a = np.concatenate([q5["L"], [g["L"]], q5["R"], [g["R"]]]).astype(np.float32)
        # env.step already computed the observations: reuse them (a 2nd _get_observations copies 3 images again and
        # doubles LeHome's success-check counter)
        act = self.torch.from_numpy(a).to(self.env.device).unsqueeze(0)
        for _ in range(self.k - 1):      # same target held for k physics steps; only the last one renders
            self.env.step(act)
        out = self.env.step(act)
        self.obs = out[0] if isinstance(out, tuple) and isinstance(out[0], dict) else self.env._get_observations()
        self.q_meas, self.g_meas = self._obs_state()
        self._verts = None
        self._hmap = None
        self.t_sim = getattr(self, "t_sim", 0.0) + time.perf_counter() - t0   # Isaac's share (teleop perf line)

    def reset(self):
        env = self.env
        env.reset()
        env._cloth_pose_cache = None      # oracle_fold.cloth_verts: the garment may have moved
        self.obs = env._get_observations()
        self.cq = {"L": HOME["left"].copy(), "R": HOME["right"].copy()}     # commanded arm joints (5 incl. roll)
        self.cg = {"L": GRIP_CLOSED, "R": GRIP_CLOSED}
        for _ in range(self.settle_ticks):
            self._send(self.cq, self.cg)
        flat = self.h["cloth_verts"](env)
        self.table_z = float(np.percentile(flat[:, 2], 1)) - 0.002
        c = flat[:, :2].mean(0)
        self.O = np.array([c[0], c[1] - SHIRT_Y, self.table_z])            # table frame -> Isaac world offset
        self.faces = self.h["cloth_faces"](env)
        self.flat_local = self.to_local(flat)
        self.cg = {"L": GRIP_OPEN, "R": GRIP_OPEN}
        self.held = {"L": 0, "R": 0}
        self.script = {"L": [], "R": []}
        self._verts = None
        self._hmap = None

    # ---------------- arms (controller interface)
    def tip(self, a):
        if self.nik is not None:
            return self.nik.frame(NSIDE[a], self.q_meas[a])[0] - self.O
        return self.to_local(self.K.tip(self.q_meas[a], SIDE[a])[0])

    def q(self, a):
        return self.q_meas[a][:4].copy()

    def roll(self, a):
        return float(self.q_meas[a][4])

    def jaw_yaw(self, a, q5=None):
        q5 = self.q_meas[a] if q5 is None else q5
        if self.nik is not None:
            x = self.nik.frame(NSIDE[a], q5)[2]                               # jaw opening axis (tip frame x)
        else:
            x = self.K.fk(q5, SIDE[a])[:3, 0]
        return float(np.arctan2(x[1], x[0]))

    def _load_native_ik(self):
        """The aim-point IK in C++ (native/so101_ik.hpp): exact and ~1 microsecond instead of milliseconds. The numpy
        solver stays as the fallback when the library is missing for this machine or disagrees with so101_kin.
        ORACLE_NATIVE_IK=0 turns it off."""
        self.ik_backend = "numpy (so101_kin)"
        if os.environ.get("ORACLE_NATIVE_IK", "1") == "0":
            return None
        try:
            from so101_native_ik import NativeIK
            nik = NativeIK(self.K)
            nik.install()                # the scripted grab / release paths (fold_plans -> SO101.ik) use it too
            self.ik_backend = "native closed form (so101_ik)"
            return nik
        except Exception as e:
            self.ik_backend += f" - native IK not loaded: {e!r}"
            return None

    def ik(self, a, goal, q4, roll, seeds=True):
        if self.nik is not None:
            # exact answer: tip on the goal with the least tilt, or err = inf when no joint angles reach it.
            # seeds (the first try of a tick) = the other elbow branch may be used if the current one cannot reach.
            q, err, tilt, _ = self.nik.solve(NSIDE[a], goal + self.O, roll, q4, seeds)
            return q, err, tilt
        side = SIDE[a]
        q0 = np.r_[q4, roll]
        q, err = self.K.ik(self.to_isaac(goal), side, q0, direction=(0, 0, -1), w_dir=0.005, iters=25)
        if seeds and err > 0.003:
            s = fp.SEED[side].copy(); s[4] = roll
            q2, e2 = self.K.ik(self.to_isaac(goal), side, s, direction=(0, 0, -1), w_dir=0.005, iters=120)
            if e2 < err:
                q, err = q2, e2
        appr = self.K.tip(q, side)[1]
        tilt = float(np.degrees(np.arccos(np.clip(-appr[2], -1, 1))))
        return q[:4], float(err), tilt

    def roll_for_yaw(self, a, q4, yaw):
        lo, hi = self.K.lo[4], self.K.hi[4]
        y0 = self.jaw_yaw(a, np.r_[q4, 0.0])
        s = np.sign(((self.jaw_yaw(a, np.r_[q4, 0.1]) - y0 + np.pi) % (2 * np.pi)) - np.pi) or 1.0
        roll = float(np.clip(((yaw - y0 + np.pi) % (2 * np.pi) - np.pi) * s, lo, hi))
        return roll, y0 + s * roll

    def set_arm(self, a, q4, roll):
        if not self.script[a]:
            self.cq[a] = np.r_[q4, roll]

    def busy(self, a):
        return bool(self.script[a])

    def _t(self, k):
        t = getattr(self, "tune", None)
        return t[k] if t is not None else SPECS[k][0]

    # ---------------- grab / release (scripted, tested moves; numbers live-tunable)
    def grab(self, a, all_layers=False):
        side = SIDE[a]
        q5 = self.q_meas[a].copy()
        aim = self.K.tip(q5, side)[0]
        d = self.K.moving_side(q5, side)[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        top = self.surface_under(self.to_local(aim)[:2]) + self.table_z
        press, half = self._t("grab_press"), self._t("grab_slide") / 2
        z_land = max(top + 0.007, fp.Z_TABLE - 0.004)
        z_press = max(top - press, fp.Z_TABLE - fp.PRESS - (press - 0.012))   # tested: 1.2 cm (stacks: from their top)
        start, end = aim[:2] - d * half, aim[:2] + d * half            # slide centred on the aim, moving finger leads
        qs, _ = fp._path(side, [aim, (start[0], start[1], z_land), (end[0], end[1], z_press)], q5, q5[4],
                         ds=self._t("grab_ds") if half > 0 else 0.004)   # no slide (MuJoCo feel): press down in a few ticks
        g0 = self.g_meas[a]
        nc = max(1, int(round(self._t("grab_close"))))
        sc = [(q, GRIP_OPEN) for q in qs]
        sc += [(qs[-1] if qs else q5, g0 + (GRIP_CLOSED - g0) * i / nc) for i in range(1, nc + 1)]
        sc += [(qs[-1] if qs else q5, GRIP_CLOSED)] * (12 if self._t("grab_slide") > 0 else 2)   # settle
        gl = self._t("grab_lift")
        lift, _ = fp._path(side, [(end[0], end[1], z_press), (end[0], end[1], z_press + gl)], sc[-1][0], q5[4], ds=0.001)             if gl > 0 else ([], 0)
        sc += [(q, GRIP_CLOSED) for q in lift]
        self.script[a] = sc
        self._grab_side = getattr(self, "_grab_side", {})
        self._grab_side[a] = True
        self.held[a] = 0          # known once the move is done (see _finish)
        return 0

    def release(self, a):
        side = SIDE[a]
        q5 = self.cq[a].copy()
        p = self.K.tip(q5, side)[0]
        no = max(1, int(round(self._t("rel_open"))))
        sc = [(q5, GRIP_CLOSED + (GRIP_OPEN - GRIP_CLOSED) * i / no) for i in range(1, no + 1)]
        sc += [(q5, GRIP_OPEN)] * int(round(self._t("rel_hold")))
        n = self.K.moving_side(q5, side)[:2]; n = n / (np.linalg.norm(n) + 1e-9)
        b = self._t("rel_backoff")
        p2 = np.array([p[0] - n[0] * b, p[1] - n[1] * b, p[2]])         # back the fixed finger off
        away, _ = fp._path(side, [p, p2], q5, q5[4], ds=0.001) if b > 0 else ([], 0)
        lift = self._t("rel_up")
        up, _ = fp._path(side, [p2, (p2[0], p2[1], p2[2] + lift)], away[-1] if away else q5, q5[4], ds=self._t("rel_lift"))             if lift > 0 else ([], 0)
        sc += [(q, GRIP_OPEN) for q in away + up]
        self.script[a] = sc
        self.held[a] = 0

    def apply_cloth(self, tuning):
        """Rebuild the shirt with the tuning's cloth physics (oracle_fold.set_cloth: no simulator restart),
        then the usual reset: settle, re-measure the table frame."""
        self.h["set_cloth"](self.env, tuning)
        self.reset()

    def _finish(self, a):
        """A scripted move just ended: hand the arm back to the controller at its current pose."""
        self.cq[a] = self.cq[a].copy()
        if getattr(self, "_grab_side", {}).pop(a, False):
            v = self.verts()
            self.held[a] = int(self.h["between_pads"](self.q_meas[a], SIDE[a], v))
            self.cg[a] = GRIP_CLOSED                  # shut until the player lets go (controller then calls release)
        else:
            self.cg[a] = GRIP_OPEN

    # ---------------- stepping
    def step(self, dt=None):
        for a in ("L", "R"):
            if self.script[a]:
                q, g = self.script[a].pop(0)
                self.cq[a], self.cg[a] = np.asarray(q, float), float(g)
                if not self.script[a]:
                    self._send_pending_finish = getattr(self, "_send_pending_finish", set()) | {a}
        self._send(self.cq, self.cg)
        for a in getattr(self, "_send_pending_finish", set()):
            self._finish(a)
        self._send_pending_finish = set()

    # ---------------- cloth
    def verts(self):
        if self._verts is None:
            self._verts = self.h["cloth_verts"](self.env)
        return self._verts

    def verts_local(self):
        return self.to_local(self.verts())

    def _buckets(self):
        """Resting cloth points indexed once per physics step (same answers as scanning all points): the native grid
        (so101_cloth.hpp) when the library is loaded, else a SciPy cKDTree."""
        key = tuple(bool(self.held[a] or self.script[a]) for a in ("L", "R"))
        if getattr(self, "_hmap", None) is not None and self._hmap[0] == key:
            return self._hmap
        if self.nik is not None:      # cloth hanging from a gripper is not a surface
            tips = [self.tip(a) for a in ("L", "R") if self.held[a] or self.script[a]]
            self._hmap = (key, self.nik.cloth_build(self.verts(), self.O, tips), None)
            return self._hmap
        v = self.verts_local()
        keep = v[:, 2] < 0.04
        for a in ("L", "R"):          # cloth hanging from a gripper is not a surface
            if self.held[a] or self.script[a]:
                keep &= np.linalg.norm(v - self.tip(a), axis=1) > 0.04
        from scipy.spatial import cKDTree
        p = v[keep]
        self._hmap = (key, p, cKDTree(p[:, :2], balanced_tree=False, compact_nodes=False))
        return self._hmap

    def surface_under(self, xy, radius=0.02):
        _, p, tree = self._buckets()
        if tree is None:                                      # native index (p = number of points kept)
            top = self.nik.cloth_top(float(xy[0]), float(xy[1]), radius)
            return top + 0.003 if top > -np.inf else 0.0
        idx = tree.query_ball_point(np.asarray(xy, float)[:2], radius) if len(p) else []
        return float(p[idx, 2].max()) + 0.003 if idx else 0.0

    def score(self):
        return {}                     # the laptop scores every drop (it has the fold-score code); keeps this fast

    # ---------------- undo
    def get_state(self):
        return dict(phys=self.h["capture"](self.env), cq={a: self.cq[a].copy() for a in self.cq}, cg=dict(self.cg),
                    held=dict(self.held))

    def set_state(self, s):
        self.h["restore"](self.env, s["phys"])
        self.cq = {a: s["cq"][a].copy() for a in s["cq"]}; self.cg = dict(s["cg"]); self.held = dict(s["held"])
        self.script = {"L": [], "R": []}
        for _ in range(10):
            self._send(self.cq, self.cg)

    # ---------------- picture for the player
    def render(self, game=None):
        """Top camera, turned so the robots are at the bottom (first-person-like), with the aim points drawn."""
        im = np.ascontiguousarray(np.asarray(self.obs["observation.images.top_rgb"])[..., :3])
        if game is not None:
            im = self._draw_aims(im, game)
        return np.ascontiguousarray(im[::-1, ::-1] if self.flip else im)

    def _draw_aims(self, im, game):
        P = self._camera_projection()
        if P is None:
            return im
        try:
            import cv2
        except ImportError:
            return im
        im = im.copy()
        for a, col in (("L", (255, 115, 25)), ("R", (50, 190, 75))):
            c = game.cmd[a]
            top = self.surface_under(c[:2])
            for z, r in ((top, 7), (c[2], 4)):
                p = P @ np.r_[self.to_isaac(np.r_[c[0], c[1], z]), 1.0]
                if p[2] > 0:
                    u, v = int(p[0] / p[2]), int(p[1] / p[2])
                    color = (220, 30, 30) if self.held[a] else col
                    cv2.circle(im, (u, v), r, color, 2 if r == 7 else -1)
        return im

    QUANT = 1e-4                      # state view: cloth points sent as int16 in 0.1 mm units (+-3.2 m)

    def state_bytes(self, game=None):
        """State view payload: measured joint angles (12 float32) + cloth points in the table frame (int16, QUANT m).
        Copies the sim's numbers now and returns a function that packs them: the server calls it on its sender
        thread, so the packing overlaps the next physics step."""
        q = np.array(self.obs["observation.state"], dtype=np.float32).ravel()[:12]
        v, O = np.array(self.verts(), dtype=float), self.O.copy()

        def pack():
            vq = np.clip(np.round((v - O[:3]) / self.QUANT), -32767, 32767).astype("<i2")
            return q.astype("<f4").tobytes() + vq.tobytes()
        return pack

    def state_hello(self):
        return dict(view="state", origin=self.O.tolist(), quant=self.QUANT)

    def set_fpv_camera(self):
        """Put the top camera where the MuJoCo game's first-person view is: head just behind and between the arms,
        looking at the table centre (lookat (0, 0.20, 0), distance 0.46 m, 49 deg down; fovy set in the cfg).
        Live view only - recorded data comes from replays with LeHome's own camera."""
        import math
        cam = self.env.top_camera
        target = self.to_isaac(np.array([0.0, 0.20, 0.0]))
        el = math.radians(49)
        eye = target + 0.46 * np.array([0.0, -math.cos(el), math.sin(el)])
        T = lambda v: self.torch.tensor(np.asarray(v, np.float32)[None], device=self.env.device)
        cam.set_world_poses_from_view(T(eye), T(target))
        self.flip = False
        if hasattr(self, "_P"):
            del self._P
        for _ in range(3):
            self._send(self.cq, self.cg)

    def camera_info(self):
        """For the laptop to draw the cursor itself: world->pixel matrix of the top camera, the table-frame origin, and
        the fact that render() flips the image both ways."""
        P = self._camera_projection()
        if P is None:
            return None
        h, w = np.asarray(self.obs["observation.images.top_rgb"]).shape[:2]
        return dict(P=np.asarray(P, float).ravel().tolist(), O=self.O.tolist(), size=[int(w), int(h)], flip=self.flip)

    def _camera_projection(self):
        """3x4 world->pixel matrix of the top camera, if the env exposes it (IsaacLab Camera sensor)."""
        if hasattr(self, "_P"):
            return self._P
        self._P = None
        try:
            cam = None
            for name in ("top_camera", "top_rgb", "top"):
                cam = getattr(self.env.scene, "sensors", {}).get(name) if hasattr(self.env, "scene") else None
                if cam is not None:
                    break
            if cam is not None:
                K = np.asarray(cam.data.intrinsic_matrices[0].cpu())
                pos = np.asarray(cam.data.pos_w[0].cpu())
                q = np.asarray(cam.data.quat_w_ros[0].cpu())               # w, x, y, z (ROS optical frame)
                w_, x, y, z = q
                R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w_), 2 * (x * z + y * w_)],
                              [2 * (x * y + z * w_), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w_)],
                              [2 * (x * z - y * w_), 2 * (y * z + x * w_), 1 - 2 * (x * x + y * y)]])
                self._P = K @ np.c_[R.T, -R.T @ pos]
        except Exception as e:
            print("[teleop] no camera projection, aim markers off:", repr(e), flush=True)
            self._P = None
        return self._P

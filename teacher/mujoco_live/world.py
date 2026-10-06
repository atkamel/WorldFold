"""The folding world: two SO-101 arms (MuJoCo) + the XPBD shirt (cloth.py) + scoring.

MuJoCo simulates the arms (they collide with each other and the table) and draws everything.
The shirt is simulated by cloth.py; MuJoCo only displays it (its flex has no physics here).
"""
import os
import numpy as np
import mujoco
from numba import njit
from cloth import Cloth
from shirt import shirt_mesh, landmark_ids
from neat_metric import flat_fraction, valley_share, inside_voids, FLOOR_AREA, AIR_ZERO_MM

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = os.path.join(HERE, "..", "teacher", "vendor", "so101_nexus", "assets", "SO101", "so101_new_calib.xml")
ARMS = ("L", "R")
ARM_BASE = {"L": (-0.17, -0.02), "R": (0.17, -0.02)}
JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OPEN, CLOSED = 1.2, 0.0
REST_Q = np.array([0.0, -1.2, 1.0, 1.3])  # folded-up parking pose
SPACING = 0.01
THICK = 0.004
TIP_DEPTH = 0.0981  # fingertip distance along the gripper's roll axis (from the SO-101 'gripperframe' site)


def build_model(pts, tris):
    point = " ".join(f"{x:.4f} {y:.4f} {THICK / 2}" for x, y in pts)
    element = " ".join(f"{a} {b} {c}" for a, b, c in tris)
    xml = f"""
<mujoco model="fold_world">
  <option timestep="0.004" integrator="implicitfast"/>
  <visual><global offwidth="1600" offheight="1000"/><quality shadowsize="1024"/>
          <headlight ambient="0.42 0.42 0.42" diffuse="0.55 0.55 0.55"/></visual>
  <asset>
    <texture name="wood" type="2d" builtin="checker" rgb1="0.80 0.74 0.64" rgb2="0.76 0.70 0.60" width="512" height="512"/>
    <material name="table" texture="wood" texrepeat="6 6" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0.1 0.1 1.0" dir="-0.1 0.05 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom name="table" type="plane" size="0.6 0.6 0.01" material="table" contype="1" conaffinity="1"/>
    <camera name="top" pos="0 0.17 0.75" xyaxes="1 0 0 0 1 0" fovy="50"/>
    <flexcomp name="shirt" type="direct" dim="2" radius="{THICK / 2}" mass="0.02" rgba="0.18 0.42 0.82 1"
              point="{point}" element="{element}">
      <contact contype="0" conaffinity="0"/>
    </flexcomp>
  </worldbody>
</mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    bits = {"L": (2, 4 | 1), "R": (4, 2 | 1)}  # each arm collides with the other arm and the table, not itself
    lo = os.path.join(HERE, "assets_lo")  # simplified meshes (decimate_meshes.py): the CAD ones are too heavy to draw
    for side, (bx, by) in ARM_BASE.items():
        robot = mujoco.MjSpec.from_file(ROBOT)
        if os.path.isdir(lo):
            robot.meshdir = lo
        for g in robot.geoms:
            if g.contype or g.conaffinity:
                g.contype, g.conaffinity = bits[side]
        frame = spec.worldbody.add_frame(pos=[bx, by, 0.0], quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        frame.attach_body(robot.body("base"), f"{side}_", "")
    return spec.compile()


@njit(cache=True)
def _raster(x, tris, lo0, lo1, res, nx, ny):
    occ = np.zeros((ny, nx), np.bool_)
    for t in range(tris.shape[0]):
        ax, ay = x[tris[t, 0], 0], x[tris[t, 0], 1]
        bx, by = x[tris[t, 1], 0], x[tris[t, 1], 1]
        cx, cy = x[tris[t, 2], 0], x[tris[t, 2], 1]
        i0 = max(int((min(ax, bx, cx) - lo0) / res), 0); i1 = min(int((max(ax, bx, cx) - lo0) / res) + 1, nx - 1)
        j0 = max(int((min(ay, by, cy) - lo1) / res), 0); j1 = min(int((max(ay, by, cy) - lo1) / res) + 1, ny - 1)
        for j in range(j0, j1 + 1):
            py = lo1 + (j + 0.5) * res
            for i in range(i0, i1 + 1):
                px = lo0 + (i + 0.5) * res
                d1 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                d2 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
                d3 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
                neg = d1 < 0 or d2 < 0 or d3 < 0
                pos = d1 > 0 or d2 > 0 or d3 > 0
                if not (neg and pos):
                    occ[j, i] = True
    return occ


class World:
    def __init__(self, **cloth_kw):
        self.pts, self.tris = shirt_mesh(SPACING)
        self.keys = landmark_ids(self.pts)
        self.m = build_model(self.pts, self.tris)
        self.d = mujoco.MjData(self.m)
        self.ik_data = mujoco.MjData(self.m)
        m = self.m
        self.act = {a: [m.actuator(f"{a}_{j}").id for j in JN] for a in ARMS}
        self.qadr = {a: [m.jnt_qposadr[m.joint(f"{a}_{j}").id] for j in JN] for a in ARMS}
        self.jrange = {a: np.array([m.jnt_range[m.joint(f"{a}_{j}").id] for j in JN[:4]]) for a in ARMS}
        self.site = {a: m.site(f"{a}_gripperframe").id for a in ARMS}
        self.gbody = {a: m.body(f"{a}_gripper").id for a in ARMS}
        self.pads = {a: [m.geom(f"{a}_static_finger_pad").id, m.geom(f"{a}_moving_finger_pad").id] for a in ARMS}
        # shirt vertices: MuJoCo only displays them, so cancel gravity on them and drive their position
        self.vbody = m.flex_vertbodyid.copy()
        m.body_gravcomp[self.vbody] = 1.0
        self.vqadr = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self.vbody])
        self.vdadr = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in self.vbody])
        self.v0 = np.array([m.body_pos[b] for b in self.vbody])
        self.cloth_kw = cloth_kw
        self.reset()

    # ---------- state
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        for a in ARMS:
            for k in range(4):
                self.d.qpos[self.qadr[a][k]] = REST_Q[k]
                self.d.ctrl[self.act[a][k]] = REST_Q[k]
            self.d.ctrl[self.act[a][5]] = OPEN
            self.d.qpos[self.qadr[a][5]] = OPEN
        self.cloth = Cloth(self.pts, self.tris, SPACING, thickness=THICK, **self.cloth_kw)
        self.held = {a: 0 for a in ARMS}
        self.sync_shirt()
        mujoco.mj_forward(self.m, self.d)
        self.area0 = None
        self.area0 = self.score()["area_cm2"]

    def get_state(self):
        st = np.empty(mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.m, self.d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
        return dict(mj=st, cloth=self.cloth.get_state(), held=dict(self.held))

    def set_state(self, s):
        mujoco.mj_setState(self.m, self.d, s["mj"], mujoco.mjtState.mjSTATE_INTEGRATION)
        self.cloth.set_state(s["cloth"])
        self.held = dict(s["held"])
        self.sync_shirt()
        mujoco.mj_forward(self.m, self.d)

    # ---------- arms
    def tip(self, a, d=None):
        """The gripper point: on the wrist-roll axis at fingertip depth. The arm aims with it, cloth is grabbed
        around it, and turning the wrist spins held cloth about it (it doesn't move when the wrist turns)."""
        dd = d or self.d
        R = dd.xmat[self.gbody[a]].reshape(3, 3)
        return dd.xpos[self.gbody[a]] + R @ np.array([0.0, 0.0, -TIP_DEPTH])

    def q(self, a):
        return np.array([self.d.qpos[k] for k in self.qadr[a][:4]])

    def roll(self, a):
        return float(self.d.qpos[self.qadr[a][4]])

    def busy(self, a):
        return False  # grabs and releases are instant here (the grasp is a pin)

    def set_arm(self, a, q4, roll=0.0, grip=None):
        for k in range(4):
            self.d.ctrl[self.act[a][k]] = q4[k]
        self.d.ctrl[self.act[a][4]] = roll
        if grip is not None:
            self.d.ctrl[self.act[a][5]] = grip

    def solve_ik(self, a, goal, q_init, roll=0.0, iters=60):
        """Tip position + gripper pointing straight down. Returns (q4, position error m, tilt deg)."""
        m, ik = self.m, self.ik_data
        idx = self.qadr[a][:4]
        lo, hi = self.jrange[a][:, 0], self.jrange[a][:, 1]
        q = np.array(q_init[:4], float)

        def resid(qq):
            # aim with the point ON the wrist-roll axis at fingertip depth (not the fingertip itself, which is
            # 8 mm off-axis): then turning the jaws never changes where the arm can reach
            ik.qpos[idx] = qq
            ik.qpos[self.qadr[a][4]] = roll
            mujoco.mj_kinematics(m, ik)
            R = ik.xmat[self.gbody[a]].reshape(3, 3)
            p = ik.xpos[self.gbody[a]] + R @ np.array([0.0, 0.0, -TIP_DEPTH])
            ax = -R[:, 2]  # approach direction (wrist -> fingertips)
            return np.concatenate([p - goal, 0.03 * (ax - np.array([0, 0, -1.0]))])

        for _ in range(iters):
            r = resid(q)
            if np.linalg.norm(r[:3]) < 5e-4 and np.linalg.norm(r[3:]) < 3e-3:
                break
            J = np.zeros((6, 4))
            for k in range(4):
                dq = np.zeros(4); dq[k] = 1e-4
                J[:, k] = (resid(q + dq) - r) / 1e-4
            step = -np.linalg.solve(J.T @ J + 1e-4 * np.eye(4), J.T @ r)
            q = np.clip(q + np.clip(step, -0.2, 0.2), lo, hi)
        r = resid(q)
        tilt = np.degrees(np.arccos(np.clip(1 - 0.5 * (np.linalg.norm(r[3:]) / 0.03) ** 2, -1, 1)))
        return q, float(np.linalg.norm(r[:3])), float(tilt)

    SEEDS = [np.array([0.0, 0.0, 0.0, 1.2]), np.array([0.0, -0.6, 0.8, 1.3]), np.array([0.0, 0.6, -0.6, 1.4])]

    def ik(self, a, goal, q, roll=0.0, seeds=True):
        """IK from the current pose; if it misses (wrong elbow branch), retry from known-good seeds."""
        best = self.solve_ik(a, goal, q, roll, iters=15)
        if seeds and best[1] > 0.003:
            for seed in self.SEEDS:
                r = self.solve_ik(a, goal, seed.copy(), roll, iters=100)
                if r[1] < best[1]:
                    best = r
        return best

    def jaw_yaw(self, a, d=None):
        """Angle (rad, world frame) of the line the jaws close along. Turns 1:1 with the wrist_roll joint."""
        R = (d or self.d).xmat[self.gbody[a]].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def roll_for_yaw(self, a, q4, yaw):
        """wrist_roll that points the jaws at world angle `yaw` for arm pose q4. Returns (roll, achieved yaw)."""
        ik = self.ik_data
        ik.qpos[self.qadr[a][:4]] = q4
        ik.qpos[self.qadr[a][4]] = 0.0
        mujoco.mj_kinematics(self.m, ik)
        y0 = self.jaw_yaw(a, ik)
        lo, hi = self.m.jnt_range[self.m.joint(f"{a}_wrist_roll").id]
        roll = float(np.clip((yaw - y0 + np.pi) % (2 * np.pi) - np.pi, lo, hi))
        return roll, y0 + roll

    # ---------- cloth
    def jaw_spheres(self):
        out = []
        for a in ARMS:
            for gid in self.pads[a]:
                out.append(np.r_[self.d.geom_xpos[gid], 0.007])
        return np.array(out)

    def grab(self, a, all_layers=False):
        gid = ARMS.index(a)
        n = self.cloth.grab(gid, self.tip(a), all_layers=all_layers, yaw=self.jaw_yaw(a))
        self.held[a] = n
        self.d.ctrl[self.act[a][5]] = CLOSED
        return n

    def release(self, a):
        self.cloth.release(ARMS.index(a))
        self.held[a] = 0
        self.d.ctrl[self.act[a][5]] = OPEN

    def step(self, dt=1 / 60):
        """Advance arms and cloth by dt (arms in 4 ms MuJoCo steps)."""
        for _ in range(max(1, int(round(dt / self.m.opt.timestep)))):
            mujoco.mj_step(self.m, self.d)
        grips = np.array([self.tip(a) for a in ARMS])
        self.cloth.step(dt, grips, self.jaw_spheres(), np.array([self.jaw_yaw(a) for a in ARMS]))
        self.sync_shirt()

    def sync_shirt(self):
        """Copy the cloth solver's particles into MuJoCo's (display-only) shirt."""
        self.d.qpos[self.vqadr[:, None] + np.arange(3)] = self.cloth.x - self.v0
        self.d.qvel[self.vdadr[:, None] + np.arange(3)] = 0.0

    def refresh(self):
        """Update positions used for drawing without stepping physics."""
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_flex(self.m, self.d)

    def surface_under(self, xy):
        return self.cloth.surface_under(np.asarray(xy, float))

    # ---------- score: how compact / rectangular / flat the shirt is
    def score(self):
        x = self.cloth.x
        res = 0.002
        lo = x[:, :2].min(0) - 0.01
        hi = x[:, :2].max(0) + 0.01
        nx, ny = (np.ceil((hi - lo) / res)).astype(int)
        occ = _raster(x, self.cloth.tris, lo[0], lo[1], res, nx, ny)
        area = occ.sum() * res * res
        jj, ii = np.nonzero(occ)
        P = np.c_[lo[0] + (ii + 0.5) * res, lo[1] + (jj + 0.5) * res]
        best = None
        for ang in np.radians(np.arange(0, 90, 2.0)):
            c, s = np.cos(ang), np.sin(ang)
            R = P @ np.array([[c, -s], [s, c]])
            ext = R.max(0) - R.min(0) + res
            if best is None or ext[0] * ext[1] < best[0]:
                best = (ext[0] * ext[1], ext)
        rect, ext = best
        out = dict(area_cm2=round(area * 1e4, 1), rect_cm=[round(float(max(ext)) * 100, 1), round(float(min(ext)) * 100, 1)],
                   rectangularity=round(float(area / rect), 3), squareness=round(float(min(ext) / max(ext)), 3),
                   height_mm=round(float(x[:, 2].max() - x[:, 2].min() + THICK) * 1000, 1))
        if self.area0:
            ratio = area * 1e4 / self.area0
            out["area_vs_flat"] = round(ratio, 3)
            # FOLD SCORE (can't be gamed by crumpling): compact (no extra credit below 25% of flat) x share of the
            # cloth lying flat (crumples are tilted everywhere) x rectangularity
            flat = flat_fraction(x, self.cloth.tris)
            vshare, vmean = valley_share(x, self.cloth.tris, THICK)
            _, air_mm, layers = inside_voids(x, self.cloth.tris, THICK)
            compact = float(np.clip((1 - ratio) / (1 - FLOOR_AREA), 0, 1))
            smooth = float(np.clip(1 - vshare / 0.5, 0, 1))
            solid = float(np.clip(1 - air_mm / AIR_ZERO_MM, 0, 1))
            out["flat"] = round(flat, 3)
            out["valleys"] = round(vshare, 3)
            out["air_mm"] = round(air_mm, 1)
            out["layers"] = round(layers, 1)
            out["fold_score"] = round(100 * compact * flat * float(area / rect) * smooth * solid, 1)
        self._occ = (occ, lo, res)
        return out

"""Attach grasp for the LeHome Isaac env: the grasp model most cloth sim-to-real work uses (FoldNet++, GarmentLab,
arXiv 2609.29340), ported from our Newton contact_grasp.py because Isaac's own particle-cloth contact cannot hold a
realistic (50 g) shirt: the driven jaw closes fully through it (jaw stop 0.05 deg short of an empty close, real
1.0-2.2 deg; measured 2026-10-19).

  capture   when a jaw closes (gripper joint below ATTACH_RAD after having been open), every cloth vertex inside the
            pad zone (grasp_log.pad_mask, the same box the grasp logger calls "pinched") is pinned to the gripper
            link: each physics step its position (and velocity) is written through the USD points / velocities
            attributes, the CPU pipeline's input path (LeHome's reset uses the same).
  release   when the jaw opens (above OPEN_RAD), or when the estimated pull on the grasp exceeds the capacity
            2 * mu_jaw * N (two pad faces): N = SO-101 squeeze (~9 N after the overload cutback), mu_jaw PLA-cotton
            0.07-0.25 (contact_params.json). The pull is estimated as in Newton: weight of the cloth off the table
            plus table friction on the cloth still on it while the jaw moves sideways (all of it assumed sliding).

Install with install(env): wraps env.sim.step (Isaac Lab calls it once per physics step) and env._reset_idx.
Events are logged as "ATTACH {json}" lines. Env: ATTACH_MU (0.15), ATTACH_N (9.0), ATTACH_MU_TABLE (0.5).
"""
import json
import os

import numpy as np

import grasp_log as gl

ATTACH_RAD = -0.10          # gripper joint (rad): pads ~5 mm apart (closed limit -0.1745); capture below this
OPEN_RAD = gl.OPEN_RAD      # 0.25: release above this (the teacher opens to ~0.5)
MIN_PINCH = gl.MIN_PINCH    # fewer vertices in the zone than this = nothing grasped
TABLE_OFF = 0.012           # m above the table's cloth level that counts as "off the table"
DRAG_V = 0.01               # m/s sideways jaw speed that counts as dragging
G = 9.81


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class AttachGrasp:
    def __init__(self, env, mu=None, N=None, mu_table=None, log=print):
        self.env, self.log = env, log
        self.mu = float(os.environ.get("ATTACH_MU", 0.15) if mu is None else mu)
        self.N = float(os.environ.get("ATTACH_N", 9.0) if N is None else N)
        self.mu_table = float(os.environ.get("ATTACH_MU_TABLE", 0.5) if mu_table is None else mu_table)
        self.capacity = 2.0 * self.mu * self.N
        self.dt = float(getattr(env, "physics_dt", 1.0 / 90.0))
        self.arms = {"left": env.left_arm, "right": env.right_arm}
        self.gidx = {a: int(arm.find_joints("gripper")[0][0]) for a, arm in self.arms.items()}
        self.t = 0
        self.faces = None
        self.gravity_scale = None
        self.reset_state()

    # ---- state ------------------------------------------------------------------------------------------------
    def reset_state(self):
        self.pins = {a: None for a in self.arms}        # per arm: dict(idx, local (n,3) in the link frame, prev_w)
        self.armed = {a: False for a in self.arms}      # the jaw has been seen open since the last capture/reset
        self.table_z = None
        self.events = []

    def _q(self, a):
        jp = _np(self.arms[a].data.joint_pos)[0]
        return jp[:5].astype(float), float(jp[self.gidx[a]])

    def _T_link(self, a, q5):
        import fold_plans as fp
        return fp.K.fk(q5, a) @ np.linalg.inv(gl.T_LINK_TIP)

    def _cloth(self):
        """(points_local float64 (n,3), A (3,3), pos (3,)): world = local @ A + pos, read live (the prim's pose follows
        the cloth's bounding-box centre every write-back)."""
        o = self.env.object
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        pos, ori = o.get_world_pose()
        sc = o.get_world_scale()
        A = (quat_to_rot_matrix(_np(ori)) * _np(sc)[None, :]).T
        pts = np.array(o._prim.GetAttribute("points").Get(), dtype=np.float64)
        return pts, np.ascontiguousarray(A, dtype=np.float64), np.asarray(_np(pos), np.float64)

    def _mass_per_point(self):
        try:
            return float(self.env.particle_config.objects["garment_config"]["particle_mass"])
        except Exception:
            return 3.1e-6

    def _gravity_scale(self):
        if self.gravity_scale is not None:            # set by the caller (oracle_fold.slip_ramp) when it scales gravity
            return self.gravity_scale
        try:
            return float(self.env.object.particle_material.get_gravity_scale())
        except Exception:
            return 1.0

    # ---- per step -------------------------------------------------------------------------------------------
    def before_step(self):
        """Write the pinned vertices to where the jaws are now (positions and velocities)."""
        if not any(p is not None for p in self.pins.values()):
            return
        from pxr import Vt
        pts, A, pos = self._cloth()
        Ainv = np.linalg.inv(A)
        vel_attr = self.env.object._prim.GetAttribute("velocities")
        vel = np.array(vel_attr.Get(), dtype=np.float64) if vel_attr and vel_attr.HasValue() else None
        for a, p in self.pins.items():
            if p is None:
                continue
            q5, _ = self._q(a)
            T = self._T_link(a, q5)
            w = (T @ np.c_[p["local"], np.ones(len(p["local"]))].T).T[:, :3]
            loc = (w - pos) @ Ainv
            if vel is not None:
                vel[p["idx"]] = (loc - pts[p["idx"]]) / self.dt
            pts[p["idx"]] = loc
            p["prev_w"], p["target_w"] = p.get("target_w", w), w
        self.env.object._prim.GetAttribute("points").Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        if vel is not None:
            vel_attr.Set(Vt.Vec3fArray.FromNumpy(vel.astype(np.float32)))

    def after_step(self):
        """Capture / release decisions from the jaw state and the estimated load."""
        self.t += 1
        pts = A = pos = None
        for a in self.arms:
            q5, g = self._q(a)
            p = self.pins[a]
            if p is None:
                if g > OPEN_RAD:
                    self.armed[a] = True
                elif g < ATTACH_RAD and self.armed[a]:
                    if pts is None:
                        pts, A, pos = self._cloth()
                    self._capture(a, q5, g, pts @ A + pos)
                continue
            if g > OPEN_RAD:
                self._release(a, "open")
                continue
            if pts is None:
                pts, A, pos = self._cloth()
            load = self._load(a, p, pts @ A + pos)
            p["last_load"] = load
            p["max_load"] = max(p.get("max_load", 0.0), load)
            if load > self.capacity:
                self._release(a, "slip", load=load)

    def _capture(self, a, q5, g, w):
        import fold_plans as fp
        self.armed[a] = False
        if self.faces is None:
            prim = self.env.object._prim
            f = np.asarray(prim.GetAttribute("faceVertexIndices").Get(), dtype=np.int64)
            c = np.asarray(prim.GetAttribute("faceVertexCounts").Get())
            self.faces = f.reshape(-1, 3) if (c == 3).all() else None
        if self.table_z is None:
            self.table_z = float(np.percentile(w[:, 2], 1))
        m = gl.pad_mask(fp.K.fk(q5, a), w)
        idx = np.nonzero(m)[0]
        ev = dict(t=self.t, arm=a, event="close", jaw_deg=round(float(np.degrees(g)), 2), n=int(len(idx)))
        if len(idx) < MIN_PINCH:
            ev["result"] = "miss"
            self.events.append(ev); self.log("ATTACH " + json.dumps(ev))
            return
        T = self._T_link(a, q5)
        local = (np.linalg.inv(T) @ np.c_[w[idx], np.ones(len(idx))].T).T[:, :3]
        self.pins[a] = dict(idx=idx, local=local, t0=self.t)
        ev.update(result="attached", layers=gl.layers(idx, self.faces)[0] if self.faces is not None else None,
                  capacity_N=round(self.capacity, 3))
        self.events.append(ev); self.log("ATTACH " + json.dumps(ev))

    def _load(self, a, p, w):
        """Newton's estimate: weight of the cloth off the table + table friction on the rest while the jaw drags."""
        m = self._mass_per_point() * self._gravity_scale()
        others = np.ones(len(w), bool); others[p["idx"]] = False
        off = w[:, 2] > self.table_z + TABLE_OFF
        hanging = m * G * int((off & others).sum())
        v = (p["target_w"].mean(0) - p["prev_w"].mean(0)) / self.dt if "target_w" in p and "prev_w" in p else np.zeros(3)
        friction = self.mu_table * m * G * int((~off & others).sum()) if np.linalg.norm(v[:2]) > DRAG_V else 0.0
        return float(np.hypot(hanging, friction))

    def _release(self, a, why, load=None):
        p = self.pins[a]
        ev = dict(t=self.t, arm=a, event="release", why=why, n=int(len(p["idx"])), held_steps=self.t - p["t0"],
                  max_load_N=round(p.get("max_load", 0.0), 4), capacity_N=round(self.capacity, 3))
        if load is not None:
            ev["load_N"] = round(load, 4)
        self.pins[a] = None
        self.events.append(ev); self.log("ATTACH " + json.dumps(ev))

    def status(self):
        return {a: None if p is None else dict(n=int(len(p["idx"])), held_steps=self.t - p["t0"],
                                               last_load_N=round(p.get("last_load", 0.0), 4))
                for a, p in self.pins.items()}


def install(env, log=print, **kw):
    """Wrap env.sim.step (one call per physics step in Isaac Lab's DirectRLEnv.step and in oracle_fold.Stepper) and
    env._reset_idx (clears the pins). Returns the AttachGrasp; also stored as env._attach."""
    ag = AttachGrasp(env, log=log, **kw)
    sim_step = env.sim.step

    def _fail(where, e):
        log(f"ATTACH {where} failed: {e!r}")
        if not getattr(ag, "_traced", False):        # the full traceback once
            import traceback
            ag._traced = True
            traceback.print_exc()

    def step(*a, **k):
        try:
            ag.before_step()
        except Exception as e:
            _fail("before_step", e)
        out = sim_step(*a, **k)
        try:
            ag.after_step()
        except Exception as e:
            _fail("after_step", e)
        return out

    env.sim.step = step
    reset_idx = env._reset_idx

    def _reset_idx(*a, **k):
        ag.reset_state()
        return reset_idx(*a, **k)

    env._reset_idx = _reset_idx
    env._attach = ag
    log(f"ATTACH installed: mu {ag.mu} N {ag.N} capacity {ag.capacity:.2f} N, mu_table {ag.mu_table}")
    return ag

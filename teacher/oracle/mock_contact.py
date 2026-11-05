"""Dry-run contact_check against a fake env (static cloth) to catch code errors without Isaac, plus a geometry check
of PadMonitor: a vertex between the pads counts as gap, one at a pad centre as inside, one moved behind a pad as crossed."""
import os, sys, json, tempfile
import numpy as np
src = open("mock_test.py").read().split("out = tempfile.mkdtemp()")[0]
exec(src)                                  # defines of (oracle_fold module), Env, Obj, d, name, W
import isaac_physics as ip
import jaw_geometry as jg
import pads

# ---- PadMonitor geometry --------------------------------------------------------------------------------------------
q = np.array([0.0, -0.5, 0.6, 1.2, 0.0]); g = np.radians(-4.0)
T_link = jg.K.fk(q, "left") @ np.linalg.inv(jg.T_LINK_TIP)
Tf = T_link @ jg.T_G_FIXEDPAD
Tm = T_link @ pads.jaw_in_gripper(jg._st, np.degrees(g)) @ jg.T_J_MOVINGPAD
cf, cm = Tf[:3, 3], Tm[:3, 3]
nf = Tf[:3, 0] / np.linalg.norm(Tf[:3, 0])
mid = (cf + cm) / 2
print("pad centres %.1f mm apart" % (np.linalg.norm(cf - cm) * 1000))
pts = {"v": np.array([mid, mid + 0.001 * Tf[:3, 1] / np.linalg.norm(Tf[:3, 1]), cf, [9, 9, 9.0]])}


class FS:
    t = 0
    def state(s):
        return {"left": q, "right": q}, {"left": g, "right": 1.0}


real_cv = of.cloth_verts
of.cloth_verts = lambda env: pts["v"]
m = of.PadMonitor(None, FS(), every=1)
m(); r1 = m.result(); print("between/inside", r1)
assert r1["pt_gap_seen"] == 2 and r1["pt_inside_max"] == 1 and r1["pt_crossed"] == 0
pts["v"] = pts["v"].copy(); pts["v"][0] = cf - nf * 0.006            # 6 mm behind the fixed pad's centre
m(); r2 = m.result(); print("after moving one behind the fixed pad", r2)
assert r2["pt_crossed"] == 1
of.cloth_verts = real_cv
print("PADMON_OK")

# ---- full contact_check on the fake env -----------------------------------------------------------------------------
class PC:
    def __init__(self):
        self.objects = {"garment_config": {}, "particle_material": {}, "particle_system": {}}
        for k, v in ip.values("orig").items():
            self.objects[ip.SECTION[k]][k] = v


class Mat:
    def __init__(s): s.g = []
    def set_gravity_scale(s, v): s.g.append(v)


class Data:
    applied_torque = np.zeros((1, 6))


class Arm:
    data = Data()
    def find_joints(s, n): return [5], ["gripper"]


class P2(Obj._P):
    def GetAttribute(s, n):
        if n == "points":
            class A:
                def Get(s2): return W
            return A()
        return super().GetAttribute(n)


Obj._prim = P2()
Obj.particle_material = Mat()


class Cfg: garment_name = name


e = Env(); e.cfg = Cfg(); e.particle_config = PC(); e.switch_garment = lambda n: None
e.left_arm = e.right_arm = Arm()
out = tempfile.mkdtemp()
# attach grasp model on the fake env: hooks on env.sim.step (the mock's env.step calls it) and env._reset_idx
import types
rot = types.ModuleType("isaacsim.core.utils.rotations")


def _q2m(q):                                   # (w, x, y, z) -> rotation matrix, numpy only (the fake torch breaks scipy)
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


rot.quat_to_rot_matrix = _q2m
for nm in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
    sys.modules.setdefault(nm, types.ModuleType(nm))
sys.modules["isaacsim.core.utils.rotations"] = rot
Arm.data.joint_pos = np.zeros((1, 6))
e.sim = types.SimpleNamespace(step=lambda render=False: None)
e._reset_idx = lambda ids=None: None
_orig_step = e.step
e.step = lambda a: (_orig_step(a), setattr(Arm.data, "joint_pos", np.asarray(e.q, np.float64).reshape(1, 12)[:, :6].copy()), e.sim.step(render=False))[0]
Obj.get_world_pose = lambda s: (np.zeros(3), np.array([1.0, 0, 0, 0]))
Obj.get_world_scale = lambda s: np.ones(3)
Obj.transform_points = lambda s, pts, pos, ori, sc: pts * sc + pos
class _VA:
    v = np.zeros_like(W, dtype=np.float32)
    def Get(s): return s.v
    def Set(s, x): s.v = np.array(x)
    def HasValue(s): return True
_P2_get = P2.GetAttribute
P2.GetAttribute = lambda s, n: _VA() if n == "velocities" else _P2_get(s, n)
import attach_grasp
ag = attach_grasp.install(e, log=of.log)
rows = of.contact_check(e, None, {"garment": name, "mode": "contact_check", "reps": 1, "variants": ["v2"]}, name, out)
print("ROWS", [(r["check"], r.get("error"), r.get("n_pinched"), sorted(r)[:6]) for r in rows])
print("gravity scales set:", e.object.particle_material.g)
assert os.path.exists(os.path.join(out, "contact_check.json"))
assert e.object.particle_material.g and e.object.particle_material.g[-1] == 1.0
print("attach events seen:", sum(len(r.get("attach_events", [])) for r in rows))
print("CONTACT_MOCK_OK")

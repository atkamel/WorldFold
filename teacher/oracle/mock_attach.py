"""Dry-run attach_grasp on a fake env (static flat cloth, a jaw driven through open -> close -> lift -> open, then a
close + overload), without Isaac: capture, follow, release on open, release on slip, reset clears."""
import json
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation

# isaacsim stub: only quat_to_rot_matrix (w, x, y, z) is used
rot = types.ModuleType("isaacsim.core.utils.rotations")
rot.quat_to_rot_matrix = lambda q: Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["isaacsim.core.utils.rotations"] = rot

import attach_grasp as ag_mod
import fold_plans as fp
import grasp_log as gl

TABLE = 0.5245
n = 61
xs = -0.2288 + 0.005 * (np.arange(n) - n // 2); ys = -0.0381 + 0.005 * (np.arange(n) - n // 2)
X, Y = np.meshgrid(xs, ys, indexing="ij")
pts = np.stack([X.ravel(), Y.ravel(), np.full(n * n, TABLE)], 1).astype(np.float32)
quad = np.array([[i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j, (i + 1) * n + j + 1, i * n + j + 1]
                 for i in range(n - 1) for j in range(n - 1)]).reshape(-1, 3)


class Attr:
    def __init__(s, v): s.v = v
    def Get(s): return s.v
    def Set(s, v): s.v = np.array(v)
    def HasValue(s): return True


class Prim:
    def __init__(s):
        s.a = {"points": Attr(pts.copy()), "velocities": Attr(np.zeros_like(pts)),
               "faceVertexIndices": Attr(quad.reshape(-1)), "faceVertexCounts": Attr(np.full(len(quad), 3))}
    def GetAttribute(s, k): return s.a[k]


class Mat:
    def get_gravity_scale(s): return 1.0


class Obj:
    _prim = Prim(); particle_material = Mat()
    def get_world_pose(s): return np.zeros(3), np.array([1.0, 0, 0, 0])
    def get_world_scale(s): return np.ones(3)


class Data: joint_pos = np.zeros((1, 6))


class Arm:
    def __init__(s): s.data = Data(); s.data.joint_pos = np.zeros((1, 6))
    def find_joints(s, nm): return [5], ["gripper"]


class Sim:
    def __init__(s): s.n = 0
    def step(s, render=False): s.n += 1


class PC: objects = {"garment_config": {"particle_mass": 3.1e-6}}


class Env:
    physics_dt = 1 / 90
    def __init__(s):
        s.left_arm, s.right_arm, s.object, s.sim, s.particle_config = Arm(), Arm(), Obj(), Sim(), PC()
        s.resets = 0
    def _reset_idx(s, ids=None): s.resets += 1


env = Env()
logs = []
ag = ag_mod.install(env, log=lambda m: logs.append(m) or print(m), mu=0.15, N=9.0)
q_home = np.concatenate([fp.HOME["left"], [fp.GRIP_CLOSED]]) if hasattr(fp, "HOME") else None
from so101_kin import HOME, GRIP_CLOSED, GRIP_OPEN
roll = None
q, err = fp._solve("left", (-0.2288, -0.0381, 0.527), fp.SEED["left"])
assert err < 1e-3, err


def set_arm(q5, g):
    env.left_arm.data.joint_pos[0, :5] = q5; env.left_arm.data.joint_pos[0, 5] = g
    env.right_arm.data.joint_pos[0, :5] = HOME["right"]; env.right_arm.data.joint_pos[0, 5] = GRIP_CLOSED


set_arm(q, GRIP_OPEN); env.sim.step()                    # open above the cloth: arms
set_arm(q, GRIP_CLOSED); env.sim.step()                  # close: capture
assert ag.pins["left"] is not None, logs[-1]
idx = ag.pins["left"]["idx"]; n_pin = len(idx)
assert n_pin >= gl.MIN_PINCH and json.loads(logs[-1].split(" ", 1)[1])["result"] == "attached", logs[-1]
z0 = env.object._prim.a["points"].v[idx, 2].mean()
tip0 = fp.K.tip(q, "left")[0]
for k in range(1, 81):                                   # lift 8 cm, 1 mm per step
    qk, e = fp._solve("left", (tip0[0], tip0[1], tip0[2] + 0.001 * k), q); q = qk
    set_arm(q, GRIP_CLOSED); env.sim.step()
z1 = env.object._prim.a["points"].v[idx, 2].mean()
tip1 = fp.K.tip(q, "left")[0]
assert abs((z1 - z0) - (tip1[2] - tip0[2])) < 0.002, (z1 - z0, tip1[2] - tip0[2])
v = env.object._prim.a["velocities"].v[idx]
assert abs(v[:, 2].mean() - 0.001 / env.physics_dt) < 0.02, v[:, 2].mean()     # ~0.09 m/s upward
others = np.setdiff1d(np.arange(n * n), idx)
assert np.all(env.object._prim.a["points"].v[others, 2] == np.float32(TABLE))  # the rest untouched (no physics here)
set_arm(q, GRIP_OPEN); env.sim.step()
assert ag.pins["left"] is None and json.loads(logs[-1].split(" ", 1)[1])["why"] == "open"
# slip: back down to the table, close again, then pretend the whole cloth hangs under 100x gravity
q, _ = fp._solve("left", (tip0[0], tip0[1] + 0.03, tip0[2]), q)      # 3 cm over: the lifted vertices stay in the air here
set_arm(q, GRIP_OPEN); env.sim.step()
set_arm(q, GRIP_CLOSED); env.sim.step()
assert ag.pins["left"] is not None
p = env.object._prim.a["points"].v; p[others, 2] = TABLE + 0.05; env.object._prim.a["points"].v = p
ag.gravity_scale = 100.0
env.sim.step()
ev = json.loads(logs[-1].split(" ", 1)[1])
assert ag.pins["left"] is None and ev["why"] == "slip" and ev["load_N"] > ag.capacity, ev
# reset clears
set_arm(q, GRIP_OPEN); env.sim.step(); set_arm(q, GRIP_CLOSED); env.sim.step(); assert ag.pins["left"] is not None
env._reset_idx(None); assert env.resets == 1 and ag.pins["left"] is None and not ag.armed["left"]
print("ATTACH_MOCK_OK pinned", n_pin, "lift follow err mm", round(abs((z1 - z0) - (tip1[2] - tip0[2])) * 1000, 2),
      "steps", env.sim.n)

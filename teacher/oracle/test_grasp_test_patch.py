"""Offline check of grasp_test_patch.py (on top of grasp_log_patch.py) with stub torch / pxr / isaacsim and a fake env:
harvest writes snapshots with the logger's episode tag; restore loads the next one, applies shift + rotation, calls
his restore_physics_state (points, velocities, pose, arm targets) and settles; select_pregrasp.py pairs them.

    python test_grasp_test_patch.py <path to lehome_solution checkout (his_repo)>
"""
import importlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parents[3] / "his_repo")
sys.path.insert(0, str(HERE))


class _T:
    def __init__(self, a): self.a = np.asarray(a, dtype=np.float32)
    def float(self): return self
    def to(self, *_, **__): return self
    def unsqueeze(self, _): return _T(self.a[None])
    def item(self): return float(self.a.ravel()[0])
    def clone(self): return _T(self.a.copy())
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a
    def reshape(self, *s): return _T(self.a.reshape(*s))
    def view(self, *s): return _T(self.a.reshape(*s))
    def __len__(self): return len(self.a)
    def __getitem__(self, i): return _T(self.a[i])
    def _b(self, o): return o.a if isinstance(o, _T) else o
    def __mul__(self, o): return _T(self.a * self._b(o))
    __rmul__ = __mul__
    def __add__(self, o): return _T(self.a + self._b(o))
    __radd__ = __add__
    def __sub__(self, o): return _T(self.a - self._b(o))
    @property
    def shape(self): return self.a.shape
    @property
    def device(self): return "cpu"


torch = types.ModuleType("torch")
torch.Tensor = _T
torch.from_numpy = lambda a: _T(a)
torch.zeros = lambda *s, **k: _T(np.zeros(s))
torch.tensor = lambda a, **k: _T(np.asarray(a))
torch.float32 = np.float32
sys.modules["torch"] = torch
for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
    sys.modules[name] = types.ModuleType(name)
rot = types.ModuleType("isaacsim.core.utils.rotations")
rot.quat_to_rot_matrix = lambda q: np.eye(3)
sys.modules["isaacsim.core.utils.rotations"] = rot
pxr = types.ModuleType("pxr"); Vt = types.ModuleType("pxr.Vt")
Vt.Vec3fArray = types.SimpleNamespace(FromNumpy=lambda a: np.asarray(a, np.float32))
pxr.Vt = Vt; sys.modules["pxr"] = pxr; sys.modules["pxr.Vt"] = Vt

tmp = pathlib.Path(tempfile.mkdtemp())
pkg = tmp / "lehome-challenge/scripts/utils"
pkg.mkdir(parents=True)
shutil.copy(SRC / "lehome-challenge/scripts/utils/remote_loop.py", pkg / "remote_loop.py")
(tmp / "lehome-challenge/scripts/__init__.py").write_text("")
(pkg / "__init__.py").write_text("")
(pkg / "common.py").write_text("def stabilize_garment_after_reset(*a, **k):\n    pass\n")
for patch in ("grasp_log_patch.py", "grasp_test_patch.py", "grasp_test_patch2.py"):
    r = subprocess.run([sys.executable, str(HERE.parent / patch), str(tmp)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-300:])
    assert r.returncode == 0, patch
sys.path.insert(0, str(tmp / "lehome-challenge"))
rl = importlib.import_module("scripts.utils.remote_loop")

# ---- fake env ---------------------------------------------------------------------------------------------------
N = 200
pts0 = np.random.rand(N, 3).astype(np.float32)


class Attr:
    def __init__(self, v): self.v = v
    def Get(self): return self.v
    def Set(self, v): self.v = np.asarray(v)
    def HasValue(self): return True


class Prim:
    def __init__(self): self.a = {"points": Attr(pts0.copy()), "velocities": Attr(np.zeros((N, 3), np.float32))}
    def GetAttribute(self, k): return self.a[k]


class Obj:
    def __init__(self): self._prim = Prim(); self.pos = np.array([0.1, 0.2, 0.6], np.float32); self.ori = np.array([1, 0, 0, 0], np.float32)
    def get_world_pose(self): return self.pos.copy(), self.ori.copy()
    def set_world_pose(self, position=None, orientation=None):
        self.pos, self.ori = np.asarray(position, np.float32), np.asarray(orientation, np.float32)


class Arm:
    def __init__(self, q):
        self.data = types.SimpleNamespace(joint_pos=_T(np.array([q])), joint_vel=_T(np.zeros((1, 6))))
        self.targets = []
    def set_joint_position_target(self, t): self.targets.append(np.asarray(t.a).ravel())
    def write_joint_state_to_sim(self, *a, **k): self.targets.append("state")
    def write_joint_position_to_sim(self, *a, **k): pass
    def write_joint_velocity_to_sim(self, *a, **k): pass


class Env:
    def __init__(self):
        self.object = Obj(); self.cfg = types.SimpleNamespace(garment_name="Top_Long_Seen_0")
        self.left_arm, self.right_arm = Arm([0.1, 0.2, 0.3, 0.4, 0.5, 0.55]), Arm([-0.1, 0.2, 0.3, 0.4, 0.5, 0.55])
        self.sim = types.SimpleNamespace(step=lambda render=False: None, render=lambda: None)


# ---- harvest ----------------------------------------------------------------------------------------------------
snapdir = tmp / "harvest" / "snapshots"
os.environ["TEACHER_SNAPSHOT_EVERY"] = "5"; os.environ["TEACHER_SNAPSHOT_DIR"] = str(snapdir)
env = Env()
lg = types.SimpleNamespace(_name="Top_Long_Seen_0", _tag=123)
for t in range(1, 31):
    rl._gt_snapshot(env, lg, t)
files = sorted((snapdir / "Top_Long_Seen_0").glob("*.npz"))
assert [f.name for f in files] == [f"Top_Long_Seen_0_123_t{t:04d}.npz" for t in (5, 10, 15, 20, 25, 30)], files
z = np.load(files[0]); assert set(z.files) >= {"garment_points", "garment_velocities", "garment_pos", "garment_ori", "left_joint_pos", "right_joint_pos", "t"}
assert np.allclose(z["garment_points"], pts0) and z["left_joint_pos"].shape == (6,)

# ---- select ---------------------------------------------------------------------------------------------------
gdir = tmp / "harvest" / "grasps"; gdir.mkdir()
recs = [dict(episode="Top_Long_Seen_0", side="left", t_close=26, t_open=40, end="opened", outcome="slipped", n_pinch=30, layers=2, max_lift_cm=7.0),
        dict(episode="Top_Long_Seen_0", side="right", t_close=60, t_open=70, end="episode_end", outcome="empty", n_pinch=0, layers=0)]
(gdir / "Top_Long_Seen_0_123.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
seldir = tmp / "selected"
r = subprocess.run([sys.executable, str(HERE.parent / "select_pregrasp.py"), str(tmp / "harvest"), str(seldir)], capture_output=True, text=True)
print(r.stdout.strip())
man = json.load(open(seldir / "manifest.json"))
assert len(man) == 1 and man[0]["t"] == 15 and man[0]["t_close"] == 26, man      # window [6, 18] before the close, nearest to 10

# ---- restore, shifted + rotated -----------------------------------------------------------------------------------
os.environ["TEACHER_RESTORE_DIR"] = str(seldir); os.environ["TEACHER_RESTORE_SHIFT_CM"] = "3"; os.environ["TEACHER_RESTORE_ROT_DEG"] = "15"
env2 = Env(); env2.object.pos[:] = 0; env2.object._prim.a["points"].v[:] = 0
info = rl._gt_restore(env2, "Top_Long_Seen_0", "test")
assert info and info["ok"] and info["t"] == 15 and abs(info["shift_cm"] - 3) < 1e-6 and abs(abs(info["rot_deg"]) - 15) < 1e-6, info
assert abs(np.linalg.norm(env2.object.pos[:2] - np.array([0.1, 0.2])) - 0.03) < 1e-4, env2.object.pos   # shifted 3 cm
assert abs(2 * np.degrees(np.arccos(abs(env2.object.ori[0]))) - 15) < 1e-3, env2.object.ori                # rotated 15 deg
assert np.allclose(env2.object._prim.a["points"].v, pts0)                                              # points restored
assert env2.left_arm.targets and np.allclose(env2.left_arm.targets[0][:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.55])
info2 = rl._gt_restore(env2, "Top_Long_Seen_0", "test")
assert info2["k"] == 1 and info2["t"] == 15                                                             # cycles the list
# ---- variant 2: move the cloth vertices at step 5 (rigid: rotate about the garment centre, then shift) -----------
os.environ["TEACHER_MOVE_AT_STEP"] = "5"; os.environ["TEACHER_MOVE_SHIFT_CM"] = "3"; os.environ["TEACHER_MOVE_ROT_DEG"] = "15"
env3 = Env(); env3.object.pos[:] = [0.1, 0.2, 0.6]
class Obj3(Obj):
    def get_world_scale(self): return np.array([0.45, 0.45, 0.45], np.float32)
env3.object = Obj3(); env3.object.pos[:] = [0.1, 0.2, 0.6]
before = env3.object._prim.a["points"].v.copy()
rl._gt_move(env3, 4, "test"); assert np.allclose(env3.object._prim.a["points"].v, before)          # not yet
rl._gt_move(env3, 5, "test"); after = env3.object._prim.a["points"].v
v0, v1 = before[1] - before[0], after[1] - after[0]                                               # pure yaw: pair vectors rotate by it
ang = float(np.degrees(np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], v0[0] * v1[0] + v0[1] * v1[1])))
assert abs(abs(ang) - 15) < 1e-3, ang
c, s_ = np.cos(np.radians(-ang)), np.sin(np.radians(-ang))
unrot = after @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]).T                                  # undo the yaw about the origin
d = (unrot - before).mean(0)
assert np.allclose(unrot - d, before, atol=1e-5)                                                      # rigid: rotation + one shift
assert abs(np.linalg.norm(d[:2]) * 0.45 - 0.03) < 1e-4 and abs(d[2]) < 1e-6, d * 0.45                 # shifted 3 cm in the world, flat
assert np.allclose(env3.object.pos, [0.1, 0.2, 0.6])                                                 # pose untouched
print("GRASP_TEST_PATCH2_MOCK_OK yaw", round(float(ang), 2))
print("GRASP_TEST_PATCH_MOCK_OK")

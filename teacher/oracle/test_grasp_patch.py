"""Offline check of grasp_log_patch.py: patch a copy of his remote_loop.py and run its real _run_steps with a fake
env, a fake policy and stub torch/isaacsim modules (no Isaac, no GPU). The fake left arm closes on a cloth patch,
lifts it 5 cm and opens.

    python test_grasp_patch.py <path to lehome_solution checkout (his_repo)>
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
import grasp_log as gl          # noqa: E402
import fold_plans as fp         # noqa: E402

# ---- stub modules -------------------------------------------------------------------------------------------
class _T:
    def __init__(self, a): self.a = np.asarray(a, dtype=np.float32)
    def float(self): return self
    def to(self, *_): return self
    def unsqueeze(self, _): return _T(self.a[None])
    def item(self): return float(self.a.ravel()[0])
    def clone(self): return _T(self.a.copy())


torch = types.ModuleType("torch")
torch.Tensor = _T
torch.from_numpy = lambda a: _T(a)
torch.zeros = lambda *s, **k: _T(np.zeros(s))
sys.modules["torch"] = torch
for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
    sys.modules[name] = types.ModuleType(name)
rot = types.ModuleType("isaacsim.core.utils.rotations")
rot.quat_to_rot_matrix = lambda q: np.eye(3)
sys.modules["isaacsim.core.utils.rotations"] = rot

# ---- patched copy of his loop, imported as a package module -------------------------------------------------
tmp = pathlib.Path(tempfile.mkdtemp())
pkg = tmp / "lehome-challenge/scripts/utils"
pkg.mkdir(parents=True)
shutil.copy(SRC / "lehome-challenge/scripts/utils/remote_loop.py", pkg / "remote_loop.py")
(tmp / "lehome-challenge/scripts/__init__.py").write_text("")
(pkg / "__init__.py").write_text("")
(pkg / "common.py").write_text("def stabilize_garment_after_reset(*a, **k):\n    pass\n")
for patch in ("verts_dump_patch.py", "grasp_log_patch.py"):
    print(subprocess.run([sys.executable, str(HERE.parent / patch), str(tmp)], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, str(tmp / "lehome-challenge"))
rl = importlib.import_module("scripts.utils.remote_loop")

# ---- fake world: a cloth patch placed between the left pads at the grasp pose ---------------------------------
q_grasp = fp.SEED["left"].copy()
T_tip = fp.K.fk(q_grasp, "left")
c = np.array([(gl.BOX["x_lo"] + gl.BOX["x_hi"]) / 2, 0.0, (gl.BOX["z_lo"] + gl.BOX["z_hi"]) / 2, 1.0])
center = (T_tip @ np.linalg.inv(gl.T_LINK_TIP) @ c)[:3]
n, h = 4, 0.004
xs = (np.arange(n) - (n - 1) / 2) * h
patch = np.array([[center[0] + x, center[1] + y, center[2]] for y in xs for x in xs])
table = np.array([[0.3 + i * 0.01, 0.3, center[2] - 0.001] for i in range(50)])      # far-away cloth on the table
faces = [[j * n + i, j * n + i + 1, (j + 1) * n + i] for j in range(n - 1) for i in range(n - 1)]
faces += [[j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i] for j in range(n - 1) for i in range(n - 1)]
faces = np.array(faces)


class Attr:
    def __init__(self, f): self.f = f
    def Get(self): return self.f()


class World:
    def __init__(self):
        self.t, self.lift = 0, 0.0
        self.attrs = {"points": lambda: np.r_[patch + [0, 0, self.lift], table],
                      "faceVertexIndices": lambda: faces.ravel(), "faceVertexCounts": lambda: np.full(len(faces), 3)}
        self._prim = types.SimpleNamespace(GetAttribute=lambda k: Attr(self.attrs[k]))
        self.get_world_pose = lambda: (np.zeros(3), np.array([1.0, 0, 0, 0]))
        self.get_world_scale = lambda: np.ones(3)
        self.get_current_mesh_points = lambda: [self.attrs["points"]()]


class Env:
    def __init__(self):
        self.object = World()
        self.cfg = types.SimpleNamespace(garment_name="Fake_Shirt")
        self.reward_buf = 0.0
        self.actions = _T(np.zeros((1, 12)))
        self.obs_buf = self._get_observations()

    def grip(self, t):                      # open 3 steps, closed 25 (lift from step 12), open after
        return 0.5 if t < 3 or t >= 28 else -0.07

    def _get_observations(self):
        t = self.object.t
        s = np.zeros(12); s[0:5] = q_grasp; s[5] = self.grip(t); s[6:11] = fp.SEED["right"]; s[11] = 0.5
        return {"observation.state": s}

    def step(self, action):
        o = self.object
        o.t += 1
        o.lift = min(max(o.t - 12, 0) * 0.005, 0.05) if self.grip(o.t) < 0 else 0.0
        self.obs_buf = self._get_observations()

    def _get_success(self):
        return _T([0.0])


class Policy:
    _garment_name = "Fake_Shirt"
    def serialize_observation(self, obs): return {}
    def post(self, path, body): return {"actions": np.zeros(12).tolist(), "n_steps": 1}


# the fake tip frame must move with the lifted cloth: lift the arm's tip by shifting FK in z while closed
env = Env()
_fk = fp.K.fk
fp.K.fk = lambda q, side: (lambda M: (M.__setitem__((2, 3), M[2, 3] + env.object.lift), M)[1])(_fk(q, side).copy())

out = pathlib.Path(tempfile.mkdtemp())
os.environ["TEACHER_GRASP_LOG"] = str(out)
os.environ["ORACLE_DIR"] = str(HERE)
args = types.SimpleNamespace(max_steps=40, device="cpu")
res = rl._run_steps(env, Policy(), args, types.SimpleNamespace(randomize_step_color=None, randomize_step_light=None),
                    "test", stop_on_success=True)
files = list(out.glob("*.jsonl"))
assert len(files) == 1, files
recs = [json.loads(l) for l in files[0].read_text().splitlines()]
print("records:", recs)
assert len(recs) == 1 and recs[0]["outcome"] == "held_lifted" and recs[0]["layers"] == 1 and recs[0]["side"] == "left", recs
assert recs[0]["n_pinch"] == n * n, recs
print("episode:", res)
print("GRASP PATCH TEST PASSED")

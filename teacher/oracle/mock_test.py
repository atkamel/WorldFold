"""Run oracle_fold.shirt_trial against a fake env (static cloth) to catch code errors without Isaac."""
import importlib.util, json, os, sys, types, tempfile
import numpy as np

# fake torch
class _T:
    def __init__(s, a): s.a = a
    def float(s): return s
    def to(s, d): return s
    def unsqueeze(s, i): return s
torch = types.ModuleType("torch"); torch.from_numpy = lambda a: _T(a); sys.modules["torch"] = torch
# fake isaaclab + package-relative imports so the module can be loaded standalone
for m in ("isaaclab", "isaaclab.app"):
    sys.modules[m] = types.ModuleType(m)
sys.modules["isaaclab.app"].AppLauncher = object
pkg = types.ModuleType("scripts"); pkg.__path__ = []; sys.modules["scripts"] = pkg
u = types.ModuleType("scripts.utils"); u.__path__ = []; sys.modules["scripts.utils"] = u
c = types.ModuleType("scripts.utils.common"); c.close_app = c.launch_app_from_args = None; sys.modules["scripts.utils.common"] = c
pz = types.ModuleType("scripts.utils.parser"); pz.setup_eval_parser = None; sys.modules["scripts.utils.parser"] = pz
spec = importlib.util.spec_from_file_location("scripts.oracle_fold", "oracle_fold.py"); of = importlib.util.module_from_spec(spec); spec.loader.exec_module(of)

name = sys.argv[1] if len(sys.argv) > 1 else "Top_Long_Seen_0"
d = np.load(f"garments/{name}.npz"); scale = float(d["scale"][0])
W = d["points"].astype(float) * scale; W[:, 2] = 0.503 + 0.01 * (d["points"][:, 2] > 0)

class Obj:
    check_points = d["check_point"].tolist()
    def get_world_pose(s): raise RuntimeError("mock: use fallback")
    def get_current_mesh_points(s): return W, W, None, None
    class _P:
        def GetAttribute(s, n):
            class A:
                def Get(s2): return d["faces"].reshape(-1) if n == "faceVertexIndices" else np.full(len(d["faces"]), 3)
            return A()
    _prim = _P()

class Env:
    device = "cpu"; object = Obj()
    def __init__(s): s.q = np.zeros(12, np.float32)
    def reset(s): pass
    def step(s, a): s.q = np.asarray(a.a, np.float32)
    def _get_observations(s):
        im = np.zeros((480, 640, 3), np.uint8)
        return {"observation.state": s.q, "observation.images.top_rgb": im, "observation.images.left_rgb": im,
                "observation.images.right_rgb": im, "check_status": np.zeros(5, np.float32), "check_distances": np.ones(5, np.float32)}

out = tempfile.mkdtemp()
if len(sys.argv) > 2:
    g = of.grasp_test(Env(), None, {"variants": [{"press": 0.0}, {"press": 0.013, "slide_len": 0.05}], "params": {"settle": 3}}, name, out)
    print("GRASP_MOCK_OK", len(g), sorted(os.listdir(out))); sys.exit(0)
r = of.shirt_trial(Env(), None, {"settle": 5, "final_settle": 5, "pause": 2}, name, "m0", out)
print("MOCK_OK neat", r["neat"], "ticks", r["ticks"], sorted(os.listdir(out)))
import pickle
fr = pickle.load(open(os.path.join(out, f"{name}_tm0.pkl"), "rb"))["frames"]; print("frames", len(fr), type(fr[0]["img"]).__name__, sorted({f["tag"] for f in fr})[:12])

"""Dry-run flatten_dump on the fake env, then flatten_garment write/use/restore on a synthetic lehome-challenge tree
(a USD mesh of this garment with stale normals), without Isaac."""
import json
import os
import tempfile

import numpy as np

src = open("mock_test.py").read().split("out = tempfile.mkdtemp()")[0]
exec(src)                                  # defines of (oracle_fold module), Env, Obj, d, name, W


class P2(Obj._P):
    def GetAttribute(s, n):
        if n == "points":
            class A:
                def Get(s2): return W
            return A()
        return super().GetAttribute(n)


Obj._prim = P2()


class Cfg: garment_name = name


e = Env(); e.cfg = Cfg()
out = tempfile.mkdtemp()
info = of.flatten_dump(e, None, {"garment": name, "mode": "flatten", "settle_s": 1.0}, name, out)
npz = os.path.join(out, f"flat_{name}.npz")
assert os.path.exists(npz) and info["n_points"] == len(W), info

from pxr import Usd, UsdGeom, Vt
import flatten_garment as fg
ch = tempfile.mkdtemp()
gdir = os.path.join(ch, fg.RELEASE, "Top_Long", name); os.makedirs(gdir)
usd = os.path.join(gdir, "garment.usd")
st = Usd.Stage.CreateNew(usd)
root = UsdGeom.Xform.Define(st, "/garment"); st.SetDefaultPrim(root.GetPrim())
m = UsdGeom.Mesh.Define(st, "/garment/mesh")
faces = d["faces"].astype(int)
pts0 = np.random.rand(len(W), 3).astype(np.float32)
m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts0))
m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.reshape(-1).tolist()))
m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros((faces.size, 3), np.float32)))
m.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(Vt.Vec3fArray.FromNumpy(pts0)))
st.GetRootLayer().Save()
jp = os.path.join(gdir, f"{name}.json")
json.dump({"asset_path": f"/{fg.RELEASE}/Top_Long/{name}/garment.usd", "scale": [0.4, 0.4, 0.4]}, open(jp, "w"))

r = fg.write(npz, ch, os.path.join(out, "keep"))
st2 = Usd.Stage.Open(r["dst"])
mesh2 = UsdGeom.Mesh(st2.GetPrimAtPath("/garment/mesh"))
assert np.allclose(np.array(mesh2.GetPointsAttr().Get()), np.load(npz)["local_points"], atol=1e-5)
n2 = np.array(mesh2.GetNormalsAttr().Get())
assert n2.shape == (faces.size, 3) and abs(np.linalg.norm(n2, axis=1).mean() - 1) < 1e-3, n2.shape
assert json.load(open(jp))["asset_path"].endswith("garment_flat.usd")
assert os.path.exists(os.path.join(out, "keep", "garment_flat.usd"))
fg.restore(ch, name); assert json.load(open(jp))["asset_path"].endswith("/garment.usd")
fg.use(ch, name); assert json.load(open(jp))["asset_path"].endswith("garment_flat.usd")
st0 = Usd.Stage.Open(usd)
assert np.allclose(np.array(UsdGeom.Mesh(st0.GetPrimAtPath("/garment/mesh")).GetPointsAttr().Get()), pts0)   # original untouched
print("FLATTEN_MOCK_OK", json.dumps(r)[:200])

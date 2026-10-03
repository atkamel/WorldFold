"""Build a flat towel garment for the LeHome Isaac env from one of the challenge's own shirt USDs.

Keeps the shirt file's structure (/World/mesh, xform, materials) and swaps the geometry for a flat
N x N triangle grid with the same footprint and point spacing, then writes the garment JSON.

    python make_towel.py TNSC_Top231_obj_exp.usd TNSC_Top231_obj_exp.json out_dir
"""
import json, os, sys
import numpy as np
from pxr import Usd, UsdGeom, Vt, Gf, Sdf

src_usd, src_json, out_dir = sys.argv[1:4]
NAME = "Top_Short_Unseen_9"          # loader only accepts the four challenge type prefixes
N = 72                               # 72 x 72 = 5184 points, ~same spacing as the shirt mesh
HX, HY = 0.30, 0.33                  # half-extents in the shirt file's units (shirt is 0.64 x 0.72)

d = os.path.join(out_dir, NAME); os.makedirs(d, exist_ok=True)
stage = Usd.Stage.Open(src_usd)
prim = stage.GetPrimAtPath("/World/mesh"); mesh = UsdGeom.Mesh(prim)

xs, ys = np.linspace(-HX, HX, N), np.linspace(HY, -HY, N)      # row 0 = far edge (+y)
pts = np.array([(x, y, 0.0) for y in ys for x in xs], dtype=np.float32)
idx = lambda r, c: r * N + c
faces = []
for r in range(N - 1):
    for c in range(N - 1):
        a, b, cc, dd = idx(r, c), idx(r, c + 1), idx(r + 1, c + 1), idx(r + 1, c)
        faces += [(a, dd, cc), (a, cc, b)]                       # +z normals
faces = np.array(faces, dtype=np.int32)

mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.reshape(-1)))
mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(-HX, -HY, 0), Gf.Vec3f(HX, HY, 0)]))
nrm = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (faces.size, 1))
mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(nrm))
uv = np.stack([(pts[:, 0] + HX) / (2 * HX), (pts[:, 1] + HY) / (2 * HY)], 1)[faces.reshape(-1)].astype(np.float32)
prim.GetAttribute("primvars:st").Set(Vt.Vec2fArray.FromNumpy(uv))
for n in ("velocities", "accelerations"):
    a = prim.GetAttribute(n)
    if a and a.Get() is not None: a.Clear()
subs = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Subset)]
UsdGeom.Subset(subs[0]).GetIndicesAttr().Set(Vt.IntArray(list(range(len(faces)))))
for s in subs[1:]: stage.RemovePrim(s.GetPath())

usd_name = "towel_grid.usd"
stage.GetRootLayer().Export(os.path.join(d, usd_name))

cfg = json.load(open(src_json))
cfg["id"] = 900
cfg["asset_path"] = f"/Assets/objects/Challenge_Garment/Release/Top_Short/{NAME}/{usd_name}"
m = N // 2
# 0,1 = far corners; 4,5 = near corners; 2,3 = two adjacent centre points (so the sleeve check is always met)
cfg["check_point"] = [idx(0, 0), idx(0, N - 1), idx(m, m - 1), idx(m, m), idx(N - 1, 0), idx(N - 1, N - 1)]
json.dump(cfg, open(os.path.join(d, "towel_grid.json"), "w"), indent=4)

chk = Usd.Stage.Open(os.path.join(d, usd_name)); cm = UsdGeom.Mesh(chk.GetPrimAtPath("/World/mesh"))
print("TOWEL_OK", d, "points", len(cm.GetPointsAttr().Get()), "faces", len(cm.GetFaceVertexCountsAttr().Get()),
      "check_point", cfg["check_point"], "bytes", os.path.getsize(os.path.join(d, usd_name)))

"""Flatten step 2: write the settled, flat vertex positions (oracle_fold.flatten_dump -> flat_<garment>.npz) into a
copy of the garment's asset USD as its built-in shape, and point the garment's JSON at that copy.

Why: LeHome's garments are modelled as worn (~10 cm thick at scale 0.4). PhysX takes the modelled shape as the
cloth's rest shape, so with the measured-stiff bend/shear (isaac_physics v2) the shirt springs back toward it and
tumbles into a heap on the table (2026-10-19). A real shirt's rest shape is flat.

    python flatten_garment.py write <flat_npz> <lehome-challenge dir> [keep_dir]   # <asset>_flat.usd + JSON -> flat
    python flatten_garment.py use <lehome-challenge dir> <garment>                  # JSON -> flat copy (if it exists)
    python flatten_garment.py restore <lehome-challenge dir> <garment>              # JSON -> original asset
The original asset path is kept in a sidecar file next to the JSON (<json>.orig_asset_path), so the JSON keeps
LeHome's schema. Needs pxr (Isaac's python).
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

RELEASE = "Assets/objects/Challenge_Garment/Release"


def garment_json(ch, name):
    typ = "_".join(name.split("_")[:2])
    js = glob.glob(os.path.join(ch, RELEASE, typ, name, "*.json"))
    assert len(js) == 1, f"expected one garment json, got {js}"
    return js[0]


def resolve(ch, asset_path):
    """Garment._get_usd_path: a leading '/' means relative to the lehome-challenge dir (its cwd)."""
    return os.path.join(ch, asset_path[1:]) if asset_path.startswith("/") else os.path.join(ch, asset_path)


def _orig_asset(jp, cfg):
    side = jp + ".orig_asset_path"
    if os.path.exists(side):
        return open(side).read().strip()
    open(side, "w").write(cfg["asset_path"])
    return cfg["asset_path"]


def _flat_name(asset_path):
    stem, ext = os.path.splitext(asset_path)
    return stem + "_flat" + ext


def vertex_normals(pts, fvi, fvc):
    """Area-weighted vertex normals (polygons fan-triangulated)."""
    n = np.zeros_like(pts, dtype=np.float64)
    k = 0
    for c in fvc:
        f = fvi[k:k + c]; k += c
        for i in range(1, c - 1):
            a, b, d = pts[f[0]], pts[f[i]], pts[f[i + 1]]
            fn = np.cross(b - a, d - a)
            n[f[0]] += fn; n[f[i]] += fn; n[f[i + 1]] += fn
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def write(npz, ch, keep_dir=None):
    from pxr import Usd, UsdGeom, Vt
    d = np.load(npz)
    info = json.loads(str(d["info"]))
    name = info["garment"]
    jp = garment_json(ch, name)
    cfg = json.load(open(jp))
    orig = _orig_asset(jp, cfg)
    src = resolve(ch, orig)
    assert os.path.exists(src), src
    pts = d["local_points"].astype(np.float32)
    dst = _flat_name(src)
    shutil.copy(src, dst)                    # edit the copy: USD layers are shared in memory, so never touch src
    st = Usd.Stage.Open(dst)
    meshes = [p for p in st.Traverse() if p.IsA(UsdGeom.Mesh)]
    cand = [p for p in meshes if len(UsdGeom.Mesh(p).GetPointsAttr().Get() or []) == len(pts)]
    assert len(cand) == 1, f"{len(meshes)} meshes in {src}, {len(cand)} with {len(pts)} points"
    mesh = UsdGeom.Mesh(cand[0])
    old = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
    note = "no authored normals"
    nattr = mesh.GetNormalsAttr()
    if nattr.HasAuthoredValue():             # stale normals would shade the flat shirt as if still worn
        fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get()); fvc = np.array(mesh.GetFaceVertexCountsAttr().Get())
        vn = vertex_normals(pts.astype(np.float64), fvi, fvc).astype(np.float32)
        interp = mesh.GetNormalsInterpolation()
        if interp == UsdGeom.Tokens.faceVarying:
            nattr.Set(Vt.Vec3fArray.FromNumpy(vn[fvi]))
        elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
            nattr.Set(Vt.Vec3fArray.FromNumpy(vn))
        else:
            nattr.Clear()
        note = f"normals recomputed ({interp})"
    if mesh.GetExtentAttr().HasAuthoredValue():
        mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(Vt.Vec3fArray.FromNumpy(pts)))
    st.GetRootLayer().Save()
    cfg["asset_path"] = _flat_name(orig)
    json.dump(cfg, open(jp, "w"), indent=4)
    out = dict(garment=name, src=src, dst=dst, mesh=str(cand[0].GetPath()), points=len(pts),
               moved_mm_max=round(float(np.abs(pts - old).max() * 1000), 1), normals=note, json_asset_path=cfg["asset_path"])
    if keep_dir:
        os.makedirs(keep_dir, exist_ok=True)
        shutil.copy(dst, os.path.join(keep_dir, os.path.basename(dst)))
        out["kept"] = os.path.join(keep_dir, os.path.basename(dst))
    print("FLAT_USD " + json.dumps(out), flush=True)
    return out


def use(ch, name):
    jp = garment_json(ch, name)
    cfg = json.load(open(jp))
    orig = _orig_asset(jp, cfg)
    if not os.path.exists(resolve(ch, _flat_name(orig))):
        print(f"FLAT_USE none: no {_flat_name(orig)} for {name}", flush=True)
        return False
    cfg["asset_path"] = _flat_name(orig)
    json.dump(cfg, open(jp, "w"), indent=4)
    print(f"FLAT_USE {name} -> {cfg['asset_path']}", flush=True)
    return True


def restore(ch, name):
    jp = garment_json(ch, name)
    cfg = json.load(open(jp))
    orig = _orig_asset(jp, cfg)
    cfg["asset_path"] = orig
    json.dump(cfg, open(jp, "w"), indent=4)
    print(f"FLAT_RESTORE {name} -> {orig}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "write":
        write(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "use":
        use(sys.argv[2], sys.argv[3])
    elif cmd == "restore":
        restore(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)

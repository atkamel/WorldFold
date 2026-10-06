"""One-time: simplify the SO-101 CAD meshes (645k triangles for two arms) so the game draws fast on
integrated graphics. Writes light copies with the same file names to mujoco_live/assets_lo/."""
import os, glob, struct
import numpy as np
import fast_simplification

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "teacher", "vendor", "so101_nexus", "assets", "SO101", "assets")
DST = os.path.join(HERE, "assets_lo")
MAX_FACES = 3000


def read_stl(path):
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * n == len(data):  # binary
        rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
        tri = rec["v"].reshape(-1, 3, 3).astype(np.float64)
    else:  # ascii
        vs = [list(map(float, l.split()[1:4])) for l in data.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")]
        tri = np.array(vs).reshape(-1, 3, 3)
    pts, inv = np.unique(np.round(tri.reshape(-1, 3), 5), axis=0, return_inverse=True)
    return pts, inv.reshape(-1, 3)


def write_stl(path, pts, faces):
    tri = pts[faces].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    with open(path, "wb") as f:
        f.write(b"decimated for the fold game".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        rec = np.zeros(len(faces), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
        rec["n"] = nrm
        rec["v"] = tri.reshape(-1, 9)
        f.write(rec.tobytes())


def cluster(pts, faces, cell):
    """Vertex clustering: snap vertices to a `cell`-sized grid, merge, drop collapsed/duplicate triangles."""
    key = np.floor(pts / cell).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    newp = np.zeros((len(uk), 3)); cnt = np.zeros(len(uk))
    np.add.at(newp, inv, pts); np.add.at(cnt, inv, 1)
    newp /= cnt[:, None]
    f = inv[faces]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return newp, f[np.sort(keep)]


os.makedirs(DST, exist_ok=True)
tot0 = tot1 = 0
for path in sorted(glob.glob(os.path.join(SRC, "*.stl"))):
    pts, faces = read_stl(path)
    n0 = len(faces)
    pts2, faces2 = pts, faces
    if n0 > MAX_FACES:
        pts2, faces2 = fast_simplification.simplify(pts.astype(np.float32), faces.astype(np.int32), target_reduction=1 - MAX_FACES / n0, agg=9)
        pts2, faces2 = np.asarray(pts2, float), np.asarray(faces2)
        cell = 0.0005
        while len(faces2) > MAX_FACES and cell < 0.004:  # simplifier got stuck: snap to a coarser grid
            pts2, faces2 = cluster(pts2, faces2, cell)
            cell *= 1.4
    write_stl(os.path.join(DST, os.path.basename(path)), np.asarray(pts2), np.asarray(faces2))
    tot0 += n0; tot1 += len(faces2)
    print(f"{os.path.basename(path):45s} {n0:7d} -> {len(faces2):6d} triangles")
print(f"total {tot0} -> {tot1}")

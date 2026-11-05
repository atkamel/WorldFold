"""How many garments can the RL env simulate per GPU? FoldNet garments (CC-BY-4.0; T-shirt, trousers) welded at the
seams, scaled to the LeHome size (x0.45, within SO-101 reach), simplified to a target vertex count, laid on a table;
one grasped patch is lifted 10 cm and carried. Newton VBD at the contact setting (10 x 10), self-contact on.

  python garment_speed.py --root ~/wf/garments_dl
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newton_contact as C  # noqa: E402

KNIT = {"density": 0.19, "tri_ke": 1121.7, "tri_ka": 721.9, "tri_kd": 0.0376, "B": 6.48e-7, "tau": 6e-4}  # T-shirt prior
SCALE = 0.45


def load_garment(path, target_verts):
    """OBJ -> welded (seams merged), scaled, simplified, flat on the table, centred at the origin."""
    import trimesh
    import fast_simplification
    m = trimesh.load(path, process=False, force="mesh")
    m.merge_vertices(digits_vertex=4)                       # weld seam vertices that coincide to 0.1 mm
    v, f = np.asarray(m.vertices, float) * SCALE, np.asarray(m.faces, np.int64)
    if target_verts and target_verts < len(v):
        v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32),
                                            target_reduction=1.0 - target_verts / len(v))
        v, f = np.asarray(v, float), np.asarray(f, np.int64)
    used = np.unique(f)                                     # drop vertices no face uses
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(len(used))
    v, f = v[used], remap[f]
    v -= [v[:, 0].mean(), v[:, 1].mean(), v[:, 2].min()]
    v[:, 2] += C.RADIUS + 0.0005
    comps = len(trimesh.Trimesh(v, f, process=False).split(only_watertight=False))
    return v, f.astype(np.int32), comps


class Garments(C.Sheets):
    t0, t1, vel = 0.2, 1.2, (0.0, 0.0, 0.10)               # lift the grasped patch 10 cm/s for 1 s
    mesh, K = None, 16

    def build(self, b):
        import newton
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=C.shape_mu(0.5), ke=C.CONTACT["shape_ke"],
                                                               kd=C.CONTACT["kd"]))
        v, f = self.mesh
        span = (v[:, :2].max(0) - v[:, :2].min(0)).max() + 0.2
        n = int(np.ceil(np.sqrt(self.K)))
        e = np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1).mean()
        self.pins, self.spans = [], []
        for k in range(self.K):
            off = np.array([span * (k % n), span * (k // n), 0.0])
            ids = C.add_sheet(b, v + off, f, e)
            top = v[:, 1].argmax()                          # grasp the top-most point (collar / waistband)
            grip = ids[np.linalg.norm(v[:, :2] - v[top, :2], axis=1) < 0.01]
            for i in grip:
                b.particle_mass[int(i)] = 0.0
            self.pins += [int(i) for i in grip]
            self.spans.append((int(ids[0]), int(ids[-1]) + 1, "garment"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.expanduser("~/wf/garments_dl"))
    p.add_argument("--garments", default="tshirt_sp/0,trousers/0")
    p.add_argument("--verts", default="500,1000,2000")
    p.add_argument("--copies", default="16,64,128,256")
    p.add_argument("--setting", default="full")
    p.add_argument("--self", dest="selfc", default="on")
    p.add_argument("--seconds", type=float, default=1.0)
    a = p.parse_args()
    wp.init()
    C.load_scale(1e-3, "_t2")
    C.CLOTH.update(KNIT)
    C.SELF[0] = a.selfc == "on"
    rows = []
    for g in a.garments.split(","):
        for nv in [int(x) for x in a.verts.split(",")]:
            v, f, comps = load_garment(os.path.join(a.root, "mesh", g, "mesh.obj"), nv)
            for K in [int(x) for x in a.copies.split(",")]:
                Garments.mesh, Garments.K = (v, f), K
                row = {"garment": g, "verts": len(v), "faces": len(f), "components": comps, "copies": K,
                       "setting": a.setting, "self_contact": a.selfc}
                try:
                    tic = time.time()
                    sc = Garments(a.setting, 1e-3, None)
                    build = time.time() - tic
                    wall = sc.run(a.seconds, lambda k: None)
                    q = sc.s0.particle_q.numpy()
                    row.update({"copy_sim_s_per_s": round(K * a.seconds / wall, 2), "wall_s": round(wall, 2),
                                "build_s": round(build, 1), "finite": bool(np.isfinite(q).all()),
                                "lift_mm": round(float(q[sc.pins, 2].mean() - v[:, 2].min()) * 1000, 1)})
                    del sc
                except Exception as e:                                      # OOM: report and stop growing K
                    row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                C.log(json.dumps(row))
                rows.append(row)
                wp.synchronize()
                if "error" in row:
                    break
    out = os.path.join(C.RES, f"garment_speed_{a.setting}_self{a.selfc}.json")
    C.jdump(rows, out)


if __name__ == "__main__":
    main()

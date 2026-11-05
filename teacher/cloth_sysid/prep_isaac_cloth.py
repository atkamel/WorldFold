"""Pack held-out Coltraro recordings for the Isaac cloth replay (isaac_cloth_replay.py): same preprocessing as
newton_cloth.prepare (first full frame, gap-filled corners, 100 Hz), but the marker grid is refined to ~5 mm spacing
like LeHome's garment meshes (Isaac particle-cloth stiffness depends on resolution). Research-only data: the npz
files stay on our machines, never in git.

    python prep_isaac_cloth.py <out dir> [per_fabric=2] [spacing_m=0.005]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor", "clothilde-sim", "python_code"))
import coltraro_data as CD  # noqa: E402
import replay  # noqa: E402
from newton_cloth_tris import quads_to_tris  # noqa: E402


def prepare(path, spacing):
    info = CD.parse_name(path)
    t, P = replay.load(path)
    N = P.shape[1]
    R, C = CD.grid_shape(N)
    f0 = CD.first_full_frame(P)
    t, P = t[f0:] - t[f0], P[f0:]
    corners = CD.corner_ids(N)
    Pc = P.copy()
    for k in corners:
        Pc[:, k] = CD.fill_track(t, P[:, k])[0]
    t_rs, P = replay.resample(t, P, 100.0)
    Pc = replay.resample(t, Pc, 100.0)[1]
    gap = float(np.nanmedian(np.linalg.norm(np.diff(P[0].reshape(R, C, 3), axis=1), axis=-1)))
    refine = max(2, int(round(gap / spacing)))
    Tc, Tf, S, X0 = replay.build(P[0], R, C, refine)
    return dict(name=info["name"], fabric=info["fabric"], rho=CD.RHO[info["fabric"]], t=t_rs, P=P, U=Pc[:, corners, :],
                X0=np.asarray(X0, float), tris=quads_to_tris(Tf), corners=np.asarray(corners), N=N,
                w=CD.lumped_areas(P[0], Tc), refine=refine, marker_gap_m=gap)


if __name__ == "__main__":
    out = sys.argv[1]
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    spacing = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    os.makedirs(out, exist_ok=True)
    held = [r for r in CD.free_hanging() if r["mode"] == "hands"]
    picked = {}
    for r in sorted(held, key=lambda r: r["name"]):
        f = r["fabric"]
        if len(picked.setdefault(f, [])) < per:
            picked[f].append(r)
    for f, rs in picked.items():
        for r in rs:
            path = r.get("path") or os.path.join(CD.ensure_data(), "Free-hanging", r["name"] + ".csv")
            d = prepare(path, spacing)
            np.savez_compressed(os.path.join(out, d["name"] + ".npz"), **{k: np.asarray(v) for k, v in d.items()})
            print(d["name"], d["fabric"], "markers", d["N"], "particles", len(d["X0"]), "refine", d["refine"],
                  "gap mm %.1f" % (d["marker_gap_m"] * 1000), "frames", len(d["t"]))

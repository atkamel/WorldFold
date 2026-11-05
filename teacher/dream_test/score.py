"""Score a Dream-1 test run (server.py's results.jsonl + start/*.npz): did each try grab the part it was asked for?

A grasp counts for a target when the median start position of its pinched cloth points (grasp_log pinch_idx) is
within HIT_M of the target point on the flat garment at the start of the try. Targets, from the start shape:
  tops   sleeve_L/R = middle of the cuff end (garment_keys.top_keys cuff_mid), hem_L/R = outer hem corner,
         collar = the segment between the two neck check points
  pants  waist_L/R = the outer ends of the waistband, leg_L/R = middle of each leg's bottom end,
         waist = the waistband segment
Left/right are the robot's: of each pair, the point with the smaller world x is left (the left arm's side).
Pant ends are found in the garment's own mesh frame (x across, y along the legs); which end is the waist is the end
whose bottom band is one piece (the legs' end splits in two). Check this on the start pictures (--plot).

    python score.py RUN_DIR [--plot]
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
HIT_M = 0.05


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _lr(v, i, j):
    """(left index, right index) by world x."""
    return (i, j) if v[i, 0] < v[j, 0] else (j, i)


def top_targets(v, local, cp):
    import garment_keys
    k = garment_keys.top_keys(local[:, :2] if local.shape[1] == 2 else local, cp)
    xy = v[:, :2]
    sl, sr = _lr(v, k["L"]["cuff_mid"], k["R"]["cuff_mid"])
    hl, hr = _lr(v, k["L"]["hem_out"], k["R"]["hem_out"])
    return {"sleeve_L": ("pt", xy[sl]), "sleeve_R": ("pt", xy[sr]), "hem_L": ("pt", xy[hl]), "hem_R": ("pt", xy[hr]),
            "collar": ("seg", xy[k["L"]["neck"]], xy[k["R"]["neck"]])}


def pant_targets(v, local):
    x, y = local[:, 0], local[:, 1]
    span = y.max() - y.min()
    ends = {}
    for name, band in (("hi", y > y.max() - 0.06 * span), ("lo", y < y.min() + 0.06 * span)):
        idx = np.nonzero(band)[0]
        xs = np.sort(x[idx])
        gap = np.max(np.diff(xs)) if len(xs) > 1 else 0
        ends[name] = (idx, gap)
    waist_end, leg_end = ("hi", "lo") if ends["hi"][1] < ends["lo"][1] else ("lo", "hi")
    wi = ends[waist_end][0]
    w1, w2 = wi[np.argmin(x[wi])], wi[np.argmax(x[wi])]
    li = ends[leg_end][0]
    legs = []
    for side in (x[li] < np.median(x[li]), x[li] >= np.median(x[li])):
        s = li[side]
        c = local[s].mean(0)
        legs.append(s[np.argmin(np.linalg.norm(local[s] - c, axis=1))])
    xy = v[:, :2]
    wl, wr = _lr(v, w1, w2)
    ll, lr = _lr(v, legs[0], legs[1])
    return {"waist_L": ("pt", xy[wl]), "waist_R": ("pt", xy[wr]), "leg_L": ("pt", xy[ll]), "leg_R": ("pt", xy[lr]),
            "waist": ("seg", xy[wl], xy[wr])}


def dist(tgt, p):
    return float(np.linalg.norm(p - tgt[1])) if tgt[0] == "pt" else _seg_dist(p, tgt[1], tgt[2])


def score_run(run, plot=False):
    run = pathlib.Path(run)
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        name = f"ep{r['idx']:03d}_{r['kind']}_{r['garment']}"
        f = run / "start" / f"{name}.npz"
        res = dict(idx=r["idx"], kind=r["kind"], garment=r["garment"], target=r["target"], lifted=r["lifted"],
                   held_2s=r["held_2s"], right_part=None, nearest=None, miss_cm=None)
        if r["kind"] in ("grasp", "both") and f.exists():
            d = np.load(f)
            v, local, cp = d["verts"], d["local"], d["check_points"]
            tg = top_targets(v, local, cp) if r["garment"].startswith("Top") else pant_targets(v, local)
            hits = {}
            for g in r.get("grasps", []):
                if g.get("n_pinch", 0) < 5 or g.get("max_lift_cm", 0) < 4.0 or not g.get("pinch_idx"):
                    continue
                p = np.median(v[np.asarray(g["pinch_idx"]), :2], axis=0)
                ds = {k: dist(t, p) for k, t in tg.items()}
                near = min(ds, key=ds.get)
                hits[g["side"]] = (near, ds)
            if r["kind"] == "grasp":
                best = min((ds[r["target"]] for _, ds in hits.values()), default=None)
                res["miss_cm"] = None if best is None else round(best * 100, 1)
                res["right_part"] = best is not None and best <= HIT_M
                res["nearest"] = [h[0] for h in hits.values()]
            else:          # both: one arm on each corner
                pair = ("hem_L", "hem_R") if r["target"] == "hem_LR" else ("waist_L", "waist_R")
                ok = all(any(ds[t] <= HIT_M for _, ds in hits.values()) for t in pair) and len(hits) == 2
                res["right_part"] = ok
                res["nearest"] = [h[0] for h in hits.values()]
            if plot:
                _plot(run, name, v, tg, hits, r)
        out.append(res)
    return out


def _plot(run, name, v, tg, hits, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(v[::7, 0], v[::7, 1], s=1, c="0.7")
    for k, t in tg.items():
        pts = [t[1]] if t[0] == "pt" else [t[1], t[2]]
        for p in pts:
            ax.plot(p[0], p[1], "o", ms=6)
            ax.annotate(k, p, fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(f"{name}\n{r['prompt']}", fontsize=8)
    fig.savefig(run / "start" / f"{name}_targets.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    res = score_run(sys.argv[1], plot="--plot" in sys.argv)
    for r in res:
        print(json.dumps(r))
    g = [r for r in res if r["kind"] == "grasp"]
    if g:
        print(f"one-arm tries: {len(g)}, lifted {sum(r['lifted'] for r in g)}, right part "
              f"{sum(bool(r['right_part']) for r in g)}, held 2 s {sum(r['held_2s'] for r in g)}")

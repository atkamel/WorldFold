"""Semantic key vertices of a top (shirt) from its rest mesh, and the fold lines built from them.

Rest mesh frame (ClothesNet exports used by the challenge): x = left-right (sleeves), y = up (collar at +y),
z = front-back. The challenge's own check points are [neck_L, neck_R, cuff_L, cuff_R, hem_L, hem_R].
Key vertices are INDICES, so they can be looked up in any later state of the cloth.
"""
import numpy as np


def _outline_low(x, y, lo, hi, nb=40):
    """Lower outline y_min(x) of the silhouette between x=lo and x=hi."""
    edges = np.linspace(lo, hi, nb + 1)
    xs, ys, idx = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = np.nonzero((x >= a) & (x <= b))[0]
        if len(m):
            i = m[np.argmin(y[m])]
            xs.append(x[i]); ys.append(y[i]); idx.append(i)
    return np.array(xs), np.array(ys), np.array(idx)


def top_keys(points, check_point):
    """points: (N,3) rest mesh. Returns dict of vertex indices for each side ('L' = -x, 'R' = +x)."""
    x, y = points[:, 0], points[:, 1]
    neck = {"L": int(check_point[0]), "R": int(check_point[1])}
    cuff = {"L": int(check_point[2]), "R": int(check_point[3])}
    hem = {"L": int(check_point[4]), "R": int(check_point[5])}
    keys = {}
    for s, sign in (("L", -1), ("R", 1)):
        # outer hem corner: lowest band of the torso, extreme x on this side (the check point sits inboard)
        band = np.nonzero((y < y.min() + 0.06 * (y.max() - y.min())) & (sign * x > 0))[0]
        hem_out = int(band[np.argmax(sign * x[band])])
        # sleeve tip: the most extreme x on this side
        side = np.nonzero(sign * x > 0)[0]
        tip = int(side[np.argmax(sign * x[side])])
        # armpit: on the LOWER outline between the hem corner and the sleeve tip, the point furthest to the
        # inner/upper side of the straight chord hem-corner -> sleeve-tip. On a normal shirt that is the armpit
        # notch; on a puff sleeve with no notch it is the knee where the torso side turns into the sleeve.
        lo, hi = sorted([x[hem_out], x[tip]])
        ox, oy, oi = _outline_low(x, y, lo, hi)
        ch = np.array([x[tip] - x[hem_out], y[tip] - y[hem_out]]); ch = ch / np.linalg.norm(ch)
        dist = sign * (ch[0] * (oy - y[hem_out]) - ch[1] * (ox - x[hem_out]))      # > 0 on the inner/upper side
        armpit = int(oi[np.argmax(dist)])
        # shoulder: top of the silhouette straight above the armpit
        col = np.nonzero(np.abs(x - x[armpit]) < 0.02 * (x.max() - x.min()))[0]
        shoulder = int(col[np.argmax(y[col])])
        # cuff grasp point: middle of the sleeve end (average of the extreme 3 % along x), nearest real vertex
        ext = side[sign * x[side] > sign * x[tip] - 0.03 * (x.max() - x.min())]
        c = points[ext].mean(0)
        cuff_mid = int(ext[np.argmin(np.linalg.norm(points[ext] - c, axis=1))])
        keys[s] = dict(neck=neck[s], cuff=cuff[s], cuff_mid=cuff_mid, sleeve_tip=tip, hem=hem[s], hem_out=hem_out,
                       armpit=armpit, shoulder=shoulder)
    return keys


def top_folds(xy, keys, sleeve="armpit"):
    """Fold lines for the current flat state. xy: (N,2) world positions; returns a list for fold_metric.ideal_fold
    plus, per fold, which vertices to grasp. Order: left sleeve, right sleeve, bottom to top.

    sleeve='armpit': fold only the sleeve across the shoulder-armpit line (Roy's rule).
    sleeve='strip' : fold along a line parallel to the body axis through the armpit (takes no torso)."""
    P = lambda s, k: np.asarray(xy[keys[s][k]], float)
    centre = (P("L", "armpit") + P("R", "armpit") + P("L", "hem_out") + P("R", "hem_out")) / 4
    up = (P("L", "shoulder") + P("R", "shoulder")) / 2 - (P("L", "hem_out") + P("R", "hem_out")) / 2
    up = up / np.linalg.norm(up)
    folds = []
    for s in ("L", "R"):
        a, sh = P(s, "armpit"), P(s, "shoulder")
        d = (sh - a) if sleeve == "armpit" else up
        d = d / np.linalg.norm(d)
        # the sleeve is the side of the line away from the garment centre
        rel = centre - a
        side_of_centre = np.sign(d[0] * rel[1] - d[1] * rel[0])
        folds.append(dict(name=f"sleeve_{s}", p=a.tolist(), d=d.tolist(), side=int(-side_of_centre), grasp=[(s, "cuff_mid")]))
    # bottom to top: line across the torso halfway between hem and shoulders; the hem side moves
    hem_mid = (P("L", "hem_out") + P("R", "hem_out")) / 2
    sh_mid = (P("L", "shoulder") + P("R", "shoulder")) / 2
    mid = (hem_mid + sh_mid) / 2
    across = P("R", "hem_out") - P("L", "hem_out"); across = across / np.linalg.norm(across)
    rel = hem_mid - mid
    folds.append(dict(name="bottom_up", p=mid.tolist(), d=across.tolist(),
                      side=int(np.sign(across[0] * rel[1] - across[1] * rel[0])), grasp=[("L", "hem_out"), ("R", "hem_out")]))
    # half again: after bottom_up the packet spans from the bottom_up line to the shoulders; fold its lower half up
    yh = (mid + sh_mid) / 2
    relh = mid - yh
    folds.append(dict(name="half", p=yh.tolist(), d=across.tolist(),
                      side=int(np.sign(across[0] * relh[1] - across[1] * relh[0])), grasp=[]))
    return folds

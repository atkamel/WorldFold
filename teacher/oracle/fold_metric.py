"""Fold-neatness metric computed from cloth vertex positions (numpy only).

Laws (Roy): area is reduced to the target; the footprint is a clean rectangle (corners visible);
low pile height (tall = crumpled). From the literature: every vertex lands where the ideal fold puts
it, and moved vertices land on their mirror partners across the fold line.

The ideal target is built from the garment's own FLAT reference state plus an ordered list of fold
lines (van den Berg 2010 style): vertices on the moving side are reflected across each line in turn.
All lengths are metres in, centimetres out.
"""
import numpy as np

CELL = 0.004  # footprint raster cell (m)


# ---- footprint -------------------------------------------------------------------------------------
def _mask(xy, faces, cell=CELL):
    """Rasterise the cloth's top-view footprint. Returns (mask, origin_xy)."""
    lo = xy.min(0) - 3 * cell
    hi = xy.max(0) + 3 * cell
    shape = np.ceil((hi - lo) / cell).astype(int) + 1
    m = np.zeros(shape, dtype=bool)
    pts = [xy]
    if faces is not None and len(faces):
        a, b, c = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
        pts += [(a + b + c) / 3, (a + b) / 2, (b + c) / 2, (a + c) / 2,
                (4 * a + b + c) / 6, (a + 4 * b + c) / 6, (a + b + 4 * c) / 6]
    for p in pts:
        ij = np.floor((p - lo) / cell).astype(int)
        m[ij[:, 0], ij[:, 1]] = True
    # close 1-cell pinholes (dilate then erode)
    d = m.copy()
    d[1:] |= m[:-1]; d[:-1] |= m[1:]; d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
    e = d.copy()
    e[1:] &= d[:-1]; e[:-1] &= d[1:]; e[:, 1:] &= d[:, :-1]; e[:, :-1] &= d[:, 1:]
    return e | m, lo


def _hull(p):
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _min_rect(p):
    """Minimum-area bounding rectangle of 2-D points: (area, long side, short side, angle_rad)."""
    h = _hull(p)
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        n = np.linalg.norm(e)
        if n < 1e-9:
            continue
        u = e / n; v = np.array([-u[1], u[0]])
        a, b = h @ u, h @ v
        w, hgt = a.max() - a.min(), b.max() - b.min()
        if best is None or w * hgt < best[0]:
            best = (w * hgt, max(w, hgt), min(w, hgt), float(np.arctan2(u[1], u[0])))
    return best


def footprint(verts, faces):
    m, lo = _mask(verts[:, :2], faces)
    area = float(m.sum()) * CELL * CELL
    ii, jj = np.nonzero(m)
    cells = np.stack([ii, jj], 1) * CELL + lo + CELL / 2
    rect_area, long_side, short_side, ang = _min_rect(cells)
    return dict(area_cm2=area * 1e4, rectangularity=float(area / max(rect_area, 1e-9)),
                long_cm=long_side * 100, short_cm=short_side * 100, angle_deg=float(np.degrees(ang)))


# ---- ideal fold target -----------------------------------------------------------------------------
def ideal_fold(flat_xy, folds):
    """folds: list of dict(p=(x,y) point on the line, d=(dx,dy) line direction, side=+1|-1).
    A vertex moves when side * cross(d, v - p) > 0. Returns (target_xy, moved_mask_per_fold)."""
    xy = np.asarray(flat_xy, float).copy()
    moved_all = []
    for f in folds:
        p = np.asarray(f["p"], float); d = np.asarray(f["d"], float); d = d / np.linalg.norm(d)
        rel = xy - p
        cr = d[0] * rel[:, 1] - d[1] * rel[:, 0]
        mv = f.get("side", 1) * cr > 0
        along = rel @ d
        refl = p + np.outer(along, d) - (rel - np.outer(along, d))
        xy[mv] = refl[mv]
        moved_all.append(mv)
    return xy, moved_all


def _kabsch2d(a, b):
    """Rigid (rotation + translation, no reflection) transform taking a onto b; returns aligned a."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return (a - ca) @ R.T + cb


def _nearest(src, dst, chunk=2000):
    out = np.empty(len(src), dtype=int)
    for i in range(0, len(src), chunk):
        d = ((src[i:i + chunk, None, :] - dst[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


# ---- the metric ------------------------------------------------------------------------------------
def score(verts, faces, flat_verts, folds, table_z, n_layers=None):
    """Returns a dict of raw terms and a 0-100 `neat` score.

    verts: final cloth vertices (N,3); flat_verts: the same garment lying flat before folding;
    folds: fold lines in the flat state's world xy; table_z: table surface height."""
    verts = np.asarray(verts, float); flat = np.asarray(flat_verts, float)
    f0, f1 = footprint(flat, faces), footprint(verts, faces)
    L = float(np.sqrt(f0["long_cm"] * f0["short_cm"]))            # garment size scale (cm)
    target_xy, moved = ideal_fold(flat[:, :2], folds)
    tf = footprint(np.c_[target_xy, np.zeros(len(target_xy))], faces)
    target_ratio = tf["area_cm2"] / f0["area_cm2"]

    # law 1: area reduction, two-sided (too big = unfinished, too small = crumpled)
    ratio = f1["area_cm2"] / f0["area_cm2"]
    e_area = abs(ratio - target_ratio) / target_ratio
    # law 2: clean rectangle. Compare with how rectangular the ideal fold itself is (a shirt is not a perfect rectangle)
    e_rect = max(0.0, tf["rectangularity"] - f1["rectangularity"])
    # law 3: pile height
    h = (verts[:, 2] - table_z) * 100
    h95, hmax = float(np.percentile(h, 95)), float(h.max())
    n_layers = n_layers or 2 ** len(folds)
    e_h = max(0.0, h95 - (0.5 + 0.4 * n_layers))                    # cm above what n flat layers need
    # literature 1: every vertex where the ideal fold puts it (shape only: rigid alignment removes drift)
    aligned = _kabsch2d(verts[:, :2], target_xy)
    e_v = float(np.linalg.norm(aligned - target_xy, axis=1).mean() * 100)
    drift = float(np.linalg.norm(verts[:, :2].mean(0) - target_xy.mean(0)) * 100)
    # literature 2: moved vertices sit on their mirror partners (nearest stationary vertex to the reflected rest position)
    mv = np.any(moved, axis=0) if moved else np.zeros(len(verts), bool)
    if mv.any() and (~mv).any():
        still = np.nonzero(~mv)[0]
        partner = still[_nearest(target_xy[mv], flat[still, :2])]
        e_m = float(np.linalg.norm(verts[mv, :2] - verts[partner, :2], axis=1).mean() * 100)
    else:
        e_m = 0.0

    s = lambda e, tau: float(np.clip(1 - e / tau, 0, 1))
    s_v, s_m = s(e_v, 0.15 * L), s(e_m, 0.15 * L)
    s_h = s(e_h, 3.0)
    s_f = 0.5 * s(e_area, 0.5) + 0.5 * s(e_rect, 0.3)
    neat = 100 * (0.4 * s_v + 0.2 * s_m + 0.2 * s_h + 0.2 * s_f)
    # area-first composite: how much the footprint shrank vs the ideal, how rectangular, how flat; exact positions minor
    s_area = s(e_area, 0.5)
    s_rect = s(e_rect, 0.3)
    neat_area = 100 * (0.35 * s_area + 0.25 * s_rect + 0.2 * s_h + 0.1 * s_v + 0.1 * s_m)
    r = lambda x: round(float(x), 2)
    return dict(neat=r(neat), neat_area=r(neat_area), s_area=r(s_area), s_rect=r(s_rect), s_vertex=r(s_v), s_mirror=r(s_m), s_height=r(s_h), s_footprint=r(s_f),
                vertex_err_cm=r(e_v), mirror_err_cm=r(e_m), drift_cm=r(drift),
                h95_cm=r(h95), hmax_cm=r(hmax), area_ratio=r(ratio), target_area_ratio=r(target_ratio),
                rectangularity=r(f1["rectangularity"]), target_rectangularity=r(tf["rectangularity"]),
                size_cm=[r(f1["long_cm"]), r(f1["short_cm"])], target_size_cm=[r(tf["long_cm"]), r(tf["short_cm"])],
                flat_size_cm=[r(f0["long_cm"]), r(f0["short_cm"])], flat_area_cm2=r(f0["area_cm2"]), L_cm=r(L))


def flat_stats(verts, faces, table_z):
    f = footprint(np.asarray(verts, float), faces)
    h = (np.asarray(verts)[:, 2] - table_z) * 100
    f.update(h95_cm=float(np.percentile(h, 95)), hmax_cm=float(h.max()))
    return {k: round(float(v), 2) for k, v in f.items()}

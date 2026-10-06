"""Fold score that can't be gamed by crumpling.

  area      footprint area / flat area (smaller = more folded)
  packing   cloth volume / bulk volume of the pile. Cloth has a fixed volume (its area x thickness);
            neat flat layers stack with no air in between (packing ~1), a crumpled ball holds the same
            cloth in a much bigger, lumpier volume (packing << 1). Bulk volume = top-surface height map
            integrated over the footprint.
  rect      footprint area / its minimum bounding rectangle (clean rectangle = 1)

  flat      share of the cloth surface lying flat (crumples are tilted everywhere)
  valleys   share of the top surface in dips deeper than 3 mm (Roy's 'depth valleys'; crumples are pitted)

  air       inside valleys: air pockets between the stacked layers (pleats, loops, hollow 'tents' - a
            crumple can hide these under a smooth top, like a stomach lining)

  score = 100 * compact * flat * rect * smooth * solid, with compact = (1 - area) / (1 - 0.25) capped at 1
  (folding below a quarter of the flat area earns nothing extra, so the only way up is neat),
  smooth = 1 - valleys / 0.5 (outside) and solid = 1 - average air / 20 mm (inside). (packing is reported but not scored: on the old puffy cloth it was ~0.5 for
  every fold and did not separate neat from crumpled.)
"""
import numpy as np
from numba import njit

FLOOR_AREA = 0.25  # no extra credit for folding smaller than this (a neat 2-fold shirt is ~25-37%)
AIR_ZERO_MM = 20.0  # average air between the layers at which the 'solid inside' factor reaches 0


@njit(cache=True)
def _height_map(x, tris, lo0, lo1, res, nx, ny):
    hm = np.full((ny, nx), -1.0)
    for t in range(tris.shape[0]):
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        ax, ay, az = x[a, 0], x[a, 1], x[a, 2]
        bx, by, bz = x[b, 0], x[b, 1], x[b, 2]
        cx, cy, cz = x[c, 0], x[c, 1], x[c, 2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-14:
            continue
        i0 = max(int((min(ax, bx, cx) - lo0) / res), 0); i1 = min(int((max(ax, bx, cx) - lo0) / res) + 1, nx - 1)
        j0 = max(int((min(ay, by, cy) - lo1) / res), 0); j1 = min(int((max(ay, by, cy) - lo1) / res) + 1, ny - 1)
        for j in range(j0, j1 + 1):
            py = lo1 + (j + 0.5) * res
            for i in range(i0, i1 + 1):
                px = lo0 + (i + 0.5) * res
                l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
                l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
                l3 = 1.0 - l1 - l2
                if l1 >= 0 and l2 >= 0 and l3 >= 0:
                    z = l1 * az + l2 * bz + l3 * cz
                    if z > hm[j, i]:
                        hm[j, i] = z
    return hm


@njit(cache=True)
def _layer_gaps(x, tris, lo0, lo1, res, nx, ny, h, maxl):
    """For every cell: all cloth layers stacked there (every triangle crossing it), sorted bottom to top.
    Returns per-cell air between consecutive layers (and under the lowest one), in metres, and layer count."""
    zs = np.zeros((ny, nx, maxl))
    cnt = np.zeros((ny, nx), np.int64)
    for t in range(tris.shape[0]):
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        ax, ay, az = x[a, 0], x[a, 1], x[a, 2]
        bx, by, bz = x[b, 0], x[b, 1], x[b, 2]
        cx, cy, cz = x[c, 0], x[c, 1], x[c, 2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-14:
            continue
        i0 = max(int((min(ax, bx, cx) - lo0) / res), 0); i1 = min(int((max(ax, bx, cx) - lo0) / res) + 1, nx - 1)
        j0 = max(int((min(ay, by, cy) - lo1) / res), 0); j1 = min(int((max(ay, by, cy) - lo1) / res) + 1, ny - 1)
        for j in range(j0, j1 + 1):
            py = lo1 + (j + 0.5) * res
            for i in range(i0, i1 + 1):
                px = lo0 + (i + 0.5) * res
                l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
                l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
                l3 = 1.0 - l1 - l2
                if l1 >= 0 and l2 >= 0 and l3 >= 0 and cnt[j, i] < maxl:
                    zs[j, i, cnt[j, i]] = l1 * az + l2 * bz + l3 * cz
                    cnt[j, i] += 1
    gap = np.zeros((ny, nx))
    nlay = np.zeros((ny, nx), np.int64)
    for j in range(ny):
        for i in range(nx):
            n = cnt[j, i]
            if n == 0:
                continue
            v = np.sort(zs[j, i, :n])
            prev = -1.0
            air = 0.0
            m = 0
            for k in range(n):
                if prev >= 0 and v[k] - prev < 0.3 * h:  # same layer seen twice (cell on a shared edge)
                    continue
                if prev < 0:
                    air += max(v[k] - 0.5 * h, 0.0)      # air under the lowest layer
                else:
                    air += max(v[k] - prev - h, 0.0)     # air between this layer and the one below
                prev = v[k]
                m += 1
            gap[j, i] = air
            nlay[j, i] = m
    return gap, nlay


def inside_voids(x, tris, thickness, deep_mm=3.0, res=0.002):
    """Inside valleys: air pockets between the stacked layers (pleats, loops, the 'stomach lining' of a
    crumple that a smooth top can hide). Returns (share of the footprint with > deep_mm of air inside,
    mean air mm, mean layer count)."""
    x = np.asarray(x, float)
    lo = x[:, :2].min(0) - 0.01
    hi = x[:, :2].max(0) + 0.01
    nx, ny = (np.ceil((hi - lo) / res)).astype(int)
    gap, nlay = _layer_gaps(x, np.asarray(tris, np.int64), lo[0], lo[1], res, nx, ny, thickness, 24)
    occ = nlay > 0
    g = gap[occ] * 1000
    return float((g > deep_mm).mean()), float(g.mean()), float(nlay[occ].mean())


def valley_share(x, tris, thickness, width_m=0.03, deep_mm=3.0, res=0.002):
    """Roy's 'depth valleys': share of the pile's top surface sitting more than deep_mm below the bumps
    around it (black top-hat: fill every dip narrower than width_m, measure the fill). Neat fold ~10-16%,
    crumple ~30%+. Returns (share, mean valley depth in mm)."""
    from numpy.lib.stride_tricks import sliding_window_view as win
    x = np.asarray(x, float)
    lo = x[:, :2].min(0) - 0.03
    hi = x[:, :2].max(0) + 0.03
    nx, ny = (np.ceil((hi - lo) / res)).astype(int)
    hm = _height_map(x, np.asarray(tris, np.int64), lo[0], lo[1], res, nx, ny)
    occ = hm > -0.5
    top = np.where(occ, np.maximum(hm, 0) + thickness / 2, 0.0)
    k = int(width_m / res) | 1

    def filt(a, fn):
        b = np.pad(a, k // 2, mode="edge")
        b = fn(win(b, k, axis=0), axis=-1)
        return fn(win(b, k, axis=1), axis=-1)
    depth = (filt(filt(top, np.max), np.min) - top)[occ] * 1000
    return float((depth > deep_mm).mean()), float(np.maximum(depth, 0).mean())


def flat_fraction(x, tris, tol_deg=25):
    """Share of the cloth surface lying flat (triangle within tol_deg of horizontal, either side up).
    A neat fold is flat layers plus a few creases; a crumple is tilted cloth everywhere."""
    a, b, c = x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    nz = np.abs(n[:, 2]) / (area + 1e-12)
    return float(area[nz > np.cos(np.radians(tol_deg))].sum() / area.sum())


def fold_score(x, tris, flat_area_m2, thickness):
    x = np.asarray(x, float)
    res = 0.002
    lo = x[:, :2].min(0) - 0.01
    hi = x[:, :2].max(0) + 0.01
    nx, ny = (np.ceil((hi - lo) / res)).astype(int)
    hm = _height_map(x, np.asarray(tris, np.int64), lo[0], lo[1], res, nx, ny)
    occ = hm > -0.5
    area = occ.sum() * res * res
    top = np.maximum(hm[occ], 0.0) + 0.5 * thickness          # top surface above the table
    bulk = float(top.sum() * res * res)
    packing = float(min(1.0, flat_area_m2 * thickness / bulk))
    jj, ii = np.nonzero(occ)
    P = np.c_[ii, jj] * res
    best = None
    for ang in np.radians(np.arange(0, 90, 2.0)):
        c, s = np.cos(ang), np.sin(ang)
        R = P @ np.array([[c, -s], [s, c]])
        ext = R.max(0) - R.min(0) + res
        if best is None or ext[0] * ext[1] < best[0]:
            best = (ext[0] * ext[1], ext)
    rect = float(area / best[0])
    ratio = float(area / flat_area_m2)
    compact = float(np.clip((1 - ratio) / (1 - FLOOR_AREA), 0, 1))
    flat = flat_fraction(x, np.asarray(tris))
    vshare, vmean = valley_share(x, tris, thickness)
    smooth = float(np.clip(1 - vshare / 0.5, 0, 1))
    _, air_mm, layers = inside_voids(x, tris, thickness)
    solid = float(np.clip(1 - air_mm / AIR_ZERO_MM, 0, 1))
    return dict(score=round(100 * compact * flat * rect * smooth * solid, 1), area_ratio=round(ratio, 3), flat=round(flat, 3),
                valleys=round(vshare, 3), valley_mm=round(vmean, 2), smooth=round(smooth, 3),
                air_mm=round(air_mm, 2), layers=round(layers, 2), solid=round(solid, 3),
                packing=round(packing, 3), rect=round(rect, 3), compact=round(compact, 3),
                height_mm=round(float(top.max()) * 1000, 1),
                box_cm=[round(float(max(best[1])) * 100, 1), round(float(min(best[1])) * 100, 1)])

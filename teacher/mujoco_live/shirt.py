"""Flat T-shirt mesh shared by the cloth solver and the renderer (hem near the robots, collar far)."""
import numpy as np

BODY = (-0.10, 0.10, 0.08, 0.30)
SLEEVE_L = (-0.20, -0.10, 0.22, 0.30)
SLEEVE_R = (0.10, 0.20, 0.22, 0.30)
RECTS = (BODY, SLEEVE_L, SLEEVE_R)


def _inside(x, y, e=1e-6):
    return any(r[0] - e <= x <= r[1] + e and r[2] - e <= y <= r[3] + e for r in RECTS)


def shirt_mesh(h=0.01):
    xs = np.round(np.arange(-0.20, 0.20 + 1e-9, h), 4)
    ys = np.round(np.arange(0.08, 0.30 + 1e-9, h), 4)
    idx, pts = {}, []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if _inside(x, y):
                idx[(i, j)] = len(pts)
                pts.append((x, y))
    tris = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if all(k in idx for k in c) and _inside((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2):
                a, b, cc, d = (idx[k] for k in c)
                # alternate the diagonal so the cloth has no preferred fold direction
                if (i + j) % 2 == 0:
                    tris += [(a, b, cc), (a, cc, d)]
                else:
                    tris += [(a, b, d), (b, cc, d)]
    return np.array(pts), np.array(tris, np.int64)


LANDMARKS = {
    "hem_L": (-0.10, 0.08), "hem_R": (0.10, 0.08), "hem_mid": (0.0, 0.08),
    "armpit_L": (-0.10, 0.22), "armpit_R": (0.10, 0.22),
    "shoulder_L": (-0.10, 0.30), "shoulder_R": (0.10, 0.30), "collar_mid": (0.0, 0.30),
    "sleeve_L_lo": (-0.20, 0.22), "sleeve_L_hi": (-0.20, 0.30),
    "sleeve_R_lo": (0.20, 0.22), "sleeve_R_hi": (0.20, 0.30),
}


def landmark_ids(pts):
    return {k: int(np.argmin(np.linalg.norm(pts - np.array(v), axis=1))) for k, v in LANDMARKS.items()}

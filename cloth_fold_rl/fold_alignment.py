"""Layer alignment of a folded cloth: how far apart the grid vertices that the folds should stack end up.

Grid vertices are row-major, index = ix * n + iy (x index first), as in ClothFoldEnv.cloth_positions() and
IsaacClothFoldEnv's 121-vertex subset. A fold is (axis, line) in grid-index units: axis "x" mirrors ix about
`line`, "y" mirrors iy. Folds apply in order to the folded coordinates; vertices that land on the same folded
coordinate form a group that should lie on one spot. A group's spread is the largest XY distance between two of
its vertices, so a rigid slide of the whole cloth scores zero. Edge groups lie on the folded shape's outline
(its open edges and creases).
"""

from __future__ import annotations

import numpy as np

MEAN_BAR = 0.010     # m, mean spread over all groups (user, 2026-10-14)
EDGE_BAR = 0.015     # m, worst spread over the edge groups (user, 2026-10-14)

QUARTER_FOLDS = (("y", 5), ("x", 5))     # 11 x 11 grid: north onto south, then west onto east
STRIP_FOLDS = (("y", 5), ("y", 2.5))     # north onto south twice: a strip a quarter as deep


def folded_coords(n, folds):
    """(n * n, 2) folded (ix, iy) of every vertex."""
    ix, iy = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    c = np.stack([ix.ravel(), iy.ravel()], axis=1)
    for axis, line in folds:
        k = 0 if axis == "x" else 1
        c[:, k] = line - np.abs(c[:, k] - line)    # every vertex onto the low side of the line
    return c


def groups(n, folds):
    """(list of vertex-index arrays, list of bools: the group lies on the folded outline)."""
    c = folded_coords(n, folds)
    keys, inverse = np.unique(c, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    lo, hi = keys.min(axis=0), keys.max(axis=0)
    members = [np.flatnonzero(inverse == g) for g in range(len(keys))]
    edge = [bool(np.any((k == lo) | (k == hi))) for k in keys]
    return members, edge


def spreads(positions, members):
    """Largest pairwise XY distance inside each group (m); 0 for a single vertex."""
    xy = np.asarray(positions, dtype=float)[:, :2]
    out = np.zeros(len(members))
    for g, idx in enumerate(members):
        if len(idx) > 1:
            d = xy[idx][:, None, :] - xy[idx][None, :, :]
            out[g] = float(np.sqrt((d ** 2).sum(-1)).max())
    return out


def score(positions, folds=QUARTER_FOLDS, n=None, mean_bar=MEAN_BAR, edge_bar=EDGE_BAR):
    """Alignment summary for one cloth state. positions: (n * n, 3) grid vertices."""
    positions = np.asarray(positions, dtype=float)
    n = n or int(round(np.sqrt(len(positions))))
    members, edge = groups(n, folds)
    s = spreads(positions, members)
    multi = np.array([len(m) > 1 for m in members])
    edge = np.array(edge) & multi
    mean, edge_max = float(s[multi].mean()), float(s[edge].max())
    worst = int(np.argmax(s))
    return {"mean": mean, "edge_max": edge_max, "worst_spread": float(s[worst]),
            "worst_group": members[worst].tolist(), "passed": mean <= mean_bar and edge_max <= edge_bar,
            "spreads": s}

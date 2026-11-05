"""quads_to_tris copied from newton_cloth.py (no newton/warp import)."""
from collections import deque

import numpy as np


def quads_to_tris(T):
    """Split each quad along alternating diagonals (checkerboard via BFS over shared edges), so the triangulation
    has no preferred shear direction."""
    T = np.asarray(T, int)
    edge_q = {}
    for f, q in enumerate(T):
        for k in range(4):
            edge_q.setdefault(tuple(sorted((q[k], q[(k + 1) % 4]))), []).append(f)
    par = -np.ones(len(T), int)
    for s in range(len(T)):
        if par[s] >= 0:
            continue
        par[s] = 0
        dq = deque([s])
        while dq:
            f = dq.popleft()
            q = T[f]
            for k in range(4):
                for g in edge_q[tuple(sorted((q[k], q[(k + 1) % 4])))]:
                    if par[g] < 0:
                        par[g] = 1 - par[f]
                        dq.append(g)
    tris = []
    for f, (a, b, c, d) in enumerate(T):
        tris += [(a, b, c), (a, c, d)] if par[f] == 0 else [(a, b, d), (b, c, d)]
    return np.array(tris, np.int32)

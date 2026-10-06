"""Cloth for the folding game: XPBD (extended position-based dynamics), compiled with numba.

Why not MuJoCo's flex cloth: in our tests folded layers sank through each other (15-30% of points),
pinned cloth was forced through the table, and waves rippled through it like water.

Model (per substep, "small steps" XPBD, Macklin et al. 2019):
  - stretch: every mesh edge is a distance constraint with zero compliance (cloth barely stretches)
  - bending: distance between the two far corners of every pair of neighbouring triangles, soft
  - table: particles stay >= thickness/2 above z=0, with Coulomb static/kinetic friction
  - self-contact: vertex-vs-triangle with a thickness gap. Each close (vertex, triangle) pair remembers
    which SIDE of the triangle the vertex is on, so layers can never pass through each other
    (the usual failure of cloth sims), with cloth-on-cloth friction
  - grippers: grabbed particles are moved to the jaw each substep, then contacts are solved last,
    so a gripper can press cloth onto the table/other layers but never push it through them
  - jaws push loose cloth away (spheres at the finger pads)
  - air drag + speed limit
"""
import numpy as np
from numba import njit


# ----------------------------------------------------------------------------- kernels
@njit(cache=True)
def _closest_bary(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """Barycentric coords (u, v, w) of the projection of p onto the plane of triangle abc (unclamped)."""
    v0x, v0y, v0z = bx - ax, by - ay, bz - az
    v1x, v1y, v1z = cx - ax, cy - ay, cz - az
    v2x, v2y, v2z = px - ax, py - ay, pz - az
    d00 = v0x * v0x + v0y * v0y + v0z * v0z
    d01 = v0x * v1x + v0y * v1y + v0z * v1z
    d11 = v1x * v1x + v1y * v1y + v1z * v1z
    d20 = v2x * v0x + v2y * v0y + v2z * v0z
    d21 = v2x * v1x + v2y * v1y + v2z * v1z
    den = d00 * d11 - d01 * d01
    if den < 1e-18:
        return -1.0, -1.0, -1.0
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return 1.0 - v - w, v, w


@njit(cache=True)
def _broad(x, tris, excl, side, stamp, frame, cand, xy_margin, z_margin, prev_cand, nprev):
    """Candidate (vertex, triangle) pairs: vertex inside the triangle's xy footprint (+margin) and
    within z_margin of it. Pairs that stopped being candidates forget which side they were on."""
    n = x.shape[0]
    T = tris.shape[0]
    # bucket vertices on a 2D grid
    cell = 0.02
    xmin = x[:, 0].min() - 0.05
    ymin = x[:, 1].min() - 0.05
    nx = int((x[:, 0].max() - xmin) / cell) + 3
    ny = int((x[:, 1].max() - ymin) / cell) + 3
    cnt = np.zeros(nx * ny + 1, np.int64)
    cid = np.empty(n, np.int64)
    for i in range(n):
        c = int((x[i, 0] - xmin) / cell) * ny + int((x[i, 1] - ymin) / cell)
        cid[i] = c
        cnt[c + 1] += 1
    for c in range(nx * ny):
        cnt[c + 1] += cnt[c]
    order = np.empty(n, np.int64)
    fill = cnt.copy()
    for i in range(n):
        order[fill[cid[i]]] = i
        fill[cid[i]] += 1
    m = 0
    cap = cand.shape[0]
    for t in range(T):
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        lx = min(x[a, 0], x[b, 0], x[c, 0]) - xy_margin
        hx = max(x[a, 0], x[b, 0], x[c, 0]) + xy_margin
        ly = min(x[a, 1], x[b, 1], x[c, 1]) - xy_margin
        hy = max(x[a, 1], x[b, 1], x[c, 1]) + xy_margin
        lz = min(x[a, 2], x[b, 2], x[c, 2]) - z_margin
        hz = max(x[a, 2], x[b, 2], x[c, 2]) + z_margin
        i0 = max(int((lx - xmin) / cell), 0)
        i1 = min(int((hx - xmin) / cell), nx - 1)
        j0 = max(int((ly - ymin) / cell), 0)
        j1 = min(int((hy - ymin) / cell), ny - 1)
        for ci in range(i0, i1 + 1):
            for cj in range(j0, j1 + 1):
                cc = ci * ny + cj
                for k in range(cnt[cc], cnt[cc + 1]):
                    vtx = order[k]
                    if excl[vtx, t]:
                        continue
                    if x[vtx, 0] < lx or x[vtx, 0] > hx or x[vtx, 1] < ly or x[vtx, 1] > hy or x[vtx, 2] < lz or x[vtx, 2] > hz:
                        continue
                    if m < cap:
                        cand[m, 0] = vtx
                        cand[m, 1] = t
                        stamp[vtx, t] = frame
                        m += 1
    for k in range(nprev):  # forget the side of pairs that drifted apart
        vtx, t = prev_cand[k, 0], prev_cand[k, 1]
        if stamp[vtx, t] != frame:
            side[vtx, t] = 0
    return m


@njit(cache=True)
def _substeps(x, v, x0, w, edges, rest_e, bpairs, rest_b, bend_alpha,
              att_i, att_g, att_off, grip_pos, jaws, attached, teth_p, teth_a, teth_d,
              cand, ncand, tris, side, h, mu_cloth, mu_table,
              g, drag, vmax, nsub, dt, stats):
    n = x.shape[0]
    dt2 = dt * dt
    fdrag = np.exp(-drag * dt)
    zmin = 0.5 * h
    for _ in range(nsub):
        # --- predict
        for i in range(n):
            x0[i, 0] = x[i, 0]; x0[i, 1] = x[i, 1]; x0[i, 2] = x[i, 2]
            v[i, 2] += g * dt
            v[i, 0] *= fdrag; v[i, 1] *= fdrag; v[i, 2] *= fdrag
            sp = np.sqrt(v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2)
            if sp > vmax:
                s = vmax / sp
                v[i, 0] *= s; v[i, 1] *= s; v[i, 2] *= s
            x[i, 0] += v[i, 0] * dt; x[i, 1] += v[i, 1] * dt; x[i, 2] += v[i, 2] * dt
        # --- grippers: grabbed particles go to the jaw
        for k in range(att_i.shape[0]):
            i = att_i[k]
            gg = att_g[k]
            x[i, 0] = grip_pos[gg, 0] + att_off[k, 0]
            x[i, 1] = grip_pos[gg, 1] + att_off[k, 1]
            x[i, 2] = grip_pos[gg, 2] + att_off[k, 2]
        # --- tethers ("long range attachments"): no point may be farther from a grabbed point than it is
        # along the fabric. Stops pulled cloth from stretching like rubber and snapping back when released.
        for k in range(teth_p.shape[0]):
            p, a = teth_p[k], teth_a[k]
            dx = x[p, 0] - x[a, 0]; dy = x[p, 1] - x[a, 1]; dz = x[p, 2] - x[a, 2]
            L = np.sqrt(dx * dx + dy * dy + dz * dz)
            if L > teth_d[k] and L > 1e-9:
                s = (L - teth_d[k]) / L
                x[p, 0] -= dx * s; x[p, 1] -= dy * s; x[p, 2] -= dz * s
        # --- stretch (stiff, 3 passes) and bending (soft) distance constraints, Gauss-Seidel
        for _it in range(3):
            for k in range(edges.shape[0]):
                i, j = edges[k, 0], edges[k, 1]
                ws = w[i] + w[j]
                dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]; dz = x[i, 2] - x[j, 2]
                L = np.sqrt(dx * dx + dy * dy + dz * dz)
                if L < 1e-9:
                    continue
                lam = -(L - rest_e[k]) / ws / L
                x[i, 0] += w[i] * lam * dx; x[i, 1] += w[i] * lam * dy; x[i, 2] += w[i] * lam * dz
                x[j, 0] -= w[j] * lam * dx; x[j, 1] -= w[j] * lam * dy; x[j, 2] -= w[j] * lam * dz
        ab = bend_alpha / dt2
        for k in range(bpairs.shape[0]):
            i, j = bpairs[k, 0], bpairs[k, 1]
            ws = w[i] + w[j] + ab
            dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]; dz = x[i, 2] - x[j, 2]
            L = np.sqrt(dx * dx + dy * dy + dz * dz)
            if L < 1e-9:
                continue
            lam = -(L - rest_b[k]) / ws / L
            x[i, 0] += w[i] * lam * dx; x[i, 1] += w[i] * lam * dy; x[i, 2] += w[i] * lam * dz
            x[j, 0] -= w[j] * lam * dx; x[j, 1] -= w[j] * lam * dy; x[j, 2] -= w[j] * lam * dz
        # --- jaws push loose cloth out of the way
        for q in range(jaws.shape[0]):
            cxj, cyj, czj, rj = jaws[q, 0], jaws[q, 1], jaws[q, 2], jaws[q, 3]
            for i in range(n):
                if attached[i]:
                    continue
                dx = x[i, 0] - cxj; dy = x[i, 1] - cyj; dz = x[i, 2] - czj
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < rj * rj and d2 > 1e-12:
                    d = np.sqrt(d2)
                    s = (rj - d) / d
                    x[i, 0] += dx * s; x[i, 1] += dy * s; x[i, 2] += dz * s
        # --- self contact: vertex vs triangle, with remembered side and friction
        for k in range(ncand):
            p, t = cand[k, 0], cand[k, 1]
            a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
            u, vv, ww = _closest_bary(x[p, 0], x[p, 1], x[p, 2], x[a, 0], x[a, 1], x[a, 2],
                                      x[b, 0], x[b, 1], x[b, 2], x[c, 0], x[c, 1], x[c, 2])
            if u < -0.12 or vv < -0.12 or ww < -0.12:
                continue
            # triangle normal
            e1x = x[b, 0] - x[a, 0]; e1y = x[b, 1] - x[a, 1]; e1z = x[b, 2] - x[a, 2]
            e2x = x[c, 0] - x[a, 0]; e2y = x[c, 1] - x[a, 1]; e2z = x[c, 2] - x[a, 2]
            nx_ = e1y * e2z - e1z * e2y; ny_ = e1z * e2x - e1x * e2z; nz_ = e1x * e2y - e1y * e2x
            nl = np.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_)
            if nl < 1e-12:
                continue
            nx_ /= nl; ny_ /= nl; nz_ /= nl
            s = (x[p, 0] - x[a, 0]) * nx_ + (x[p, 1] - x[a, 1]) * ny_ + (x[p, 2] - x[a, 2]) * nz_
            sd = side[p, t]
            if sd == 0:
                if abs(s) < 0.25 * h:  # ambiguous first contact: put the vertex on its upper side
                    sd = 1 if nz_ * (x[p, 2] - (u * x[a, 2] + vv * x[b, 2] + ww * x[c, 2])) >= 0 else -1
                else:
                    sd = 1 if s > 0 else -1
                side[p, t] = sd
            C = s * sd - h
            if C >= 0:
                continue
            if s * sd < -0.5 * h:
                # already deep on the wrong side (another constraint dragged it through): accept the new
                # side instead of yanking it back a centimetre in one step, which explodes the cloth
                side[p, t] = -sd
                stats[0] += 1
                continue
            if C < -0.5 * h:
                C = -0.5 * h  # push at most half a thickness per substep
            uc = min(max(u, 0.0), 1.0); vc = min(max(vv, 0.0), 1.0); wc = min(max(ww, 0.0), 1.0)
            ssum = uc + vc + wc
            uc /= ssum; vc /= ssum; wc /= ssum
            W = w[p] + uc * uc * w[a] + vc * vc * w[b] + wc * wc * w[c]
            lam = -C / W
            ox = nx_ * sd; oy = ny_ * sd; oz = nz_ * sd
            x[p, 0] += w[p] * lam * ox; x[p, 1] += w[p] * lam * oy; x[p, 2] += w[p] * lam * oz
            for (q, bq) in ((a, uc), (b, vc), (c, wc)):
                x[q, 0] -= w[q] * bq * lam * ox; x[q, 1] -= w[q] * bq * lam * oy; x[q, 2] -= w[q] * bq * lam * oz
            # friction on the relative sliding during this substep
            rx = (x[p, 0] - x0[p, 0]) - (uc * (x[a, 0] - x0[a, 0]) + vc * (x[b, 0] - x0[b, 0]) + wc * (x[c, 0] - x0[c, 0]))
            ry = (x[p, 1] - x0[p, 1]) - (uc * (x[a, 1] - x0[a, 1]) + vc * (x[b, 1] - x0[b, 1]) + wc * (x[c, 1] - x0[c, 1]))
            rz = (x[p, 2] - x0[p, 2]) - (uc * (x[a, 2] - x0[a, 2]) + vc * (x[b, 2] - x0[b, 2]) + wc * (x[c, 2] - x0[c, 2]))
            rn = rx * ox + ry * oy + rz * oz
            tx = rx - rn * ox; ty = ry - rn * oy; tz = rz - rn * oz
            tl = np.sqrt(tx * tx + ty * ty + tz * tz)
            if tl > 1e-12:
                f = 1.0 if tl < mu_cloth * (-C) else mu_cloth * (-C) / tl
                lt = f / W
                x[p, 0] -= w[p] * lt * tx; x[p, 1] -= w[p] * lt * ty; x[p, 2] -= w[p] * lt * tz
                for (q, bq) in ((a, uc), (b, vc), (c, wc)):
                    x[q, 0] += w[q] * bq * lt * tx; x[q, 1] += w[q] * bq * lt * ty; x[q, 2] += w[q] * bq * lt * tz
        # --- table: never below, with friction
        for i in range(n):
            if x[i, 2] < zmin:
                pen = zmin - x[i, 2]
                x[i, 2] = zmin
                tx = x[i, 0] - x0[i, 0]; ty = x[i, 1] - x0[i, 1]
                tl = np.sqrt(tx * tx + ty * ty)
                if tl > 1e-12:
                    f = 1.0 if tl < mu_table * pen else mu_table * pen / tl
                    x[i, 0] -= tx * f; x[i, 1] -= ty * f
        # --- velocities (clamped: a correction must never turn into a launch)
        for i in range(n):
            v[i, 0] = (x[i, 0] - x0[i, 0]) / dt
            v[i, 1] = (x[i, 1] - x0[i, 1]) / dt
            v[i, 2] = (x[i, 2] - x0[i, 2]) / dt
            sp = np.sqrt(v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2)
            if sp > vmax:
                s = vmax / sp
                v[i, 0] *= s; v[i, 1] *= s; v[i, 2] *= s


# ----------------------------------------------------------------------------- cloth object
class Cloth:
    def __init__(self, pts2d, tris, spacing, thickness=0.004, mass=0.02, bend_compliance=20.0,
                 drag=1.5, mu_cloth=1.0, mu_table=0.6, substeps=16, vmax=1.2, excl_radius=0.04, teth_scale=1.01,
                 sleep_speed=0.005, sleep_time=0.5):
        # bend_compliance 20 / mu_cloth 1.0: soft like cotton, so folds stay put (flowback tests: worst 5% of
        # points move 7-13 mm after release, vs 33-37 mm with stiff bending)
        n = len(pts2d)
        self.teth_scale = teth_scale
        self.sleep_speed, self.sleep_time = sleep_speed, sleep_time
        self.still_for = 0.0
        self.asleep = False
        self.n, self.tris = n, np.ascontiguousarray(tris, np.int64)
        self.stats = np.zeros(4, np.int64)  # [side flips, ...] counters for debugging
        self.h, self.substeps = thickness, substeps
        self.bend_alpha, self.drag, self.vmax = bend_compliance, drag, vmax
        self.mu_cloth, self.mu_table = mu_cloth, mu_table
        self.x = np.c_[pts2d, np.full(n, thickness / 2)].astype(np.float64)
        self.rest = self.x.copy()
        self.v = np.zeros((n, 3))
        self.x0 = np.zeros((n, 3))
        self.w = np.full(n, n / mass)
        # edges and bending pairs from the triangles
        emap = {}
        for t, (a, b, c) in enumerate(self.tris):
            for i, j, o in ((a, b, c), (b, c, a), (c, a, b)):
                emap.setdefault((min(i, j), max(i, j)), []).append(o)
        self.edges = np.array(list(emap.keys()), np.int64)
        self.rest_e = np.linalg.norm(self.x[self.edges[:, 0]] - self.x[self.edges[:, 1]], axis=1)
        bp = [(o[0], o[1]) for o in emap.values() if len(o) == 2]
        self.bpairs = np.array(bp, np.int64).reshape(-1, 2)
        self.rest_b = np.linalg.norm(self.x[self.bpairs[:, 0]] - self.x[self.bpairs[:, 1]], axis=1)
        # self-contact bookkeeping: skip vertex/triangle pairs that are neighbours on the cloth itself
        cen = self.x[self.tris].mean(1)
        # (4 cm: a fold crease needs that much cloth to turn around; contacts inside it fight the crease)
        self.excl = np.linalg.norm(self.x[:, None, :2] - cen[None, :, :2], axis=2) < excl_radius
        T = len(self.tris)
        self.side = np.zeros((n, T), np.int8)
        self.stamp = np.zeros((n, T), np.int64)
        self.cand = np.zeros((400000, 2), np.int64)
        self.ncand = 0
        self.frame = 0
        # grippers
        self.att = {}  # gid -> (particle ids, offsets)
        self._pack_att()

    # ---- grabbing
    def _pack_att(self):
        ids, gs, offs = [], [], []
        for g, (i, o) in self.att.items():
            ids.append(i); gs.append(np.full(len(i), g)); offs.append(o)
        self.att_i = np.concatenate(ids).astype(np.int64) if ids else np.zeros(0, np.int64)
        self.att_g = np.concatenate(gs).astype(np.int64) if gs else np.zeros(0, np.int64)
        self.att_off = np.concatenate(offs).astype(np.float64) if offs else np.zeros((0, 3))
        self.attached = np.zeros(self.n, np.bool_)
        self.attached[self.att_i] = True
        # tethers: every loose point to the nearest grabbed point of each gripper; max length = shortest path
        # ALONG THE FABRIC (not a straight line: from a sleeve to the hem it goes around the armpit) + 2%
        tp, ta, td = [], [], []
        loose = ~self.attached
        for g, (i, o) in self.att.items():
            dist, src = self._fabric_distance(i)
            sel = np.where(loose & np.isfinite(dist))[0]
            tp.append(sel); ta.append(src[sel]); td.append(dist[sel] * self.teth_scale + 0.0003)
        self.teth_p = np.concatenate(tp).astype(np.int64) if tp else np.zeros(0, np.int64)
        self.teth_a = np.concatenate(ta).astype(np.int64) if ta else np.zeros(0, np.int64)
        self.teth_d = np.concatenate(td).astype(np.float64) if td else np.zeros(0)
        # the "flap": cloth within 5 cm (on the shirt itself) of anything a gripper holds - it hangs from the
        # gripper, so it must not count as the surface under the gripper (that made the arm climb forever)
        if len(self.att_i):
            d = np.linalg.norm(self.rest[:, None, :2] - self.rest[None, self.att_i, :2], axis=2).min(1)
            self.flap = d < 0.05
        else:
            self.flap = np.zeros(self.n, np.bool_)

    def _fabric_distance(self, sources):
        """Shortest path along the mesh edges (in the flat shirt) from any of `sources` to every point.
        Returns (distance, which source is nearest)."""
        import heapq
        if not hasattr(self, "_adj"):
            self._adj = [[] for _ in range(self.n)]
            for (i, j), L in zip(self.edges, self.rest_e):
                self._adj[i].append((j, L)); self._adj[j].append((i, L))
        dist = np.full(self.n, np.inf)
        src = np.full(self.n, -1, np.int64)
        heap = []
        for s in sources:
            dist[s] = 0.0; src[s] = s
            heap.append((0.0, int(s)))
        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, L in self._adj[u]:
                nd = d + L
                if nd < dist[v]:
                    dist[v] = nd; src[v] = src[u]
                    heapq.heappush(heap, (nd, v))
        return dist, src

    def surface_under(self, xy, radius=0.02):
        """Height of the top of the RESTING cloth under xy (0 = bare table): ignores the carried flap and
        anything more than 4 cm up (cloth in the air, not a stack)."""
        d = np.linalg.norm(self.x[:, :2] - xy, axis=1)
        near = (d < radius) & ~self.flap & (self.x[:, 2] < 0.04)
        return float(self.x[near, 2].max() + 0.5 * self.h) if near.any() else 0.0

    def grab(self, gid, jaw, radius=0.016, all_layers=False, yaw=0.0):
        """Grab cloth under the jaw (within `radius` in xy): only the TOP layer (points within one layer of
        the highest one there), or the whole stack (all_layers). `yaw` = jaw angle now; turning the wrist
        later turns the held cloth with it. Returns how many points were grabbed."""
        d = np.linalg.norm(self.x[:, :2] - jaw[:2], axis=1)
        near = (d < radius) & ~self.attached
        if not near.any():
            return 0
        top = self.x[near, 2].max()
        sel = np.where(near if all_layers else near & (self.x[:, 2] > top - 0.75 * self.h))[0]
        self.att[gid] = (sel, self.x[sel] - jaw)
        self.att_yaw0 = getattr(self, "att_yaw0", {})
        self.att_yaw0[gid] = yaw
        self._pack_att()
        return len(sel)

    def release(self, gid):
        self.att.pop(gid, None)
        self._pack_att()

    # ---- stepping
    def step(self, dt, grip_pos, jaws=None, grip_yaw=None):
        """Advance dt seconds. grip_pos: (2,3) jaw positions of gripper 0 and 1 (used for grabbed cloth).
        jaws: (k,4) spheres (x, y, z, r) that push loose cloth away. grip_yaw: (2,) jaw angles now."""
        self.frame += 1
        # sleeping: once ~all the cloth has been nearly motionless for sleep_time, freeze it exactly (real
        # fabric on a table comes to a dead stop; the solver otherwise creeps by ~1 mm/s). Wakes on any grab
        # or when a jaw touches the cloth.
        jaws_arr = np.zeros((0, 4)) if jaws is None else np.asarray(jaws, np.float64)
        touching = False
        if len(jaws_arr):
            d = np.linalg.norm(self.x[None, :, :] - jaws_arr[:, None, :3], axis=2) - jaws_arr[:, 3:4]
            touching = bool((d < 0.004).any())
        if len(self.att_i) or touching:
            self.asleep, self.still_for = False, 0.0
        elif self.asleep:
            self.v[:] = 0.0
            return
        prev = self.cand[:self.ncand].copy()
        self.ncand = _broad(self.x, self.tris, self.excl, self.side, self.stamp, self.frame, self.cand,
                            0.01, 0.03, prev, len(prev))
        jaws = np.zeros((0, 4)) if jaws is None else np.ascontiguousarray(jaws, np.float64)
        off = self.att_off
        if grip_yaw is not None and len(self.att_i):  # held cloth turns with the wrist
            off = off.copy()
            for g in self.att:
                ang = grip_yaw[g] - self.att_yaw0.get(g, grip_yaw[g])
                c, s = np.cos(ang), np.sin(ang)
                k = self.att_g == g
                ox, oy = off[k, 0].copy(), off[k, 1].copy()
                off[k, 0] = c * ox - s * oy
                off[k, 1] = s * ox + c * oy
        _substeps(self.x, self.v, self.x0, self.w, self.edges, self.rest_e, self.bpairs, self.rest_b,
                  self.bend_alpha, self.att_i, self.att_g, off, np.ascontiguousarray(grip_pos, np.float64),
                  jaws, self.attached, self.teth_p, self.teth_a, self.teth_d, self.cand, self.ncand, self.tris,
                  self.side, self.h, self.mu_cloth, self.mu_table, -9.81, self.drag, self.vmax, self.substeps,
                  dt / self.substeps, self.stats)
        if not len(self.att_i):
            sp = np.linalg.norm(self.v, axis=1)
            if np.percentile(sp, 99) < self.sleep_speed:
                self.still_for += dt
                if self.still_for >= self.sleep_time:
                    self.asleep = True
                    self.v[:] = 0.0
            else:
                self.still_for = 0.0

    # ---- save / restore (exact)
    def get_state(self):
        return dict(x=self.x.copy(), v=self.v.copy(), side=self.side.copy(),
                    cand=self.cand[:self.ncand].copy(), att={g: (i.copy(), o.copy()) for g, (i, o) in self.att.items()},
                    yaw0=dict(getattr(self, "att_yaw0", {})))

    def set_state(self, s):
        self.x[:] = s["x"]; self.v[:] = s["v"]; self.side[:] = s["side"]
        self.ncand = len(s["cand"]); self.cand[:self.ncand] = s["cand"]
        self.att = {g: (i.copy(), o.copy()) for g, (i, o) in s["att"].items()}
        self.att_yaw0 = dict(s.get("yaw0", {}))
        self.asleep, self.still_for = False, 0.0
        self._pack_att()

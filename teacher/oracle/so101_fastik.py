"""Faster IK for so101_kin.SO101.ik, used by the teleop server (installed with FastIK(K).install()).
Audit 2026-10-05: fd variant matches the original to 4e-11 rad, ~20x faster; analytic ~55x faster, <=1.6e-4 rad.

Same algorithm as the original (damped least squares on [tip position (3), w_dir * (approach - direction) (3)] over
joints 0..3, same damping, same step clip 0.2, same exits: pos < tol and dir < 0.03*w_dir, or |step| < 2e-4, or
iteration cap; joint clipping each iteration). Two variants:
  fd        the SAME forward-difference Jacobian (eps 1e-5), just compiled with numba
  analytic  geometric Jacobian (axis x (p - o) and axis x approach): one FK per iteration instead of five
Both fall back to the original for call shapes they don't cover (direction None, other free sets).
"""
import numpy as np
from numba import njit
import so101_kin as sk


@njit(cache=True)
def _fk(base, Tfix, rev, jq, q, O, Z):
    """Tip pose; O[k], Z[k] = origin and axis (world) of the k-th revolute joint in the chain (before its rotation)."""
    M = base.copy()
    k = 0
    for j in range(Tfix.shape[0]):
        M = M @ Tfix[j]
        if rev[j]:
            for r in range(3):
                O[k, r] = M[r, 3]; Z[k, r] = M[r, 2]
            a = q[jq[j]]
            c, s = np.cos(a), np.sin(a)
            for r in range(3):          # M = M @ rotz(a): only columns 0 and 1 change
                m0, m1 = M[r, 0], M[r, 1]
                M[r, 0] = c * m0 + s * m1
                M[r, 1] = -s * m0 + c * m1
            k += 1
    return M


@njit(cache=True)
def _res(base, Tfix, rev, jq, q, pos, d, w_dir, O, Z, r):
    M = _fk(base, Tfix, rev, jq, q, O, Z)
    for i in range(3):
        r[i] = M[i, 3] - pos[i]
        r[3 + i] = w_dir * (M[i, 2] - d[i])
    return M


@njit(cache=True)
def _ik(base, Tfix, rev, jq, colj, q0, lo, hi, pos, d, w_dir, iters, damp, tol, analytic):
    """colj[c] = index (in chain order of revolute joints) of free joint c; free joints are q[0..3]."""
    q = np.minimum(np.maximum(q0, lo), hi)
    O = np.zeros((5, 3)); Z = np.zeros((5, 3)); O2 = np.zeros((5, 3)); Z2 = np.zeros((5, 3))
    r = np.zeros(6); r2 = np.zeros(6); J = np.zeros((6, 4)); A = np.zeros((4, 4)); g = np.zeros(4)
    eps = 1e-5
    n_it = 0; why = 0                      # 0 cap, 1 converged, 2 small step
    for _ in range(iters):
        M = _res(base, Tfix, rev, jq, q, pos, d, w_dir, O, Z, r)
        if np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2) < tol and np.sqrt(r[3] ** 2 + r[4] ** 2 + r[5] ** 2) < 0.03 * w_dir:
            why = 1; break
        n_it += 1
        if analytic:
            for c in range(4):
                k = colj[c]
                zx, zy, zz = Z[k, 0], Z[k, 1], Z[k, 2]
                px, py, pz = M[0, 3] - O[k, 0], M[1, 3] - O[k, 1], M[2, 3] - O[k, 2]
                J[0, c] = zy * pz - zz * py; J[1, c] = zz * px - zx * pz; J[2, c] = zx * py - zy * px
                ax, ay, az = M[0, 2], M[1, 2], M[2, 2]
                J[3, c] = w_dir * (zy * az - zz * ay); J[4, c] = w_dir * (zz * ax - zx * az); J[5, c] = w_dir * (zx * ay - zy * ax)
        else:
            for c in range(4):
                dq = q.copy(); dq[c] += eps
                _res(base, Tfix, rev, jq, dq, pos, d, w_dir, O2, Z2, r2)
                for i in range(6):
                    J[i, c] = (r2[i] - r[i]) / eps
        for a in range(4):
            g[a] = 0.0
            for i in range(6):
                g[a] += J[i, a] * r[i]
            for b in range(4):
                s = 0.0
                for i in range(6):
                    s += J[i, a] * J[i, b]
                A[a, b] = s + (damp if a == b else 0.0)
        step = -np.linalg.solve(A, g)
        n = np.sqrt(step[0] ** 2 + step[1] ** 2 + step[2] ** 2 + step[3] ** 2)
        if n < 2e-4:
            why = 2; break
        if n > 0.2:
            step *= 0.2 / n
        for c in range(4):
            q[c] = min(max(q[c] + step[c], lo[c]), hi[c])
    M = _fk(base, Tfix, rev, jq, q, O, Z)
    err = np.sqrt((M[0, 3] - pos[0]) ** 2 + (M[1, 3] - pos[1]) ** 2 + (M[2, 3] - pos[2]) ** 2)
    return q, err, n_it, why


class FastIK:
    def __init__(self, K, analytic=False, stats=None):
        self.K, self.analytic, self.stats = K, analytic, stats
        self.orig = sk.SO101.ik
        chain = K.chain
        self.Tfix = np.ascontiguousarray(np.array([K.joints[n]["T"] for n in chain]))
        self.rev = np.array([K.joints[n]["type"] == "revolute" for n in chain])
        self.jq = np.array([sk.ARM_JOINTS.index(n) if n in sk.ARM_JOINTS else -1 for n in chain])
        revnames = [n for n in chain if K.joints[n]["type"] == "revolute"]
        assert all(n in sk.ARM_JOINTS for n in revnames), revnames
        self.colj = np.array([revnames.index(sk.ARM_JOINTS[c]) for c in range(4)])
        self.base = {s: np.ascontiguousarray(sk._T(sk.BASE_ROT, sk.BASE_POS[s])) for s in ("left", "right")}
        # compile
        self.ik(K, np.array([0.0, 0.0, 0.6]), "left", sk.HOME["left"])

    def ik(self, K, pos, side, q0, direction=(0, 0, -1), w_dir=0.005, iters=200, damp=1e-4, tol=3e-4, free=(0, 1, 2, 3)):
        if direction is None or w_dir <= 0 or tuple(free) != (0, 1, 2, 3):
            return self.orig(K, pos, side, q0, direction=direction, w_dir=w_dir, iters=iters, damp=damp, tol=tol, free=free)
        d = np.asarray(direction, float); d = d / np.linalg.norm(d)
        q, err, n_it, why = _ik(self.base[side], self.Tfix, self.rev, self.jq, self.colj, np.asarray(q0, float).copy(),
                                K.lo, K.hi, np.asarray(pos, float), d, float(w_dir), int(iters), float(damp), float(tol), self.analytic)
        if self.stats is not None:
            self.stats.append((int(iters), int(n_it), int(why)))
        return q, float(err)

    def install(self):
        f = self
        sk.SO101.ik = lambda K, pos, side, q0, **kw: f.ik(K, pos, side, q0, **kw)

    def uninstall(self):
        sk.SO101.ik = self.orig

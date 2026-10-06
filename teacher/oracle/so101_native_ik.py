"""Exact IK for the SO-101 in C++ (native/so101_ik.hpp), loaded with ctypes: no numba, no cffi, no compiler needed
where it runs (the built library is in native/lib/<platform>/, rebuilt with native/build.sh).

    nik = NativeIK(K)                                    # loads the library and checks it against so101_kin
    q4, err, tilt_deg, status = nik.solve(side, pos_world, roll, q_prev4)

It answers the teleop question in closed form: tip exactly at pos_world, wrist_roll as given, gripper as close to
straight down as the joint limits allow. status 0 = no joint angles put the tip there (err = inf, q4 = q_prev).
One solve is ~50 ns in C++, ~1 microsecond through this wrapper (the numpy IK it replaces: milliseconds).
"""
import ctypes
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIDE = {"left": 0, "right": 1}


def lib_path():
    if platform.machine().lower() not in ("amd64", "x86_64"):
        raise OSError(f"no so101_ik build for CPU type {platform.machine()}")
    sub, name = ("win-amd64", "so101_ik.dll") if sys.platform.startswith("win") else ("linux-x86_64", "libso101_ik.so")
    return os.environ.get("SO101_IK_LIB") or os.path.join(HERE, "native", "lib", sub, name)


class NativeIK:
    def __init__(self, K=None, path=None, selftest=True):
        self.path = path or lib_path()
        self.lib = ctypes.CDLL(self.path)
        self._fn = self.lib.so101_ik_solve_io
        self._fn.argtypes, self._fn.restype = [ctypes.c_void_p], ctypes.c_int
        self.lib.so101_fk.argtypes, self.lib.so101_fk.restype = [ctypes.c_int] + [ctypes.c_void_p] * 3, None
        self.lib.so101_ik_solve_n.argtypes, self.lib.so101_ik_solve_n.restype = [ctypes.c_int] + [ctypes.c_void_p] * 2, None
        self.lib.so101_ik_geom_id.restype = ctypes.c_uint64
        self.geom_id = int(self.lib.so101_ik_geom_id())
        # one buffer shared with the library: [side, x, y, z, roll, q0..q3, allow_switch | q0..q3, err, tilt, status]
        self._io = np.zeros(17)
        self._ptr = ctypes.c_void_p(self._io.ctypes.data)
        self._pos, self._qin, self._qout = self._io[1:4], self._io[5:9], self._io[10:14]   # views, made once
        self._fkb = np.zeros(11)                          # [q0..q4 | pos3 | approach3]
        self._fkp = [ctypes.c_void_p(self._fkb.ctypes.data + 8 * o) for o in (0, 5, 8)]
        # tip frame through one buffer: [side, q0..q4 | tip3 | approach3 | jaw3]
        self.lib.so101_fk_io.argtypes, self.lib.so101_fk_io.restype = [ctypes.c_void_p], None
        self._fio = np.zeros(15)
        self._fiop = ctypes.c_void_p(self._fio.ctypes.data)
        # cloth surface index
        self.lib.so101_cloth_build.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_double,
                                               ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        self.lib.so101_cloth_build.restype = ctypes.c_int
        self.lib.so101_cloth_top.argtypes, self.lib.so101_cloth_top.restype = [ctypes.c_double] * 3, ctypes.c_double
        self._cloth_top = self.lib.so101_cloth_top
        self._origin = np.zeros(3)
        self._excl = np.zeros(6)
        if selftest and K is not None:
            self.selftest(K)

    def solve(self, side, pos_world, roll, q_prev, allow_switch=True):
        """side 0/1 (left/right). -> (q4 [pan, lift, elbow, wrist_flex], position error m, tilt deg, status)."""
        io = self._io
        io[0] = side
        self._pos[:] = pos_world
        io[4] = roll
        self._qin[:] = q_prev if len(q_prev) == 4 else q_prev[:4]
        io[9] = allow_switch
        st = self._fn(self._ptr)
        return self._qout.copy(), io.item(14), io.item(15), st

    def solve_many(self, sides, pos_world, rolls, q_prev, allow_switch=True):
        """n problems in one call. -> array (n, 7): q4, err, tilt_deg, status."""
        n = len(pos_world)
        a = np.empty((n, 10))
        a[:, 0], a[:, 1:4], a[:, 4], a[:, 5:9], a[:, 9] = sides, pos_world, rolls, np.asarray(q_prev)[:, :4], allow_switch
        out = np.empty((n, 7))
        self.lib.so101_ik_solve_n(n, ctypes.c_void_p(a.ctypes.data), ctypes.c_void_p(out.ctypes.data))
        return out

    def frame(self, side, q5):
        """-> the library's (tip position, approach direction, jaw opening axis), world frame (views, overwritten by
        the next call: copy what you keep)."""
        f = self._fio
        f[0] = side
        f[1:6] = q5
        self.lib.so101_fk_io(self._fiop)
        return f[6:9], f[9:12], f[12:15]

    def cloth_build(self, pts_world, origin, excl_tips=(), zmax=0.04, excl_r=0.04):
        """Index the resting cloth points for cloth_top: local = world - origin, keep local z < zmax and not within
        excl_r of the given gripper tips (local frame). pts_world: (n, 3) float64. Returns the number kept."""
        pts = np.ascontiguousarray(pts_world, dtype=np.float64)
        self._origin[:] = origin
        k = len(excl_tips)
        for j, t in enumerate(excl_tips):
            self._excl[3 * j:3 * j + 3] = t
        n = self.lib.so101_cloth_build(pts.ctypes.data, len(pts), self._origin.ctypes.data, zmax,
                                       self._excl.ctypes.data, k, excl_r)
        if n < 0:
            raise ValueError(f"too many cloth points for the native index: {len(pts)}")
        return n

    def cloth_top(self, x, y, r):
        """Highest indexed cloth point within horizontal distance r of (x, y) (local frame); -inf if none."""
        return self._cloth_top(x, y, r)

    def fk(self, side, q5):
        """-> (tip position, approach direction), world frame: the library's forward kinematics."""
        self._fkb[:5] = q5
        self.lib.so101_fk(side, *self._fkp)
        return self._fkb[5:8].copy(), self._fkb[8:11].copy()

    def install(self):
        """Use this solver inside so101_kin.SO101.ik for the calls it covers (gripper straight down as a soft goal,
        the usual four free joints): the scripted grab / release paths. Points it cannot reach, and every other
        kind of call, go to the solver that was there before (numpy, or the numba one if that is installed)."""
        import so101_kin as sk
        orig, nik = sk.SO101.ik, self

        def ik(K, pos, side, q0, direction=(0, 0, -1), w_dir=0.005, iters=200, damp=1e-4, tol=3e-4, free=(0, 1, 2, 3)):
            if direction is not None and w_dir > 0 and tuple(free) == (0, 1, 2, 3) and tuple(direction) == (0, 0, -1):
                q = np.clip(np.asarray(q0, float), K.lo, K.hi)
                q4, err, _, st = nik.solve(SIDE[side], pos, q[4], q, False)
                if st:
                    q[:4] = q4
                    return q, err
            return orig(K, pos, side, q0, direction=direction, w_dir=w_dir, iters=iters, damp=damp, tol=tol, free=free)
        sk.SO101.ik, self._orig_ik = ik, orig

    def uninstall(self):
        import so101_kin as sk
        if getattr(self, "_orig_ik", None) is not None:
            sk.SO101.ik, self._orig_ik = self._orig_ik, None

    def selftest(self, K, n=64, seed=1):
        """The library must agree with so101_kin (it was generated from it): forward kinematics, and every answer
        of the solver must put so101_kin's tip on the target. Raises if the library is stale or broken."""
        rng = np.random.default_rng(seed)
        for i in range(n):
            name = ("left", "right")[i % 2]
            q = rng.uniform(K.lo, K.hi)
            p, a = K.tip(q, name)
            pf, af = self.fk(SIDE[name], q)
            if np.abs(pf - p).max() > 1e-9 or np.abs(af - a).max() > 1e-9:
                raise RuntimeError(f"so101_ik library does not match so101_kin (FK differs by {np.abs(pf - p).max():.2e} m): "
                                   "regenerate with native/gen_so101_ik.py and rebuild with native/build.sh")
            q4, err, tilt, st = self.solve(SIDE[name], p, q[4], q, True)
            if st:
                e = float(np.linalg.norm(K.tip(np.r_[q4, q[4]], name)[0] - p))
                if e > 1e-8 or not ((q4 >= K.lo[:4] - 1e-12) & (q4 <= K.hi[:4] + 1e-12)).all():
                    raise RuntimeError(f"so101_ik self-test failed: tip error {e:.2e} m, joints {q4}")
        return True

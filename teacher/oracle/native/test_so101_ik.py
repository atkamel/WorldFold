"""Checks of the C++ SO-101 IK (so101_ik.hpp) against things that do not share its code:
    python test_so101_ik.py          (from teacher/oracle/native; needs the built library, see build.sh)

 1. forward kinematics    C++ vs so101_kin (numpy, straight from the URDF)
 2. round trip            random reachable points: the answer must put so101_kin's tip on the point, inside the joint
                          limits, tilted no more than the pose the point came from
 3. brute force           for random points (reachable or not) scan every hand angle with a plain trigonometric
                          two-link solver: the C++ answer must tilt no more than the best of the scan, and must not
                          call a point unreachable that the scan reaches
 4. old solver            the numpy solver teleop used (damped least squares): same decisions where both succeed
 5. smoothness            sweep the table in 2 mm steps: joint angles must not jump between neighbouring points
 6. speed                 one solve through the Python wrapper, vs the numpy solver
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import so101_kin as sk  # noqa: E402
import fold_plans as fp  # noqa: E402
import gen_so101_ik as gen  # noqa: E402
from so101_native_ik import NativeIK, SIDE  # noqa: E402

K, GEO = gen.build()
NIK = NativeIK(K)
FAILS = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def tilt_of(a):
    return float(np.degrees(np.arccos(np.clip(-a[2], -1, 1))))


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def hand(rho):
    c, s = np.cos(rho), np.sin(rho)
    g = GEO
    return (g["h"][0] + c * g["h"][1] + s * g["h"][2], g["y"][0] + c * g["y"][1] + s * g["y"][2],
            g["a"][0] + c * g["a"][1] + s * g["a"][2], g["b"][0] + c * g["b"][1] + s * g["b"][2])


def brute(side, p, rho, n=7200):
    """Scan the hand's angle in the arm plane (n steps); for each, the upper arm + forearm are a plain two-link arm
    solved with the law of cosines (trigonometry, nothing shared with the C++). Returns per elbow branch (-1, +1):
    smallest tilt (deg) of any pose inside the limits, and that pose's joints; None if the scan finds none."""
    g, sd = GEO, GEO["sides"][side]
    sig, lo, hi = g["sig"], g["lo"], g["hi"]
    X, Y, Z = sd["M"] @ p + sd["c"]
    h, ylat, a, alat = hand(rho)
    R2 = X * X + Y * Y
    out = {-1.0: None, 1.0: None}
    if R2 - ylat * ylat <= 1e-10:
        return out
    r = np.sqrt(R2 - ylat * ylat)
    pan = np.arctan2(Y, X) - np.arctan2(ylat, r)
    q0 = sig[0] * wrap(pan)
    if not lo[0] <= q0 <= hi[0]:
        return out
    E0 = np.exp(1j * pan)
    V = complex(r, Z) - g["S"]
    psi = np.linspace(-np.pi, np.pi, n, endpoint=False)
    u3 = np.exp(1j * psi)
    W = V - h * u3
    A1, A2 = abs(g["l1"]), abs(g["l2"])
    cg = (np.abs(W) ** 2 - A1 * A1 - A2 * A2) / (2 * A1 * A2)
    feas = np.abs(cg) <= 1
    gam0 = np.arccos(np.clip(cg, -1, 1))
    A = a * u3
    axy = (A.real + 1j * alat) * E0
    tilt = np.degrees(np.arccos(np.clip(axy.real * sd["d"][0] + axy.imag * sd["d"][1] + A.imag * sd["d"][2], -1, 1)))
    for sgn in (1.0, -1.0):
        gam = sgn * gam0
        th1 = np.angle(W) - np.arctan2(A2 * np.sin(gam), A1 + A2 * np.cos(gam))
        r1 = wrap(th1 - np.angle(g["l1"]))
        r2 = wrap(th1 + gam - np.angle(g["l2"]))
        q1, q2, q3 = sig[1] * r1, sig[2] * wrap(r2 - r1), sig[3] * wrap(psi - r2)
        ok = feas & (q1 >= lo[1]) & (q1 <= hi[1]) & (q2 >= lo[2]) & (q2 <= hi[2]) & (q3 >= lo[3]) & (q3 <= hi[3])
        br = sig[2] * np.sign(q2 - g["q2S"])
        for b in (-1.0, 1.0):
            m = ok & (br == b)
            if m.any():
                i = np.flatnonzero(m)[np.argmin(tilt[m])]
                if out[b] is None or tilt[i] < out[b][0]:
                    out[b] = (float(tilt[i]), np.array([q0, q1[i], q2[i], q3[i]]))
    return out


def old_ik(side, p, q4, roll, seeds=True):
    """What IsaacWorld.ik did before (so101_kin's damped least squares, then a retry from the standard pose)."""
    q0 = np.r_[q4, roll]
    q, err = K.ik(p, side, q0, direction=(0, 0, -1), w_dir=0.005, iters=25)
    if seeds and err > 0.003:
        s = fp.SEED[side].copy(); s[4] = roll
        q2, e2 = K.ik(p, side, s, direction=(0, 0, -1), w_dir=0.005, iters=120)
        if e2 < err:
            q, err = q2, e2
    return q[:4], float(err), tilt_of(K.tip(q, side)[1])


def t1_fk(n=20000):
    rng = np.random.default_rng(0)
    wp = wa = 0.0
    for i in range(n):
        side = ("left", "right")[i % 2]
        q = rng.uniform(K.lo, K.hi)
        p, a = K.tip(q, side)
        pf, af = NIK.fk(SIDE[side], q)
        wp, wa = max(wp, np.abs(pf - p).max()), max(wa, np.abs(af - a).max())
    print(f"1. forward kinematics, {n} random poses: position differs by at most {wp:.1e} m, direction {wa:.1e}")
    check(wp < 1e-10 and wa < 1e-10, "C++ forward kinematics = so101_kin")


def t2_round_trip(n=60000):
    rng = np.random.default_rng(1)
    sig0 = GEO["sig"][0]
    stat = dict(front=0, behind=0, missed=0, down=0, worse=0)
    we = wt = 0.0
    for i in range(n):
        side = ("left", "right")[i % 2]
        q = rng.uniform(K.lo, K.hi)
        p, a = K.tip(q, side)
        sd = GEO["sides"][side]
        pA = sd["M"] @ p + sd["c"]
        if (complex(pA[0], pA[1]) * np.exp(-1j * sig0 * q[0])).real <= 1e-4:
            stat["behind"] += 1          # tip behind the base in this pose: the solver only answers in front
            continue
        stat["front"] += 1
        q4, err, tilt, st = NIK.solve(SIDE[side], p, q[4], q, False)
        if not st:
            stat["missed"] += 1
            continue
        stat["down"] += st == 1
        p2, a2 = K.tip(np.r_[q4, q[4]], side)
        we = max(we, float(np.linalg.norm(p2 - p)))
        wt = max(wt, abs(tilt_of(a2) - tilt))
        stat["worse"] += tilt > tilt_of(a) + 1e-3
        if not ((q4 >= K.lo[:4]) & (q4 <= K.hi[:4])).all():
            FAILS.append(f"joint outside its limits: {q4}")
    print(f"2. round trip, {stat['front']} reachable points ({stat['behind']} with the arm reaching backwards skipped): "
          f"{stat['down']} answered straight down, the rest tilted")
    check(stat["missed"] == 0, f"every reachable point answered (missed {stat['missed']})")
    check(we < 1e-9, f"tip lands on the point: worst error {we:.1e} m (checked with so101_kin)")
    check(stat["worse"] == 0, f"never tilts more than the pose the point came from ({stat['worse']} cases)")
    check(wt < 1e-6, f"reported tilt = so101_kin's tilt (max difference {wt:.1e} deg)")


def t3_brute(n=3000):
    rng = np.random.default_rng(2)
    q2S, sig2 = GEO["q2S"], GEO["sig"][2]
    n_reach = n_unreach = miss = worse = extra = 0
    gaps, we = [], 0.0
    for i in range(n):
        side = ("left", "right")[i % 2]
        base = sk.BASE_POS[side]
        if i % 3 == 0:                   # points on real poses: mostly reachable
            p = K.tip(rng.uniform(K.lo, K.hi), side)[0] + rng.normal(0, 0.01, 3)
        else:                            # points in a box around the robot: reachable or not
            p = base + np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.15, 0.50), rng.uniform(-0.10, 0.35)])
        rho = rng.uniform(K.lo[4], K.hi[4])
        ref = brute(side, p, rho)
        for b in (-1.0, 1.0):
            qprev = np.array([0.0, 0.0, q2S + sig2 * b * 0.5, 0.0])     # an elbow on branch b
            q4, err, tilt, st = NIK.solve(SIDE[side], p, rho, qprev, False)
            if st:
                n_reach += 1
                p2, a2 = K.tip(np.r_[q4, rho], side)
                we = max(we, float(np.linalg.norm(p2 - p)))
                if ref[b] is None:
                    extra += 1           # the 7200-step scan stepped over a narrow reachable window
                else:
                    gaps.append(ref[b][0] - tilt)
                    worse += tilt > ref[b][0] + 1e-3
            else:
                n_unreach += 1
                miss += ref[b] is not None
    gaps = np.array(gaps)
    print(f"3. brute force, {n} points x 2 elbow branches: {n_reach} reachable, {n_unreach} not")
    check(miss == 0, f"never calls a point unreachable that the scan reaches ({miss} cases)")
    check(worse == 0, f"never tilts more than the best pose of the scan (worst excess {max(0.0, -gaps.min()):.1e} deg: "
                      "'down' is taken along the pan axis, which the URDF has 1.5e-4 deg off vertical)")
    check(we < 1e-9, f"tip lands on the point: worst error {we:.1e} m")
    print(f"       scan's best tilt minus the C++ tilt: median {np.median(gaps):.4f} deg, max {gaps.max():.4f} deg "
          f"(the scan's step is 0.05 deg); {extra} points only the C++ reached (windows narrower than the scan's step)")
    check(np.percentile(gaps, 99) < 0.2, "the scan's best pose is the C++ answer (within the scan's resolution)")


def t4_old(n_paths=24, steps=120):
    """Teleop-like: the aim point walks over the table in 4 mm steps; each solver starts from its own last answer,
    as the controller does. The old solver may stop up to 4 mm short of a point (the controller's rule), the new one
    lands on it, so tilts are compared only where the old one actually reached the point."""
    rng = np.random.default_rng(3)
    O = np.array([0.0, -0.19, 0.53])     # table frame -> world, as in a live session
    both = only_new = only_old = neither = n = 0
    dq, dt, old_err, t_old, cut = [], [], [], 0.0, []
    for k in range(n_paths):
        side = ("left", "right")[k % 2]
        sx = -1 if side == "left" else 1
        roll = fp.ROLL_NEAR[side] + rng.uniform(-0.4, 0.4)
        p = O + np.array([sx * 0.15, 0.20, 0.06])
        qo = qn = sk.HOME[side][:4].copy()
        goal = p.copy()
        for _ in range(steps):
            if np.linalg.norm(goal - p) < 0.004:
                goal = O + np.array([sx * rng.uniform(-0.04, 0.33), rng.uniform(-0.07, 0.44), rng.uniform(0.01, 0.18)])
            p = p + (goal - p) / np.linalg.norm(goal - p) * 0.004
            t = time.perf_counter()
            q1, eo, to = old_ik(side, p, qo, roll)
            t_old += time.perf_counter() - t
            q2, en, tn, st = NIK.solve(SIDE[side], p, roll, qn, True)
            a_old, a_new = (eo < 0.004 and to < 45.0), (st != 0 and tn < 45.0)
            n += 1
            both += a_old and a_new; only_new += a_new and not a_old; only_old += a_old and not a_new
            neither += not a_old and not a_new
            if a_old:
                qo = q1; old_err.append(eo)
            if a_new:
                qn = q2
            if a_old and a_new and eo < 5e-5:
                dq.append(np.abs(q1 - q2).max()); dt.append(tn - to)
            if a_old and not a_new:
                cut.append(eo)
    dq, dt, old_err = np.array(dq), np.array(dt), np.array(old_err)
    print(f"4. old numpy solver on {n_paths} teleop-like paths ({n} steps of 4 mm; controller's rule: error < 4 mm, tilt < 45 deg):")
    print(f"       both accept {both}, both refuse {neither}, only the new accepts {only_new}, only the old accepts {only_old}")
    print(f"       old solver's position error where it accepts: median {np.median(old_err) * 1000:.3f} mm, max {old_err.max() * 1000:.2f} mm "
          f"(new: exact)")
    if len(cut):
        print(f"       'only the old accepts': the old answer was {np.min(cut) * 1000:.2f}-{np.max(cut) * 1000:.2f} mm off the point "
              f"(points just outside the reachable area)")
    print(f"       where the old one reached the point ({len(dq)} steps): joints differ by median {np.median(dq):.1e} rad, "
          f"max {dq.max():.3f}; new tilt minus old tilt: median {np.median(dt):+.4f} deg, worst {dt.max():+.4f} deg")
    print(f"       old solver: {t_old / n * 1000:.2f} ms per step on this machine")
    check(dt.max() < 0.05, "where the old solver reached the point, the new answer tilts no more")
    check(len(cut) == 0 or np.min(cut) > 2e-5, "the old solver only 'wins' where it did not reach the point")
    return t_old / n


def t5_smooth():
    """Sweep the table on straight lines in 2 mm steps (each solve starts from the previous answer). Away from the
    edge of the reachable area, neighbouring points must give neighbouring joint angles. Within 4 mm of the edge an
    exact solver's joints move fast by nature (like an elbow snapping straight): reported, not asserted."""
    O = np.array([0.0, -0.19, 0.53])
    inner, edge, inner_arc, n_pts, n_reach = (0.0, None), 0.0, 0.0, 0, 0
    for side in ("left", "right"):
        sx = -1 if side == "left" else 1
        sd = GEO["sides"][side]
        for roll in (fp.ROLL_NEAR[side], 0.0):
            for z in (0.015, 0.05, 0.10, 0.16):
                for axis in (0, 1):                      # lines along x, then along y
                    for other in np.arange(-0.06, 0.45, 0.03) if axis == 0 else np.arange(-0.34, 0.341, 0.03):
                        line = np.arange(-0.34, 0.341, 0.002) if axis == 0 else np.arange(-0.08, 0.441, 0.002)
                        q, sols = sk.HOME[side][:4].copy(), []
                        for v in line:
                            loc = np.array([v, other, z]) if axis == 0 else np.array([other, v, z])
                            if sx * loc[0] < -0.06:      # far across the centre line: the other arm's side
                                sols.append(None)
                                continue
                            n_pts += 1
                            q4, err, tilt, st = NIK.solve(SIDE[side], O + loc, roll, q, False)
                            if not st or tilt >= 45.0:
                                sols.append(None)
                                continue
                            n_reach += 1
                            pa = sd["M"] @ (O + loc) + sd["c"]
                            sols.append((q4, float(np.hypot(pa[0], pa[1])), loc))
                            q = q4
                        for i in range(len(sols) - 1):
                            if sols[i] is None or sols[i + 1] is None:
                                continue
                            d = float(np.abs(sols[i + 1][0][1:] - sols[i][0][1:]).max())      # lift, elbow, wrist
                            arc = abs(sols[i + 1][0][0] - sols[i][0][0]) * sols[i + 1][1]       # pan, as tip travel
                            if all(0 <= j < len(sols) and sols[j] is not None for j in range(i - 2, i + 4)):
                                inner_arc = max(inner_arc, arc)
                                if d > inner[0]:
                                    inner = (d, (side, roll, sols[i + 1][2].round(3).tolist()))
                            else:
                                edge = max(edge, d)
    print(f"5. smoothness: {n_pts} points on 2 mm lines over the table ({n_reach} reachable within 45 deg of tilt)")
    print(f"       away from the edge: largest change of lift / elbow / wrist between neighbours {inner[0]:.4f} rad "
          f"({np.degrees(inner[0]):.2f} deg) at {inner[1]}; largest pan move, as tip travel, {inner_arc * 1000:.2f} mm")
    print(f"       within 4 mm of the edge of the reachable area: up to {edge:.3f} rad ({np.degrees(edge):.1f} deg) per 2 mm")
    check(inner[0] < 0.12 and inner_arc < 0.004, "no jumps: neighbouring points give neighbouring joint angles")


def t6_speed(t_old):
    rng = np.random.default_rng(4)
    O = np.array([0.0, -0.19, 0.53])
    P = O + np.column_stack([rng.uniform(-0.3, 0.0, 2000), rng.uniform(0.0, 0.3, 2000), rng.uniform(0.02, 0.1, 2000)])
    q = sk.HOME["left"][:4].copy()
    t = time.perf_counter()
    for _ in range(50):
        for p in P:
            NIK.solve(0, p, -0.8, q, True)
    us = (time.perf_counter() - t) / (50 * len(P)) * 1e6
    t = time.perf_counter()
    for _ in range(50):
        NIK.solve_many(np.zeros(len(P)), P, np.full(len(P), -0.8), np.tile(q, (len(P), 1)))
    us_n = (time.perf_counter() - t) / (50 * len(P)) * 1e6
    print(f"6. speed from Python: {us:.2f} microseconds per solve ({us_n * 1000:.0f} ns each when batched); "
          f"old numpy solver {t_old * 1e6:.0f} microseconds: {t_old * 1e6 / us:.0f}x")


if __name__ == "__main__":
    print(f"library: {NIK.path}")
    t1_fk()
    t2_round_trip()
    t3_brute()
    t_old = t4_old()
    t5_smooth()
    t6_speed(t_old)
    print("\nALL CHECKS PASSED" if not FAILS else f"\n{len(FAILS)} CHECK(S) FAILED:\n  " + "\n  ".join(FAILS))
    sys.exit(1 if FAILS else 0)

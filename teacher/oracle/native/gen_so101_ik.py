"""Generator for the SO-101 closed-form IK kernel (so101_ik.hpp): reads the arm model once (so101_kin: the challenge
URDF + where the two bases stand) and writes so101_ik_gen.h, the constants the C++ needs. Same idea as IKFast's
generation step, specialised to this arm's layout, and it runs in milliseconds:

    python gen_so101_ik.py            (from teacher/oracle/native; then build.sh)

The arm: pan (vertical axis) -> lift, elbow, wrist_flex (three PARALLEL axes) -> wrist_roll -> jaw tip. For a given
roll the tip therefore moves like a planar 3-link chain inside a vertical plane that the pan joint turns. In that plane
every rotation is a unit complex number, so the solver needs no trigonometry until the final joint angles.

The generator checks the layout it relies on (parallel / perpendicular axes) and that its planar model reproduces
so101_kin's forward kinematics to 1e-10 m (measured: ~5e-12 m, the URDF's own rounding of its angles); it refuses to
write the header otherwise.
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import so101_kin as sk  # noqa: E402

N_ATAN = 128                       # atan2 lookup table size (see atan2u in so101_ik.hpp)


def build():
    K = sk.SO101()
    names = K.chain
    assert names == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper_frame_joint"], names
    T0, T1, T2, T3, T4, T5 = (K.joints[n]["T"] for n in names)
    G1 = T0 @ T1; G2 = G1 @ T2; G3 = G2 @ T3            # joint frames at the zero pose (base_link coordinates)
    o0, k0 = T0[:3, 3], T0[:3, 2]                        # a point on the pan axis, the pan axis
    zA = -k0 if k0[2] < 0 else k0.copy()                 # frame A: z along the pan axis (up), y = lift axis, x = y x z
    sig0 = float(np.sign(k0 @ zA))                       # +q0 turns the arm by sig0*q0 about zA
    h1 = G1[:3, 2]
    assert abs(h1 @ zA) < 1e-9, "lift axis is not perpendicular to the pan axis"
    yA = h1 - (h1 @ zA) * zA; yA /= np.linalg.norm(yA)
    xA = np.cross(yA, zA)
    RA = np.column_stack([xA, yA, zA])
    sig = [sig0]
    for G in (G1, G2, G3):                               # pitch joints: axis = +-yA; +q about +y turns x towards -z
        s = G[:3, 2] @ yA
        assert abs(abs(s) - 1) < 1e-12, "lift / elbow / wrist_flex axes are not parallel"
        sig.append(float(-np.sign(s)))

    def cz(v):                                           # in-plane part of a frame-A vector as a complex number
        return complex(v[0], v[2])

    pS, pE, pW = (RA.T @ (G[:3, 3] - o0) for G in (G1, G2, G3))
    assert max(abs(pE[1] - pS[1]), abs(pW[1] - pS[1])) < 1e-10, "the three pitch joints are not in one plane"
    S, l1, l2 = cz(pS), cz(pE) - cz(pS), cz(pW) - cz(pE)
    # hand: tip offset from the wrist_flex joint and approach direction, as (const + cos(roll)*.. + sin(roll)*..)
    B = RA.T @ G3[:3, :3]
    R4, t4, t5, n5 = T4[:3, :3], T4[:3, 3], T5[:3, 3], T5[:3, 2]
    V = [B @ (t4 + R4 @ [0, 0, t5[2]]), B @ R4 @ [t5[0], t5[1], 0], B @ R4 @ [-t5[1], t5[0], 0]]
    Nv = [B @ R4 @ [0, 0, n5[2]], B @ R4 @ [n5[0], n5[1], 0], B @ R4 @ [-n5[1], n5[0], 0]]
    h = [cz(v) for v in V]
    y = [pW[1] + V[0][1], V[1][1], V[2][1]]
    a = [cz(v) for v in Nv]
    b = [v[1] for v in Nv]
    x5 = T5[:3, 0]                                       # jaw opening axis (tip frame x), same form as the approach
    Xv = [B @ R4 @ [0, 0, x5[2]], B @ R4 @ [x5[0], x5[1], 0], B @ R4 @ [-x5[1], x5[0], 0]]
    xk = [cz(v) for v in Xv]
    xb = [v[1] for v in Xv]

    lo, hi = K.lo.copy(), K.hi.copy()
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    assert (half[:4] < np.pi - 1e-6).all()
    assert np.abs(mid[:4]).max() < 1e-12, "pan / lift / elbow / wrist_flex limits are not symmetric: the kernel assumes +-HI"
    e = lambda j, q: np.exp(1j * sig[j] * q)             # the model's unit complex number for joint j at angle q
    E2S = (l1 / abs(l1)) * np.conj(l2 / abs(l2))          # elbow rotation that makes upper arm and forearm collinear
    q2S = sig[2] * np.angle(E2S)
    q2F = (q2S + 2 * np.pi) % (2 * np.pi) - np.pi         # folded = stretched + 180 deg
    assert not (lo[2] <= q2F <= hi[2]), "folded elbow inside the joint limits: branch test needs the general form"
    assert lo[2] - 2.9 < q2S < hi[2] + 2.9
    W2 = [e(2, lo[2]), e(2, hi[2])] + ([E2S] if lo[2] <= q2S <= hi[2] else [])
    geo = dict(
        sig=sig, lo=lo, hi=hi, S=S, l1=l1, l2=l2, h=h, y=y, a=a, b=b, xk=xk, xb=xb,
        lim_c=[float(np.cos(half[j])) for j in range(4)],
        E2S=E2S, q2S=float(q2S), br_home=float(sig[2] * np.sign(sk.HOME["left"][2] - q2S)),
        U1=[e(1, lo[1]), e(1, hi[1])], W2=W2, W3=[e(3, lo[3]), e(3, hi[3])],
        o0=o0, RA=RA,
    )
    sides = {}
    for i, side in enumerate(("left", "right")):
        Mw = RA.T @ sk.BASE_ROT.T
        sides[side] = dict(M=Mw, c=-RA.T @ (sk.BASE_ROT.T @ sk.BASE_POS[side] + o0), d=Mw @ np.array([0.0, 0.0, -1.0]))
    geo["sides"] = sides
    return K, geo


def model_fk(geo, side, q):
    """Tip position and approach direction (world) from the planar model: what the C++ computes."""
    sig, sd = geo["sig"], geo["sides"][side]
    c, s = np.cos(q[4]), np.sin(q[4])
    h = geo["h"][0] + c * geo["h"][1] + s * geo["h"][2]
    ylat = geo["y"][0] + c * geo["y"][1] + s * geo["y"][2]
    a = geo["a"][0] + c * geo["a"][1] + s * geo["a"][2]
    alat = geo["b"][0] + c * geo["b"][1] + s * geo["b"][2]
    E0, u1 = np.exp(1j * sig[0] * q[0]), np.exp(1j * sig[1] * q[1])
    u2 = u1 * np.exp(1j * sig[2] * q[2]); u3 = u2 * np.exp(1j * sig[3] * q[3])
    T = geo["S"] + geo["l1"] * u1 + geo["l2"] * u2 + h * u3
    xy, A = complex(T.real, ylat) * E0, a * u3
    axy = complex(A.real, alat) * E0
    Minv = np.linalg.inv(sd["M"])
    xa = (geo["xk"][0] + c * geo["xk"][1] + s * geo["xk"][2]) * u3
    xl = geo["xb"][0] + c * geo["xb"][1] + s * geo["xb"][2]
    xxy = complex(xa.real, xl) * E0
    return (Minv @ (np.array([xy.real, xy.imag, T.imag]) - sd["c"]), Minv @ np.array([axy.real, axy.imag, A.imag]),
            Minv @ np.array([xxy.real, xxy.imag, xa.imag]))


def check(K, geo, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    worst_p = worst_a = 0.0
    for i in range(n):
        side = ("left", "right")[i % 2]
        q = rng.uniform(K.lo, K.hi)
        p, ap, xp = model_fk(geo, side, q)
        P, A = K.tip(q, side)
        X = K.fk(q, side)[:3, 0]
        worst_p = max(worst_p, np.abs(p - P).max())
        worst_a = max(worst_a, np.abs(ap - A).max(), np.abs(xp - X).max())
    assert worst_p < 1e-10 and worst_a < 1e-10, (worst_p, worst_a)
    return worst_p, worst_a


def atan_table():
    """N_ATAN + 1 rows: the extra one catches min(|cos|, |sin|) = sin(45 deg) exactly, so the index needs no upper clamp
    beyond one unsigned compare."""
    d = np.sin(np.pi / 4) / N_ATAN
    th = np.arcsin((np.arange(N_ATAN + 1) + 0.5) * d)
    return np.column_stack([np.cos(th), np.sin(th), th, np.zeros(N_ATAN + 1)]), 1.0 / d


def emit(geo, path):
    vals = []                                            # every number written, for the geometry id

    def f(x):
        x = float(x); vals.append(x)
        return float.hex(x)

    def cx(z):
        return "{" + f(z.real) + ", " + f(z.imag) + "}"

    def arr(v):
        return "{" + ", ".join(f(x) for x in v) + "}"

    def cxarr(v):
        return "{" + ", ".join(cx(z) for z in v) + "}"

    tab, tab_scale = atan_table()
    L = []
    w = L.append
    w("// GENERATED by gen_so101_ik.py from so101_kin (so101_new_calib.urdf + base placement). Do not edit: regenerate.")
    w("// Planar model of the SO-101: see gen_so101_ik.py. All numbers are exact doubles (hex float literals).")
    w("#pragma once")
    w("#include <cstdint>")
    w("namespace so101 { namespace gen {")
    w("struct C { double re, im; };")
    w(f"constexpr double SIG[4] = {arr(geo['sig'])};            // joint j in the model is exp(i * SIG[j] * q_j)")
    w(f"constexpr double LO[5] = {arr(geo['lo'])};")
    w(f"constexpr double HI[5] = {arr(geo['hi'])};")
    w(f"constexpr double LIM_C[4] = {arr(geo['lim_c'])};          // cos(HI): joint j is inside its limits when Re(e_j) >= LIM_C[j]")
    w(f"constexpr C S = {cx(geo['S'])};                        // lift joint in the arm plane (r + i z), frame A")
    w(f"constexpr C L1 = {cx(geo['l1'])};                       // lift -> elbow at the zero pose")
    w(f"constexpr C L2 = {cx(geo['l2'])};                       // elbow -> wrist_flex at the zero pose")
    w(f"constexpr C HK[3] = {cxarr(geo['h'])};                  // wrist_flex -> tip: HK[0] + cos(roll) HK[1] + sin(roll) HK[2]")
    w(f"constexpr double YK[3] = {arr(geo['y'])};               // sideways offset of the tip from the arm plane, same form")
    w(f"constexpr C AK[3] = {cxarr(geo['a'])};                  // approach direction, in-plane part, same form")
    w(f"constexpr double BK[3] = {arr(geo['b'])};               // approach direction, sideways part")
    w(f"constexpr C XK[3] = {cxarr(geo['xk'])};                 // jaw opening axis (tip frame x), in-plane part, same form")
    w(f"constexpr double XB[3] = {arr(geo['xb'])};              // jaw opening axis, sideways part")
    w(f"constexpr C E2S = {cx(geo['E2S'])};                     // elbow rotation with upper arm and forearm collinear")
    w(f"constexpr double Q2S = {f(geo['q2S'])};                 // ... as an elbow joint angle (the two elbow branches meet here)")
    w(f"constexpr double BR_HOME = {f(geo['br_home'])};         // elbow branch of the home pose")
    n1, n2 = abs(geo['l1']), abs(geo['l2'])
    hinv, sh = 0.5 / (n1 * n2), (n1 * n1 + n2 * n2) * 0.5 / (n1 * n2)
    w("// upper arm + forearm as a two-link arm reaching a point at distance^2 = d2: cos(elbow from straight) = d2*HINV - SH")
    w(f"constexpr double P0_HINV = {f(hinv)}, P0_SH = {f(sh)}, P0_K1 = {f(1 + sh)}, P0_K2 = {f(1 - sh)};")
    w(f"constexpr double P0_RHO = {f(n1 / n2)}, P0_RINV = {f(n2 / n1)};       // |L1| / |L2| and its inverse")
    w(f"constexpr C U1[2] = {cxarr(geo['U1'])};                 // lift joint at its lower / upper limit")
    w(f"constexpr int NE = {len(geo['W2'])};")
    w(f"constexpr C W2[{len(geo['W2'])}] = {cxarr(geo['W2'])};  // elbow at its limits (and collinear, if inside the limits)")
    w(f"constexpr C W3[2] = {cxarr(geo['W3'])};                 // wrist_flex at its lower / upper limit")
    w("struct Side { double M[9]; double c[3]; double d[3]; };  // world -> frame A: M p + c ; d = straight down in frame A")
    rows = []
    for side in ("left", "right"):
        sd = geo["sides"][side]
        rows.append("{" + arr(sd["M"].ravel()) + ", " + arr(sd["c"]) + ", " + arr(sd["d"]) + "}")
    w("constexpr Side SIDES[2] = {" + ",\n                            ".join(rows) + "};")
    w(f"constexpr int N_ATAN = {N_ATAN};")
    w(f"constexpr double ATAN_SCALE = {f(tab_scale)};          // table index = (int)(min(|cos|,|sin|) * ATAN_SCALE)")
    w("alignas(64) constexpr double ATAN_TAB[N_ATAN + 1][4] = {  // {cos t, sin t, t, 0} at the centre of each bin")
    for r in tab:
        w("    " + arr(r) + ",")
    w("};")
    gid = 0xcbf29ce484222325                             # FNV-1a over the bytes of every constant
    for b in struct.pack(f"<{len(vals)}d", *vals):
        gid = ((gid ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    w(f"constexpr std::uint64_t GEOM_ID = 0x{gid:016x}ULL;")
    w("}}  // namespace so101::gen")
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    return gid


if __name__ == "__main__":
    K, geo = build()
    wp, wa = check(K, geo)
    out = os.path.join(HERE, "so101_ik_gen.h")
    gid = emit(geo, out)
    print(f"model vs so101_kin FK over 4000 poses: position {wp:.2e} m, approach {wa:.2e}")
    print(f"links: upper arm {abs(geo['l1']) * 1000:.2f} mm, forearm {abs(geo['l2']) * 1000:.2f} mm; "
          f"elbow collinear at {np.degrees(geo['q2S']):.2f} deg; {len(geo['W2'])} elbow edge cases")
    print(f"wrote {out} (geometry id {gid:016x})")

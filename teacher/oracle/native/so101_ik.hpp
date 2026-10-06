// Closed-form inverse kinematics for the SO-101 arm. Header-only, no heap, no locks, no libm: + - * / sqrt and one
// small lookup table. Constants come from so101_ik_gen.h (gen_so101_ik.py reads them from the URDF).
//
// Question answered (what the teleop controller asks every tick):
//     put the jaw tip at point p, wrist_roll given, gripper pointing as straight down as the arm allows.
// Answer: pan, lift, elbow, wrist_flex with the tip EXACTLY at p and the smallest possible tilt, or "unreachable".
//
// How (see gen_so101_ik.py for the geometry):
//   * pan is one closed formula: the tip's sideways offset from the arm plane is a known constant for a given roll.
//   * In the arm plane the three parallel joints are a planar chain  V = L1 u1 + L2 u2 + h u3  where u1, u2, u3 are
//     unit complex numbers (the absolute rotations of upper arm, forearm, hand). Rotations stay complex numbers the
//     whole way, so there is no sin / cos / atan in the search, only at the very end for the four joint angles.
//   * The best hand rotation u3* (pointing down) is a constant. If the arm can do it: two-link formula, done.
//     If not, the best answer has one joint on a limit or the elbow straight, and every one of those cases is again
//     the SAME two-link formula with different link constants. So: evaluate the handful of cases, keep the one that
//     tilts least. No iteration, no starting guess, no local minima.
//   * Angles come from angle_of(): a 129-entry table + one complex multiply + a 2-term series, exact to ~2e-16 rad.
//
// Speed (it is all one dependency chain: two square roots, then the angles):
//   * values that only change with the roll are cached per arm (RollCache);
//   * the common case ("can point straight down") is one predictable branch; the edge cases only run at the edge of
//     the reachable area;
//   * everything that does not depend on a square root's result is computed while the square root is still running:
//     the two-link answer is written as (ready early) + (ready early) * S, with S = +-sqrt the last thing to arrive.
// "Down" for choosing the tilt is taken along the pan axis (the URDF has it 1.5e-4 deg off the world's vertical);
// the tilt that is reported is measured against the world's vertical.
// Not thread-safe (per-arm cache); the teleop loop is single-threaded.
#pragma once
#include <cstdint>
#include <cstring>

#include "so101_ik_gen.h"

#if defined(__GNUC__) || defined(__clang__)
#define SO101_LIKELY(x) __builtin_expect(!!(x), 1)
#define SO101_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define SO101_INLINE inline __attribute__((always_inline))
#define SO101_COLD __attribute__((noinline, cold))
#else
#define SO101_LIKELY(x) (x)
#define SO101_UNLIKELY(x) (x)
#define SO101_INLINE inline
#define SO101_COLD
#endif

namespace so101 {

using Cx = gen::C;                                      // complex number = planar vector = planar rotation

constexpr Cx operator+(Cx a, Cx b) { return {a.re + b.re, a.im + b.im}; }
constexpr Cx operator-(Cx a, Cx b) { return {a.re - b.re, a.im - b.im}; }
constexpr Cx mul(Cx a, Cx b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }      // a * b
constexpr Cx mulc(Cx a, Cx b) { return {a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im}; }     // a * conj(b)
constexpr Cx conj(Cx a) { return {a.re, -a.im}; }
constexpr Cx scale(Cx a, double k) { return {a.re * k, a.im * k}; }
constexpr double dot(Cx a, Cx b) { return a.re * b.re + a.im * b.im; }                               // Re(a conj(b))
constexpr double cross(Cx a, Cx b) { return a.im * b.re - a.re * b.im; }                             // Im(a conj(b))
constexpr double norm2(Cx a) { return a.re * a.re + a.im * a.im; }

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double HALF_PI = PI / 2;
constexpr double RAD2DEG = 180.0 / PI;
constexpr double C_TOL = 1e-10;       // slack on "is the point within the two links' reach" (in 1 - cos^2)
constexpr double LIM_TOL = 1e-12;     // slack on joint limits (in cos of the joint angle)
constexpr double BR_TOL = 1e-9;       // elbow this close to straight counts as either elbow branch

// links of the edge cases (compile-time): lift on a limit, elbow on a limit / straight
constexpr Cx L1 = gen::L1, L2 = gen::L2, E2S = gen::E2S;
constexpr Cx LU1[2] = {mul(gen::L1, gen::U1[0]), mul(gen::L1, gen::U1[1])};
constexpr Cx B12[3] = {gen::L1 + mul(gen::L2, gen::W2[0]), gen::L1 + mul(gen::L2, gen::W2[1]),
                       gen::L1 + mul(gen::L2, gen::W2[gen::NE - 1])};
static_assert(gen::NE == 2 || gen::NE == 3, "elbow edge cases: both limits, plus 'straight' when it is inside the limits");

// ---- elementary functions without libm -----------------------------------------------------------------------------

// sig * atan2(e.im, e.re) for a UNIT complex number e, clamped to [-hmax, hmax]; sig = +-1.
// Fold into the first octant, pick the nearest of 129 reference directions by table lookup, rotate onto it (one complex
// multiply, which leaves sin of the small remaining angle) and add that angle's asin series (|x| < 0.004: two terms).
// The octant bookkeeping (offset, sign) does not depend on the table, so it is ready before the series is.
SO101_INLINE double angle_of(Cx e, double sig, double hmax) noexcept {
    const double ac = __builtin_fabs(e.re), as = __builtin_fabs(e.im);
    const bool swap = as > ac, cneg = e.re < 0;
    const double hi = swap ? as : ac, lo = swap ? ac : as;
    unsigned k = static_cast<unsigned>(static_cast<int>(lo * gen::ATAN_SCALE));
    k = k < static_cast<unsigned>(gen::N_ATAN) ? k : static_cast<unsigned>(gen::N_ATAN);
    const double* t = gen::ATAN_TAB[k];
    const double x = lo * t[0] - hi * t[1];             // sin(angle - table angle)
    const double x2 = x * x;
    const double phi = (t[2] + x) + (x * x2) * (1.0 / 6.0 + x2 * (3.0 / 40.0));    // angle in the first octant
    const double off = cneg ? (swap ? HALF_PI : PI) : (swap ? HALF_PI : 0.0);
    double a = off + (swap != cneg ? -phi : phi);       // 0 .. pi
    a = a < hmax ? a : hmax;
    return ((e.im < 0) != (sig < 0)) ? -a : a;
}

// sin and cos together: quadrant reduction + Taylor series on [-pi/4, pi/4]. Only runs when a roll changes.
SO101_INLINE void sincos(double x, double& s, double& c) noexcept {
    constexpr double PIO2_HI = 0x1.921fb544p+0;         // first 33 bits of pi/2: k * PIO2_HI is exact
    constexpr double PIO2_LO = 0x1.0b4611a626331p-34;   // pi/2 - PIO2_HI
    const double kf = x * (2.0 / PI);
    const int k = static_cast<int>(kf + (kf >= 0 ? 0.5 : -0.5));
    const double r = (x - k * PIO2_HI) - k * PIO2_LO, r2 = r * r;
    const double sn = r * (1.0 + r2 * (-1.0 / 6 + r2 * (1.0 / 120 + r2 * (-1.0 / 5040 + r2 * (1.0 / 362880 + r2 * (-1.0 / 39916800
                      + r2 * (1.0 / 6227020800.0 + r2 * (-1.0 / 1307674368000.0 + r2 * (1.0 / 355687428096000.0)))))))));
    const double cs = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24 + r2 * (-1.0 / 720 + r2 * (1.0 / 40320 + r2 * (-1.0 / 3628800
                      + r2 * (1.0 / 479001600.0 + r2 * (-1.0 / 87178291200.0 + r2 * (1.0 / 20922789888000.0))))))));
    switch (k & 3) {
        case 0: s = sn; c = cs; break;
        case 1: s = cs; c = -sn; break;
        case 2: s = -sn; c = -cs; break;
        default: s = -cs; c = sn; break;
    }
}

// ---- two links reaching a point ------------------------------------------------------------------------------------
// Links a, b (complex, at zero rotation) and a point P:  a x + b y = P  with x, y unit complex numbers.
// With c = cos of the angle between the links (law of cosines) and S = +-sqrt(1 - c^2) for the two elbow sides:
//     x = P conj(a) / |P|^2 * ((1 + c / rho) + i S / rho)        rho = |a| / |b|
//     y = P conj(b) / |P|^2 * ((1 + c rho) - i S rho)
//     x conj(y) = N (c + i S)                                    N = conj(a) b / (|a| |b|)
struct PairK {
    Cx a, b, N;
    double rho, rinv;       // |a| / |b| and its inverse
    double hinv, sh;        // c = |P|^2 * hinv - sh
    double k1, k2;          // 1 - c = k1 - |P|^2 hinv,  1 + c = k2 + |P|^2 hinv
};

struct Pre {                // everything about one (pair, point) that does not depend on the elbow side
    double c, s;            // s = sqrt(1 - c^2) >= 0
    double xr, xi, xa, xb;  // x = (xr - xa S) + i (xi + xb S)
    double yr, yi, ya, yb;  // y = (yr + ya S) + i (yi - yb S)
    bool ok;                // the point is within the two links' reach
};

SO101_INLINE PairK make_pair(Cx a, Cx b) noexcept {
    const double na2 = norm2(a), nb2 = norm2(b), nab = __builtin_sqrt(na2 * nb2), inv = 1.0 / nab;
    const Cx n = mulc(b, a);                            // conj(a) b
    const double sh = (na2 + nb2) * 0.5 * inv;
    return {a, b, {n.re * inv, n.im * inv}, na2 * inv, nb2 * inv, 0.5 * inv, sh, 1.0 + sh, 1.0 - sh};
}

SO101_INLINE Pre prepare(const PairK& k, Cx P, double Pn2, double iP) noexcept {
    Pre r;
    const double t = Pn2 * k.hinv, s2 = (k.k1 - t) * (k.k2 + t);
    r.c = t - k.sh;
    r.ok = s2 >= -C_TOL;
    r.s = __builtin_sqrt(s2 > 0 ? s2 : 0.0);
    const Cx Pa = scale(mulc(P, k.a), iP), Pb = scale(mulc(P, k.b), iP);
    const double ux = 1.0 + r.c * k.rinv, uy = 1.0 + r.c * k.rho;
    r.xr = Pa.re * ux; r.xi = Pa.im * ux; r.xa = Pa.im * k.rinv; r.xb = Pa.re * k.rinv;
    r.yr = Pb.re * uy; r.yi = Pb.im * uy; r.ya = Pb.im * k.rho; r.yb = Pb.re * k.rho;
    return r;
}
SO101_INLINE Cx lane_x(const Pre& r, double S) noexcept { return {r.xr - r.xa * S, r.xi + r.xb * S}; }
SO101_INLINE Cx lane_y(const Pre& r, double S) noexcept { return {r.yr + r.ya * S, r.yi - r.yb * S}; }
SO101_INLINE Cx lane_z(const PairK& k, const Pre& r, double S) noexcept {
    return {k.N.re * r.c - k.N.im * S, k.N.im * r.c + k.N.re * S};
}

// ---- per-arm cache of everything that depends only on the wrist roll -----------------------------------------------

struct alignas(64) RollCache {
    std::uint64_t key;      // bit pattern of the roll these values are for
    std::uint32_t ready, edge;
    double ylat, ylat2;     // sideways offset of the tip from the arm plane
    Cx h;                   // wrist_flex -> tip in the arm plane (zero pitch)
    Cx a;                   // approach direction in the arm plane (zero pitch)
    double alat;            // approach direction, sideways part
    Cx u3s;                 // hand rotation that points the gripper down
    Cx K0;                  // S + h u3s: lift joint -> wrist point is (r, Z) - K0 when the hand points down
    Cx UL;                  // u3s L2: wrist_flex rotation = UL conj(P) / |P|^2 * (...)
    // edge cases only (filled the first time an edge case is needed for this roll)
    PairK L;                // links (L2, h): lift on a limit
    PairK E[3];             // links (B12[k], h): elbow on a limit / straight
    PairK W[2];             // links (L1, L2 + h W3[k]): wrist_flex on a limit
};

inline RollCache g_cache[2] = {};

SO101_INLINE std::uint64_t bits_of(double x) noexcept {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

SO101_COLD inline void roll_update(RollCache& rc, double rho) noexcept {
    double s, c;
    sincos(rho, s, c);
    rc.h = {gen::HK[0].re + c * gen::HK[1].re + s * gen::HK[2].re, gen::HK[0].im + c * gen::HK[1].im + s * gen::HK[2].im};
    rc.a = {gen::AK[0].re + c * gen::AK[1].re + s * gen::AK[2].re, gen::AK[0].im + c * gen::AK[1].im + s * gen::AK[2].im};
    rc.ylat = gen::YK[0] + c * gen::YK[1] + s * gen::YK[2];
    rc.ylat2 = rc.ylat * rc.ylat;
    rc.alat = gen::BK[0] + c * gen::BK[1] + s * gen::BK[2];
    const double ia = 1.0 / __builtin_sqrt(norm2(rc.a));
    rc.u3s = {-rc.a.im * ia, -rc.a.re * ia};            // -i conj(a) / |a|: turns the approach direction to -z
    rc.K0 = gen::S + mul(rc.h, rc.u3s);
    rc.UL = mul(rc.u3s, L2);
    rc.key = bits_of(rho);
    rc.ready = 1;
    rc.edge = 0;
}

SO101_COLD inline void roll_update_edge(RollCache& rc) noexcept {
    rc.L = make_pair(L2, rc.h);
    for (int k = 0; k < gen::NE; ++k) rc.E[k] = make_pair(B12[k], rc.h);
    for (int k = 0; k < 2; ++k) rc.W[k] = make_pair(L1, L2 + mul(rc.h, gen::W3[k]));
    rc.edge = 1;
}

// ---- the solver ----------------------------------------------------------------------------------------------------

enum Status : int {
    UNREACHABLE = 0,        // no joint angles inside the limits put the tip there (out[6] says which test failed)
    OK_DOWN = 1,            // gripper pointing straight down
    OK_TILTED = 2,          // reached with the smallest possible tilt (a joint limit or a straight elbow is active)
    OK_SWITCHED = 3,        // reached only with the other elbow branch (returned when allow_switch is set)
};

struct Lane {               // one candidate answer: the three pitch joints as rotations
    Cx e1, e2, e3;
    double score;           // cos of the hand's angle from pointing down: higher = less tilt
};

SO101_INLINE bool in_lim(int j, Cx e) noexcept { return e.re >= gen::LIM_C[j] - LIM_TOL; }   // limits are +-HI[j]

SO101_INLINE int fail(const double* qprev, double code, double* out) noexcept {
    for (int i = 0; i < 4; ++i) out[i] = qprev ? qprev[i] : 0.0;
    out[4] = __builtin_inf();
    out[5] = 180.0;
    out[6] = code;
    return UNREACHABLE;
}

SO101_INLINE int finish(const gen::Side& sd, const RollCache& rc, Cx E0, Cx V, const Lane& ln, int status,
                        double* out) noexcept {
    out[0] = angle_of(E0, gen::SIG[0], gen::HI[0]);
    out[1] = angle_of(ln.e1, gen::SIG[1], gen::HI[1]);
    out[2] = angle_of(ln.e2, gen::SIG[2], gen::HI[2]);
    out[3] = angle_of(ln.e3, gen::SIG[3], gen::HI[3]);
    // what this answer really does (forward kinematics from the rotations being returned)
    const Cx u1 = ln.e1, u2 = mul(u1, ln.e2), u3 = mul(u2, ln.e3);
    const Cx res = mul(L1, u1) + mul(L2, u2) + mul(rc.h, u3) - V;
    out[4] = __builtin_sqrt(norm2(res));                // position error (m): rounding only
    const Cx A = mul(rc.a, u3);                         // approach direction: in the plane, then turned by the pan
    const Cx Axy = mul(Cx{A.re, rc.alat}, E0);
    const double ax = Axy.re, ay = Axy.im, az = A.im;
    const double cosT = ax * sd.d[0] + ay * sd.d[1] + az * sd.d[2];
    const double cx = ay * sd.d[2] - az * sd.d[1], cy = az * sd.d[0] - ax * sd.d[2], cz = ax * sd.d[1] - ay * sd.d[0];
    out[5] = angle_of(Cx{cosT, __builtin_sqrt(cx * cx + cy * cy + cz * cz)}, 1.0, PI) * RAD2DEG;   // tilt (deg)
    out[6] = static_cast<double>(status);
    return status;
}

// side: 0 left, 1 right.  p: tip target, world frame (m).  rho: wrist_roll (rad).
// qprev: current pan, lift, elbow, wrist_flex (rad) or null; only the elbow is used, to stay on the same elbow branch.
// allow_switch: if that branch cannot reach p, may the other branch be returned?
// out[0..3] joint angles, out[4] position error (m), out[5] tilt (deg), out[6] status / failed test. Returns Status.
SO101_INLINE int solve(int side, const double* p, double rho, const double* qprev, int allow_switch, double* out) noexcept {
    const gen::Side& sd = gen::SIDES[side];
    RollCache& rc = g_cache[side];
    if (SO101_UNLIKELY(rc.key != bits_of(rho) || !rc.ready)) roll_update(rc, rho);

    // world -> frame A (z = pan axis). Pan: the tip sits rc.ylat beside the arm plane, which fixes the plane's heading.
    const double X = (sd.M[0] * p[0] + sd.M[1] * p[1]) + (sd.M[2] * p[2] + sd.c[0]);
    const double Y = (sd.M[3] * p[0] + sd.M[4] * p[1]) + (sd.M[5] * p[2] + sd.c[1]);
    const double Z = (sd.M[6] * p[0] + sd.M[7] * p[1]) + (sd.M[8] * p[2] + sd.c[2]);
    const double R2 = X * X + Y * Y, r2 = R2 - rc.ylat2;
    if (SO101_UNLIKELY(!(r2 > 1e-10))) return fail(qprev, -1.0, out);         // on the pan axis (or NaN input)
    const double r = __builtin_sqrt(r2), iR2 = 1.0 / R2;
    const Cx E0 = {(X * r + Y * rc.ylat) * iR2, (Y * r - X * rc.ylat) * iR2};  // pan as a rotation
    if (SO101_UNLIKELY(!in_lim(0, E0))) return fail(qprev, -2.0, out);         // pan limit
    const Cx V = {r - gen::S.re, Z - gen::S.im};                               // lift joint -> tip, in the arm plane

    // elbow branch to stay on: which side of "straight" the elbow is now (+1 / -1 = sign of sin(elbow from straight))
    double bpref = gen::BR_HOME;
    if (qprev) {
        const double d = qprev[2] - gen::Q2S;
        if (d > BR_TOL) bpref = gen::SIG[2];
        else if (d < -BR_TOL) bpref = -gen::SIG[2];
    }

    // ---- common case: hand straight down; upper arm + forearm reach the wrist point P ----
    const Cx P = {r - rc.K0.re, Z - rc.K0.im};
    const double Pn2 = norm2(P);
    const double t = Pn2 * gen::P0_HINV, s20 = (gen::P0_K1 - t) * (gen::P0_K2 + t), c0 = t - gen::P0_SH;
    const bool c0_ok = s20 >= -C_TOL && Pn2 > 1e-12;
    const double s0 = __builtin_sqrt(s20 > 0 ? s20 : 0.0), iP = 1.0 / Pn2;
    // ready before s0 is: upper arm u1 = (xr - xa S) + i (xi + xb S), wrist_flex e3 = (gr - ga S) + i (gi + gb S),
    // elbow e2 = E2S (c0 - i S)
    const Cx Pa = scale(mulc(P, L1), iP), G3 = scale(mul(conj(P), rc.UL), iP);
    const double ux = 1.0 + c0 * gen::P0_RINV, uy = 1.0 + c0 * gen::P0_RHO;
    const double xr = Pa.re * ux, xi = Pa.im * ux, xa = Pa.im * gen::P0_RINV, xb = Pa.re * gen::P0_RINV;
    const double gr = G3.re * uy, gi = G3.im * uy, ga = G3.im * gen::P0_RHO, gb = G3.re * gen::P0_RHO;
    const double er = E2S.re * c0, ei = E2S.im * c0;
    Lane other{};                                        // best answer on the other elbow branch
    bool has_other = false;
    if (SO101_LIKELY(c0_ok)) {
        for (int pass = 0; pass < 2; ++pass) {           // pass 0: the branch the elbow is on (its side = -sign(S))
            const double S = ((pass == 0) == (bpref > 0)) ? -s0 : s0;
            Lane ln;
            ln.e1 = {xr - xa * S, xi + xb * S};
            ln.e2 = {er + E2S.im * S, ei - E2S.re * S};
            ln.e3 = {gr - ga * S, gi + gb * S};
            ln.score = 1.0;
            if (in_lim(1, ln.e1) && in_lim(2, ln.e2) && in_lim(3, ln.e3)) {
                if (SO101_LIKELY(pass == 0)) return finish(sd, rc, E0, V, ln, OK_DOWN, out);
                other = ln; has_other = true;
            }
            if (s0 == 0.0) break;                        // elbow exactly straight: both branches are the same pose
        }
    }

    // ---- edge of the reachable area: the least-tilted answer has one joint on a limit or the elbow straight ----
    if (SO101_UNLIKELY(!rc.edge)) roll_update_edge(rc);
    Lane best{};
    bool has_best = false;
    const auto consider = [&](const Lane& ln) noexcept {
        const bool same = cross(ln.e2, E2S) * bpref >= -BR_TOL;
        Lane& slot = same ? best : other;
        bool& has = same ? has_best : has_other;
        if (!has || ln.score > slot.score) { slot = ln; has = true; }
    };
    for (int k = 0; k < 2; ++k) {                        // lift on a limit: forearm + hand reach from the fixed elbow
        const Cx Pk = V - LU1[k];
        const double n2 = norm2(Pk);
        if (n2 < 1e-12) continue;
        const Pre pr = prepare(rc.L, Pk, n2, 1.0 / n2);
        if (!pr.ok) continue;
        for (int sg = 0; sg < 2; ++sg) {
            const double S = sg ? -pr.s : pr.s;
            const Cx x = lane_x(pr, S), y = lane_y(pr, S);          // forearm u2, hand u3
            Lane ln;
            ln.e1 = gen::U1[k]; ln.e2 = mulc(x, gen::U1[k]); ln.e3 = conj(lane_z(rc.L, pr, S)); ln.score = dot(y, rc.u3s);
            if (in_lim(2, ln.e2) && in_lim(3, ln.e3)) consider(ln);
        }
    }
    const double Vn2 = norm2(V);
    if (SO101_LIKELY(Vn2 > 1e-12)) {
        const double iV = 1.0 / Vn2;
        for (int k = 0; k < gen::NE; ++k) {              // elbow on a limit / straight: arm as one link + hand
            const Pre pr = prepare(rc.E[k], V, Vn2, iV);
            if (!pr.ok) continue;
            for (int sg = 0; sg < 2; ++sg) {
                const double S = sg ? -pr.s : pr.s;
                const Cx x = lane_x(pr, S), y = lane_y(pr, S);      // upper arm u1, hand u3
                Lane ln;
                ln.e1 = x; ln.e2 = gen::W2[k]; ln.e3 = conj(mul(lane_z(rc.E[k], pr, S), gen::W2[k])); ln.score = dot(y, rc.u3s);
                if (in_lim(1, ln.e1) && in_lim(3, ln.e3)) consider(ln);
            }
        }
        for (int k = 0; k < 2; ++k) {                    // wrist_flex on a limit: upper arm + (forearm and hand as one)
            const Pre pr = prepare(rc.W[k], V, Vn2, iV);
            if (!pr.ok) continue;
            for (int sg = 0; sg < 2; ++sg) {
                const double S = sg ? -pr.s : pr.s;
                const Cx x = lane_x(pr, S), y = lane_y(pr, S);      // upper arm u1, forearm u2
                Lane ln;
                ln.e1 = x; ln.e2 = conj(lane_z(rc.W[k], pr, S)); ln.e3 = gen::W3[k]; ln.score = dot(mul(y, gen::W3[k]), rc.u3s);
                if (in_lim(1, ln.e1) && in_lim(2, ln.e2)) consider(ln);
            }
        }
    }
    if (has_best) return finish(sd, rc, E0, V, best, OK_TILTED, out);
    if (has_other && allow_switch) return finish(sd, rc, E0, V, other, OK_SWITCHED, out);
    return fail(qprev, has_other ? -4.0 : -3.0, out);    // -4: only the other elbow branch reaches it
}

// Forward kinematics from the same model (the controller's tip / jaw-direction queries, and the tests).
// q5: pan, lift, elbow, wrist_flex, wrist_roll.  World frame: pos3 tip position, appr3 approach direction (tip z),
// jaw3 jaw opening axis (tip x). appr3 / jaw3 may be null.
inline void fk(int side, const double* q5, double* pos3, double* appr3, double* jaw3 = nullptr) noexcept {
    const gen::Side& sd = gen::SIDES[side];
    Cx e[4];
    for (int j = 0; j < 4; ++j) sincos(gen::SIG[j] * q5[j], e[j].im, e[j].re);
    double s, c;
    sincos(q5[4], s, c);
    const Cx h = {gen::HK[0].re + c * gen::HK[1].re + s * gen::HK[2].re, gen::HK[0].im + c * gen::HK[1].im + s * gen::HK[2].im};
    const Cx a = {gen::AK[0].re + c * gen::AK[1].re + s * gen::AK[2].re, gen::AK[0].im + c * gen::AK[1].im + s * gen::AK[2].im};
    const double ylat = gen::YK[0] + c * gen::YK[1] + s * gen::YK[2], alat = gen::BK[0] + c * gen::BK[1] + s * gen::BK[2];
    const Cx u1 = e[1], u2 = mul(u1, e[2]), u3 = mul(u2, e[3]);
    const Cx T = gen::S + mul(L1, u1) + mul(L2, u2) + mul(h, u3);
    const Cx xy = mul(Cx{T.re, ylat}, e[0]), A = mul(a, u3), axy = mul(Cx{A.re, alat}, e[0]);
    const Cx xk = {gen::XK[0].re + c * gen::XK[1].re + s * gen::XK[2].re, gen::XK[0].im + c * gen::XK[1].im + s * gen::XK[2].im};
    const Cx X = mul(xk, u3), xxy = mul(Cx{X.re, gen::XB[0] + c * gen::XB[1] + s * gen::XB[2]}, e[0]);
    const double pa[3] = {xy.re - sd.c[0], xy.im - sd.c[1], T.im - sd.c[2]}, aa[3] = {axy.re, axy.im, A.im};
    const double xa[3] = {xxy.re, xxy.im, X.im};
    for (int i = 0; i < 3; ++i) {                        // frame A -> world: M is a rotation, its inverse is its transpose
        pos3[i] = sd.M[i] * pa[0] + sd.M[3 + i] * pa[1] + sd.M[6 + i] * pa[2];
        if (appr3) appr3[i] = sd.M[i] * aa[0] + sd.M[3 + i] * aa[1] + sd.M[6 + i] * aa[2];
        if (jaw3) jaw3[i] = sd.M[i] * xa[0] + sd.M[3 + i] * xa[1] + sd.M[6 + i] * xa[2];
    }
}

}  // namespace so101

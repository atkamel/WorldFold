// Correctness sweep + latency benchmark for the SO-101 closed-form IK (so101_ik.hpp).
//   ./bench_so101_ik            sweep 4M random poses, then per-call latency percentiles (rdtscp) and throughput
//   ./bench_so101_ik 20000000   bigger sweep
// Sweep: random joint angles inside the limits -> forward kinematics -> ask the solver for that tip point.
// It must (1) find an answer, (2) land on the point, (3) tilt no more than the pose the point came from.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "so101_ik.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
// Fenced timestamps: without the lfence the CPU starts executing the next call before the previous one's end stamp
// is taken, and independent calls look faster than one call really is.
static inline std::uint64_t tsc_begin() {
    _mm_lfence();
    const std::uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}
static inline std::uint64_t tsc_end() {
    unsigned aux;
    const std::uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}
#else
static inline std::uint64_t tsc_begin() {
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
}
static inline std::uint64_t tsc_end() { return tsc_begin(); }
#endif
#if defined(_WIN32)
#include <windows.h>
static void pin() {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
}
#else
#include <sched.h>
static void pin() {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(0, &s);
    sched_setaffinity(0, sizeof s, &s);
}
#endif

using namespace so101;

struct Rng {                                             // xorshift64*: fast, good enough for test inputs
    std::uint64_t s;
    double u() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return static_cast<double>((s * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
    }
    double in(double lo, double hi) { return lo + (hi - lo) * u(); }
};

static double ticks_per_ns() {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t c0 = tsc_begin();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(200)) {}
    const std::uint64_t c1 = tsc_end();
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    return static_cast<double>(c1 - c0) / ns;
}

static int sweep(long n) {
    Rng rng{0x9E3779B97F4A7C15ULL};
    long ok_down = 0, ok_tilt = 0, behind = 0, fails = 0, worse = 0, n_front = 0;
    double max_err = 0, max_pan = 0, max_tilt_excess = 0, max_fk_err = 0;
    for (long i = 0; i < n; ++i) {
        const int side = static_cast<int>(i & 1);
        double q[5], p[3], ap[3], out[7];
        for (int j = 0; j < 5; ++j) q[j] = rng.in(gen::LO[j], gen::HI[j]);
        fk(side, q, p, ap);
        const double tilt_gen = std::acos(std::max(-1.0, std::min(1.0, -ap[2]))) * RAD2DEG;
        // is the tip in front of the pan axis in this pose? (the solver only returns front solutions)
        const gen::Side& sd = gen::SIDES[side];
        const double X = sd.M[0] * p[0] + sd.M[1] * p[1] + sd.M[2] * p[2] + sd.c[0];
        const double Y = sd.M[3] * p[0] + sd.M[4] * p[1] + sd.M[5] * p[2] + sd.c[1];
        const double r_gen = X * std::cos(gen::SIG[0] * q[0]) + Y * std::sin(gen::SIG[0] * q[0]);
        const int st = solve(side, p, q[4], q, 0, out);
        if (r_gen <= 1e-4) { ++behind; continue; }
        ++n_front;
        if (st == UNREACHABLE) {
            if (++fails <= 5) std::printf("  MISSED a reachable point: side %d q = %.6f %.6f %.6f %.6f %.6f code %.0f\n",
                                          side, q[0], q[1], q[2], q[3], q[4], out[6]);
            continue;
        }
        (st == OK_DOWN ? ok_down : ok_tilt)++;
        double q2[5] = {out[0], out[1], out[2], out[3], q[4]}, p2[3];
        fk(side, q2, p2, nullptr);
        const double e = std::sqrt((p2[0] - p[0]) * (p2[0] - p[0]) + (p2[1] - p[1]) * (p2[1] - p[1]) + (p2[2] - p[2]) * (p2[2] - p[2]));
        max_fk_err = std::max(max_fk_err, e);
        max_err = std::max(max_err, out[4]);
        max_pan = std::max(max_pan, std::fabs(out[0] - q[0]));
        if (out[5] > tilt_gen + 1e-3) {
            max_tilt_excess = std::max(max_tilt_excess, out[5] - tilt_gen);
            if (++worse <= 5) std::printf("  TILTS MORE than a known pose: side %d q = %.6f %.6f %.6f %.6f %.6f: %.5f vs %.5f deg\n",
                                          side, q[0], q[1], q[2], q[3], q[4], out[5], tilt_gen);
        }
    }
    std::printf("sweep: %ld random poses (%ld with the tip in front of the base, %ld behind: skipped)\n", n, n_front, behind);
    std::printf("  answered %ld (straight down %ld, least tilt %ld), missed %ld, tilted more than the source pose %ld\n",
                ok_down + ok_tilt, ok_down, ok_tilt, fails, worse);
    std::printf("  tip error: max %.3e m (solver's own figure %.3e m); pan differs from the source pose by max %.2e rad\n",
                max_fk_err, max_err, max_pan);
    return (fails == 0 && worse == 0 && max_fk_err < 1e-9) ? 0 : 1;
}

struct Case { int side; double p[3], rho, q[4]; };

// The core's real clock right now, GHz: a chain of dependent 1-cycle adds runs at one add per cycle. Laptops move
// between turbo and throttled speeds, so nanoseconds vary run to run while the cycle counts do not.
static double core_ghz(double tpn) {
#if defined(__x86_64__) || defined(_M_X64)
    const long n = 300000000;
    std::uint64_t x = 0;
    const std::uint64_t c0 = tsc_begin();
    for (long i = 0; i < n; ++i) __asm__ volatile("add $1, %0" : "+r"(x));
    const std::uint64_t c1 = tsc_end();
    return (x ? static_cast<double>(n) : 0.0) / (static_cast<double>(c1 - c0) / tpn);
#else
    (void)tpn;
    return 0.0;
#endif
}

static double g_ghz = 0.0;

static void report(const char* name, std::vector<double>& ns, double thr_ns, const long st[4]) {
    std::sort(ns.begin(), ns.end());
    const auto pct = [&](double p) { return ns[static_cast<size_t>(p / 100.0 * (ns.size() - 1))]; };
    double mean = 0;
    for (double v : ns) mean += v;
    mean /= ns.size();
    std::printf("%-34s| %6.1f %6.1f %6.1f %6.1f %7.1f %8.1f | %6.1f | %6.0f | %5.1f M/s | down %ld tilt %ld switch %ld none %ld\n",
                name, ns.front(), pct(50), pct(90), pct(99), pct(99.9), ns.back(), mean, pct(50) * g_ghz, 1e3 / thr_ns, st[1], st[2],
                st[3], st[0]);
}

// Per-call latency: each call alone between fenced timestamps, the cost of the empty pair subtracted. Throughput: the
// same calls back to back, no timers. chain = feed each answer back as the next call's current joints (what the controller does).
static void scenario(const char* name, std::vector<Case>& cs, double tpn, double overhead, bool chain) {
    const size_t n = cs.size();
    if (n == 0) { std::printf("%-34s| no cases\n", name); return; }
    std::vector<double> ns(n);
    long st[4] = {0, 0, 0, 0};
    double out[7], sink = 0, q[4];
    for (int rep = 0; rep < 3; ++rep) {                  // rep 0-1 warm up caches and the branch predictor
        for (int j = 0; j < 4; ++j) q[j] = cs[0].q[j];
        for (size_t i = 0; i < n; ++i) {
            const Case& c = cs[i];
            const double* qp = chain ? q : c.q;
            const std::uint64_t t0 = tsc_begin();
            const int s = solve(c.side, c.p, c.rho, qp, 1, out);
            const std::uint64_t t1 = tsc_end();
            if (rep == 2) { ns[i] = std::max(0.0, (static_cast<double>(t1 - t0) - overhead) / tpn); st[s]++; }
            if (s != UNREACHABLE) for (int j = 0; j < 4; ++j) q[j] = out[j];
            sink += out[0];
        }
    }
    const auto a = std::chrono::steady_clock::now();
    const int reps = 20;
    for (int rep = 0; rep < reps; ++rep) {
        for (int j = 0; j < 4; ++j) q[j] = cs[0].q[j];
        for (size_t i = 0; i < n; ++i) {
            const Case& c = cs[i];
            const int s = solve(c.side, c.p, c.rho, chain ? q : c.q, 1, out);
            if (s != UNREACHABLE) for (int j = 0; j < 4; ++j) q[j] = out[j];
            sink += out[1];
        }
    }
    const double thr = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - a).count() / (reps * n);
    report(name, ns, thr, st);
    if (sink == 12345.678) std::printf("%f", sink);
}

int main(int argc, char** argv) {
    pin();
    const long n_sweep = argc > 1 ? std::atol(argv[1]) : 4000000;
    const int bad = sweep(n_sweep);
    std::printf("%s\n\n", bad ? "SWEEP FAILED" : "sweep passed");

    const double tpn = ticks_per_ns();
    std::vector<double> ov(200000);
    for (auto& v : ov) { const std::uint64_t t0 = tsc_begin(); const std::uint64_t t1 = tsc_end(); v = static_cast<double>(t1 - t0); }
    std::sort(ov.begin(), ov.end());
    const double overhead = ov[ov.size() / 2];
    std::printf("timer: %.3f ticks/ns. Latency = one call alone between fenced timestamps (empty pair %.1f ns, subtracted).\n",
                tpn, overhead / tpn);
    std::printf("Throughput = the same calls back to back, no timers (the CPU overlaps neighbouring calls).\n");
    g_ghz = core_ghz(tpn);
    std::printf("Core clock during this run: %.2f GHz (nanoseconds move with it; cycles = p50 x clock).\n", g_ghz);
    std::printf("%-34s| %6s %6s %6s %6s %7s %8s | %6s | %6s | %9s |\n", "per call, nanoseconds", "min", "p50", "p90", "p99", "p99.9", "max",
                "mean", "cycles", "throughput");

    const size_t N = 200000;
    const double home[4] = {-1.2363, -1.7135, 1.4979, 1.0534};
    Rng rng{0xD1B54A32D192ED03ULL};
    // The aim point of each arm wanders over the table (world frame) a few mm per call, out to where the gripper has
    // to tilt and beyond the reach. Sorted by what the solver has to do, so each line below times one kind of work.
    std::vector<Case> down, tilt, none, mixed;
    double q[2][4] = {{home[0], home[1], home[2], home[3]}, {-home[0], home[1], home[2], home[3]}}, out[7];
    for (size_t i = 0; i < 60 * N && (mixed.size() < N || down.size() < N || tilt.size() < N || none.size() < N); ++i) {
        const int side = static_cast<int>(i & 1);
        const double t = (i / 2) * 0.004, reach = 0.25 + 0.20 * std::sin(0.37 * t), dir = 0.9 * std::sin(t) + (side ? 0.5 : -0.5);
        Case c = {side, {(side ? 0.2092 : -0.2508) + reach * std::sin(dir), -0.2269 + reach * std::cos(dir), 0.56 + 0.03 * std::sin(0.9 * t)},
                  side ? 0.7 : -0.8, {q[side][0], q[side][1], q[side][2], q[side][3]}};
        const int st = solve(side, c.p, c.rho, c.q, 1, out);
        if (st != UNREACHABLE) for (int j = 0; j < 4; ++j) q[side][j] = out[j];
        if (mixed.size() < N) mixed.push_back(c);
        std::vector<Case>& v = st == OK_DOWN ? down : (st == UNREACHABLE ? none : tilt);
        if (v.size() < N) v.push_back(c);
    }
    scenario("gripper straight down", down, tpn, overhead, false);
    scenario("edge of reach: least tilt", tilt, tpn, overhead, false);
    scenario("beyond reach: refuse", none, tpn, overhead, false);
    scenario("teleop mix, as it comes", mixed, tpn, overhead, false);
    // the wheel is turning: a new roll on every call (the per-roll cache misses every time)
    std::vector<Case> cs = down;
    for (size_t i = 0; i < cs.size(); ++i) cs[i].rho += 0.3 * std::sin(0.01 * i);
    scenario("straight down, roll changing", cs, tpn, overhead, false);
    // random points in a box around the robot, random current joints: nothing for the branch predictor to learn
    cs.resize(N);
    for (size_t i = 0; i < N; ++i) {
        cs[i] = {static_cast<int>(i & 1), {rng.in(-0.5, 0.5), rng.in(-0.5, 0.3), rng.in(0.45, 0.85)}, rng.in(-2.5, 2.5),
                 {rng.in(-1.5, 1.5), rng.in(-1.5, 1.5), rng.in(-1.5, 1.5), rng.in(-1.5, 1.5)}};
    }
    scenario("random points, random rolls", cs, tpn, overhead, false);
    return bad;
}

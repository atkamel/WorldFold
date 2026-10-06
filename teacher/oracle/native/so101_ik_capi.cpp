// C interface of the SO-101 closed-form IK (so101_ik.hpp), built as a small shared library with no dependencies
// beyond libc, so Python loads it with ctypes (no cffi / numba / compiler needed where it runs).
#include "so101_cloth.hpp"
#include "so101_ik.hpp"

#if defined(_WIN32)
#define SO101_API extern "C" __declspec(dllexport)
#else
#define SO101_API extern "C" __attribute__((visibility("default")))
#endif

// See so101::solve. Returns 0 unreachable, 1 straight down, 2 least tilt, 3 other elbow branch.
SO101_API int so101_ik_solve(int side, const double* p_world, double roll, const double* q_prev, int allow_switch,
                             double* out7) {
    return so101::solve(side, p_world, roll, q_prev, allow_switch, out7);
}

// Same call through ONE pointer, for callers where every argument costs time (Python):
//   io[0] side, io[1..3] target, io[4] roll, io[5..8] current joints, io[9] allow_switch   ->   io[10..16] = out7
SO101_API int so101_ik_solve_io(double* io) {
    return so101::solve(static_cast<int>(io[0]), io + 1, io[4], io + 5, io[9] != 0.0, io + 10);
}

// n problems in one call: in = n rows of io[0..9], out = n rows of out7.
SO101_API void so101_ik_solve_n(int n, const double* in, double* out) {
    for (int i = 0; i < n; ++i, in += 10, out += 7)
        so101::solve(static_cast<int>(in[0]), in + 1, in[4], in + 5, in[9] != 0.0, out);
}

// Tip position and approach direction (world) for pan, lift, elbow, wrist_flex, wrist_roll. appr3 may be null.
SO101_API void so101_fk(int side, const double* q5, double* pos3, double* appr3) { so101::fk(side, q5, pos3, appr3); }

// Same through one buffer: io[0] side, io[1..5] q5  ->  io[6..8] tip, io[9..11] approach, io[12..14] jaw axis (world)
SO101_API void so101_fk_io(double* io) { so101::fk(static_cast<int>(io[0]), io + 1, io + 6, io + 9, io + 12); }

// Cloth surface index (so101_cloth.hpp): rebuild from n world points, then query.
SO101_API int so101_cloth_build(const double* pts, int n, const double* O, double zmax, const double* excl, int k, double excl_r) {
    return so101::cloth::build(pts, n, O, zmax, excl, k, excl_r);
}
SO101_API double so101_cloth_top(double x, double y, double r) { return so101::cloth::top(x, y, r); }

// Identifies the geometry the library was generated for (gen_so101_ik.py writes the same id into the header).
SO101_API std::uint64_t so101_ik_geom_id() { return so101::gen::GEOM_ID; }

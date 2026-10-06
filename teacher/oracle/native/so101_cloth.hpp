// "How high is the cloth under this point?" for the teleop controller, once per physics step, no heap.
//
// build(): drop the cloth points into a fixed grid of CELL x CELL cells by counting sort (two linear passes), keeping
// only resting points (local z below zmax, and not within `excl_r` of a gripper that is holding cloth).
// top(x, y, r): highest kept point within horizontal distance r of (x, y): checks only the cells that circle touches.
// Same answer as testing every point (the grid only decides which points are worth testing).
#pragma once
#include <cstdint>

namespace so101 { namespace cloth {

constexpr int GRID = 128;                       // cells per side
constexpr double CELL = 0.02;                   // m
constexpr double HALF = GRID * CELL / 2;        // grid covers [-HALF, HALF)^2 in the table frame; outside -> edge cells
constexpr int MAXP = 1 << 16;                   // cloth points (the long-sleeve shirt has 14 746)

struct Index {
    std::uint32_t start[GRID * GRID + 1];       // points of cell c: x/y/z[start[c] .. start[c+1])
    std::uint16_t cell[MAXP];                   // build scratch: cell of each kept input point
    std::int32_t src[MAXP];                     // build scratch: input index of each kept point
    double x[MAXP], y[MAXP], z[MAXP];           // kept points, grouped by cell
    int n;
};

inline Index g_index;                           // one cloth, rebuilt every physics step (single-threaded caller)

inline int cell_of(double v) noexcept {
    int i = static_cast<int>((v + HALF) * (1.0 / CELL));
    i = i < 0 ? 0 : i;
    return i < GRID ? i : GRID - 1;
}

// pts: n world points (x y z interleaved, float64). O: table frame origin in world (local = world - O).
// excl: k gripper tips (local frame) whose surroundings are not a surface (cloth hanging from a gripper).
// Returns the number of points kept, or -1 if n is too large.
inline int build(const double* pts, int n, const double* O, double zmax, const double* excl, int k, double excl_r) noexcept {
    Index& g = g_index;
    if (n < 0 || n > MAXP) return -1;
    for (int c = 0; c <= GRID * GRID; ++c) g.start[c] = 0;
    const double r2 = excl_r * excl_r;
    int m = 0;
    for (int i = 0; i < n; ++i) {               // pass 1: filter, find the cell, count
        const double px = pts[3 * i] - O[0], py = pts[3 * i + 1] - O[1], pz = pts[3 * i + 2] - O[2];
        if (!(pz < zmax)) continue;
        bool near = false;
        for (int j = 0; j < k; ++j) {
            const double dx = px - excl[3 * j], dy = py - excl[3 * j + 1], dz = pz - excl[3 * j + 2];
            near |= !(dx * dx + dy * dy + dz * dz > r2);
        }
        if (near) continue;
        const int c = cell_of(py) * GRID + cell_of(px);
        g.cell[m] = static_cast<std::uint16_t>(c);
        g.src[m] = i;
        ++g.start[c + 1];
        ++m;
    }
    for (int c = 0; c < GRID * GRID; ++c) g.start[c + 1] += g.start[c];    // prefix sum: start of each cell
    std::uint32_t fill[GRID * GRID];
    for (int c = 0; c < GRID * GRID; ++c) fill[c] = g.start[c];
    for (int j = 0; j < m; ++j) {               // pass 2: place the kept points cell by cell
        const int i = g.src[j];
        const std::uint32_t d = fill[g.cell[j]]++;
        g.x[d] = pts[3 * i] - O[0];
        g.y[d] = pts[3 * i + 1] - O[1];
        g.z[d] = pts[3 * i + 2] - O[2];
    }
    g.n = m;
    return m;
}

// Highest kept point with (dx^2 + dy^2) <= r^2 around (x, y); -inf when there is none.
inline double top(double x, double y, double r) noexcept {
    const Index& g = g_index;
    const double r2 = r * r;
    const int i0 = cell_of(x - r), i1 = cell_of(x + r), j0 = cell_of(y - r), j1 = cell_of(y + r);
    double best = -__builtin_inf();
    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const int c = j * GRID + i;
            for (std::uint32_t p = g.start[c]; p < g.start[c + 1]; ++p) {
                const double dx = g.x[p] - x, dy = g.y[p] - y;
                const double zz = g.z[p];
                best = (dx * dx + dy * dy <= r2 && zz > best) ? zz : best;
            }
        }
    }
    return best;
}

}}  // namespace so101::cloth

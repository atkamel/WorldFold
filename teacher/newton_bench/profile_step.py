#!/usr/bin/env python3
"""GPU time per kernel for a few frames of the grasp bench (step 3: where the time goes).

  python profile_step.py --worlds 1 --frames 20
"""
import argparse
import collections
import time

import numpy as np
import warp as wp

import grasp_sampler as gs
import grip_bench as gb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=1)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--warm", type=int, default=90, help="frames to run first (the shirt drops and flattens)")
    args, rest = p.parse_known_args()
    import sys
    sys.argv = [sys.argv[0], "--worlds", str(args.worlds), "--no-graph"] + rest
    ap = None
    # reuse the sampler's argument defaults
    import argparse as _a
    parser = _a.ArgumentParser()
    for k, v in (("density", 0.16), ("tri_ke", 400.0), ("tri_ka", 400.0), ("tri_kd", 1e-3), ("edge_ke", 2e-3),
                 ("edge_kd", 5e-6), ("radius", 0.00075), ("self_margin", 0.001), ("self_gap", 0.001),
                 ("rest_exclusion", 0.003), ("contact_ke", 1e4), ("shape_ke", 1e5), ("arm_kz", 1.4e4),
                 ("arm_fz", 15.0), ("arm_kdz", 40.0), ("contact_kd", 1e-2), ("contact_gap", 0.002),
                 ("mu_cloth", 0.5), ("servo_ke", 20.0), ("servo_kd", gb.SERVO_DAMPING), ("sdf_voxel", 0.0005),
                 ("body_contact_buffer", 4096), ("substeps", 10), ("iterations", 10), ("drop_gap", 0.005)):
        parser.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    parser.add_argument("--garment", default=gs.os.path.join(gb.HERE, "..", "oracle", "garments", "Top_Long_Seen_0.npz"))
    parser.add_argument("--graph", default=False)
    parser.add_argument("--self-detect", default="default")
    a = parser.parse_args(rest)
    a.graph = False
    B = gs.Batch(a, args.worlds)
    s = gs.sample(np.random.default_rng(0), args.worlds)
    B.make_solver(s)
    B.poses = np.zeros((args.worlds, 7))
    B.poses[:, :3] = B.centers + np.array([0.0, 0.0, 0.40]) - np.asarray(gb.TCP)
    B.poses[:, 6] = 1.0
    jaw = np.full(args.worlds, gb.GRIP_OPEN)
    for _ in range(args.warm):
        B.frame(B.poses.copy(), jaw)
    wp.synchronize()
    wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
    t0 = time.time()
    for _ in range(args.frames):
        B.frame(B.poses.copy(), jaw)
    wp.synchronize()
    wall = time.time() - t0
    res = wp.timing_end()
    tot = collections.defaultdict(float)
    cnt = collections.Counter()
    for r in res:
        tot[r.name] += r.elapsed
        cnt[r.name] += 1
    gpu = sum(tot.values())
    print(f"worlds {args.worlds}: {args.frames} frames, wall {wall:.2f} s ({wall / args.frames * 1000:.0f} ms/frame), "
          f"GPU kernel time {gpu / 1000:.2f} s ({gpu / args.frames:.0f} ms/frame), launches {sum(cnt.values())}")
    for name, ms in sorted(tot.items(), key=lambda x: -x[1])[:25]:
        print(f"  {ms / args.frames:8.2f} ms/frame  {100 * ms / gpu:5.1f} %  x{cnt[name] // args.frames:<5d} {name[:100]}")


if __name__ == "__main__":
    main()

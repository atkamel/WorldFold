"""Diagnoses the anchor grasp (isaac_env GRASP_MODE "anchor") on Isaac Sim: one scripted episode with, per step, the wall
time, each arm's phase, how far each anchor sits from its corner and from the gripper, and how far the near (anchor)
corners have drifted, plus a video.

    ISAAC_ENV_PARAMS='{"GRASP_MODE": "anchor"}' python isaac/anchor_probe.py --steps 120 --video-dir /vol/probe
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--video-dir", default=None)
    args = ap.parse_args()
    from imitation.tasks.isaac_half_fold import make_isaac_env
    env = make_isaac_env(video_size=256 if args.video_dir else None)
    from isaac.half_fold_expert import IsaacHalfFoldExpert
    expert = IsaacHalfFoldExpert(env)
    base = env.unwrapped
    t0 = time.time()
    env.reset(seed=args.seed)
    expert.reset()
    print(f"reset {time.time() - t0:.1f}s anchors {sorted(base._anchor_vtx)}", flush=True)
    frames = []
    for t in range(args.steps):
        t1 = time.time()
        a = expert.act()
        phases = expert.phases()
        _, _, term, trunc, info = env.step(a)
        row = {"t": t, "s": round(time.time() - t1, 2), "phase": phases, "drift": round(info["anchor_drift"], 3),
               "fold": round(info["fold_score"], 3), "grasp": info["grasped"]}
        for p, vtx in base._anchor_vtx.items():
            corner = base._particles[base._grid[vtx]]
            apos, _ = base.lab.anchor_pose(p)
            row[p] = {"anchor-corner": round(float(np.linalg.norm(apos - corner)), 3),
                      "anchor-tip": round(float(np.linalg.norm(apos - base.gripper_position(p))), 3),
                      "held": base._held[p] is not None}
        print("probe:", json.dumps(row), flush=True)
        if args.video_dir:
            frames.append(base._render_image()[0])
        if term or trunc:
            print("ended:", info["termination_reason"], flush=True)
            break
    if frames:
        import imageio
        os.makedirs(args.video_dir, exist_ok=True)
        imageio.mimwrite(f"{args.video_dir}/anchor_probe_seed{args.seed}.mp4", frames, fps=20, macro_block_size=1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

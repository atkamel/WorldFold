"""Checks IsaacHalfFoldExpert on Isaac Sim: per episode its outcome, phase timeline and how well its kinematic look-ahead
labels predict what it actually does next (the DAgger labels depend on that), optionally with a video.

    python isaac/expert_check.py --seeds 0 1 --video-dir /vol/check
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABEL_EVERY = 8
HORIZON = 16


def run(env, expert, seed, frames=None):
    base = env.unwrapped
    env.reset(seed=seed)
    expert.reset()
    actions, labels, timeline, info, prev = [], {}, [], {}, None
    t0 = time.time()
    for t in range(base.max_episode_steps):
        if t % LABEL_EVERY == 0:
            labels[t] = expert.label_chunk(HORIZON)
        phases = expert.phases()
        if phases != prev:
            jaws = {p: round(float(base.joint_positions(p)[5]), 3) for p in ("left_", "right_")}
            timeline.append((t, phases["left_"], phases["right_"], jaws["left_"], jaws["right_"]))
            prev = phases
        a = expert.act()
        actions.append(a)
        _, _, term, trunc, info = env.step(a)
        if frames is not None:
            frames.append(base._render_image()[0])
        if term or trunc:
            break
    actions = np.array(actions)
    # mean |label - executed action| over the arm joints, by how far ahead in the chunk
    err = {k: [] for k in (1, 4, 8, 16)}
    grip_agree = []
    joints = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    for t, chunk in labels.items():
        for k in err:
            if t + k <= len(actions):
                err[k].append(float(np.abs(chunk[:k][:, joints] - actions[t:t + k][:, joints]).mean()))
        n = min(HORIZON, len(actions) - t)
        grip_agree.append(float((np.sign(chunk[:n][:, [5, 11]]) == np.sign(actions[t:t + n][:, [5, 11]])).mean()))
    return {"seed": seed, "success": bool(info["success"]), "steps": len(actions),
            "fold_score": round(float(info["fold_score"]), 3), "reason": info["termination_reason"],
            "anchor_drift": round(float(info["anchor_drift"]), 3), "steps_per_s": round(len(actions) / (time.time() - t0), 2),
            "label_joint_mae": {k: round(float(np.mean(v)), 4) for k, v in err.items() if v},
            "label_grip_agree": round(float(np.mean(grip_agree)), 3), "timeline": timeline}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--video-dir", default=None)
    args = ap.parse_args()
    from imitation.tasks.isaac_half_fold import make_isaac_env
    env = make_isaac_env(video_size=360 if args.video_dir else None)
    from isaac.half_fold_expert import IsaacHalfFoldExpert
    expert = IsaacHalfFoldExpert(env)
    for seed in args.seeds:
        frames = [] if args.video_dir else None
        print("expert check:", json.dumps(run(env, expert, seed, frames)), flush=True)
        if frames:
            import imageio
            imageio.mimwrite(f"{args.video_dir}/expert_seed{seed}.mp4", frames + [frames[-1]] * 15, fps=20,
                             macro_block_size=1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac Sim 5.1's Kit logs an uncaught exception and still exits 0, so exit non-zero ourselves
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

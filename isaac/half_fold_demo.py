"""Scripted two-arm half fold on Isaac with friction grasps (cloth_fold_rl.quarter_fold_env.HalfFoldEnv).

Each arm pinches its far corner top-down (isaac.pinch, on the corner's actual position after the reset drop), lifts
it, carries it OVERSHOOT past its goal (the start of the near corner on the same side) to cover spring-back, lowers
it, releases and backs off, then both hold still for HalfFoldEnv's settle window. Joint targets are servoed through
HalfFoldEnv's 12-dim action (per arm: 5 joint deltas, gripper).

    modal run isaac/modal_isaac.py::half_fold --episodes 3      # on Modal, with videos
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mujuco.cloth_params import JOINT_DELTA_SCALE, TABLE_TOP_Z   # noqa: E402

OVERSHOOT = 0.03       # carry past the goal, along the carry direction
PINCH_HEIGHT = 0.010   # fixed fingertip above the table at the pinch: its pad rests on the table
PLACE_HEIGHT = 0.011
LIFT_HEIGHT = 0.05
# IK seeds above each far corner, fingers down
SEED_Q = {"left_": [0.2614, 1.0654, -1.1908, 1.6581, -2.046], "right_": [-0.2619, 1.0622, -1.1853, 1.6581, 2.1429]}
# (pose, gripper, dwell steps or None to servo until the slowest joint arrives)
SCHEDULE = (("above", 1.0, None), ("pinch", 1.0, None), ("pinch", -1.0, 8), ("lift", -1.0, None),
            ("carry", -1.0, None), ("place", -1.0, None), ("place", 1.0, 6), ("retreat", 1.0, None),
            ("retreat", 1.0, 25))


def plan(env, ik):
    """Joint targets per arm and pose, solved on the cloth as it lies now."""
    base = env.unwrapped
    center = base.cloth_positions().mean(axis=0)
    poses = {}
    for move in env.stages[0].moves:
        p = move.prefix
        corner = np.mean([env._vertex(c) for c in move.corners], axis=0)
        goal = env.goal(move)
        jaw = np.r_[center[:2] - corner[:2], 0.0]
        jaw /= np.linalg.norm(jaw)
        carry = np.r_[goal[:2] - corner[:2], 0.0]
        carry /= np.linalg.norm(carry)
        pinch = np.array([corner[0], corner[1], TABLE_TOP_Z + PINCH_HEIGHT]) - 0.005 * jaw
        place = np.array([goal[0], goal[1], TABLE_TOP_Z + PLACE_HEIGHT]) - 0.005 * jaw + OVERSHOOT * carry
        q, poses[p] = SEED_Q[p], {}
        # the fingers point straight down at the pinch and the place; lifting and carrying they may tilt a little
        for name, target, weight in (("above", pinch + [0, 0, 0.04], 0.1), ("pinch", pinch, 0.1),
                                     ("lift", pinch + [0, 0, LIFT_HEIGHT], 0.02),
                                     ("carry", place + [0, 0, 0.045], 0.02), ("place", place, 0.05),
                                     ("retreat", place + [0, 0, 0.05], 0.02)):
            q, _ = ik.solve(p, target, jaw, q, orientation_weight=weight)
            poses[p][name] = q
    return poses


def run_episode(env, ik, seed, frames=None):
    base = env.unwrapped
    env.reset(seed=seed)
    poses = plan(env, ik)
    prev = {p: base.joint_positions(p)[:5] for p in poses}
    info = {}
    for name, grip, dwell in SCHEDULE:
        steps = dwell or math.ceil(max(float(np.max(np.abs(poses[p][name] - prev[p]))) for p in poses)
                                   / JOINT_DELTA_SCALE) + 4
        for _ in range(steps):
            action = np.zeros(12, dtype=np.float32)
            for p, off in (("left_", 0), ("right_", 6)):
                delta = (poses[p][name] - base.joint_positions(p)[:5]) / JOINT_DELTA_SCALE
                action[off:off + 5] = np.clip(delta, -1.0, 1.0)
                action[off + 5] = grip
            _, _, terminated, truncated, info = env.step(action)
            if frames is not None:
                frames.append(base._render_image()[0])
            if terminated or truncated:
                break
        prev = {p: poses[p][name] for p in poses}
        if terminated or truncated:
            break
    return {"seed": seed, "success": bool(info["success"]), "fold_score": round(float(info["fold_score"]), 3),
            "move_distance": [round(d, 3) for d in info["move_distance"]],
            "anchor_drift": round(info["anchor_drift"], 3), "reason": info["termination_reason"] or "running"}


def make_env(video):
    from isaac.isaac_env import IsaacClothFoldEnv
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS, HALF_FOLD_MAX_STEPS, HalfFoldEnv
    base = IsaacClothFoldEnv(observation_mode="hybrid" if video else "state", image_size=(360, 360),
                             max_episode_steps=HALF_FOLD_MAX_STEPS, grasp_corners=GRASP_CORNERS,
                             grasp_radius=GRASP_RADIUS)
    return HalfFoldEnv(base_env=base, seed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--video-dir", default=None, help="write half_fold_seed<k>.mp4 here")
    args = ap.parse_args()
    env = make_env(video=args.video_dir is not None)
    from isaac.pinch import PinchIK
    ik = PinchIK()
    for seed in range(args.episodes):
        frames = [] if args.video_dir else None
        row = run_episode(env, ik, seed, frames)
        print("half fold:", json.dumps(row), flush=True)
        if frames:
            import imageio
            imageio.mimwrite(f"{args.video_dir}/half_fold_seed{seed}.mp4", frames + [frames[-1]] * 15, fps=20,
                             macro_block_size=1)


if __name__ == "__main__":
    import os
    try:
        main()
    except BaseException:
        # Isaac Sim 5.1's Kit logs an uncaught exception and still exits 0, so exit non-zero ourselves
        import traceback
        traceback.print_exc()
        os._exit(1)
    # leaving the interpreter with Isaac still up hangs at shutdown, so exit directly
    sys.stdout.flush()
    os._exit(0)

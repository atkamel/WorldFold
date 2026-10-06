"""Scripted two-arm half fold on Isaac with friction grasps (cloth_fold_rl.quarter_fold_env.HalfFoldEnv).

Each arm pinches its far corner top-down (isaac.pinch, on the corner's actual position after the reset drop), then
carries it along an arc over the fold line onto the start of the near corner on the same side, through IK waypoints
about 2 cm apart, each seeded from the last so the arm keeps one configuration. It releases, backs off, and both arms
hold still for HalfFoldEnv's settle window. Joint targets are servoed through HalfFoldEnv's 12-dim action (per arm:
5 joint deltas, gripper), each arm's joints scaled together so they arrive at once, and each pose held until every
joint is within TRACK_TOL of it: the env sets a joint's target at most JOINT_DELTA_SCALE from where it is now, and
the joints lag by different amounts, so a fixed step count lets the gripper wander off the planned path. Waypoints
inside the arc only need WAYPOINT_TOL. A pose pressed against the table or the cloth may never get that close, so
each segment's last pose gets POSE_SETTLE_STEPS more than its largest joint move needs, then the next one starts.

The result is scored on the whole cloth (fold_error): an ideal half fold puts each far-half grid vertex on the start
of its mirror image across the middle row and leaves the near half where it started.

    modal run isaac/modal_isaac.py::half_fold --episodes 3      # on Modal, with videos
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mujuco.cloth_params import CLOTH_COUNT, JOINT_DELTA_SCALE, TABLE_TOP_Z   # noqa: E402

PINCH_HEIGHT = 0.010   # fixed fingertip above the table at the pinch: its pad rests on the table
PLACE_HEIGHT = 0.011
PINCH_OFFSET = 0.005   # fingertip this far outside the corner (along the jaw direction), so the jaw sweeps the corner in
PLACE_OVERSHOOT = 0.0  # carry the corner this far past its goal (along the carry), for the crease to spring back onto it
ARC_HEIGHT = 0.12      # peak of the carry arc above the fold line; under the cloth's half length, so it never pulls
ARC_WAYPOINTS = 20
TRACK_TOL = 0.02       # rad
WAYPOINT_TOL = 0.06    # rad
POSE_SETTLE_STEPS = 8
# episode length (steps of 0.05 s): HalfFoldEnv's 250 is about what the fold alone takes at the action's speed limit
# (0.05 rad per step: ~60 steps from home to above the corner, ~95 along the arc), leaving no room to settle
MAX_STEPS = 400
FOLDED_MEAN = 0.02     # fold_error thresholds (m) for a folded cloth
FOLDED_MAX = 0.05
# IK seeds above each far corner
SEED_Q = {"left_": [0.2614, 1.0654, -1.1908, 1.6581, -2.046], "right_": [-0.2619, 1.0622, -1.1853, 1.6581, 2.1429]}
# (segment, gripper, dwell steps on its last pose, or None to servo through its poses)
SCHEDULE = (("above", 1.0, None), ("pinch", 1.0, None), ("pinch", -1.0, 8), ("arc", -1.0, None),
            ("arc", 1.0, 6), ("retreat", 1.0, None), ("retreat", 1.0, 25))


def fold_error(start, now):
    """xy distance (m) of each cloth grid vertex from where an ideal half fold puts it."""
    n = CLOTH_COUNT
    start, now = start.reshape(n, n, 3), now.reshape(n, n, 3)      # [x index, y index]
    target = start.copy()
    target[:, n // 2 + 1:] = start[:, :n // 2][:, ::-1]
    return np.linalg.norm((now - target)[..., :2], axis=-1)


def plan_arm(ik, prefix, corner, goal, center, seed_q=None):
    """One arm's joint targets per segment (and the arc's tip positions), to pinch `corner` and lay it on `goal`.
    `center` (the cloth's middle) sets the jaw direction."""
    jaw = np.r_[center[:2] - corner[:2], 0.0]
    jaw /= np.linalg.norm(jaw)
    offset = -PINCH_OFFSET * jaw[:2]
    pinch = np.array([corner[0] + offset[0], corner[1] + offset[1], TABLE_TOP_Z + PINCH_HEIGHT])
    # the held point's mirror image across the fold line, so the corner lands on the goal (plus any overshoot)
    carry = np.r_[(goal - corner)[:2], 0.0]
    carry /= max(np.linalg.norm(carry), 1e-9)
    target = np.asarray(goal, float) + PLACE_OVERSHOOT * carry
    place = np.array([target[0] + offset[0], target[1] - offset[1], TABLE_TOP_Z + PLACE_HEIGHT])
    q, poses = SEED_Q[prefix] if seed_q is None else seed_q, {}
    for name, target in (("above", pinch + [0, 0, 0.04]), ("pinch", pinch)):
        q, _ = ik.solve(prefix, target, jaw, q)
        poses[name] = [q]
    # half an ellipse from the pinch to the place, rising ARC_HEIGHT over the fold line; along it the fingers may
    # tilt a little to keep the position
    poses["arc"], poses["arc_tips"] = [], []
    for s in np.linspace(0.0, 1.0, ARC_WAYPOINTS + 1)[1:]:
        target = pinch + (place - pinch) * (1.0 - math.cos(math.pi * s)) / 2.0
        target[2] = pinch[2] + (place[2] - pinch[2]) * s + ARC_HEIGHT * math.sin(math.pi * s)
        q, _ = ik.solve(prefix, target, jaw, q, orientation_weight=0.05)
        poses["arc"].append(q)
        poses["arc_tips"].append(target)
    q, _ = ik.solve(prefix, place + [0, 0, 0.05], jaw, q, orientation_weight=0.02)
    poses["retreat"] = [q]
    poses["pinch_tip"] = pinch
    poses["corner_target"] = target       # where the corner should be when the jaw lets go
    return poses


def plan(env, ik):
    """Joint targets per arm and segment, solved on the cloth as it lies now."""
    center = env.unwrapped.cloth_positions().mean(axis=0)
    return {move.prefix: plan_arm(ik, move.prefix, np.mean([env._vertex(c) for c in move.corners], axis=0),
                                  env.goal(move), center)
            for move in env.stages[0].moves}


def run_episode(env, ik, seed, frames=None):
    base = env.unwrapped
    env.reset(seed=seed)
    start = base.cloth_positions().copy()
    poses = plan(env, ik)
    info, done, steps, track, used = {}, False, 0, 0.0, {}

    def step(target, grip):
        nonlocal info, done, steps
        action = np.zeros(12, dtype=np.float32)
        for p, off in (("left_", 0), ("right_", 6)):
            delta = (target[p] - base.joint_positions(p)[:5]) / JOINT_DELTA_SCALE
            action[off:off + 5] = delta / max(1.0, float(np.max(np.abs(delta))))
            action[off + 5] = grip
        _, _, terminated, truncated, info = env.step(action)
        if frames is not None:
            frames.append(base._render_image()[0])
        steps += 1
        done = terminated or truncated

    for name, grip, dwell in SCHEDULE:
        before = steps
        for k in range(1 if dwell else len(poses["left_"][name])):
            target = {p: poses[p][name][-1 if dwell else k] for p in poses}
            last = k == len(poses["left_"][name]) - 1
            budget = math.ceil(max(float(np.max(np.abs(target[p] - base.joint_positions(p)[:5]))) for p in target)
                               / JOINT_DELTA_SCALE) + (POSE_SETTLE_STEPS if last else 0)
            tol = TRACK_TOL if last else WAYPOINT_TOL
            for _ in range(dwell or budget):
                if not dwell and all(np.max(np.abs(target[p] - base.joint_positions(p)[:5])) < tol for p in target):
                    break
                step(target, grip)
                if done:
                    break
            if name == "arc" and not dwell:
                track = max(track, max(float(np.linalg.norm(base.gripper_position(p) - poses[p]["arc_tips"][k]))
                                       for p in poses))
            if done:
                break
        used[name] = used.get(name, 0) + steps - before
        if done:
            break
    err = fold_error(start, base.cloth_positions())
    return {"seed": seed, "folded": bool(err.mean() < FOLDED_MEAN and err.max() < FOLDED_MAX),
            "fold_score": round(1.0 - float(err.mean() / fold_error(start, start).mean()), 3),
            "error_mean_cm": round(100 * float(err.mean()), 1), "error_max_cm": round(100 * float(err.max()), 1),
            "arc_tracking_cm": round(100 * track, 1), "steps": used,
            "half_fold_env_success": bool(info["success"]), "anchor_drift": round(info["anchor_drift"], 3),
            "reason": info["termination_reason"] or "running"}


def make_env(video):
    from isaac.isaac_env import CLOTH_JITTER, IsaacClothFoldEnv
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS, HalfFoldEnv
    base = IsaacClothFoldEnv(observation_mode="hybrid" if video else "state", image_size=(360, 360),
                             max_episode_steps=MAX_STEPS, grasp_corners=GRASP_CORNERS,
                             grasp_radius=GRASP_RADIUS)
    return HalfFoldEnv(base_env=base, seed=0, cloth_jitter=CLOTH_JITTER)


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

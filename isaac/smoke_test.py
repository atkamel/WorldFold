"""Runnable check for IsaacClothFoldEnv on Isaac Sim 5.1; on Modal: modal run isaac/modal_isaac.py

Exits non-zero if the observation contract drifts, a friction pinch fails to grasp, lift and release the near-left
corner, or the fold / half-fold wrappers cannot run on the env. State mode runs the physics checks; hybrid mode
checks the contract, throughput and the RGB/depth frame.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from isaac.isaac_env import IsaacClothFoldEnv   # noqa: E402
from mujuco.cloth_params import JOINT_DELTA_SCALE, TABLE_TOP_Z, check_contract   # noqa: E402

MOVING_CORNER = 0          # cloth_0, the left arm's near corner: inside its top-down reach
OTHER_CORNERS = [1, 2, 3]
PINCH_HEIGHT = 0.010       # fixed fingertip above the table: its pad rests on the table
LIFT_HEIGHT = 0.10
INWARD = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)     # jaw opens across the corner, toward the cloth centre
IK_SEED = [0.5717, -0.3365, 0.8595, 1.0478, -0.165]  # left arm above cloth_0, fingers down


def drive_to(env, q_goal, grip, n_steps):
    action = np.zeros(14, dtype=np.float32)
    action[6] = grip
    action[13] = 1.0
    for _ in range(n_steps):
        q = env.joint_positions("left_")[:5]
        action[0:5] = np.clip((q_goal - q) / JOINT_DELTA_SCALE, -1.0, 1.0)
        env.step(action)


def grasp_and_lift(env):
    from isaac.pinch import PinchIK
    ik = PinchIK()
    env.grasp_corners = {"left_": (0,), "right_": (120,)}
    env.reset(seed=0)
    corners0 = env.corner_positions().copy()
    corner = corners0[MOVING_CORNER]
    tip = np.array([corner[0], corner[1], TABLE_TOP_Z + PINCH_HEIGHT]) - 0.005 * INWARD
    q_above, e_above = ik.solve("left_", tip + [0.0, 0.0, 0.06], INWARD, IK_SEED)
    q_pinch, e_pinch = ik.solve("left_", tip, INWARD, q_above)
    q_lift, e_lift = ik.solve("left_", tip + [0.0, 0.0, LIFT_HEIGHT], INWARD, q_pinch)
    print(f"pinch IK err: above {e_above * 1000:.1f} mm, pinch {e_pinch * 1000:.1f} mm, lift {e_lift * 1000:.1f} mm")
    assert max(e_above, e_pinch, e_lift) < 0.002, "IK could not reach the pinch poses"

    def drift():
        return float(np.linalg.norm(env.corner_positions()[OTHER_CORNERS] - corners0[OTHER_CORNERS], axis=1).max())

    drive_to(env, q_above, grip=1.0, n_steps=35)
    drive_to(env, q_pinch, grip=1.0, n_steps=15)
    gap = float(np.linalg.norm(env.gripper_position("left_") - tip))
    drive_to(env, q_pinch, grip=-1.0, n_steps=12)
    print(f"pinch: fingertip {gap * 1000:.1f} mm from its target, grasp {env.grasp_active('left_')}")
    assert env.grasp_active("left_"), "gripper closed but the corner is not between the jaws"

    drive_to(env, q_lift, grip=-1.0, n_steps=30)
    lifted = env.corner_positions()[MOVING_CORNER, 2]
    rise = lifted - corners0[MOVING_CORNER, 2]
    print(f"lift: corner rise {rise:.4f} m, other corners drift {drift():.4f} m")
    assert rise > 0.05, "the friction pinch did not lift the corner"
    assert drift() < 0.10, "the other corners were dragged during the lift (fold wrapper terminates at 0.20)"

    drive_to(env, q_lift, grip=1.0, n_steps=20)
    dropped = lifted - env.corner_positions()[MOVING_CORNER, 2]
    print(f"release: corner fell {dropped:.4f} m")
    assert dropped > 0.03, "opening the gripper did not release the corner"
    print("grasp/lift ok")


def wrapper_runs(env):
    sys.path.insert(0, str(REPO / "cloth_fold_rl"))
    from cloth_fold_rl.fold_env import SingleCornerFoldEnv
    wrapped = SingleCornerFoldEnv(base_env=env, seed=0)
    obs, info = wrapped.reset(seed=0)
    for _ in range(50):
        obs, reward, terminated, truncated, info = wrapped.step(wrapped.action_space.sample())
        if terminated or truncated:
            obs, info = wrapped.reset()
    assert obs.shape == wrapped.observation_space.shape
    print("fold wrapper ok, obs", obs.shape, "fold_score", round(info["fold_score"], 4))


def half_fold_runs(env):
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS, HalfFoldEnv
    # one World per process, so reuse the env with the half fold's weld corners instead of building another
    env.grasp_corners = dict(GRASP_CORNERS)
    env.grasp_radius = GRASP_RADIUS
    wrapped = HalfFoldEnv(base_env=env, seed=0)
    obs, info = wrapped.reset(seed=0)
    assert info["fold_score"] < 0.05, f"half fold starts {info['fold_score']:.3f} folded"
    for _ in range(30):
        obs, reward, terminated, truncated, info = wrapped.step(wrapped.action_space.sample())
        if terminated or truncated:
            obs, info = wrapped.reset()
    assert obs.shape == wrapped.observation_space.shape
    print("half fold wrapper ok, fold_score", round(info["fold_score"], 4),
          "move_distance", [round(d, 3) for d in info["move_distance"]])


def throughput(env, n_steps=30):
    env.reset(seed=1)
    action = np.zeros(14, dtype=np.float32)
    t0 = time.time()
    for _ in range(n_steps):
        env.step(action)
    dt = time.time() - t0
    print(f"throughput ({env.observation_mode}): {n_steps / dt:.1f} control steps/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="state", choices=["state", "hybrid"])
    parser.add_argument("--save-frame", default=None, help="hybrid only: write <path>_rgb.png and <path>_depth.png")
    args = parser.parse_args()
    env = IsaacClothFoldEnv(observation_mode=args.mode)
    check_contract(env)
    if args.mode == "state":     # physics checks; hybrid only adds rendering, checked below
        grasp_and_lift(env)
        wrapper_runs(env)
        half_fold_runs(env)
    throughput(env)
    if args.mode == "hybrid":
        obs, _ = env.reset(seed=2)
        depth = obs["depth"][:, :, 0]
        valid = depth[depth > 0]
        print(f"depth: {valid.size}/{depth.size} valid px, range {valid.min():.3f}..{valid.max():.3f} m, "
              f"image mean {obs['image'].mean():.1f}")
        assert valid.size > 0.5 * depth.size, "depth image is mostly invalid; camera pose or clipping is wrong"
        if args.save_frame:
            from PIL import Image
            Image.fromarray(obs["image"]).save(args.save_frame + "_rgb.png")
            shade = (np.clip(depth / max(valid.max(), 1e-6), 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(shade).save(args.save_frame + "_depth.png")
            print("saved", args.save_frame + "_rgb.png")
    print("SMOKE OK")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac Sim 5.1's Kit logs an uncaught exception and still exits 0, so exit non-zero ourselves
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)

"""Runnable check for IsaacClothFoldEnv. Needs Isaac Sim, so it runs on the cluster:

    ./python.sh /path/to/WorldFold/isaac/smoke_test.py --mode state
    ./python.sh /path/to/WorldFold/isaac/smoke_test.py --mode hybrid

Exits non-zero if the observation contract drifts, the weld cheat fails to
grasp and lift the corner, or the fold wrapper cannot run on the env.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from isaac.isaac_env import IsaacClothFoldEnv, _dev, _npy   # noqa: E402
from mujuco.cloth_params import JOINT_DELTA_SCALE, check_contract   # noqa: E402

MOVING_CORNER = 1          # cloth_10, the left arm's grasp corner
ANCHOR_CORNERS = [0, 2]    # cloth_0 and cloth_110 must stay put
LIFT_HEIGHT = 0.08


def teleport_and_measure(env, q5, target):
    q = np.zeros(6, dtype=np.float32)
    q[:5] = q5
    arm = env.arms["left_"]
    arm.set_joint_positions(_dev(q[None, :]))
    arm.set_joint_velocities(_dev(np.zeros((1, 6))))
    arm.set_joint_position_targets(_dev(q[None, :]))
    env.world.step(render=False)
    return float(np.linalg.norm(env.gripper_position("left_") - target))


def search_joint_config(env, target, rng, n_random=300, n_refine=300):
    """Random search + hill climb over the 5 arm joints for a config that reaches target."""
    limits = env._dof_limits[:5]
    best_q = None
    best_d = np.inf
    for _ in range(n_random):
        q = rng.uniform(limits[:, 0], limits[:, 1])
        d = teleport_and_measure(env, q, target)
        if d < best_d:
            best_d = d
            best_q = q
    for _ in range(n_refine):
        q = np.clip(best_q + rng.normal(0.0, 0.08, size=5), limits[:, 0], limits[:, 1])
        d = teleport_and_measure(env, q, target)
        if d < best_d:
            best_d = d
            best_q = q
    return best_q, best_d


def drive_to(env, q_goal, grip, n_steps):
    action = np.zeros(14, dtype=np.float32)
    action[6] = grip
    action[13] = 1.0
    for _ in range(n_steps):
        q = env.joint_positions("left_")[:5]
        action[0:5] = np.clip((q_goal - q) / JOINT_DELTA_SCALE, -1.0, 1.0)
        env.step(action)


def grasp_and_lift(env):
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    corner = env.corner_positions()[MOVING_CORNER]
    q_reach, d_reach = search_joint_config(env, corner, rng)
    q_lift, d_lift = search_joint_config(env, corner + np.array([0.0, 0.0, LIFT_HEIGHT]), rng)
    print(f"IK search: reach err {d_reach:.4f} m, lift err {d_lift:.4f} m")
    assert d_reach < env.grasp_radius, "search could not bring the gripper within grasp radius of the corner"

    env.reset(seed=0)
    anchors0 = env.corner_positions()[ANCHOR_CORNERS].copy()
    corner0 = env.corner_positions()[MOVING_CORNER].copy()
    def anchor_drift():
        return float(np.linalg.norm(env.corner_positions()[ANCHOR_CORNERS] - anchors0, axis=1).max())

    drive_to(env, q_reach, grip=1.0, n_steps=80)
    gap = float(np.linalg.norm(env.gripper_position("left_") - env.corner_positions()[MOVING_CORNER]))
    print(f"approach: gripper to corner {gap:.4f} m, anchor drift {anchor_drift():.4f} m")
    drive_to(env, q_reach, grip=-1.0, n_steps=3)
    print(f"close: grasp {env.grasp_active('left_')}, anchor drift {anchor_drift():.4f} m")
    assert env.grasp_active("left_"), f"gripper closed {gap:.4f} m from the corner but no pin engaged"

    drive_to(env, q_lift, grip=-1.0, n_steps=60)
    corner1 = env.corner_positions()[MOVING_CORNER]
    rise = corner1[2] - corner0[2]
    drift = anchor_drift()
    print(f"lift: corner rise {rise:.4f} m, anchor drift {drift:.4f} m, grasp {env.grasp_active('left_')}")
    assert rise > 0.03, "pinned corner did not lift with the gripper"
    assert drift < 0.10, "anchor corners were dragged during the lift (fold wrapper terminates at 0.20)"

    drive_to(env, q_lift, grip=1.0, n_steps=20)
    assert not env.grasp_active("left_"), "opening the gripper did not release the pin"
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
    grasp_and_lift(env)
    wrapper_runs(env)
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
    main()

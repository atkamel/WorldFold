"""Runnable check for IsaacClothFoldEnv on Isaac Sim 5.1; on Modal: modal run isaac/modal_isaac.py::main

Exits non-zero if the observation contract drifts, the scripted friction half fold (isaac/half_fold_demo.py) leaves
the cloth mostly unfolded, or the fold / half-fold wrappers cannot run on the env. State mode runs
the physics checks; hybrid mode checks the contract, throughput and the RGB/depth frame.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from isaac.isaac_env import CLOTH_JITTER, IsaacClothFoldEnv   # noqa: E402
from mujuco.cloth_params import check_contract   # noqa: E402


def half_fold_episode(env):
    # the scripted friction half fold must get the cloth part of the way to an ideal half fold (whole-cloth
    # fold_score, see half_fold_demo.fold_error): it scores 0.40 to 0.61, an untouched cloth 0
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS, HalfFoldEnv
    from isaac.half_fold_demo import MAX_STEPS, run_episode
    from isaac.pinch import PinchIK
    env.grasp_corners = dict(GRASP_CORNERS)
    env.grasp_radius = GRASP_RADIUS
    env.max_episode_steps = MAX_STEPS
    row = run_episode(HalfFoldEnv(base_env=env, seed=0, cloth_jitter=CLOTH_JITTER), PinchIK(), seed=0)
    print("scripted half fold:", row)
    assert row["fold_score"] > 0.3, "the scripted half fold left the cloth mostly unfolded"
    print("half fold ok")


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
        half_fold_episode(env)
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

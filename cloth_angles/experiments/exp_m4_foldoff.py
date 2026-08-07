"""M4 fold-off: world-model planner vs scripted expert on matched starts.

Arms (same seeds, same cloth offsets -- paired):

    scripted -- the full ScriptedFoldPolicy (the choreography baseline)
    planner  -- scripted reach/grasp/lift (IK-precision territory), then the
                CEM planner controls the CARRY from imagination: replan every
                step over imagined rollouts scored by the keypoint head, then
                scripted release + settle

Metrics: success, final fold_score, moving-corner->goal distance (cm), steps.
The deliverable is folding-from-imagination at all; the honest expectation is
that choreography beats planner v1.

Usage:
    python -m cloth_angles.experiments.exp_m4_foldoff --config cloth_angles/config.yaml \
        --wm-checkpoint outputs/cloth_angles/checkpoints/checkpoint_020000.pt \
        --keypoint-checkpoint outputs/cloth_angles/keypoint/keypoint_head.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mujuco"))

from cloth_angles.collect_episodes import angle_obs
from cloth_angles.model.checkpoint import ExpectedSchema, load_checkpoint
from cloth_angles.model.keypoint_head import load_keypoint_checkpoint
from cloth_angles.planning.cem import CEMCarryPlanner

SEEDS = list(range(5000, 5010))    # 10 matched starts
CARRY_BUDGET = 70                  # planner-controlled steps
MAX_STEPS = 300


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--keypoint-checkpoint", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--output-dir", default="outputs/cloth_angles/m4_foldoff")
    return parser.parse_args()


def moving_corner_dist_cm(env) -> float:
    dists = env._corner_dists()
    return float((dists[1] + dists[3]) / 2 * 100.0)


def run_scripted(env, seed: int) -> dict:
    from sim_main import ScriptedFoldPolicy  # noqa: PLC0415
    rng = np.random.default_rng(seed)
    env.reset(seed=seed, options={"cloth_pose": rng.uniform(-0.03, 0.03, size=2), "task": 0})
    policy = ScriptedFoldPolicy(env)
    success, fold_score, steps = False, 0.0, 0
    for _ in range(MAX_STEPS):
        _, _, terminated, truncated, info = env.step(policy.act())
        steps += 1
        success, fold_score = bool(info["success"]), float(info["fold_score"])
        if terminated or truncated:
            break
    return {"success": int(success), "fold_score": round(fold_score, 4),
            "moving_corner_cm": round(moving_corner_dist_cm(env), 2), "steps": steps}


def run_planner(env, seed: int, wm, head, grid_size: int, horizon: int, population: int,
                 device) -> dict:
    from sim_main import ScriptedFoldPolicy  # noqa: PLC0415
    rng = np.random.default_rng(seed)
    env.reset(seed=seed, options={"cloth_pose": rng.uniform(-0.03, 0.03, size=2), "task": 0})
    corners0 = env.data.xpos[env._corner_ids].reshape(-1).copy()
    goal = env._goal_corners.reshape(-1).copy()

    policy = ScriptedFoldPolicy(env)
    success, fold_score, steps = False, 0.0, 0
    phase_reached = "grasp"

    # phase 1: scripted reach/grasp/lift until both arms are in (or past) carry
    for _ in range(160):
        _, _, terminated, truncated, info = env.step(policy.act())
        steps += 1
        success, fold_score = bool(info["success"]), float(info["fold_score"])
        if terminated or truncated:
            return {"success": int(success), "fold_score": round(fold_score, 4),
                    "moving_corner_cm": round(moving_corner_dist_cm(env), 2),
                    "steps": steps, "handover": 0}
        ready = all(policy.phase[p] in ("carry", "place", "lower", "release", "done")
                    for p in env.prefixes)
        if ready:
            phase_reached = "handover"
            break

    # phase 2: the brain drives -- CEM over imagined rollouts, replanned every step
    planner = CEMCarryPlanner(wm, head, corners0, goal, horizon=horizon,
                               population=population, device=device)
    plan_rng = np.random.default_rng(seed + 42)
    for _ in range(CARRY_BUDGET):
        field = angle_obs(env, grid_size, signed=True)
        action = planner.plan(field, plan_rng)
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        success, fold_score = bool(info["success"]), float(info["fold_score"])
        if terminated or truncated:
            break

    # phase 3: scripted release + settle (give the success-hold a chance to fire)
    if not success:
        release = np.zeros(14, dtype=np.float32)
        release[6] = release[13] = 1.0
        release[2] = release[9] = 0.4
        idle = np.zeros(14, dtype=np.float32)
        idle[6] = idle[13] = 1.0
        for k in range(50):
            _, _, terminated, truncated, info = env.step(release if k < 10 else idle)
            steps += 1
            success, fold_score = bool(info["success"]), float(info["fold_score"])
            if terminated or truncated:
                break

    return {"success": int(success), "fold_score": round(fold_score, 4),
            "moving_corner_cm": round(moving_corner_dist_cm(env), 2),
            "steps": steps, "handover": int(phase_reached == "handover")}


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    data_cfg = config["data"]
    grid_size = data_cfg["grid_size"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    expected = ExpectedSchema(grid_size=grid_size, action_dim=data_cfg["action_dim"],
                               angle_unit=data_cfg["angle_unit"],
                               angle_convention=data_cfg["angle_convention"])
    wm, _ = load_checkpoint(args.wm_checkpoint, expected, device=device)
    wm.eval()
    head = load_keypoint_checkpoint(args.keypoint_checkpoint,
                                     expected_latent_dim=wm.rssm.h_dim + wm.rssm.z_dim,
                                     device=device)

    from sim_main import ClothFoldEnv  # noqa: PLC0415
    env = ClothFoldEnv(max_episode_steps=MAX_STEPS, observation_mode="state")

    rows = []
    for seed in SEEDS:
        r_s = run_scripted(env, seed)
        r_s.update({"seed": seed, "arm": "scripted"})
        rows.append(r_s)
        print(f"[scripted] seed={seed} success={r_s['success']} fold={r_s['fold_score']:.3f} "
              f"corner={r_s['moving_corner_cm']:.1f}cm steps={r_s['steps']}")

        r_p = run_planner(env, seed, wm, head, grid_size, args.horizon, args.population, device)
        r_p.update({"seed": seed, "arm": "planner"})
        rows.append(r_p)
        print(f"[planner ] seed={seed} success={r_p['success']} fold={r_p['fold_score']:.3f} "
              f"corner={r_p['moving_corner_cm']:.1f}cm steps={r_p['steps']} "
              f"handover={r_p.get('handover', '-')}")

    csv_path = output_dir / "per_episode.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== fold-off summary ===")
    results = {}
    for arm in ("scripted", "planner"):
        sel = [r for r in rows if r["arm"] == arm]
        results[arm] = {
            "success_rate": float(np.mean([r["success"] for r in sel])),
            "mean_fold_score": float(np.mean([r["fold_score"] for r in sel])),
            "mean_moving_corner_cm": float(np.mean([r["moving_corner_cm"] for r in sel])),
            "mean_steps": float(np.mean([r["steps"] for r in sel])),
        }
        print(f"{arm}: success={results[arm]['success_rate']:.2f} "
              f"fold={results[arm]['mean_fold_score']:.3f} "
              f"corner={results[arm]['mean_moving_corner_cm']:.1f}cm "
              f"steps={results[arm]['mean_steps']:.0f}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()

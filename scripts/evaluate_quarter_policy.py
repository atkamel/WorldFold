"""Evaluate a staged quarter-fold checkpoint and report a failure funnel."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloth_angles.data.fold_observation import observe_state
from cloth_angles.model.actor_critic import policy_features
from cloth_angles.tasks import QUARTER
from scripts.train_imagined_actor import load_policy_actors


def run_episode(job) -> dict:
    checkpoint, seed, cloth_jitter, mass_scale = job
    torch.set_num_threads(1)
    env = QUARTER.make_env(cloth_jitter=cloth_jitter)
    env.unwrapped.domain_randomization = True
    base = env.unwrapped
    if mass_scale != 1.0:
        for body_id in base._cloth_body_ids:
            base._base_body_mass[body_id] *= mass_scale
    _, info = env.reset(seed=seed)
    saved = torch.load(checkpoint, map_location="cpu")
    actors = load_policy_actors(saved, QUARTER)
    goal = torch.as_tensor(QUARTER.goals(env))[None]
    ever_grasped = {arm.prefix: False for arm in QUARTER.arms}
    stage1_right_grasp = False
    stage1_placement = False
    reached_stage1 = False
    total = 0.0

    for step in range(base.max_episode_steps):
        stage_id = QUARTER.stage(info)
        state = torch.as_tensor(observe_state(base, QUARTER))[None]
        stage = torch.tensor([stage_id])
        actor = actors[stage_id] if len(actors) > 1 else actors[0]
        with torch.no_grad():
            action = actor(policy_features(state, goal, saved["state_mean"], saved["state_scale"],
                                           QUARTER, stage))[0].numpy()
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        reached_stage1 |= QUARTER.stage(info) >= 1
        for arm in QUARTER.arms:
            ever_grasped[arm.prefix] |= bool(info["grasped"][arm.prefix])
        if QUARTER.stage(info) == 1:
            stage1_right_grasp |= bool(info["grasped"]["right_"])
            stage1_placement |= int(info["settle_steps"]) > 0
        if terminated or truncated:
            break
    env.close()

    if info["success"]:
        failure = "success"
    elif not any(ever_grasped.values()):
        failure = "no_grasp"
    elif not reached_stage1:
        failure = "stage0_incomplete"
    elif not stage1_right_grasp:
        failure = "stage1_no_stack_grasp"
    elif not stage1_placement:
        failure = "stage1_no_placement"
    else:
        failure = info["termination_reason"] or "stage1_unsettled_or_timeout"
    return {"seed": seed, "success": bool(info["success"]), "steps": step + 1, "return": total,
            "fold_score": float(info["fold_score"]), "reached_stage1": reached_stage1,
            "left_grasp": ever_grasped["left_"], "right_grasp": ever_grasped["right_"],
            "stage1_right_grasp": stage1_right_grasp, "stage1_placement": stage1_placement,
            "failure": failure, "termination_reason": info["termination_reason"]}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    failures: dict[str, int] = {}
    for row in rows:
        failures[row["failure"]] = failures.get(row["failure"], 0) + 1
    reached = sum(row["reached_stage1"] for row in rows)
    return {
        "episodes": n,
        "left_grasp_rate": sum(row["left_grasp"] for row in rows) / n,
        "right_grasp_rate": sum(row["right_grasp"] for row in rows) / n,
        "any_grasp_rate": sum(row["left_grasp"] or row["right_grasp"] for row in rows) / n,
        "stage0_completion_rate": reached / n,
        "stage1_stack_grasp_rate": sum(row["stage1_right_grasp"] for row in rows) / n,
        "stage1_placement_rate": sum(row["stage1_placement"] for row in rows) / n,
        "success_rate": sum(row["success"] for row in rows) / n,
        "stage1_success_given_reached": sum(row["success"] for row in rows) / reached if reached else 0.0,
        "mean_fold_score": float(np.mean([row["fold_score"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "failure_counts": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=60000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--cloth-jitter", type=float, default=0.025)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("episodes must be positive")
    output = args.output or args.checkpoint.parent / "evaluation.json"
    jobs = [(str(args.checkpoint), args.seed_base + i, args.cloth_jitter, args.mass_scale)
            for i in range(args.episodes)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(run_episode, jobs))
    rows.sort(key=lambda row: row["seed"])
    summary = summarize(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Validate and summarize full-state fold demonstration datasets.

Example:
    .venv/bin/python scripts/audit_fold_dataset.py \
        outputs/cloth_angles/quarter_expert_v1 --task quarter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloth_angles.data.state_episode import StateEpisodeStore
from cloth_angles.tasks import TASKS, VERTEX_DIM


def audit_dataset(path: Path, task_name: str) -> tuple[dict, list[str]]:
    task = TASKS[task_name]
    episodes = StateEpisodeStore(path).load_all()
    errors: list[str] = []
    if not episodes:
        return {"path": str(path), "episodes": 0}, ["dataset contains no episode_*.npz files"]

    stage_transitions = {str(i): 0 for i in range(len(task.stages))}
    reached_stage = {str(i): 0 for i in range(len(task.stages))}
    grasp_events = {arm.prefix: 0 for arm in task.arms}
    gripper_command_changes = {arm.prefix: 0 for arm in task.arms}
    split_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    successes = 0
    total_transitions = 0

    for i, episode in enumerate(episodes):
        label = f"episode_{i:06d}"
        t = len(episode)
        total_transitions += t
        split = str(episode.metadata.get("split", "missing"))
        kind = str(episode.metadata.get("kind", "missing"))
        split_counts[split] = split_counts.get(split, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        successes += int(bool(episode.metadata.get("success", False)))

        if episode.metadata.get("task") != task_name:
            errors.append(f"{label}: task is {episode.metadata.get('task')!r}, expected {task_name!r}")
        if episode.metadata.get("length") != t:
            errors.append(f"{label}: metadata length does not match actions")
        if episode.states().shape != (t + 1, task.state_dim):
            errors.append(f"{label}: state shape is {episode.states().shape}, expected {(t + 1, task.state_dim)}")
        if episode.actions.shape != (t, task.action_dim):
            errors.append(f"{label}: action shape is {episode.actions.shape}, expected {(t, task.action_dim)}")
        if episode.stage is None or episode.stage.shape != (t + 1,):
            errors.append(f"{label}: missing or misaligned stage array")
            continue
        if episode.rewards is None or episode.rewards.shape != (t,):
            errors.append(f"{label}: missing or misaligned rewards")
        if episode.terminated is None or episode.terminated.shape != (t,):
            errors.append(f"{label}: missing or misaligned termination flags")
        if not all(np.isfinite(x).all() for x in (episode.vertices, episode.robot, episode.actions)):
            errors.append(f"{label}: non-finite state or action value")
        if float(np.abs(episode.actions).max(initial=0.0)) > 1.00001:
            errors.append(f"{label}: action outside [-1, 1]")
        delta_stage = np.diff(episode.stage)
        if np.any(delta_stage < 0) or np.any(delta_stage > 1):
            errors.append(f"{label}: invalid stage transition")
        if int(episode.stage[-1]) != int(episode.metadata.get("final_stage", episode.stage[-1])):
            errors.append(f"{label}: final_stage metadata mismatch")
        if len(episode.metadata.get("goal", [])) != task.goal_dim:
            errors.append(f"{label}: goal has the wrong dimension")
        if np.asarray(episode.metadata.get("anchors0", [])).shape != (4, 3):
            errors.append(f"{label}: anchors0 must have shape (4, 3)")

        for stage_id in range(len(task.stages)):
            stage_transitions[str(stage_id)] += int(np.sum(episode.stage[:-1] == stage_id))
            reached_stage[str(stage_id)] += int(np.any(episode.stage == stage_id))
        for arm in task.arms:
            grasp = episode.robot[:, arm.grasp_index - VERTEX_DIM] > 0.5
            grasp_events[arm.prefix] += int(np.sum((~grasp[:-1]) & grasp[1:]))
            commands = episode.actions[:, arm.gripper]
            gripper_command_changes[arm.prefix] += int(np.sum(np.abs(np.diff(commands)) > 0.5))

    manifest = path / "manifest.json"
    if manifest.exists():
        rows = json.loads(manifest.read_text())
        if len(rows) != len(episodes):
            errors.append(f"manifest has {len(rows)} rows for {len(episodes)} episodes")
    else:
        errors.append("manifest.json is missing")

    summary = {
        "path": str(path),
        "task": task_name,
        "episodes": len(episodes),
        "transitions": total_transitions,
        "successes": successes,
        "success_rate": successes / len(episodes),
        "splits": split_counts,
        "kinds": kind_counts,
        "episodes_reaching_stage": reached_stage,
        "transitions_by_stage": stage_transitions,
        "grasp_events": grasp_events,
        "gripper_command_changes": gripper_command_changes,
        "errors": len(errors),
    }
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--task", choices=TASKS, default="quarter")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    summary, errors = audit_dataset(args.data, args.task)
    print(json.dumps(summary, indent=2))
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"- {error}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({**summary, "validation_errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

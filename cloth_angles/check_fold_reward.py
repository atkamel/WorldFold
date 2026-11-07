"""Recompute FoldReward on recorded task episodes and compare it with the env's own rewards and terminations.

    python -m cloth_angles.check_fold_reward --data outputs/cloth_angles/isaac_wm_a

Every episode that stores the env's rewards (the collector's task kinds) is scored by FoldReward(task) over its
recorded states and actions: goal from its metadata, starts its first state's vertices, from its first state's stage,
hard grasp flags. Up to FoldReward's first termination (its later rewards are unmasked by design), reported per
episode: the largest reward difference and its step, whether FoldReward terminates exactly where the env did, and
whether the stages match; then the same per termination reason. FoldReward cannot apply the env's "unstable" rule (its
instability check reads what the state omits), so an episode the env ended as unstable differs by the penalty at its
last step and in its last termination flag; any other difference is a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloth_angles.data.state_episode import StateEpisodeStore  # noqa: E402
from cloth_angles.model.fold_reward import FoldReward  # noqa: E402
from cloth_angles.tasks import TASKS  # noqa: E402


@torch.no_grad()
def check_episode(ep, task):
    """One episode against FoldReward: its worst reward difference and step, whether it terminates where the env
    did (the env stops there, so only at the last step), and whether the stages agree."""
    spec = FoldReward(goal=ep.metadata["goal"], starts=ep.vertices[0], task=task)
    stage0 = int(ep.stage[0]) if ep.stage is not None else 0
    rewards, terminated, stages = spec.rollout(torch.as_tensor(ep.states())[None], torch.as_tensor(ep.actions)[None],
                                               stage=torch.tensor([stage0]))
    ends = np.flatnonzero(terminated[0].numpy())
    end = int(ends[0]) if len(ends) else None
    env_end = len(ep) - 1 if ep.terminated[-1] else None
    upto = len(ep) if end is None else end + 1
    diff = np.abs(rewards[0].numpy()[:upto] - ep.rewards[:upto])
    return {"file": ep.metadata.get("file"), "kind": ep.metadata.get("kind"),
            "reason": ep.metadata.get("termination_reason"), "steps": len(ep), "fold_reward_end": end,
            "max_diff": float(diff.max()), "worst_step": int(diff.argmax()), "terminations_agree": end == env_end,
            "stages_agree": ep.stage is None or bool((stages[0].numpy()[:upto + 1] == ep.stage[:upto + 1]).all())}


def summarize(rows):
    """Overall and per termination reason: episodes, largest difference, episodes whose terminations or stages
    disagree."""
    def one(rs):
        return {"episodes": len(rs), "max_diff": max(r["max_diff"] for r in rs),
                "termination_mismatches": sum(not r["terminations_agree"] for r in rs),
                "stage_mismatches": sum(not r["stages_agree"] for r in rs)}
    return {"all": one(rows), "by_reason": {str(k): one([r for r in rows if r["reason"] == k])
                                           for k in dict.fromkeys(r["reason"] for r in rows)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="StateEpisodeStore directories")
    ap.add_argument("--task", default="half", choices=sorted(TASKS))
    ap.add_argument("--output", type=Path, help="write every episode's row and the summary here (.json)")
    args = ap.parse_args()
    task = TASKS[args.task]
    episodes = []
    for d in args.data:
        for path in sorted(Path(d).glob("episode_*.npz")):
            ep = StateEpisodeStore(d).load(path)
            ep.metadata["file"] = f"{Path(d).name}/{path.name}"
            if ep.rewards is not None:
                episodes.append(ep)
    if not episodes:
        ap.error("no episodes with recorded rewards")
    rows = [check_episode(ep, task) for ep in episodes]
    summary = summarize(rows)
    if args.output:
        args.output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        if not (r["terminations_agree"] and r["stages_agree"]) or r["max_diff"] > 1e-4:
            print("DIFFERS", json.dumps(r))


if __name__ == "__main__":
    main()

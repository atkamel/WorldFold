"""Evaluate the scripted half fold, the cloned actor and the imagined actor on Isaac Sim 5.1, the world model scoring
every step alongside.

    python -m isaac.world_model.evaluate --checkpoint outputs/cloth_angles/isaac_wm_model/state_l1_seed0.pt \
        --actors outputs/cloth_angles/isaac_imagined/clone.pt outputs/cloth_angles/isaac_imagined/imagined.pt \
        --episodes 20 --out outputs/cloth_angles/isaac_imagined/evaluation.json

Every policy runs the same seeds (from --seed, outside the collection's) on the collector's env, QuarterFoldEnv's
stage 0 over make_base (friction profile, dynamics DR, the half fold's +-1 cm cloth offset). "scripted" is the
collector's half fold; an actor (cloth_angles.train_imagined_actor's file, named by its stem) acts every step,
deterministically, on observe_state, the episode's goal and its stage. Once an episode ends (termination, truncation
or the cap) every policy gets the same finish, the point precision_env scores at: each arm still holding opens and
lifts with the place servo's gentle release from where it is, then both hold until the cloth has been still for
SETTLE_STEPS (capped at FINISH_CAP). Per episode: the env's success and fold score, then after the finish the cloth's
gate (isaac.rl.precision_env.gate: flat, on the table, still, both jaws open and clear) and its alignment at the half
fold's bar (cloth_fold_rl.fold_alignment, mean <= 1.0 cm and edge <= 1.5 cm), which counts only where the gate passes,
the finish's length, cap hit and mean cloth displacement, and the world model's one-step errors and grasp accuracy
(run.py's run_episode). Per policy: n, seeds, counts with Wilson 95% intervals, means. One env, episodes in sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloth_angles.tasks import HALF  # noqa: E402
from cloth_fold_rl.fold_alignment import score  # noqa: E402
from isaac.rl.precision_env import FOLDS, GATE_STILL, SETTLE_STEPS, gate  # noqa: E402
from isaac.tools.grasp_metrics import wilson  # noqa: E402
from isaac.world_model.adapter import IsaacWorldModel, observe_state  # noqa: E402
from isaac.world_model.collect import HALF_STEPS  # noqa: E402
from isaac.world_model.half_fold import ScriptedHalfFold  # noqa: E402
from isaac.world_model.play import ARMS, expand_action, make_base  # noqa: E402
from isaac.world_model.run import run_episode  # noqa: E402

FINISH_CAP = 200         # control steps the finish may take before the cloth is scored anyway (recorded as a cap hit)


class ActorPolicy:
    """An actor file acting on the half fold: observe_state, the episode's goal and stage -> the 12-dim action."""

    def __init__(self, path, half):
        import torch
        from cloth_angles.model.actor_critic import policy_features
        from cloth_angles.train_imagined_actor import load_actor
        self.torch, self.policy_features, self.half = torch, policy_features, half
        self.actor, self.mean, self.scale, self.task = load_actor(path)
        if self.task is not HALF:
            raise ValueError(f"{path} is an actor for task {self.task.name}, not the half fold")

    def __call__(self):
        env, t = self.half.task, self.torch
        state = t.as_tensor(observe_state(env.unwrapped, self.task.arms))[None]
        goal = t.as_tensor(self.task.goals(env))[None]
        with t.no_grad():
            features = self.policy_features(state, goal, self.mean, self.scale, self.task, t.tensor([env.stage]))
            return self.actor(features)[0].numpy()


def release_here(expert):
    """Point an IsaacPlaceServoExpert's gentle release at the arm's current pose, its jaw axis as it is now; an arm
    already releasing, retreating or done carries on."""
    if expert.PHASES[expert.phase] in ("release", "retreat", "done"):
        return
    jaw = -expert._ik.site_pose(expert.prefix, expert._q())[1][:, 0]        # the site's -x axis is the jaw direction
    jaw = np.r_[jaw[:2], 0.0]
    expert.plan = dict(expert.plan or {}, jaw=jaw / max(np.linalg.norm(jaw), 1e-9))
    expert.release_site = None                                              # release where the arm is
    expert._go("release")


def finish(base, experts):
    """Release every arm (release_here), then step until both are done and the cloth has moved less than GATE_STILL
    per step for SETTLE_STEPS steps, or FINISH_CAP steps -> (final cloth, the cloth a step before, steps, whether the
    cap was hit, the cloth's mean vertex displacement over the finish in mm)."""
    for e in experts.values():
        release_here(e)
    start = np.asarray(base.cloth_positions(), dtype=float).copy()
    prev, cloth, still, steps = start, start, 0, 0
    while still < SETTLE_STEPS and steps < FINISH_CAP:
        base.step(expand_action(np.concatenate([experts[a.prefix].act() for a in ARMS])))
        prev, cloth = cloth, np.asarray(base.cloth_positions(), dtype=float).copy()
        done = all(e.PHASES[e.phase] == "done" for e in experts.values())
        still = still + 1 if done and np.linalg.norm(cloth - prev, axis=1).max() < GATE_STILL else 0
        steps += 1
    return cloth, prev, steps, still < SETTLE_STEPS, 1000 * float(np.linalg.norm(cloth - start, axis=1).mean())


def score_final(cloth, prev_cloth, sites, jaws):
    """The final cloth's gate and its half-fold alignment (cm); passed_valid only where the gate passes."""
    g = gate(cloth, prev_cloth, sites, jaws)
    s = score(cloth, folds=FOLDS)
    return {"gate": g, "mean_cm": round(100 * s["mean"], 3), "edge_cm": round(100 * s["edge_max"], 3),
            "passed": bool(s["passed"]), "passed_valid": bool(s["passed"] and g["valid"])}


def _mean(values):
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


def summarize(rows):
    """Per policy: n, seeds, success / gated / passed / finish-cap counts with Wilson 95% intervals (percent), the mean
    fold score and finish displacement, the mean alignment over gated episodes, the world model's mean one-step
    scores."""
    out = []
    for name in dict.fromkeys(r["policy"] for r in rows):
        mine = [r for r in rows if r["policy"] == name]
        gated = [r for r in mine if r["gate"]["valid"]]

        def count(flags):
            k, n = int(sum(flags)), len(mine)
            lo, hi = wilson(k, n)
            return {"k": k, "n": n, "wilson95": [round(lo, 1), round(hi, 1)]}

        out.append({"policy": name, "n": len(mine), "seeds": [r["seed"] for r in mine],
                    "success": count(r["success"] for r in mine), "gated": count(r["gate"]["valid"] for r in mine),
                    "passed_valid": count(r["passed_valid"] for r in mine),
                    "fold_score": _mean([r["fold_score"] for r in mine]),
                    "finish_capped": count(r["finish_capped"] for r in mine),
                    "finish_disp_mm": _mean([r["finish_disp_mm"] for r in mine]),
                    "gated_mean_cm": _mean([r["mean_cm"] for r in gated]),
                    "gated_edge_cm": _mean([r["edge_cm"] for r in gated]),
                    **{k: _mean([r[k] for r in mine])
                       for k in ("vertex_mae_mm", "manipulated_vertex_mm", "grasp_accuracy", "control_hz")}})
    return out


def run(args):
    base = make_base(HALF_STEPS)
    half = ScriptedHalfFold(base)
    world_model = IsaacWorldModel(args.checkpoint, device=args.device)     # torch after Kit is up
    policies = [("scripted", None)] + [(Path(a).stem, ActorPolicy(a, half)) for a in args.actors]
    seeds = [args.seed + i for i in range(args.episodes)]
    rows = []
    for name, actor in policies:
        for seed in seeds:
            out, _ = run_episode(base, half, world_model, "half", seed, HALF_STEPS, actor=actor)
            cloth, prev, steps, capped, disp = finish(base, half.experts)
            final = score_final(cloth, prev, [base.gripper_position(a.prefix) for a in ARMS],
                                [base.joint_positions(a.prefix)[5] for a in ARMS])
            rows.append({"policy": name, **out, **final, "finish_steps": steps, "finish_capped": capped,
                         "finish_disp_mm": round(disp, 2)})
            print("EPISODE", json.dumps(rows[-1]), flush=True)
            args.out.write_text(json.dumps({"rows": rows}, indent=1))
    summary = summarize(rows)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    for row in summary:
        print("POLICY", json.dumps({k: v for k, v in row.items() if k != "seeds"}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="world model + grasp head checkpoint")
    ap.add_argument("--actors", type=Path, nargs="*", default=[], help="actor files (train_imagined_actor's)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3_000_000, help="first seed (outside the collection's)")
    ap.add_argument("--out", type=Path, required=True, help="rows and summary (.json)")
    ap.add_argument("--device", default="cpu", help="torch device for the world model")
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("--episodes must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)                  # Kit can hang at interpreter shutdown


if __name__ == "__main__":
    main()

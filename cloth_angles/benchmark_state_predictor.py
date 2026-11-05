"""Residual state predictors on stored episodes, scored against persistence and constant-velocity extrapolation.

    python -m cloth_angles.benchmark_state_predictor --data outputs/cloth_angles/isaac_wm_a \
        outputs/cloth_angles/isaac_wm_b --output outputs/cloth_angles/isaac_wm_benchmark
    python -m cloth_angles.benchmark_state_predictor --task single --data outputs/cloth_angles/fold_state_v2 \
        --output outputs/cloth_angles/state_benchmark

docs/world_model.md's protocol. Each variant is trained once per seed (cloth_angles.train_state_predictor). Evaluation
uses fixed windows: ten observed transitions, then 1, 5 and 10 steps predicted from the window's origin with the
recorded actions, fed back autoregressively, stride ten. Metrics are in millimetres: mean absolute vertex error and
the position error of the manipulated vertices (each arm's, averaged: the vertex its play cycle works on at the
origin, its half-fold corner, cloth_10 in the MuJoCo single-corner data), on all windows and on windows where an
arm's grasp flag is set at the origin. Test and training episodes are each scored in two groups, play episodes and
task episodes (every other kind). Results are averaged per seed with episode-window weights, then mean and sample
standard deviation are taken across seeds. Persistence repeats the origin; constant velocity extrapolates the last
observed change (vertex error only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloth_angles.data.state_episode import GRASP_INDEX, ROBOT_DIM, manipulated_vertices  # noqa: E402
from cloth_angles.tasks import TASKS, VERTEX_DIM  # noqa: E402
from cloth_angles.train_state_predictor import (  # noqa: E402
    VARIANTS, check_layout, load_split, save_checkpoint, sequence, train,
)

BURN_IN, HORIZON, STRIDE = 10, 10, 10
HORIZONS = (1, 5, 10)
METRICS = ("vertex_mae", "manip_err")
BASELINES = ("persistence_vertex_mae", "persistence_manip_err", "const_velocity_vertex_mae")
GROUPS = ("play", "task")


def group(ep):
    return "play" if ep.metadata.get("kind") == "play" else "task"


@torch.no_grad()
def evaluate(model, variant, episodes):
    """One row per (episode, horizon): window-mean metrics, overall and on grasped origins."""
    rows = []
    for index, ep in enumerate(episodes):
        starts = list(range(0, len(ep) - BURN_IN - HORIZON + 1, STRIDE))
        if not starts:
            continue
        seq = torch.as_tensor(sequence(ep, variant))
        origins = [s + BURN_IN for s in starts]
        actions = torch.as_tensor(np.stack([ep.actions[o:o + HORIZON] for o in origins]))
        predicted = model.rollout(seq[origins], seq[[o - 1 for o in origins]], actions)[..., :VERTEX_DIM]
        predicted = predicted.reshape(len(starts), HORIZON, -1, 3)
        vertices = torch.as_tensor(ep.vertices)
        v_origin, v_prev = vertices[origins], vertices[[o - 1 for o in origins]]
        manip = torch.as_tensor(manipulated_vertices(ep)[origins])          # [windows, arms]
        window = torch.arange(len(starts))[:, None]
        grasped = torch.as_tensor((ep.robot[origins, GRASP_INDEX::ROBOT_DIM] > 0.5).any(axis=1))
        for h in HORIZONS:
            target = vertices[[o + h for o in origins]]
            err = predicted[:, h - 1] - target
            persist = v_origin - target
            metrics = {"vertex_mae": err.abs().mean((-1, -2)), "manip_err": err[window, manip].norm(dim=-1).mean(-1),
                       "persistence_vertex_mae": persist.abs().mean((-1, -2)),
                       "persistence_manip_err": persist[window, manip].norm(dim=-1).mean(-1),
                       "const_velocity_vertex_mae": (v_origin + h * (v_origin - v_prev) - target).abs().mean((-1, -2))}
            row = {"episode": index, "kind": ep.metadata.get("kind"), "group": group(ep), "horizon": h,
                   "windows": len(starts), "grasped_windows": int(grasped.sum())}
            for name, values in metrics.items():
                row[name] = values.mean().item()
                row["grasped_" + name] = values[grasped].mean().item() if grasped.any() else None
            rows.append(row)
    return rows


def weighted(rows, name, count):
    return sum(r[name] * r[count] for r in rows) / sum(r[count] for r in rows)


def summarize(results, variants):
    """Per split, group, subset and horizon: the baselines and each variant's mean and sample std across seeds."""
    summary = []
    for split in ("test", "train"):
        for grp in GROUPS:
            for subset in ("all", "grasped"):
                prefix = "grasped_" if subset == "grasped" else ""
                count = "grasped_windows" if subset == "grasped" else "windows"
                for h in HORIZONS:
                    summary.append(_summary_row(results, variants, split, grp, subset, prefix, count, h))
    return summary


def _summary_row(results, variants, split, grp, subset, prefix, count, h):
    row = {"split": split, "group": grp, "subset": subset, "horizon": h}
    for variant in variants:
        per_seed = {m: [] for m in METRICS}
        for run in results:
            if run["variant"] != variant:
                continue
            rows = [r for r in run[split] if r["group"] == grp and r["horizon"] == h and r[count] > 0]
            if not rows:
                continue
            row["windows"] = sum(r[count] for r in rows)
            for b in BASELINES:
                row[b] = weighted(rows, prefix + b, count)
            for m in METRICS:
                per_seed[m].append(weighted(rows, prefix + m, count))
        for m, values in per_seed.items():
            if values:
                row[f"{variant}_{m}_mean"] = float(np.mean(values))
                row[f"{variant}_{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="one or more StateEpisodeStore directories")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--task", default="half", choices=sorted(TASKS), help="the stores' state layout")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["state_l1"], choices=VARIANTS)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--device", default="cpu", help="a torch device for training")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_eps, test_eps = load_split(args.data)
    check_layout(train_eps + test_eps, TASKS[args.task])
    results = []
    for variant in args.variants:
        for seed in args.seeds:
            started = time.perf_counter()
            model, loss = train(train_eps, variant, args.steps, seed, args.hidden_dim, args.device, log_every=0)
            save_checkpoint(model, args.output / f"{variant}_seed{seed}.pt", variant, args.task, args.steps, seed,
                            args.hidden_dim)
            results.append({"variant": variant, "seed": seed, "steps": args.steps, "train_episodes": len(train_eps),
                            "final_loss": loss, "elapsed_seconds": time.perf_counter() - started,
                            "test": evaluate(model, variant, test_eps), "train": evaluate(model, variant, train_eps)})
            print(f"done {variant} seed={seed} loss={loss:.4f} elapsed={time.perf_counter() - started:.0f}s",
                  flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=1))
    summary = summarize(results, args.variants)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    for row in summary:
        if "windows" not in row:
            continue
        cells = [f"{v}={1000 * row[f'{v}_vertex_mae_mean']:.3f}/{1000 * row[f'{v}_manip_err_mean']:.2f}"
                 for v in args.variants if f"{v}_vertex_mae_mean" in row]
        print(row["split"], row["group"], row["subset"], f"h={row['horizon']}", f"n={row['windows']}",
              "vertex/manip mm:", *cells,
              f"persist={1000 * row['persistence_vertex_mae']:.3f}/{1000 * row['persistence_manip_err']:.2f}",
              f"constvel={1000 * row['const_velocity_vertex_mae']:.3f}")


if __name__ == "__main__":
    main()

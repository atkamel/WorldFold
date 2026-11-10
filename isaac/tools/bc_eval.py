"""Closed-loop evaluation of a behaviour-cloned half-fold policy (overnight plan 2026-11-09) in the batched precision
env the expert demos came from (bc_collect.py), scored the way the expert is: at the end, the full validity gate and the
alignment score, and a success when the fold is gate-valid within MEAN_BAR / EDGE_BAR.

Each copy resets from its seed, then the policy alone acts for --steps control steps: it predicts a chunk from the
last obs_horizon observations (and the rig frames, for a camera policy) and executes the first --replan actions of it.
Rows go to <out>/eval.jsonl, a top-camera video per seed with --video.

  isaac/watcloud.sbatch isaac/tools/bc_eval.py --ckpt outputs/isaac/bc/runs/state_s0/final.pt \\
      --seeds 2000000:2000064 --out outputs/isaac/bc/eval/state_s0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from isaac.tools.bc_collect import EDGE_BAR, MEAN_BAR, parse_cameras   # noqa: E402


def score_final(cloth, prev_cloth, sites, jaws):
    """The expert's end-of-window scoring for a student episode: gate, alignment, and the working-bar success."""
    from cloth_fold_rl.fold_alignment import score
    from isaac.rl.precision_env import FOLDS, gate
    g = gate(cloth, prev_cloth, sites, jaws)
    s = score(cloth, folds=FOLDS)
    ok = bool(g["valid"] and s["mean"] <= MEAN_BAR and s["edge_max"] <= EDGE_BAR)
    return {"mean": round(float(s["mean"]), 4), "edge_max": round(float(s["edge_max"]), 4), "gate": g,
            "gate_valid": bool(g["valid"]), "success": ok, "passed_valid": bool(s["passed"] and g["valid"])}


def history(hist, obs, horizon):
    """Append obs to the per-copy history (filled with the first obs at the start); -> [horizon, D]."""
    if not hist:
        hist.extend([obs] * horizon)
    hist.append(obs)
    del hist[:-horizon]
    return np.stack(hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", required=True, help="a:b")
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600, help="control steps per episode")
    ap.add_argument("--replan", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    sys.argv = sys.argv[:1]
    from imitation.policies.common import load_policy
    from imitation.rollout import padded_predict
    from isaac.rl.precision_env import ARMS, PrecisionBatch
    policy = load_policy(args.ckpt)
    cams = {name: size for name, size in getattr(policy, "cameras", [])} if policy.needs_images else {}
    if args.video:
        cams["top"] = 400
    pb = PrecisionBatch(args.copies, cameras=cams or None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a, b = (int(v) for v in args.seeds.split(":"))
    for start in range(a, b, pb.n):
        seeds = list(range(start, start + pb.n))
        for i, seed in enumerate(seeds):
            pb.tasks[i].reset(seed=int(seed))
        pb.frames = [[] for _ in range(pb.n)] if args.video else None
        hists, chunk, prev = [[] for _ in range(pb.n)], None, None
        for t in range(args.steps):
            obs = np.stack([history(hists[i], pb.tasks[i]._observe(), policy.obs_horizon) for i in range(pb.n)])
            if t % args.replan == 0:
                images = [pb.subs[i].render_rig() for i in range(pb.n)] if policy.needs_images else None
                chunk = padded_predict(policy, obs, images)
            prev = [pb.subs[i].cloth_positions().copy() for i in range(pb.n)]
            pb.drive(chunk[:, t % args.replan])
        for i, seed in enumerate(seeds):
            sub = pb.subs[i]
            row = {"seed": seed, "ckpt": args.ckpt, "steps": args.steps, "replan": args.replan,
                   **score_final(sub.cloth_positions(), prev[i], [sub.gripper_position(p) for p in ARMS],
                                 [sub.joint_positions(p)[5] for p in ARMS]),
                   "cloth": np.round(sub.cloth_positions(), 4).tolist()}
            print("BCEVAL", json.dumps({k: v for k, v in row.items() if k != "cloth"}), flush=True)
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if args.video:
                import imageio
                clip = pb.frames[i]
                imageio.mimwrite(str(out / f"seed{seed}.mp4"), clip + clip[-1:] * 20, fps=20, macro_block_size=1)
    os._exit(0)                                   # Kit can hang on shutdown


if __name__ == "__main__":
    main()

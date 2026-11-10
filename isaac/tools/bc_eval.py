"""Closed-loop evaluation of a behaviour-cloned half-fold policy (overnight plan 2026-11-09) in the batched precision
env the expert demos came from (bc_collect.py), scored the way the expert is: at the end, the full validity gate and the
alignment score, and a success when the fold is gate-valid within MEAN_BAR / EDGE_BAR.

Each copy resets from its seed, then the policy alone acts for --steps control steps: it predicts a chunk from the
last obs_horizon observations (and the rig frames, for a camera policy) and executes the first --replan actions of it.
--ckpt takes several checkpoints (comma-separated), evaluated one after another in one scene that renders every camera
any of them needs. Rows go to <out>/<run name>/eval.jsonl, a top-camera video per seed with --video.

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

from isaac.tools.bc_collect import EDGE_BAR, MEAN_BAR   # noqa: E402


def score_final(cloth, prev_cloth, sites, jaws):
    """The expert's end-of-window scoring for a student episode: gate, alignment, and the working-bar success."""
    from cloth_fold_rl.fold_alignment import score
    from isaac.rl.precision_env import FOLDS, gate
    g = gate(cloth, prev_cloth, sites, jaws)
    s = score(cloth, folds=FOLDS)
    ok = bool(g["valid"] and s["mean"] <= MEAN_BAR and s["edge_max"] <= EDGE_BAR)
    return {"mean": round(float(s["mean"]), 4), "edge_max": round(float(s["edge_max"]), 4), "gate": g,
            "gate_valid": bool(g["valid"]), "success": ok, "passed_valid": bool(s["passed"] and g["valid"])}


GRIPPER_DIMS = (5, 11)          # the 12-D action's left and right gripper commands


def grip_gain(action, gain):
    """The action with its gripper commands scaled by gain (the env clips to [-1, 1]); the demos' are exactly +-1."""
    a = np.array(action, dtype=np.float32)
    a[list(GRIPPER_DIMS)] *= gain
    return a


def history(hist, obs, horizon):
    """Append obs to the per-copy history (filled with the first obs at the start); -> [horizon, D]."""
    if not hist:
        hist.extend([obs] * horizon)
    hist.append(obs)
    del hist[:-horizon]
    return np.stack(hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="one or more checkpoints, comma-separated")
    ap.add_argument("--seeds", required=True, help="a:b (a multiple of --copies long)")
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--steps", type=int, default=900, help="control steps per episode")
    ap.add_argument("--replan", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--grip-gain", type=float, default=1.0,
                    help="scale the two gripper outputs before the env's binary hysteresis (+-0.3) reads them")
    args = ap.parse_args()
    sys.argv = sys.argv[:1]
    import torch
    from cloth_fold_rl.fold_alignment import score
    from imitation.policies.common import load_policy
    from imitation.rollout import padded_predict
    from isaac.rl.precision_env import ARMS, FOLDS, PrecisionBatch
    ckpts = args.ckpt.split(",")
    policies = [load_policy(c, device="cpu") for c in ckpts]     # Kit starts with the scene below; CUDA after it
    cams = {name: size for p in policies if p.needs_images for name, size in p.cameras}
    video_cam = next(c for c in ("top", "main", "side_left") if c not in cams)    # one no policy reads
    if args.video:
        cams[video_cam] = 400
    pb = PrecisionBatch(args.copies, cameras=cams or None)
    pb.video_cam = video_cam
    knobs = {k: os.environ.get(k, "") for k in ("WORLDFOLD_GRASP_KNOBS", "WORLDFOLD_PLACE_PARAMS")}
    for ckpt, policy in zip(ckpts, policies):
        policy.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        name = Path(ckpt).parent.name + (f"_g{args.grip_gain:g}" if args.grip_gain != 1.0 else "")
        run_policy(pb, policy, ckpt, Path(args.out) / name, args, knobs, score, padded_predict, ARMS, FOLDS)
    os._exit(0)                                   # Kit can hang on shutdown


def run_policy(pb, policy, ckpt, out, args, knobs, score, padded_predict, ARMS, FOLDS):
    out.mkdir(parents=True, exist_ok=True)
    a, b = (int(v) for v in args.seeds.split(":"))
    for start in range(a, b, pb.n):
        seeds = list(range(start, start + pb.n))
        for i, seed in enumerate(seeds):
            pb.tasks[i].reset(seed=int(seed))
        pb.frames = [[] for _ in range(pb.n)] if args.video else None
        hists, chunk, prev = [[] for _ in range(pb.n)], None, None
        alive, trace = [True] * pb.n, [[] for _ in range(pb.n)]
        for t in range(args.steps):
            obs = np.stack([history(hists[i], pb.tasks[i]._observe(), policy.obs_horizon) for i in range(pb.n)])
            if t % args.replan == 0:
                images = [pb.subs[i].render_rig() for i in range(pb.n)] if policy.needs_images else None
                chunk = padded_predict(policy, np.nan_to_num(obs), images)
            prev = [pb.subs[i].cloth_positions().copy() for i in range(pb.n)]
            pb.drive([grip_gain(chunk[i, t % args.replan], args.grip_gain) if alive[i] else np.zeros(12)
                      for i in range(pb.n)])
            for i in range(pb.n):
                if alive[i] and pb.subs[i]._failed():      # a blown-up cloth: park it before it drifts into others
                    pb.subs[i].park()
                    alive[i] = False
                elif alive[i] and t % 50 == 49:
                    s = score(pb.subs[i].cloth_positions(), folds=FOLDS)
                    trace[i].append([t + 1, round(float(s["mean"]), 4), round(float(s["edge_max"]), 4)])
        for i, seed in enumerate(seeds):
            sub = pb.subs[i]
            row = {"seed": seed, "ckpt": ckpt, "steps": args.steps, "replan": args.replan, "grip_gain": args.grip_gain,
                   **knobs,
                   "unstable": not alive[i], "trace": trace[i]}
            if alive[i]:
                row |= score_final(sub.cloth_positions(), prev[i], [sub.gripper_position(p) for p in ARMS],
                                   [sub.joint_positions(p)[5] for p in ARMS])
                row["cloth"] = np.round(sub.cloth_positions(), 4).tolist()
            else:
                row |= {"success": False, "gate_valid": False, "mean": None, "edge_max": None}
            print("BCEVAL", json.dumps({k: v for k, v in row.items() if k not in ("cloth", "trace")}), flush=True)
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if args.video:
                import imageio
                clip = pb.frames[i]
                imageio.mimwrite(str(out / f"seed{seed}.mp4"), clip + clip[-1:] * 20, fps=20, macro_block_size=1)


if __name__ == "__main__":
    main()

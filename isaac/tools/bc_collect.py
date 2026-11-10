"""Expert half-fold demos for behaviour cloning (overnight plan 2026-11-09): the place-servo expert run in the batched
precision env (PrecisionBatch, the same env the expert's numbers come from), recorded from each copy's reset to the end
of its window episode as imitation-pipeline episodes (139-D observation, 12-D action, optional rig camera frames).

An episode is a demo when it ends settled, passes the validity gate except its stillness check, and lands within
MEAN_BAR / EDGE_BAR (the user's working bar, 2026-11-09: a gate-valid fold within 2 / 3 cm; evaluation keeps the full
gate). Only demos go into the dataset (the trainer clones successes); every seed gets a row in <root>/<version>.rows.jsonl.
Versions roll every --freeze-every chunks (<version>_r0, _r1, ...) so training can start on the frozen ones.

The expert's gentle release normally sets the jaw's open target through a side channel (base.jaw_open) that its
action doesn't carry; here it only commands "open" and WORLDFOLD_GRASP_KNOBS must set jaw_open_rate, so the env ramps
the jaw exactly as it will for a student (review 2026-11-09).

  isaac/watcloud.sbatch isaac/tools/bc_collect.py --seeds 1000000:1000200 --version bc_stiff_c1 \\
      --cameras main=128,left_wrist_cam=64,right_wrist_cam=64

The cloth and expert settings come from WORLDFOLD_GRASP_KNOBS / WORLDFOLD_PLACE_PARAMS, as for precision_rl sweep.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MEAN_BAR, EDGE_BAR = 0.02, 0.03
ROOT = "outputs/isaac/bc/datasets"


def success(final, mean_bar=MEAN_BAR, edge_bar=EDGE_BAR, still=True):
    """A settled, gate-valid fold within the bar; still=False drops the gate's stillness check (demo filter)."""
    from isaac.rl.precision_env import GATE_CLEAR, GATE_Z
    if not (final and final.get("reason") == "settled" and final.get("mean") is not None):
        return False
    g = final.get("gate") or {}
    valid = g.get("valid") if still else (g.get("on_table") and g.get("jaws_open") and g.get("z_max_cm", 99) <= 100 * GATE_Z
                                          and g.get("clear_cm", 0) >= 100 * GATE_CLEAR)
    return bool(valid and final["mean"] <= mean_bar and final["edge_max"] <= edge_bar)


def build_episode(steps, final, final_obs, seed, meta):
    """steps: [(obs 139, action 12, images {cam: uint8 [3, H, W]} or None)] -> an imitation Episode (source expert)."""
    from imitation.data.schema import Episode
    T = len(steps)
    term = final.get("reason") in ("settled", "cloth_dragged")
    terminated, truncated = np.zeros(T, bool), np.zeros(T, bool)
    (terminated if term else truncated)[-1] = True
    images = None
    if steps[0][2] is not None:
        images = {cam: np.stack([s[2][cam] for s in steps]).astype(np.uint8) for cam in steps[0][2]}
    return Episode(obs=np.stack([s[0] for s in steps]).astype(np.float32),
                   actions=np.clip(np.stack([s[1] for s in steps]), -1.0, 1.0).astype(np.float32),
                   rewards=np.zeros(T, np.float32), stage=np.zeros(T, np.int8), fold_score=np.zeros(T, np.float32),
                   grasped=np.zeros((T, 2), bool), actor=np.zeros(T, np.int8),
                   final_obs=np.asarray(final_obs, np.float32), terminated=terminated, truncated=truncated,
                   discount=np.where(terminated, 0.0, 1.0).astype(np.float32),
                   meta={"seed": int(seed), "source": "expert", "success": success(final, still=False),
                         "termination_reason": final.get("reason"), **meta}, images=images)


def parse_cameras(text):
    return {k: int(v) for k, v in (item.split("=") for item in text.split(","))} if text else {}


def split_seeds(a, b, k, chunk):
    """[a, b) into k contiguous ranges of whole chunks (the last takes the remainder)."""
    n = -(-(b - a) // chunk)
    per = -(-n // k)
    return [(a + j * per * chunk, min(b, a + (j + 1) * per * chunk)) for j in range(k) if a + j * per * chunk < b]


def run_parallel(args):
    import subprocess
    import time
    a, b = (int(v) for v in args.seeds.split(":"))
    Path(args.root).mkdir(parents=True, exist_ok=True)
    procs = []
    for j, (lo, hi) in enumerate(split_seeds(a, b, args.parallel, args.copies)):
        cmd = [sys.executable, "-u", __file__, "--seeds", f"{lo}:{hi}", "--copies", str(args.copies),
               "--version", f"{args.version}_p{j}", "--root", args.root, "--cameras", args.cameras,
               "--freeze-every", str(args.freeze_every)] + (["--resume"] if args.resume else []) \
            + (["--keep-failures"] if args.keep_failures else [])
        log = open(Path(args.root) / f"{args.version}_p{j}.log", "a")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        print("BC child", j, lo, hi, flush=True)
    end = time.time() + 3600 * args.child_hours
    while any(p.poll() is None for p in procs) and time.time() < end:
        time.sleep(30)
    for p in procs:
        if p.poll() is None:             # a hung child (Kit): its frozen parts are kept, the rest resumes later
            p.kill()
    print("BC children done", [p.wait() for p in procs], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", required=True, help="a:b")
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--version", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--cameras", default="", help="rig cameras to record, e.g. main=128,left_wrist_cam=64,...")
    ap.add_argument("--merge", default="", help="v1,v2,...: no simulation, write --version as one version listing them")
    ap.add_argument("--parallel", type=int, default=1, help="k > 1: k child collectors on this GPU, seed ranges and "
                                                             "versions <version>_p0.. split between them")
    ap.add_argument("--child-hours", type=float, default=7.0, help="--parallel: kill children still running after")
    ap.add_argument("--freeze-every", type=int, default=4, help="chunks per frozen version (<version>_r0, _r1, ...)")
    ap.add_argument("--resume", action="store_true", help="skip seeds that already have a row")
    ap.add_argument("--keep-failures", action="store_true",
                    help="store every episode that reached its window, demo or not (meta success marks the demos)")
    args = ap.parse_args()
    if args.parallel > 1:
        run_parallel(args)
        return
    sys.argv = sys.argv[:1]
    if args.merge:
        from imitation.data.schema import merge_versions
        m = merge_versions(args.root, args.merge.split(","), args.version)
        print("BC merged", args.version, m["n_episodes"], "episodes", m["n_steps"], "steps")
        return
    from imitation.data.schema import DatasetWriter
    from imitation.spec import ACTION_DIM, OBS_DIM
    from isaac.experts.place_servo import IsaacPlaceServoExpert
    from isaac.rl.precision_env import ACT_DIM, PrecisionBatch
    from isaac.rl.precision_rl import run_episodes
    if "jaw_open_rate" not in os.environ.get("WORLDFOLD_GRASP_KNOBS", ""):
        sys.exit("bc_collect needs jaw_open_rate in WORLDFOLD_GRASP_KNOBS (the release ramp a student can reproduce)")
    IsaacPlaceServoExpert._set_jaw = open_only
    cams = parse_cameras(args.cameras)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = root / f"{args.version}.rows.jsonl"
    done = {json.loads(l)["seed"] for l in open(rows)} if args.resume and rows.exists() else set()
    meta = {"grasp_knobs": os.environ.get("WORLDFOLD_GRASP_KNOBS", ""),
            "place_params": os.environ.get("WORLDFOLD_PLACE_PARAMS", ""), "cameras": cams}
    part = 0
    while (root / f"{args.version}_r{part}" / "manifest.json").exists():   # resume: after the frozen parts
        part += 1
    writer, chunks = None, 0
    pb = PrecisionBatch(args.copies, cameras=cams or None)
    buf = [[] for _ in range(pb.n)]
    pb.recorder = lambda i, obs, act, images: buf[i].append((obs, act, images))
    pb.record_images = bool(cams)
    zero = lambda obs: np.zeros((pb.n, ACT_DIM))
    a, b = (int(v) for v in args.seeds.split(":"))
    try:
        for start in range(a, b, pb.n):
            seeds = list(range(start, start + pb.n))
            if set(seeds) <= done:
                continue
            if writer is None:
                writer = DatasetWriter(root, f"{args.version}_r{part}", config={"tool": "isaac/tools/bc_collect.py",
                                                                                  **meta}, resume=args.resume)
            for k in buf:
                k.clear()
            snaps, _ = pb.scripted_to_window(seeds)
            ok = [i for i in range(pb.n) if snaps[i] is not None]
            finals = run_episodes(pb, [snaps[i] if snaps[i] is not None else snaps[ok[0]] for i in range(pb.n)],
                                  zero) if ok else [None] * pb.n
            for i, seed in enumerate(seeds):
                final = finals[i] if i in ok else None
                row = {"seed": seed, "no_window": final is None, "steps": len(buf[i]),
                       "demo": success(final, still=False), "success": success(final)}
                if final is not None:
                    row |= {k: final.get(k) for k in ("reason", "mean", "edge_max", "passed_valid", "dropped")}
                    row["gate_valid"] = bool((final.get("gate") or {}).get("valid"))
                    if row["demo"] or args.keep_failures:
                        ep = build_episode(buf[i], final, pb.tasks[i]._observe(), seed,
                                           {k: final.get(k) for k in ("mean", "edge_max", "passed_valid")})
                        writer.add(ep, OBS_DIM, ACTION_DIM)
                print("BC", json.dumps(row), flush=True)
                with open(rows, "a") as f:
                    f.write(json.dumps(row) + "\n")
            chunks += 1
            if chunks % args.freeze_every == 0:
                m = writer.freeze()
                print("BC frozen", writer.version, m["n_episodes"], "episodes", m["n_steps"], "steps", flush=True)
                writer, part = None, part + 1
        code = 0
    except Exception:
        import traceback
        traceback.print_exc()
        code = 3
    if writer is not None:
        m = writer.freeze()
        print("BC frozen", writer.version, m["n_episodes"], "episodes", m["n_steps"], "steps", flush=True)
    os._exit(code)                                # Kit can hang on shutdown


def open_only(self, action, q):
    """IsaacPlaceServoExpert._set_jaw for demos: command "open" and let the env's jaw_open_rate ramp the jaw, since a
    student can't set the side-channel cap (base.jaw_open) the gentle release uses."""
    action[5] = 1.0


if __name__ == "__main__":
    main()

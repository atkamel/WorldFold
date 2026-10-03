"""Throughput of N concurrent Isaac processes (Phase I, I0.3): each process builds the state-mode env, resets and
takes `--steps` zero-action control steps; the parent reports per-process and aggregate steps/s and peak VRAM.

    python isaac/bench_parallel.py --procs 1 2 --steps 60 --out outputs/isaac/runtime.json   (Isaac venv)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def child(steps, result):
    sys.path.insert(0, str(REPO))
    import numpy as np
    from isaac.isaac_env import IsaacClothFoldEnv
    env = IsaacClothFoldEnv(observation_mode="state")
    env.reset(seed=1)
    action = np.zeros(14, dtype=np.float32)
    t0 = time.time()
    for _ in range(steps):
        env.step(action)
    Path(result).write_text(json.dumps({"steps_per_s": steps / (time.time() - t0)}))
    os._exit(0)


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", default="outputs/isaac/runtime.json")
    ap.add_argument("--child", default=None, help="internal: result file of one child process")
    ap.add_argument("--deadline", type=float, default=900.0, help="seconds before a child counts as hung")
    args = ap.parse_args()
    if args.child:
        child(args.steps, args.child)
    rows = []
    for n in args.procs:
        # Kit is chatty: its output goes to a log file (a pipe nobody drains fills up and blocks the child)
        out_dir = Path(args.out).parent / "bench"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = [out_dir / f"n{n}_p{i}.json" for i in range(n)]
        for r in results:
            r.unlink(missing_ok=True)
        procs = [subprocess.Popen([sys.executable, "-u", __file__, "--child", str(r), "--steps", str(args.steps)],
                                  stdout=open(out_dir / f"n{n}_p{i}.log", "w"), stderr=subprocess.STDOUT)
                 for i, r in enumerate(results)]
        peak, t0 = 0, time.time()
        while any(p.poll() is None for p in procs) and time.time() - t0 < args.deadline:
            peak = max(peak, vram_mib())
            time.sleep(2)
        for p in procs:
            if p.poll() is None:
                p.kill()
        rates = [json.loads(r.read_text())["steps_per_s"] for r in results if r.exists()]
        rows.append({"procs": n, "per_process": [round(r, 2) for r in rates], "aggregate": round(sum(rates), 2),
                     "peak_vram_mib": peak, "completed": len(rates)})
        print(rows[-1], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()

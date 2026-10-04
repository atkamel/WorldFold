"""Throughput of the vectorised Isaac env (milestone V): expert half-fold episodes per hour with B envs in ONE Isaac
process (backend isaac_weld), through the real rollout worker code (imitation.rollout._serve_slots with the lockstep
scheduler on the main thread) and driver (_rollout, ExpertController). Run in the Isaac venv from the repo root,
one B per process:

    python -u isaac/bench_vec.py --B 4 --episodes 8 --out outputs/isaac/vec/bench.jsonl

Appends one JSON line: B, episodes, successes, wall seconds (first reset to last episode end; Kit start-up
excluded), episodes/hour, env-steps/second, global physics steps, GPU memory in use (nvidia-smi, all processes:
WDDM reports no per-process figure) and the other Isaac processes running meanwhile (contention).
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def gpu_mem_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else None


def other_isaac_procs():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine "
                              "-match '.venv-isaac' -and $_.CommandLine -notmatch 'pytest' }).Count"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out) - 1
    except Exception:
        return None


class _Pool:
    def __init__(self, pipes):
        self.pipes = pipes

    def __len__(self):
        return len(self.pipes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=None, help="default 2*B (two waves)")
    ap.add_argument("--seed0", type=int, default=700000)
    ap.add_argument("--out", default="outputs/isaac/vec/bench.jsonl")
    args = ap.parse_args()
    n_eps = args.episodes or 2 * args.B

    from imitation.rollout import ExpertController, _rollout, _serve_slots
    from imitation.tasks import make_env_batch
    batch, envs = make_env_batch(backend="isaac_weld", n=args.B)
    ends = [mp.Pipe() for _ in range(args.B)]
    pool = _Pool([a for a, _ in ends])
    result = {}

    def drive():
        for p in pool.pipes:
            assert p.recv() == ("ready", None)
        mem = []
        stop = threading.Event()

        def sample():
            while not stop.wait(20.0):
                mem.append(gpu_mem_mb())
        threading.Thread(target=sample, daemon=True).start()
        others = other_isaac_procs()
        t0 = time.perf_counter()
        eps = _rollout(pool, range(args.seed0, args.seed0 + n_eps), ExpertController(),
                       progress=lambda d, n, ep: print(f"  {d}/{n} seed {ep.meta['seed']} "
                                                       f"{'ok' if ep.meta['success'] else ep.meta['termination_reason']}"
                                                       f" {ep.steps} steps", flush=True))
        wall = time.perf_counter() - t0
        stop.set()
        steps = sum(e.steps for e in eps)
        result.update({"B": args.B, "episodes": n_eps, "successes": sum(e.meta["success"] for e in eps),
                       "wall_s": round(wall, 1), "episodes_per_hour": round(3600 * n_eps / wall, 1),
                       "env_steps": steps, "env_steps_per_s": round(steps / wall, 3),
                       "gpu_mem_mb_max": max([m for m in mem if m] or [gpu_mem_mb()]),
                       "other_isaac_procs": others})
        for p in pool.pipes:
            p.send(("close", None))
            p.recv()

    t = threading.Thread(target=drive, daemon=True)
    t.start()
    lockstep = _serve_slots([b for _, b in ends], batch, envs, isaac=True)
    t.join()
    result["global_steps"] = lockstep.steps
    result["physics_s_per_global_step"] = round(lockstep.advance_s / max(lockstep.steps, 1), 4)
    result["physics_share"] = round(lockstep.advance_s / result["wall_s"], 3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf8") as f:
        f.write(json.dumps(result) + "\n")
    print("BENCH", json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

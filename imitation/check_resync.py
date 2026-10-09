"""M1 exit check: does the expert still finish the half fold after someone else
has been driving? At a random step the expert hands over to a perturbation
(none / noisy expert / random actions) for `--k` steps, then resyncs its phases
from the sim and finishes the episode.

    python -m imitation.check_resync --episodes 24 --workers 8
    python -u -m imitation.check_resync --backend isaac --episodes 100 --out outputs/imitation/isaac/ig3/check_resync.json

Isaac (IG.3): one process, one env, the episodes run in sequence (there is no state snapshot to restore: the check
never needed one). `<out>.rows.jsonl` holds one row per finished episode, and a re-run skips the (mode, seed) pairs
already in it, so a job cut off by the 2 h limit resumes. The handover step is drawn from
imitation.seeds.ISAAC_RECOVERY_T (the Isaac fold takes ~230 steps; MuJoCo uses 5-80).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODES = ("resync_only", "noisy", "random")
MUJOCO_SWITCH_T = (5, 80)


def episode(env, teacher, job, switch_t=MUJOCO_SWITCH_T):
    """One check_resync episode on any backend: the expert drives until a random step, `mode` drives for k steps,
    then the expert resyncs its phases from the sim and finishes. Returns the result row."""
    seed, mode, k = job
    rng = np.random.default_rng(10_000 + seed)
    env.reset(seed=seed)
    teacher.reset()
    t_switch = int(rng.integers(*switch_t))
    info, phase_at_resync = {}, None
    for t in range(env.unwrapped.max_episode_steps):
        if t_switch <= t < t_switch + k and mode != "resync_only":
            a = teacher.act() if mode == "noisy" else rng.uniform(-1, 1, 12)
            if mode == "noisy":
                a = np.clip(a + rng.normal(0, 0.5, 12), -1, 1)
        else:
            if t == t_switch + (0 if mode == "resync_only" else k):
                phase_at_resync = teacher.expert.resync()
            a = teacher.act()
        _, _, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float32))
        if terminated or truncated:
            break
    return {"seed": seed, "mode": mode, "t_switch": t_switch, "success": bool(info["success"]),
            "reason": info["termination_reason"] or "truncated", "phases": phase_at_resync}


def _run(job):
    from imitation.tasks import HalfFoldEnv
    from imitation.teachers import ScriptedTeacher

    env = HalfFoldEnv()
    return episode(env, ScriptedTeacher(env), job)


def read_rows(path):
    """Rows already finished (the Isaac run's resume file); [] when there is none."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _isaac_make(backend):
    from imitation.tasks import make_env
    from imitation.teachers import ScriptedTeacher

    env = make_env(backend)
    return env, ScriptedTeacher(env)


def run_isaac(backend, episodes, k, rows_path, make=None, switch_t=None, log=print):
    """Sequential, resumable run with one env in this process. `make` -> (env, teacher) replaces the Isaac env (tests)."""
    if switch_t is None:
        from imitation.seeds import ISAAC_RECOVERY_T
        switch_t = ISAAC_RECOVERY_T
    rows = read_rows(rows_path)
    done = {(r["seed"], r["mode"]) for r in rows}
    todo = [(s, m, k) for m in MODES for s in range(episodes) if (s, m) not in done]
    if todo:
        env, teacher = make() if make else _isaac_make(backend)
        Path(rows_path).parent.mkdir(parents=True, exist_ok=True)
        for job in todo:
            row = episode(env, teacher, job, switch_t)
            rows.append(row)
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            log(f"ROW {row['seed']} {row['mode']} t_switch {row['t_switch']} success {row['success']} {row['reason']}")
    return rows


def summarize_rows(rows, episodes):
    """{mode: {"n", "success", "reasons"}} over the first `episodes` seeds of each mode."""
    summary = {}
    for m in MODES:
        rs = [r for r in rows if r["mode"] == m and r["seed"] < episodes]
        summary[m] = {"n": len(rs), "success": sum(r["success"] for r in rs),
                      "reasons": dict(Counter(r["reason"] for r in rs))}
    return summary


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--backend", choices=("mujoco", "isaac", "isaac_weld", "isaac_friction"), default="mujoco")
    ap.add_argument("--out", default=None, help="JSON summary (counts per mode); required for Isaac backends")
    return ap


def main():
    args = build_parser().parse_args()
    from imitation.tasks.half_fold import is_isaac
    if is_isaac(args.backend):
        if not args.out:
            raise SystemExit("isaac backends need --out (the resume file sits next to it)")
        rows = run_isaac(args.backend, args.episodes, args.k, str(args.out) + ".rows.jsonl")
        summary = summarize_rows(rows, args.episodes)
        for m, v in summary.items():
            print(f"{m:12s} success {v['success']}/{v['n']}  reasons {v['reasons']}")
        with open(args.out, "w") as f:
            json.dump({"episodes": args.episodes, "k": args.k, "backend": args.backend, "modes": summary}, f, indent=1)
        return
    jobs = [(s, m, args.k) for m in MODES for s in range(args.episodes)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_run, jobs)
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    summary = {}
    for m in MODES:
        rs = by_mode[m]
        summary[m] = {"n": len(rs), "success": sum(r["success"] for r in rs),
                      "reasons": dict(Counter(r["reason"] for r in rs))}
        print(f"{m:12s} success {sum(r['success'] for r in rs)}/{len(rs)}  "
              f"reasons {dict(Counter(r['reason'] for r in rs))}")
        for r in rs:
            if not r["success"]:
                print(f"    fail seed {r['seed']} t_switch {r['t_switch']} phases {r['phases']} {r['reason']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"episodes": args.episodes, "k": args.k, "modes": summary}, f, indent=1)


if __name__ == "__main__":
    main()

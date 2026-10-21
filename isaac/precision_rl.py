"""Residual RL on the half fold's placement window (docs/superpowers/plans/2026-10-20-closed-loop-rl.md, R1-R4).

One Isaac process holds B scene copies (isaac.precision_env.PrecisionBatch, GPU `friction` profile, dynamics DR off)
and the learner (isaac.rlpd.RLPDAgent, SAC with RLPD settings) on the same GPU. Episodes start from snapshots taken at
the start of the carry, so only the window (carry, place, hold, release, settle) is simulated per episode.

    python isaac/precision_rl.py check-restore --copies 8 --seeds 810000:810008       # R1
    python isaac/precision_rl.py train --run outputs/isaac/rl/p1 --copies 8 --hours 8   # R2 (version 0) + R4

`train` builds (or reloads) two snapshot libraries in the run directory, train (seeds 800000+) and eval (900000+);
evaluates version 0, the zero residual, which is the scripted base (R2's parity number); collects PRIOR_EPISODES
episodes of the base with Ornstein-Uhlenbeck noise on the action; then trains online, UTD gradient updates per new
transition, evaluating the deterministic policy on the eval library every EVAL_EVERY env steps. learner.jsonl gets
one row per batch of episodes and one per evaluation; agent.pt is the latest learner, best.pt the best evaluation.
A child process does the work: if a copy goes unstable within two steps of a restore (a corrupt scene; p1), it saves
the learner, the online buffer and its state and exits CORRUPT, and the parent restarts it with the hours left.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAIN_SEED, EVAL_SEED = 800000, 900000
N_TRAIN_SNAPS, N_EVAL_SNAPS = 160, 16
PRIOR_EPISODES = 48
ACTOR_WARMUP = 3000              # online transitions of critic-only updates before the actor moves (p3: the critic
                                 # started at Q ~ +15 against returns ~ -1 and the actor chased it)
EVAL_EVERY = 10_000              # env steps
OU_THETA, OU_SIGMA = 0.15, 0.3   # prior-data noise, action units (RES_SCALE m each)
ONLINE_CAPACITY = 300_000
CORRUPT = 3                      # exit code: the scene is corrupt, the supervisor restarts the process
MAX_RESTARTS = 20


class SceneCorrupted(RuntimeError):
    """A copy went unstable within two steps of a restore: the scene, not the policy, is broken (p1, results.md)."""


def _wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def summarize(finals):
    f = [x for x in finals if x is not None]
    if not f:
        return {"n": 0}
    reasons = {}
    for x in f:
        reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    k = sum(x["passed"] for x in f)
    f = [x for x in f if x["mean"] is not None]          # unstable episodes have no alignment
    if not f:
        return {"n": 0, "reasons": reasons}
    m = np.array([x["mean"] for x in f]) * 100
    e = np.array([x["edge_max"] for x in f]) * 100
    return {"n": len(f), "mean_cm": round(float(m.mean()), 2), "mean_sd": round(float(m.std()), 2),
            "best_mean_cm": round(float(m.min()), 2), "edge_cm": round(float(e.mean()), 2), "passed": k,
            "pass_wilson": _wilson(k, len(f)), "reasons": reasons}


def snapshot_library(pb, path, seed0, count):
    """count snapshots from seeds seed0, seed0+1, ... (seeds whose copy never reaches the window are skipped)."""
    path = Path(path)
    if path.exists():
        return pickle.loads(path.read_bytes())
    snaps, seed = [], seed0
    while len(snaps) < count:
        got, _ = pb.scripted_to_window(list(range(seed, seed + pb.n)))
        snaps += [s for s in got if s is not None]
        seed += pb.n
        print(f"SNAPS {path.name} {len(snaps)}/{count} (seeds to {seed})", flush=True)
    snaps = snaps[:count]
    path.write_bytes(pickle.dumps(snaps))
    return snaps


def run_episodes(pb, snaps, policy, on_step=None):
    """One window episode per copy from snaps (len n); policy(obs (n, D)) -> actions (n, A). on_step(transitions)
    gets [(obs, act, rew, next_obs, terminal)] for the live copies after every step. Returns the n finals."""
    from isaac.precision_env import ACT_DIM
    obs = pb.reset_window(snaps)
    finals = [None] * pb.n
    while any(pb.state[i]["alive"] for i in range(pb.n)):
        act = np.clip(np.asarray(policy(obs), dtype=np.float32).reshape(pb.n, ACT_DIM), -1.0, 1.0)
        nobs, rew, term, done, alive, infos = pb.step(act)
        if on_step is not None:
            on_step([(obs[i], act[i], rew[i], nobs[i], term[i]) for i in range(pb.n) if alive[i]])
        for i in range(pb.n):
            if infos[i] is not None:
                finals[i] = infos[i]
                if infos[i]["reason"] == "unstable" and infos[i]["steps"] <= 2:
                    raise SceneCorrupted(json.dumps(infos[i]))
        obs = nobs
    return finals


def evaluate(pb, snaps, policy):
    finals = []
    for k in range(0, len(snaps), pb.n):
        chunk = snaps[k:k + pb.n]
        finals += run_episodes(pb, chunk + [chunk[0]] * (pb.n - len(chunk)), policy)[:len(chunk)]
    return finals


def check_restore(args):
    """R1: per copy, the base run straight through the window vs twice from the restored snapshot."""
    from isaac.precision_env import ACT_DIM, PrecisionBatch
    pb = PrecisionBatch(args.copies)
    a, b = (int(v) for v in args.seeds.split(":"))
    zero = lambda obs: np.zeros((pb.n, ACT_DIM))
    for start in range(a, b, pb.n):
        seeds = list(range(start, start + pb.n))
        snaps, direct = pb.scripted_to_window(seeds, finish=True)
        ok = [i for i in range(pb.n) if snaps[i] is not None]
        if not ok:
            continue
        fill = [snaps[i] if snaps[i] is not None else snaps[ok[0]] for i in range(pb.n)]
        t0, s0 = time.time(), pb.batch.steps
        r1 = run_episodes(pb, fill, zero)
        print("THROUGHPUT", json.dumps({"copies": pb.n, "batch_steps": pb.batch.steps - s0,
                                        "wall_s": round(time.time() - t0, 1),
                                        "env_steps_per_s": round(pb.n * (pb.batch.steps - s0) / (time.time() - t0), 2)}),
              flush=True)
        r2 = run_episodes(pb, fill, zero)
        for i in ok:
            row = {"seed": seeds[i], "direct": direct[i], "restored_1": r1[i], "restored_2": r2[i]}
            print("RESTORE", json.dumps(row), flush=True)
        for i in set(range(pb.n)) - set(ok):
            print("RESTORE", json.dumps({"seed": seeds[i], "no_window": True}), flush=True)


def check_recover(args):
    """Can a copy come back after its cloth blows up? Clean run, a forced blow-up (the jaws driven into the table with
    the floor off), plain restores, then park + settle + restore; and the noisy prior with the floor on."""
    from isaac.place_servo import IsaacPlaceServoExpert
    from isaac.precision_env import ACT_DIM, PrecisionBatch, instability
    pb = PrecisionBatch(args.copies)
    seeds = list(range(820000, 820000 + pb.n))
    snaps, _ = pb.scripted_to_window(seeds)
    ok = [i for i in range(pb.n) if snaps[i] is not None]
    fill = [snaps[i] if snaps[i] is not None else snaps[ok[0]] for i in range(pb.n)]
    zero = lambda obs: np.zeros((pb.n, ACT_DIM))
    down = np.zeros((pb.n, ACT_DIM))
    down[:, 2] = down[:, 6] = -1.0

    def report(tag, finals):
        for i in range(pb.n):
            print("RECOVER", tag, i, json.dumps({k: finals[i][k] for k in ("reason", "steps", "mean", "instability")}),
                  json.dumps(instability(pb.subs[i])), flush=True)

    report("clean", run_episodes(pb, fill, zero))
    floor, IsaacPlaceServoExpert.Z_FLOOR = IsaacPlaceServoExpert.Z_FLOOR, None
    report("forced", run_episodes(pb, fill, lambda obs: down))
    IsaacPlaceServoExpert.Z_FLOOR = floor
    report("restore_1", run_episodes(pb, fill, zero))
    report("restore_2", run_episodes(pb, fill, zero))
    for sub in pb.subs:
        sub.park()
    for e in (e for ex in pb.experts for e in ex.values()):
        e.phase = e.PHASES.index("done")
    for _ in range(30):
        pb._scripted_step()
    report("after_park", run_episodes(pb, fill, zero))
    rng = np.random.default_rng(0)
    ou = np.zeros((pb.n, ACT_DIM))

    def noisy(obs):
        ou[:] += -OU_THETA * ou + OU_SIGMA * rng.standard_normal(ou.shape)
        return ou.copy()
    report("ou_floor", run_episodes(pb, fill, noisy))


def train(args):
    import torch
    from isaac.precision_env import ACT_DIM, OBS_DIM, PrecisionBatch
    from isaac.rlpd import ReplayBuffer, RLPDAgent
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    log = open(run / "learner.jsonl", "a")

    def write(row):
        row["time"] = time.strftime("%H:%M:%S")
        log.write(json.dumps(row) + "\n")
        log.flush()
        print("LOG", json.dumps(row), flush=True)

    t_end = time.time() + 3600 * args.hours
    pb = PrecisionBatch(args.copies)
    train_snaps = snapshot_library(pb, run / "snaps_train.pkl", TRAIN_SEED, N_TRAIN_SNAPS)
    eval_snaps = snapshot_library(pb, run / "snaps_eval.pkl", EVAL_SEED, N_EVAL_SNAPS)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = RLPDAgent(OBS_DIM, ACT_DIM, device=device, init_std=args.init_std, seed=args.seed,
                      target_entropy_scale=args.entropy_scale)
    if (run / "agent.pt").exists():
        agent.load(run / "agent.pt")
    online = ReplayBuffer(OBS_DIM, ACT_DIM, ONLINE_CAPACITY, seed=args.seed)
    if (run / "online.npz").exists():
        online.load(run / "online.npz")
    prior = ReplayBuffer(OBS_DIM, ACT_DIM, 200_000, seed=args.seed + 1)
    rng = np.random.default_rng(args.seed)
    state = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else \
        {"env_steps": 0, "episodes": 0, "version": 0, "best": None, "next_eval": 0}

    def save():
        agent.save(run / "agent.pt")
        online.save(run / "online.npz")
        (run / "state.json").write_text(json.dumps(state))

    try:
        loop(args, pb, agent, online, prior, rng, state, train_snaps, eval_snaps, run, write, save, t_end)
    except SceneCorrupted as err:
        write({"corrupted": str(err), "env_steps": state["env_steps"]})
        save()
        sys.exit(CORRUPT)


def loop(args, pb, agent, online, prior, rng, state, train_snaps, eval_snaps, run, write, save, t_end):
    from isaac.precision_env import ACT_DIM

    def det(obs):
        return agent.act(obs, deterministic=True)

    def do_eval():
        t0 = time.time()
        s = summarize(evaluate(pb, eval_snaps, det))
        write({"eval": s, "version": state["version"], "env_steps": state["env_steps"],
               "wall_s": round(time.time() - t0, 1)})
        agent.save(run / f"v{state['version']}.pt")
        if s["n"] and (state["best"] is None or s["mean_cm"] < state["best"]):
            state["best"] = s["mean_cm"]
            agent.save(run / "best.pt")
        state["version"] += 1
        state["next_eval"] = state["env_steps"] + EVAL_EVERY

    if state["version"] == 0:
        do_eval()                       # version 0: the zero residual = the scripted base (R2 parity)
        (run / "state.json").write_text(json.dumps(state))
    if (run / "prior.npz").exists():
        prior.load(run / "prior.npz")
    else:
        ou = np.zeros((pb.n, ACT_DIM))
        def noisy(obs):
            ou[:] += -OU_THETA * ou + OU_SIGMA * rng.standard_normal(ou.shape)
            return ou.copy()
        for k in range(0, PRIOR_EPISODES, pb.n):
            ou[:] = 0.0
            batch = [train_snaps[j] for j in rng.integers(len(train_snaps), size=pb.n)]
            finals = run_episodes(pb, batch, noisy, on_step=lambda tr: [prior.add(*t) for t in tr])
            write({"prior": summarize(finals), "transitions": len(prior)})
        prior.save(run / "prior.npz")

    while time.time() < t_end:
        stats = {}

        def learn(tr):
            for t in tr:
                online.add(*t)
            state["env_steps"] += len(tr)
            if len(online) >= args.batch_size // 2:
                for _ in tr:                # RLPD's rate: one actor update and UTD critic updates per transition
                    stats.update(agent.update(online, prior if len(prior) else None, batch_size=args.batch_size,
                                              utd=args.utd, update_actor=state["env_steps"] >= ACTOR_WARMUP))
        t0 = time.time()
        batch = [train_snaps[j] for j in rng.integers(len(train_snaps), size=pb.n)]
        finals = run_episodes(pb, batch, lambda obs: agent.act(obs), on_step=learn)
        state["episodes"] += pb.n
        write({"train": summarize(finals), "env_steps": state["env_steps"], "episodes": state["episodes"],
               "wall_s": round(time.time() - t0, 1), **{k: round(float(v), 4) for k, v in stats.items()}})
        agent.save(run / "agent.pt")
        (run / "state.json").write_text(json.dumps(state))
        if state["env_steps"] >= state["next_eval"]:
            do_eval()
            save()
    do_eval()
    save()


def demo(args):
    """Full half folds from cloth seeds: the base to the carry, then the policy (or the base, --base) through the
    window, restored from the carry-start snapshot so the video runs on without a cut. One row per seed in
    <run>/demo/demo.jsonl, a top-camera video per seed with --video."""
    from isaac.precision_env import ACT_DIM, OBS_DIM, PrecisionBatch
    from isaac.rlpd import RLPDAgent
    run = Path(args.run)
    out = run / ("demo_base" if args.base else "demo")
    out.mkdir(parents=True, exist_ok=True)
    pb = PrecisionBatch(args.copies, cameras={"top": 400} if args.video else None)
    if args.base:
        policy = lambda obs: np.zeros((pb.n, ACT_DIM))
    else:
        agent = RLPDAgent(OBS_DIM, ACT_DIM, device="cuda:0")
        agent.load(run / args.ckpt)
        policy = lambda obs: agent.act(obs, deterministic=True)
    a, b = (int(v) for v in args.seeds.split(":"))
    for start in range(a, b, pb.n):
        seeds = list(range(start, start + pb.n))
        pb.frames = [[] for _ in range(pb.n)] if args.video else None
        pb.cut = [None] * pb.n
        snaps, _ = pb.scripted_to_window(seeds)
        ok = [i for i in range(pb.n) if snaps[i] is not None]
        if not ok:
            continue
        pre = [f[:c] for f, c in zip(pb.frames, pb.cut)] if args.video else None
        if args.video:
            pb.frames = [[] for _ in range(pb.n)]
        finals = run_episodes(pb, [snaps[i] if snaps[i] is not None else snaps[ok[0]] for i in range(pb.n)], policy)
        for i in ok:
            row = {"seed": seeds[i], "ckpt": None if args.base else args.ckpt, **finals[i]}
            print("DEMO", json.dumps(row), flush=True)
            with open(out / "demo.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if args.video:
                import imageio
                clip = pre[i] + pb.frames[i][:finals[i]["steps"]]
                imageio.mimwrite(str(out / f"seed{seeds[i]}.mp4"), clip + clip[-1:] * 20, fps=20, macro_block_size=1)
        for i in set(range(pb.n)) - set(ok):
            print("DEMO", json.dumps({"seed": seeds[i], "no_window": True}), flush=True)


def supervise(argv, hours):
    """Run `train` in child processes, restarting one that exits CORRUPT (it saved its learner, buffers and state)
    with the hours left."""
    t_end = time.time() + 3600 * hours
    for k in range(MAX_RESTARTS + 1):
        left = (t_end - time.time()) / 3600
        if left < 0.25:
            return
        args = list(argv)
        args[args.index("--hours") + 1] = f"{left:.3f}"
        rc = subprocess.run([sys.executable, "-u", __file__] + args,
                            env=dict(os.environ, PRECISION_CHILD="1")).returncode
        print(f"SUPERVISOR child {k} exit {rc}", flush=True)
        if rc != CORRUPT:
            sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--run", required=True)
    d.add_argument("--ckpt", default="best.pt")
    d.add_argument("--seeds", default="910000:910008")
    d.add_argument("--copies", type=int, default=8)
    d.add_argument("--video", action="store_true")
    d.add_argument("--base", action="store_true", help="the zero residual (the scripted base) instead of a policy")
    r = sub.add_parser("check-recover")
    r.add_argument("--copies", type=int, default=8)
    c = sub.add_parser("check-restore")
    c.add_argument("--copies", type=int, default=8)
    c.add_argument("--seeds", default="810000:810008")
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--copies", type=int, default=8)
    t.add_argument("--hours", type=float, default=8.0)
    t.add_argument("--utd", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--init-std", type=float, default=0.1)
    t.add_argument("--entropy-scale", type=float, default=0.5)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "train" and not os.environ.get("PRECISION_CHILD"):
        argv = sys.argv[1:] if "--hours" in sys.argv else sys.argv[1:] + ["--hours", str(args.hours)]
        supervise(argv, args.hours)
        return
    sys.argv = sys.argv[:1]              # AppLauncher reads sys.argv
    {"check-restore": check_restore, "check-recover": check_recover, "train": train, "demo": demo}[args.cmd](args)


if __name__ == "__main__":
    main()

"""Train and demo the RL corrector (isaac/corrector_env.py) on WATcloud, one Isaac process with B copies.

    python isaac/corrector_rl.py train --run outputs/isaac/rl/c1 --copies 8 --hours 8
    python isaac/corrector_rl.py demo --run outputs/isaac/rl/c1 --ckpt best.pt --seeds 950000:950064 --video

Run with WORLDFOLD_PLACE_PARAMS for the base (latch + g-fold 1.2). `train` builds settled-fold libraries (train from
seed 960000, eval from 970000), scores the no-op and the scripted push on the eval library, collects PRIOR_EPISODES of
the scripted push with noise and of random pushes, then trains SAC with RLPD settings (isaac/rlpd.py), evaluating the
deterministic policy every EVAL_EVERY transitions; learner.jsonl, best.pt (most passes, then lowest mean), agent.pt.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaac.precision_rl import summarize  # noqa: E402

TRAIN_SEED, EVAL_SEED = 960000, 970000
N_TRAIN, N_EVAL = 160, 32
PRIOR_EPISODES = 64
EVAL_EVERY = 400                 # transitions (decisions)


def library(cb, path, seed0, count):
    path = Path(path)
    if path.exists():
        return pickle.loads(path.read_bytes())
    snaps, seed = [], seed0
    while len(snaps) < count:
        snaps += [s for s in cb.settled_snapshots(list(range(seed, seed + cb.n))) if s is not None]
        seed += cb.n
        print(f"SNAPS {path.name} {len(snaps)}/{count}", flush=True)
    path.write_bytes(pickle.dumps(snaps[:count]))
    return snaps[:count]


def run(cb, snaps, policy, on_step=None, ends=None):
    """One corrector episode per copy; ends[i] gets len(pb.frames[i]) when copy i finishes (demo video)."""
    from isaac.corrector_env import ACT_DIM
    obs = cb.reset(snaps)
    finals = [None] * cb.n
    while cb.alive.any():
        act = np.clip(np.asarray(policy(obs), np.float32).reshape(cb.n, ACT_DIM), -1.0, 1.0)
        nobs, rew, term, done, alive, infos = cb.step(act)
        if on_step is not None:
            on_step([(obs[i], act[i], rew[i], nobs[i], term[i]) for i in range(cb.n) if alive[i]])
        for i in range(cb.n):
            if infos[i] is not None:
                finals[i] = infos[i]
                if ends is not None:
                    ends[i] = len(cb.pb.frames[i])
        obs = nobs
    return finals


def evaluate(cb, snaps, policy):
    finals = []
    for k in range(0, len(snaps), cb.n):
        chunk = snaps[k:k + cb.n]
        finals += run(cb, chunk + [chunk[0]] * (cb.n - len(chunk)), policy)[:len(chunk)]
    return finals


def scripted_policy(cb):
    from isaac.corrector_env import scripted_action
    return lambda obs: np.stack([scripted_action(cb._cloth(i)) for i in range(cb.n)])


def train(args):
    import torch
    from isaac.corrector_env import ACT_DIM, OBS_DIM, CorrectorBatch
    from isaac.rlpd import ReplayBuffer, RLPDAgent
    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "learner.jsonl", "a")

    def write(row):
        row["time"] = time.strftime("%H:%M:%S")
        log.write(json.dumps(row) + "\n")
        log.flush()
        print("LOG", json.dumps(row), flush=True)

    t_end = time.time() + 3600 * args.hours
    cb = CorrectorBatch(args.copies)
    train_snaps = library(cb, run_dir / "settled_train.pkl", TRAIN_SEED, N_TRAIN)
    eval_snaps = library(cb, run_dir / "settled_eval.pkl", EVAL_SEED, N_EVAL)
    agent = RLPDAgent(OBS_DIM, ACT_DIM, device="cuda:0" if torch.cuda.is_available() else "cpu",
                      init_std=args.init_std, seed=args.seed, target_entropy_scale=0.5, critic_clip=10.0)
    online = ReplayBuffer(OBS_DIM, ACT_DIM, 200_000, seed=args.seed)
    prior = ReplayBuffer(OBS_DIM, ACT_DIM, 50_000, seed=args.seed + 1)
    rng = np.random.default_rng(args.seed)
    zero = lambda obs: np.zeros((cb.n, ACT_DIM))
    write({"baseline": "no-op", "eval": summarize(evaluate(cb, eval_snaps, zero))})
    write({"baseline": "scripted push", "eval": summarize(evaluate(cb, eval_snaps, scripted_policy(cb)))})
    script = scripted_policy(cb)
    for k in range(0, PRIOR_EPISODES, cb.n):
        noisy = (lambda obs: script(obs) + 0.2 * rng.standard_normal((cb.n, ACT_DIM))) if (k // cb.n) % 2 == 0 \
            else (lambda obs: rng.uniform(-0.6, 0.6, (cb.n, ACT_DIM)))
        batch = [train_snaps[j] for j in rng.integers(len(train_snaps), size=cb.n)]
        finals = run(cb, batch, noisy, on_step=lambda tr: [prior.add(*t) for t in tr])
        write({"prior": summarize(finals), "transitions": len(prior)})
    state = {"transitions": 0, "episodes": 0, "version": 0, "best": None, "next_eval": 0}

    def det(obs):
        return agent.act(obs, deterministic=True)

    def do_eval():
        s = summarize(evaluate(cb, eval_snaps, det))
        write({"eval": s, "version": state["version"], "transitions": state["transitions"]})
        agent.save(run_dir / f"v{state['version']}.pt")
        key = (s.get("passed", 0), -s.get("mean_cm", 99.0))
        if state["best"] is None or key > tuple(state["best"]):
            state["best"] = list(key)
            agent.save(run_dir / "best.pt")
        state["version"] += 1
        state["next_eval"] = state["transitions"] + EVAL_EVERY

    while time.time() < t_end:
        stats = {}

        def learn(tr):
            for t in tr:
                online.add(*t)
            state["transitions"] += len(tr)
            if len(online) >= 32:
                for _ in tr:
                    stats.update(agent.update(online, prior, batch_size=args.batch_size, utd=args.utd))
        batch = [train_snaps[j] for j in rng.integers(len(train_snaps), size=cb.n)]
        finals = run(cb, batch, lambda obs: agent.act(obs), on_step=learn)
        state["episodes"] += cb.n
        write({"train": summarize(finals), "transitions": state["transitions"], "episodes": state["episodes"],
               **{k: round(float(v), 4) for k, v in stats.items()}})
        agent.save(run_dir / "agent.pt")
        (run_dir / "state.json").write_text(json.dumps(state))
        if state["transitions"] >= state["next_eval"]:
            do_eval()
    do_eval()


def demo(args):
    """Full folds from cloth seeds: the base fold, its settle, then the corrector (or the scripted push, --scripted),
    as one continuous video per seed (each stage restores the state the previous one ended on)."""
    from isaac.corrector_env import ACT_DIM, OBS_DIM, CorrectorBatch
    from isaac.rlpd import RLPDAgent
    run_dir = Path(args.run)
    out = run_dir / ("demo_scripted" if args.scripted else "demo")
    out.mkdir(parents=True, exist_ok=True)
    cb = CorrectorBatch(args.copies, cameras={"top": 400} if args.video else None)
    pb = cb.pb
    if args.scripted:
        policy = scripted_policy(cb)
    else:
        agent = RLPDAgent(OBS_DIM, ACT_DIM, device="cuda:0")
        agent.load(run_dir / args.ckpt)
        policy = lambda obs: agent.act(obs, deterministic=True)
    a, b = (int(v) for v in args.seeds.split(":"))
    for start in range(a, b, cb.n):
        seeds = list(range(start, start + cb.n))
        pb.frames = [[] for _ in range(cb.n)] if args.video else None
        pb.cut = [None] * cb.n
        carry, _ = pb.scripted_to_window(seeds)
        seg1 = [f[:c] if c else [] for f, c in zip(pb.frames, pb.cut)] if args.video else None
        if args.video:
            pb.frames = [[] for _ in range(cb.n)]
        settled = cb.settle(carry, seeds)
        ok = [i for i in range(cb.n) if settled[i] is not None]
        if not ok:
            continue
        seg2 = [f[:c] for f, c in zip(pb.frames, cb.settle_cut)] if args.video else None
        if args.video:
            pb.frames = [[] for _ in range(cb.n)]
        ends = [0] * cb.n
        finals = run(cb, [settled[i] if settled[i] is not None else settled[ok[0]] for i in range(cb.n)], policy,
                     ends=ends)
        for i in ok:
            row = {"seed": seeds[i], "policy": "scripted" if args.scripted else args.ckpt,
                   **{k: v for k, v in finals[i].items() if k != "cloth"}}
            print("DEMO", json.dumps(row), flush=True)
            with open(out / "demo.jsonl", "a") as f:
                f.write(json.dumps(dict(row, cloth=finals[i]["cloth"])) + "\n")
            if args.video:
                import imageio
                clip = seg1[i] + seg2[i] + pb.frames[i][:ends[i]]
                imageio.mimwrite(str(out / f"seed{seeds[i]}.mp4"), clip + clip[-1:] * 20, fps=20, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--copies", type=int, default=8)
    t.add_argument("--hours", type=float, default=8.0)
    t.add_argument("--utd", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--init-std", type=float, default=0.3)
    t.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("demo")
    d.add_argument("--run", required=True)
    d.add_argument("--ckpt", default="best.pt")
    d.add_argument("--seeds", default="950000:950064")
    d.add_argument("--copies", type=int, default=8)
    d.add_argument("--video", action="store_true")
    d.add_argument("--scripted", action="store_true")
    args = ap.parse_args()
    sys.argv = sys.argv[:1]
    {"train": train, "demo": demo}[args.cmd](args)
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()

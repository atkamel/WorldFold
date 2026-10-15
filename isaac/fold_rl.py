"""Fold-move RL for the Isaac quarter fold (docs/superpowers/plans/2026-10-14-quarter-fold-rl.md §5).

A policy picks each fold's per-arm move parameters (ARM_PARAMS) from the cloth's state at the start of that fold;
IsaacArmExpert executes them inside QuarterFoldExpert on the "quarter" profile. Two decisions per episode, PPO,
rewards from the layer alignment (cloth_fold_rl.fold_alignment). The mean is the scripted value plus a network
whose last layer starts at zero, so version 0 is the scripted fold.

The learner and the workers share a run directory: the learner writes policy/v<k>.pt, each worker (one Isaac
process) writes episodes/w<worker>-<n>.json. `launch` starts both on one machine (a Modal container, a SLURM job):

    python isaac/fold_rl.py launch --run outputs/isaac/rl/r1 --workers 8 --gpus 4 --hours 6
    python isaac/fold_rl.py eval --run outputs/isaac/rl/r1 --seeds 900000-900019 --video
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (expert attribute or overshoot axis, low, high), per arm; the scripted value is each one's default
ARM_PARAMS = (
    ("PINCH_INSET", -0.005, 0.015),
    ("PINCH_HEIGHT", 0.002, 0.012),
    ("SQUEEZE", -0.10, 0.15),        # jaw target (rad); the jaws meet at about -0.015
    ("ARC_HEIGHT", 0.06, 0.16),
    ("PLACE_HEIGHT", 0.004, 0.03),
    ("CARRY_SPEED", 0.4, 1.0),
    ("RELEASE_RATE", 0.02, 1.2),     # rad per control step; 1.2 opens in one step, as the scripted release
    ("place_dx", -0.04, 0.04),       # added to the scripted placement overshoot (fold_expert.OVERSHOOTS), world axes
    ("place_dy", -0.04, 0.04),
)
ARMS = ("left_", "right_")
N_ARM = len(ARM_PARAMS)
ACT_DIM = len(ARMS) * N_ARM
KEYPOINTS = [ix * 11 + iy for ix in (0, 5, 10) for iy in (0, 5, 10)]   # corners, edge midpoints, centre
OBS_DIM = 3 * len(KEYPOINTS) + 1
FOLD1 = (("y", 5),)
INIT_STD = 0.25                  # in [-1, 1] units per parameter
SEED_BASE = 700000
FAIL = 10.0                      # reward penalty: the env failed the fold (drag, unstable, timeout)
PASS_BONUS = 20.0


def defaults(stage):
    """The scripted values, in ARM_PARAMS order for each arm."""
    from isaac.fold_expert import OVERSHOOTS, IsaacArmExpert
    from isaac.isaac_env import PROFILES
    out = []
    for p in ARMS:
        for name, _, _ in ARM_PARAMS:
            if name == "SQUEEZE":
                out.append(PROFILES["quarter"]["friction_grasp"]["gripper_closed"])
            elif name == "RELEASE_RATE":
                out.append(1.2)
            elif name.startswith("place_"):
                out.append(0.0)
            else:
                out.append(float(getattr(IsaacArmExpert, name)))
    return np.array(out)


def _bounds():
    lo = np.array([lo for _, lo, _ in ARM_PARAMS] * len(ARMS))
    hi = np.array([hi for _, _, hi in ARM_PARAMS] * len(ARMS))
    return lo, hi


def to_unit(v):
    lo, hi = _bounds()
    return 2.0 * (np.asarray(v) - lo) / (hi - lo) - 1.0


def from_unit(u):
    lo, hi = _bounds()
    return lo + (np.clip(u, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)


def apply(expert, stage, values):
    """Set one fold's move parameters on its two IsaacArmExperts (before their first act())."""
    for k, p in enumerate(ARMS):
        v = dict(zip([n for n, _, _ in ARM_PARAMS], values[k * N_ARM:(k + 1) * N_ARM]))
        e = expert.experts[(stage, p)]
        for name in ("PINCH_INSET", "PINCH_HEIGHT", "SQUEEZE", "ARC_HEIGHT", "PLACE_HEIGHT", "CARRY_SPEED",
                     "RELEASE_RATE"):
            setattr(e, name, float(v[name]))
        from isaac.fold_expert import OVERSHOOTS
        expert.overshoot[(stage, p)] = OVERSHOOTS["quarter"][(stage, p)] + np.array([v["place_dx"], v["place_dy"], 0.0])


def features(env, stage):
    base = env.unwrapped
    c = base._prof["cloth_center"]
    from mujuco.cloth_params import TABLE_TOP_Z
    kp = base.cloth_positions()[KEYPOINTS] - np.array([c[0], c[1], TABLE_TOP_Z])
    return np.r_[10.0 * kp.ravel(), float(stage)].astype(np.float32)


# ---- policy --------------------------------------------------------------------------------------------------------

def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def make_policy():
    torch = _torch()
    nn = torch.nn

    def mlp(out):
        net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))
        return net

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = nn.ModuleList([mlp(ACT_DIM) for _ in range(2)])     # one head per fold
            self.v = nn.ModuleList([mlp(1) for _ in range(2)])
            for head in self.pi:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
            self.log_std = nn.Parameter(torch.full((2, ACT_DIM), math.log(INIT_STD)))
            self.register_buffer("u0", torch.tensor(np.stack([to_unit(defaults(s)) for s in (0, 1)]),
                                                    dtype=torch.float32))

        def dist(self, obs, stage):
            mean = self.u0[stage] + self.pi[stage](obs)
            return torch.distributions.Normal(mean, self.log_std[stage].exp())

    return Policy()


def save_policy(pol, path):
    torch = _torch()
    tmp = str(path) + ".tmp"
    torch.save(pol.state_dict(), tmp)
    os.replace(tmp, path)


def latest_version(run):
    vs = [int(Path(f).stem[1:]) for f in glob.glob(str(Path(run, "policy", "v*.pt")))]
    return max(vs) if vs else None


def load_policy(run, version):
    torch = _torch()
    pol = make_policy()
    pol.load_state_dict(torch.load(Path(run, "policy", f"v{version}.pt"), map_location="cpu"))
    return pol


def decide(pol, obs, stage, explore):
    torch = _torch()
    with torch.no_grad():
        d = pol.dist(torch.as_tensor(obs)[None], stage)
        u = d.sample() if explore else d.mean
        return u[0].numpy(), float(d.log_prob(u).sum())


# ---- one episode ---------------------------------------------------------------------------------------------------

def episode(env, expert, pol, seed, explore=True, frames=None):
    from cloth_fold_rl.fold_alignment import score
    base = env.unwrapped
    env.reset(seed=seed)
    expert.reset()
    decisions = []

    def choose(stage):
        o = features(env, stage)
        u, logp = decide(pol, o, stage, explore)
        apply(expert, stage, from_unit(u))
        decisions.append({"stage": stage, "obs": o.tolist(), "act": u.tolist(), "logp": logp})

    choose(0)
    fold1 = None
    for t in range(base.max_episode_steps):
        _, _, terminated, truncated, info = env.step(expert.act())
        if frames is not None:
            frames.append(base.render_rig()["top"].transpose(1, 2, 0))
        if env.stage == 1 and fold1 is None:
            s = score(base.cloth_positions(), folds=FOLD1)
            fold1 = {"mean": s["mean"], "edge_max": s["edge_max"]}
            choose(1)
        if terminated or truncated:
            break
    final = score(base.cloth_positions())
    success = bool(info["success"])
    if fold1 is None:     # fold 1 never settled: score it as it lies
        s = score(base.cloth_positions(), folds=FOLD1)
        r1 = -100.0 * (s["mean"] + s["edge_max"]) - FAIL
        decisions[0]["reward"] = r1
    else:
        decisions[0]["reward"] = -100.0 * (fold1["mean"] + fold1["edge_max"])
        decisions[1]["reward"] = (-100.0 * (final["mean"] + final["edge_max"]) + PASS_BONUS * final["passed"]
                                  - FAIL * (not success))
    return {"seed": seed, "steps": t + 1, "stage": int(info["stage"]), "success": success,
            "reason": info["termination_reason"] or "truncated", "fold1": fold1,
            "align": {"mean": final["mean"], "edge_max": final["edge_max"], "passed": bool(final["passed"])},
            "decisions": decisions}


def _make_env_expert(video=False):
    from isaac.isaac_env import make_quarter_fold_env
    from isaac.fold_expert import IsaacQuarterFoldExpert, _apply_env_params
    _apply_env_params()
    env = make_quarter_fold_env(cameras={"top": 400} if video else None)
    expert = IsaacQuarterFoldExpert(env)        # its own overshoot table copy: apply() edits it per episode
    return env, expert


def worker(args):
    run = Path(args.run)
    (run / "episodes").mkdir(parents=True, exist_ok=True)
    env, expert = _make_env_expert()
    t_end, n = time.time() + 3600 * args.hours, 0
    while time.time() < t_end:
        v = latest_version(run)
        if v is None:
            time.sleep(5)
            continue
        if (run / "STOP").exists():
            break
        pol = load_policy(run, v)
        seed = SEED_BASE + 10000 * args.worker + n
        rec = episode(env, expert, pol, seed)
        rec.update(version=v, worker=args.worker)
        path = run / "episodes" / f"w{args.worker}-{n:05d}.json"
        path.with_suffix(".tmp").write_text(json.dumps(rec))
        os.replace(path.with_suffix(".tmp"), path)
        print(f"EP w{args.worker} n{n} v{v} stage {rec['stage']} {rec['reason']} mean {rec['align']['mean']:.4f} "
              f"edge {rec['align']['edge_max']:.4f} pass {rec['align']['passed']}", flush=True)
        n += 1
    env.close()


# ---- learner -------------------------------------------------------------------------------------------------------

def ppo_update(pol, opt, batch, epochs=8, clip=0.2):
    torch = _torch()
    rows = []          # (stage, obs, act, logp, return)
    for rec in batch:
        ds = rec["decisions"]
        ret = 0.0
        for d in reversed(ds):
            ret += d["reward"]
            rows.append((d["stage"], d["obs"], d["act"], d["logp"], ret))
    stats = {}
    for s in (0, 1):
        r = [x for x in rows if x[0] == s]
        if not r:
            continue
        obs = torch.tensor([x[1] for x in r], dtype=torch.float32)
        act = torch.tensor([x[2] for x in r], dtype=torch.float32)
        old = torch.tensor([x[3] for x in r], dtype=torch.float32)
        ret = torch.tensor([x[4] for x in r], dtype=torch.float32)
        for _ in range(epochs):
            v = pol.v[s](obs).squeeze(-1)
            adv = ret - v.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6) if len(r) > 1 else adv
            d = pol.dist(obs, s)
            ratio = (d.log_prob(act).sum(-1) - old).exp()
            loss_pi = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            loss_v = ((v - ret) ** 2).mean()
            opt.zero_grad()
            (loss_pi + 0.5 * loss_v).backward()
            opt.step()
        stats[f"fold{s + 1}_return"] = float(ret.mean())
        stats[f"fold{s + 1}_std"] = float(pol.log_std[s].exp().mean())
    return stats


def learner(args):
    torch = _torch()
    run = Path(args.run)
    (run / "policy").mkdir(parents=True, exist_ok=True)
    k = latest_version(run)
    if k is None:
        k = 0
        save_policy(make_policy(), run / "policy" / "v0.pt")
    pol = load_policy(run, k)
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    seen = set()
    while k < args.iters:
        batch = []
        while len(batch) < args.batch:
            for f in sorted(glob.glob(str(run / "episodes" / "*.json"))):
                if f in seen:
                    continue
                seen.add(f)
                rec = json.loads(Path(f).read_text())
                if rec["version"] >= k - 1:          # one version of lag; PPO's ratio uses the logged logp
                    batch.append(rec)
            if len(batch) < args.batch:
                time.sleep(10)
        stats = ppo_update(pol, opt, batch)
        k += 1
        save_policy(pol, run / "policy" / f"v{k}.pt")
        a = [r["align"] for r in batch]
        stats.update(version=k, n=len(batch), time=time.strftime("%H:%M:%S"),
                     success=sum(r["success"] for r in batch), passed=sum(x["passed"] for x in a),
                     stage1=sum(r["stage"] >= 1 for r in batch),
                     mean_cm=round(100 * float(np.mean([x["mean"] for x in a])), 2),
                     edge_cm=round(100 * float(np.mean([x["edge_max"] for x in a])), 2),
                     best_mean_cm=round(100 * min(x["mean"] for x in a), 2))
        print("ITER", json.dumps(stats), flush=True)
        with open(run / "learner.jsonl", "a") as f:
            f.write(json.dumps(stats) + "\n")


def launch(args):
    """The learner plus args.workers Isaac workers on this machine, spread over args.gpus GPUs."""
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    me = [sys.executable, "-u", str(Path(__file__).resolve())]
    procs = [subprocess.Popen(me + ["learner", "--run", str(run), "--batch", str(args.batch), "--iters",
                                    str(args.iters), "--lr", str(args.lr)],
                              stdout=open(run / "learner.log", "a"), stderr=subprocess.STDOUT)]
    for w in range(args.workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w % args.gpus))
        procs.append(subprocess.Popen(me + ["worker", "--run", str(run), "--worker", str(args.first_worker + w),
                                            "--hours", str(args.hours)],
                                      stdout=open(run / f"worker{args.first_worker + w}.log", "a"),
                                      stderr=subprocess.STDOUT, env=env))
        time.sleep(20)        # Isaac starts one at a time (Kit start-up contends)
    t_end = time.time() + 3600 * args.hours
    while time.time() < t_end and any(p.poll() is None for p in procs[1:]):
        time.sleep(30)
    (run / "STOP").touch()
    for p in procs:
        try:
            p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()


def evaluate(args):
    run = Path(args.run)
    v = args.version if args.version is not None else latest_version(run)
    pol = load_policy(run, v)
    env, expert = _make_env_expert(video=args.video)
    out = run / "eval" / f"v{v}"
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in args.seeds.split("-"))
    for seed in range(lo, hi + 1):
        frames = [] if args.video else None
        rec = episode(env, expert, pol, seed, explore=False, frames=frames)
        rec["version"] = v
        print("EVAL", json.dumps({k: rec[k] for k in ("seed", "stage", "success", "reason", "align", "fold1")}),
              flush=True)
        with open(out / "episodes.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if frames and (rec["align"]["passed"] or args.keep_all_videos):
            import imageio
            imageio.mimwrite(str(out / f"seed{seed}.mp4"), frames + [frames[-1]] * 20, fps=20, macro_block_size=1)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "learner", "worker", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        if name in ("launch", "learner"):
            p.add_argument("--batch", type=int, default=32)
            p.add_argument("--iters", type=int, default=1000)
            p.add_argument("--lr", type=float, default=3e-4)
        if name in ("launch", "worker"):
            p.add_argument("--hours", type=float, default=6.0)
        if name == "launch":
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--gpus", type=int, default=1)
            p.add_argument("--first-worker", type=int, default=0)
        if name == "worker":
            p.add_argument("--worker", type=int, default=0)
        if name == "eval":
            p.add_argument("--version", type=int, default=None)
            p.add_argument("--seeds", default="900000-900019")
            p.add_argument("--video", action="store_true")
            p.add_argument("--keep-all-videos", action="store_true")
    args = ap.parse_args()
    {"launch": launch, "learner": learner, "worker": worker, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()

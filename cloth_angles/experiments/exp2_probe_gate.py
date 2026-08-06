"""Experiment 2: uncertainty-gated probing (EXPERIMENTS.md section 2).

Causal chain under test: crumpled starts are physically harder to fold AND
are where the DPM's K-sample spread is high (Exp 1: rho=0.85 between spread
and error). The gate spends a physical probe (grasp far corners, lift, drop,
settle -- flattens wrinkles) only when spread says the state is ambiguous,
then folds. Arms:

    gate    -- probe iff spread > tau (at most once), then fold
    always  -- fold immediately, never probe
    random  -- probe with fixed probability p regardless of spread
               (frequency-matched control; set --random-p to gate's rate)

Matched starts: every arm replays the same seeds/perturbations. Strata:
easy (flat, small offset) vs ambiguous (offset + random-action crumpling).
Responses: success, final fold_score, probes used, episode steps. Stats:
McNemar (success) and Wilcoxon (fold_score) on the gate-vs-always pair.

Calibration mode (--calibrate) measures spread distributions per stratum
without folding, to choose tau on data disjoint from the main battery.

Usage:
    python -m cloth_angles.experiments.exp2_probe_gate --config cloth_angles/config.yaml \
        --dpm-checkpoint outputs/cloth_angles/diffusion/dpm_060000.pt --calibrate
    python -m cloth_angles.experiments.exp2_probe_gate ... --tau 1.0 --arms gate,always
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mujuco"))

from cloth_angles.collect_episodes import angle_obs
from cloth_angles.data.occlusion import sample_mask
from cloth_angles.model.diffusion import load_dpm_checkpoint

# main battery seeds; tuning/calibration uses TUNE_SEEDS (disjoint)
MAIN_SEEDS = list(range(3000, 3040))      # 40 starts: 20 easy + 20 ambiguous
TUNE_SEEDS = list(range(3100, 3124))      # 24 starts: 12 + 12
PERTURB_STEPS = 25
SETTLE_STEPS = 30
PROBE_BUDGET = 130                        # max control steps a probe may take
MAX_TOTAL_STEPS = 400


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dpm-checkpoint", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--tau", type=float, default=None, help="spread gate threshold")
    parser.add_argument("--arms", default="gate,always", help="comma list: gate,always,random")
    parser.add_argument("--random-p", type=float, default=0.5,
                         help="probe probability for the random arm")
    parser.add_argument("--n-per-stratum", type=int, default=20)
    parser.add_argument("--calibrate", action="store_true",
                         help="only measure spread per stratum on tuning seeds")
    parser.add_argument("--output-dir", default="outputs/cloth_angles/exp2")
    return parser.parse_args()


def make_env():
    from sim_main import ClothFoldEnv  # noqa: PLC0415
    return ClothFoldEnv(max_episode_steps=MAX_TOTAL_STEPS, observation_mode="state")


def build_start(env, seed: int, stratum: str) -> int:
    """Reset into a matched start. Returns steps consumed. easy: flat cloth,
    small offset. ambiguous: offset + random-action crumpling + brief settle."""
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-0.03, 0.03, size=2)
    env.reset(seed=seed, options={"cloth_pose": offset, "task": 0})
    steps = 0
    if stratum == "ambiguous":
        for _ in range(PERTURB_STEPS):
            env.step(rng.uniform(-1.0, 1.0, size=14).astype(np.float32))
            steps += 1
        open_action = np.zeros(14, dtype=np.float32)
        open_action[6] = open_action[13] = 1.0    # force grippers open post-perturb
        for _ in range(SETTLE_STEPS):
            env.step(open_action)
            steps += 1
    return steps


def perceive_spread(env, dpm, grid_size: int, seed: int, k: int) -> float:
    """Heavy flap occlusion (seeded per episode) + DPM K-sample spread on the
    hidden cells -- the gate signal validated in Exp 1."""
    rng = np.random.default_rng(seed + 500_000)
    field = angle_obs(env, grid_size, signed=True)
    mask = sample_mask(grid_size, "heavy", rng)
    masked = field.copy()
    masked[mask.reshape(-1)] = 0.0
    f_t = torch.as_tensor(masked)
    m_t = torch.as_tensor(mask.reshape(-1).astype(np.float32))
    samples = dpm.sample_k(f_t, m_t, k=k)
    return float(samples.var(dim=0)[torch.as_tensor(mask.reshape(-1))].mean())


def run_probe(env) -> int:
    """Physical uncertainty-reducing action: grasp far corners, lift, release,
    settle. Flattens wrinkles. Returns steps consumed (capped at PROBE_BUDGET)."""
    from sim_main import ScriptedFoldPolicy  # noqa: PLC0415
    policy = ScriptedFoldPolicy(env)
    steps = 0
    # phase 1: reach + lift via the fold policy, stop once both arms are past lift
    while steps < PROBE_BUDGET - 55:
        env.step(policy.act())
        steps += 1
        past_lift = all(policy.phase[p] in ("carry", "place", "lower", "release", "done")
                        for p in env.prefixes)
        if past_lift:
            break
    # phase 2: release and retreat upward
    release = np.zeros(14, dtype=np.float32)
    release[6] = release[13] = 1.0
    release[2] = release[9] = 0.5      # ee up
    for _ in range(15):
        env.step(release)
        steps += 1
    # phase 3: hands-off settle
    idle = np.zeros(14, dtype=np.float32)
    idle[6] = idle[13] = 1.0
    for _ in range(40):
        env.step(idle)
        steps += 1
    return steps


def run_fold(env, steps_used: int):
    """Scripted fold until termination or total budget. Returns (success,
    fold_score, steps_consumed)."""
    from sim_main import ScriptedFoldPolicy  # noqa: PLC0415
    policy = ScriptedFoldPolicy(env)
    success, fold_score = False, 0.0
    steps = 0
    while steps_used + steps < MAX_TOTAL_STEPS:
        _, _, terminated, truncated, info = env.step(policy.act())
        steps += 1
        success = bool(info["success"])
        fold_score = float(info["fold_score"])
        if terminated or truncated:
            break
    return success, fold_score, steps


def run_episode(env, dpm, grid_size: int, seed: int, stratum: str, arm: str,
                 tau: float, random_p: float, k: int) -> dict:
    steps = build_start(env, seed, stratum)
    spread0 = perceive_spread(env, dpm, grid_size, seed, k)

    probes = 0
    spread_after = spread0
    do_probe = False
    if arm == "gate":
        do_probe = spread0 > tau
    elif arm == "random":
        do_probe = bool(np.random.default_rng(seed + 900_000).uniform() < random_p)

    if do_probe:
        steps += run_probe(env)
        probes = 1
        spread_after = perceive_spread(env, dpm, grid_size, seed + 1, k)

    success, fold_score, fold_steps = run_fold(env, steps)
    steps += fold_steps
    return {
        "seed": seed, "stratum": stratum, "arm": arm,
        "spread0": round(spread0, 4), "spread_after_probe": round(spread_after, 4),
        "probes": probes, "steps": steps,
        "success": int(success), "fold_score": round(fold_score, 4),
    }


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    grid_size = config["data"]["grid_size"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpm, _ = load_dpm_checkpoint(args.dpm_checkpoint, expected_grid_size=grid_size)
    env = make_env()

    if args.calibrate:
        print("=== calibration: spread per stratum (tuning seeds, no folding) ===")
        rows = []
        for i, seed in enumerate(TUNE_SEEDS):
            stratum = "easy" if i < len(TUNE_SEEDS) // 2 else "ambiguous"
            build_start(env, seed, stratum)
            spread = perceive_spread(env, dpm, grid_size, seed, args.k)
            rows.append({"seed": seed, "stratum": stratum, "spread": spread})
            print(f"  seed={seed} [{stratum}] spread={spread:.4f}")
        easy = [r["spread"] for r in rows if r["stratum"] == "easy"]
        amb = [r["spread"] for r in rows if r["stratum"] == "ambiguous"]
        print(f"\neasy:      mean={np.mean(easy):.4f} median={np.median(easy):.4f} max={np.max(easy):.4f}")
        print(f"ambiguous: mean={np.mean(amb):.4f} median={np.median(amb):.4f} min={np.min(amb):.4f}")
        suggested = float((np.max(easy) + np.min(amb)) / 2) if np.min(amb) > np.max(easy) \
            else float(np.median(easy + amb))
        print(f"suggested tau: {suggested:.4f} "
              f"({'clean separation' if np.min(amb) > np.max(easy) else 'overlapping -- midpoint of pooled median'})")
        with open(output_dir / "calibration.json", "w") as f:
            json.dump({"rows": rows, "suggested_tau": suggested}, f, indent=2)
        return

    if args.tau is None:
        raise SystemExit("--tau required for the main battery (run --calibrate first)")

    arms = args.arms.split(",")
    n = args.n_per_stratum
    starts = [(s, "easy") for s in MAIN_SEEDS[:n]] + [(s, "ambiguous") for s in MAIN_SEEDS[20:20 + n]]

    rows = []
    for arm in arms:
        for seed, stratum in starts:
            row = run_episode(env, dpm, grid_size, seed, stratum, arm,
                               args.tau, args.random_p, args.k)
            rows.append(row)
            print(f"[{arm}] seed={seed} [{stratum}] spread={row['spread0']:.3f} "
                  f"probes={row['probes']} success={row['success']} fold={row['fold_score']:.3f} "
                  f"steps={row['steps']}")

    csv_path = output_dir / "per_episode.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # summary + paired stats (gate vs always on matched seeds)
    from scipy import stats  # noqa: PLC0415
    print("\n=== summary ===")
    results = {"tau": args.tau, "k": args.k}
    for stratum in ("easy", "ambiguous"):
        for arm in arms:
            sel = [r for r in rows if r["stratum"] == stratum and r["arm"] == arm]
            if not sel:
                continue
            sr = np.mean([r["success"] for r in sel])
            fs = np.mean([r["fold_score"] for r in sel])
            pr = np.mean([r["probes"] for r in sel])
            st = np.mean([r["steps"] for r in sel])
            results[f"{stratum}_{arm}"] = {"success_rate": sr, "mean_fold_score": fs,
                                             "probe_rate": pr, "mean_steps": st}
            print(f"[{stratum}] {arm}: success={sr:.2f} fold_score={fs:.3f} "
                  f"probe_rate={pr:.2f} steps={st:.0f}")
        if "gate" in arms and "always" in arms:
            g = {r["seed"]: r for r in rows if r["stratum"] == stratum and r["arm"] == "gate"}
            a = {r["seed"]: r for r in rows if r["stratum"] == stratum and r["arm"] == "always"}
            common = sorted(set(g) & set(a))
            if common:
                gf = np.array([g[s]["fold_score"] for s in common])
                af = np.array([a[s]["fold_score"] for s in common])
                if np.any(gf != af):
                    w = stats.wilcoxon(gf, af)
                    print(f"[{stratum}] gate vs always fold_score: mean diff {float((gf - af).mean()):+.4f}, "
                          f"wilcoxon p={w.pvalue:.4g}")
                    results[f"{stratum}_gate_vs_always"] = {
                        "mean_diff": float((gf - af).mean()), "wilcoxon_p": float(w.pvalue)}
                # McNemar on success (paired binary)
                b = sum(1 for s in common if g[s]["success"] and not a[s]["success"])
                c = sum(1 for s in common if a[s]["success"] and not g[s]["success"])
                if b + c > 0:
                    p_mc = stats.binomtest(b, b + c, 0.5).pvalue
                    print(f"[{stratum}] McNemar success (gate+/always- = {b}, gate-/always+ = {c}): p={p_mc:.4g}")
                    results[f"{stratum}_mcnemar"] = {"b": b, "c": c, "p": float(p_mc)}

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {csv_path}")
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

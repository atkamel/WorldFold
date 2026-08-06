"""Experiment 2 (redesigned): uncertainty-gated ACTIVE PERCEPTION.

Calibration on tuning seeds killed the original premise: DPM sample spread
does not track physical crumpledness -- it tracks PERCEPTUAL AMBIGUITY given
the occlusion (bimodal: ~0.01 when the visible cells pin the state, >0.8 when
flat-vs-folded is genuinely undecidable under the flap; three FLAT starts hit
spreads of 3.1-3.9). So the probe matching the signal is re-observation (a
second look with a different occlusion), not physical flattening.

Design (single sim pass per unit; all arms derived from the same two looks,
perfectly paired):

    look1: heavy flap mask m1 -> DPM K-sample belief B1, spread s1
    look2: independent flap mask m2 -> conditioning on the UNION of visible
           cells (hidden only where BOTH looks hide) -> belief B12
    arms:  single = B1
           gated  = B12 if s1 > tau else B1     (pays for look2 only when ambiguous)
           always = B12                          (upper bound, pays 2 looks everywhere)

Response: wrapped MAE on the cells hidden in look1 (the initial blind spot).
Cost: mean number of looks. Hypothesis: gated ~= always in accuracy at ~1.3
looks. Stats: paired Wilcoxon per pair, Bonferroni alpha 0.05/3.

Usage:
    python -m cloth_angles.experiments.exp2_active_perception \
        --config cloth_angles/config.yaml \
        --dpm-checkpoint outputs/cloth_angles/diffusion/dpm_060000.pt --tau 0.5
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
from cloth_angles.experiments.exp2_probe_gate import build_start
from cloth_angles.model.diffusion import load_dpm_checkpoint, wrapped_abs_error

MAIN_SEEDS = list(range(3000, 3040))   # 40 units: 20 easy + 20 ambiguous starts
ALPHA = 0.05
N_COMPARISONS = 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dpm-checkpoint", required=True)
    parser.add_argument("--tau", type=float, default=0.5,
                         help="spread gate; 0.5 sits in the bimodal gap measured at calibration")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/cloth_angles/exp2")
    return parser.parse_args()


def circular_mean(samples: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(samples).mean(0), torch.cos(samples).mean(0))


def belief(dpm, field: np.ndarray, hidden: np.ndarray, k: int):
    """DPM K-sample belief conditioned on `field` with `hidden` cells masked.
    Returns (belief field [N*N] tensor, spread on hidden cells)."""
    masked = field.copy()
    masked[hidden] = 0.0
    f_t = torch.as_tensor(masked)
    m_t = torch.as_tensor(hidden.astype(np.float32))
    samples = dpm.sample_k(f_t, m_t, k=k)
    spread = float(samples.var(dim=0)[torch.as_tensor(hidden)].mean())
    return circular_mean(samples), spread


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    grid_size = config["data"]["grid_size"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpm, _ = load_dpm_checkpoint(args.dpm_checkpoint, expected_grid_size=grid_size)

    from sim_main import ClothFoldEnv  # noqa: PLC0415
    env = ClothFoldEnv(max_episode_steps=400, observation_mode="state")

    rows = []
    for i, seed in enumerate(MAIN_SEEDS):
        stratum = "easy" if i < 20 else "ambiguous"
        build_start(env, seed, stratum)
        field = angle_obs(env, grid_size, signed=True)
        true_t = torch.as_tensor(field)

        rng = np.random.default_rng(seed + 500_000)
        m1 = sample_mask(grid_size, "heavy", rng).reshape(-1)
        m2 = sample_mask(grid_size, "heavy", rng).reshape(-1)   # independent second look
        both_hidden = m1 & m2

        b1, s1 = belief(dpm, field, m1, args.k)
        b12, s12 = belief(dpm, field, both_hidden, args.k)

        gated_uses_look2 = s1 > args.tau
        b_gated = b12 if gated_uses_look2 else b1

        h1 = torch.as_tensor(m1)
        row = {
            "seed": seed, "stratum": stratum,
            "spread1": round(s1, 4), "spread_fused": round(s12, 4),
            "gate_fired": int(gated_uses_look2),
            "looks_gated": 2 if gated_uses_look2 else 1,
            "single_mae": float(wrapped_abs_error(b1, true_t)[h1].mean()),
            "gated_mae": float(wrapped_abs_error(b_gated, true_t)[h1].mean()),
            "always_mae": float(wrapped_abs_error(b12, true_t)[h1].mean()),
        }
        rows.append(row)
        print(f"seed={seed} [{stratum}] s1={s1:.3f} fired={row['gate_fired']} "
              f"single={row['single_mae']:.4f} gated={row['gated_mae']:.4f} "
              f"always={row['always_mae']:.4f}")

    csv_path = output_dir / "active_perception_per_unit.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    from scipy import stats  # noqa: PLC0415
    results = {"tau": args.tau, "k": args.k, "n_units": len(rows)}
    print("\n=== belief MAE on look-1 hidden cells (rad) ===")
    for scope in ("easy", "ambiguous", "all"):
        sel = [r for r in rows if scope == "all" or r["stratum"] == scope]
        arm = {a: np.array([r[f"{a}_mae"] for r in sel]) for a in ("single", "gated", "always")}
        looks = float(np.mean([r["looks_gated"] for r in sel]))
        fired = float(np.mean([r["gate_fired"] for r in sel]))
        results[scope] = {
            "means": {a: float(v.mean()) for a, v in arm.items()},
            "gated_mean_looks": looks, "gate_fire_rate": fired, "n": len(sel),
        }
        print(f"[{scope}] n={len(sel)} single={arm['single'].mean():.4f} "
              f"gated={arm['gated'].mean():.4f} always={arm['always'].mean():.4f} "
              f"| gated looks={looks:.2f} fire_rate={fired:.2f}")
        for pair in (("single", "gated"), ("single", "always"), ("gated", "always")):
            a, b = arm[pair[0]], arm[pair[1]]
            if np.any(a != b):
                w = stats.wilcoxon(a, b)
                sig = "SIG" if w.pvalue < ALPHA / N_COMPARISONS else "ns"
                results[scope][f"{pair[0]}_vs_{pair[1]}"] = {
                    "mean_diff": float((a - b).mean()), "wilcoxon_p": float(w.pvalue)}
                print(f"    {pair[0]} vs {pair[1]}: diff {float((a - b).mean()):+.4f}, "
                      f"p={w.pvalue:.4g} [{sig}]")
            else:
                print(f"    {pair[0]} vs {pair[1]}: identical (gate never fired here)")

    with open(output_dir / "active_perception_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {csv_path}")
    print(f"wrote {output_dir / 'active_perception_results.json'}")


if __name__ == "__main__":
    main()

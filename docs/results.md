# Results

Append-only. Every number reported anywhere else must appear here first, with the run id,
dataset version and n that produced it. Rates carry Wilson 95% intervals — see
[imitation.md §5.3](imitation.md) for the statistical rules.

Audit trail for any row: `run.json` (git commit, dataset hash, config, checkpoint hash) in
the run directory, plus the dataset `manifest.json`.

---

## Expert benchmark

| date | task | seeds | n | success | mean steps | mean fold score | notes |
|---|---|---|---|---|---|---|---|
| pre-2026-09-23 | half fold | **0-49 (training seeds)** | 50 | 48/50 = 96.0% [86.5, 98.9] | 96.7 | 0.904 | mujoco 3.10.0. Both failures (seeds 24, 35) truncated at the 250-step cap with fold score ≈0.76 — reached the goal region but never settled. Mean 0.2 retries/episode. |

**This row is not a ceiling.** Seeds 0-49 are training seeds (`TRAIN_SEED_BASE = 0`), so it
cannot be compared against any student number. The expert ceiling on `id_easy` / `id_hard` /
`recovery` is roadmap M1.7 and is still unmeasured — in particular **the scripted teacher's
own recovery rate is unknown**, which gates the whole recovery arm of the plan.

| 2026-09-23 | half fold | 0-49 (training seeds) | 50 | 48/50 = 96.0% [86.5, 98.9] | 96.7 | 0.904 | After M1.1 (139-D obs, per-stage goals). Identical to the row above — same failing seeds 24, 35 — as expected: the expert reads sim state, not the observation. |

| 2026-09-23 | half fold | 0-49 (training seeds) | 50 | 48/50 = 96.0% [86.5, 98.9] | 96.7 | — | After M1.5 (IK scratch cache + per-episode IK rng reseed). Per-episode rows identical to M1.1. |

Artifacts: `outputs/imitation/expert_benchmark.json`, `outputs/imitation/expert_benchmark_m1_1.json`,
`outputs/imitation/expert_benchmark_m1_5.json`

### Expert ceiling — M1.7 gate (2026-09-23)

Scripted expert (`evaluate --ckpt expert`), mujoco 3.10.0, commit after M1.6, eval seed
bases from `imitation/seeds.py`. Wilson 95%. **This is the ceiling students are compared to.**

| set | seeds | n | success | fold score | failures |
|---|---|---|---|---|---|
| id_easy | 100000-100099 | 100 | 100/100 = 100% [96.3, 100.0] | 0.911 | — |
| id_hard | 200000-200099 | 100 | 97/100 = 97% [91.5, 99.0] | 0.902 | S1 ×2, F1 ×1 |
| recovery | 300000-300099 | 100 | 96/100 = 96% [90.2, 98.4] | 0.898 | R1 ×4 |

`check_resync` (seeds 0-99, handover at a random step, then `resync`): resync_only 92/100
[85.0, 95.9], noisy 94/100 [87.5, 97.2], random 93/100 [86.3, 96.6]; every failure is a
step-cap truncation, most with an arm stuck in `lift`.

**Gate verdict: pass.** Teacher recovery (96%) is within the ID intervals, so the recovery
arm stands as designed. Artifacts: `outputs/imitation/gate_m1_7/`.

### Collection throughput (M1.5)

20-episode expert collect, seeds 0-19, `--workers 10 --mixed`, recovery fraction 0.3, two
runs each. Both configurations produce the identical dataset hash `ac2fb3892c0f`.

| config | wall time | notes |
|---|---|---|
| fresh `MjData` per IK call | 83 s, 81 s | after the rng-reseed fix |
| cached scratch `MjData` | 82 s, 93 s | **no measurable speedup** — allocation is not the bottleneck; the IK iterations are |

Determinism: before the reseed fix, two identical collects gave different hashes (seeds 12,
13, 15 differed in length) because each worker's `FoldExpert.rng` carried over between
episodes, so a seed's demo depended on which worker ran it before.

## Imitation baselines

_Nothing measured yet — no dataset has been frozen._

## Ablations

_Pending. First planned: privileged-features ablation (proprio-only / +corners / full 139),
roadmap M2.3._

## DAgger rounds

_Pending._

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

Artifact: `outputs/imitation/expert_benchmark.json`

## Imitation baselines

_Nothing measured yet — no dataset has been frozen._

## Ablations

_Pending. First planned: privileged-features ablation (proprio-only / +corners / full 141),
roadmap M2.3._

## DAgger rounds

_Pending._

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

### v1 collection outcome — M1.8 (2026-09-23)

Expert, seeds 0-399, recovery fraction 0.3, 14 workers, 1250 s. Successes → `v1`
(`769dd372719c`), failures → `v1_failures` (`699a1dc71c71`).

| subset | n | success |
|---|---|---|
| clean | 279 | 271 = 97.1% [94.4, 98.5] |
| perturbed (recovery demos) | 121 | 113 = 93.4% [87.5, 96.6] |
| all | 400 | 384 = 96.0% [93.6, 97.5] |

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

### M2.1 — chunk-MLP BC on v1 (2026-09-24)

Dataset `v1` (`769dd372719c`, 384 eps; hashed split → 340 train / 44 val), chunk_mlp
8.65M params, K=16, replan 8, 30k steps, batch 1024, EMA 0.999. Eval n=200 per set, held-out
seeds (`imitation/seeds.py`), Wilson 95%. Failure codes are mechanistic (M2.2); on
`recovery` every failure is also a perturbed failure. Code at commit `33067c6`.

| run | set | n | success | fold score | failure codes |
|---|---|---|---|---|---|
| bc_v1_s0 | id_easy | 200 | 195/200 = 97.5% [94.3, 98.9] | 0.907 | F1 4, S1 1 |
| bc_v1_s0 | id_hard | 200 | 145/200 = 72.5% [65.9, 78.2] | 0.806 | G1 30, F1 17, S1 8 |
| bc_v1_s0 | recovery | 200 | 74/200 = 37.0% [30.6, 43.9] | 0.735 | G1 69, F1 53, S1 4 |
| bc_v1_s1 | id_easy | 200 | 193/200 = 96.5% [93.0, 98.3] | 0.907 | F1 6, G1 1 |
| bc_v1_s1 | id_hard | 200 | 137/200 = 68.5% [61.8, 74.5] | 0.770 | G1 44, F1 16, S1 3 |
| bc_v1_s1 | recovery | 200 | 61/200 = 30.5% [24.5, 37.2] | 0.710 | G1 79, F1 58, S1 2 |

Reading: BC matches the expert in distribution (97% vs 100%) but falls off under shift:
id_hard −25 pp and recovery −60 pp against the expert ceiling (97% / 96%). Recovery
failures are mostly missed or dropped grasps (G1) and misplaced corners (F1). This
compounding error is the gap DAgger (M3.2) targets. Seeds agree within their intervals.
Artifacts: `outputs/imitation/runs/bc_v1_s{0,1}/` (`run.json`, `eval_r8.json`).

## Ablations

### M2.3 — privileged-features ablation (2026-09-24)

Same protocol as M2.1 (v1, chunk_mlp, 30k steps, seed 0, n=200 per set). Hidden dims are
zeroed after normalization (`--obs-subset`, `imitation/spec.py::OBS_SUBSETS`).
`proprio` = 48 sensor-available proprio dims (no weld grasp flags). `proprio_corners` =
proprio + 4 corner positions + goal keypoints + stage/settle (74 dims).

| obs | id_easy | id_hard | recovery |
|---|---|---|---|
| full (139) | 195/200 = 97.5% [94.3, 98.9] | 145/200 = 72.5% [65.9, 78.2] | 74/200 = 37.0% [30.6, 43.9] |
| proprio_corners (74) | 192/200 = 96.0% [92.3, 98.0] | 90/200 = 45.0% [38.3, 51.9] | 54/200 = 27.0% [21.3, 33.5] |
| proprio (48) | 184/200 = 92.0% [87.4, 95.0] | 97/200 = 48.5% [41.7, 55.4] | 52/200 = 26.0% [20.4, 32.5] |

Reading:
- **In distribution, cloth state barely matters** (92% blind). From nominal starts the fold
  can be replayed almost open-loop from proprioception.
- **Under shift it's worth ~25 pp on id_hard**, but corner positions + goals alone buy
  nothing over proprio-only (45.0 vs 48.5, overlapping intervals). The gain comes from
  the remaining cloth dims: corner-to-goal vectors, corner velocities, sampled vertices,
  z statistics, progress. For Phase 4 this means a vision student must recover *relative*
  cloth geometry (corner-to-goal, shape), not just corner keypoints.
- The extra G1 (grasp) failures on id_hard without cloth state (63-69 vs 30) show the
  shifted corner is where the grasp misses.
- Single seed per arm. The seed-to-seed spread on full is ~4 pp (M2.1), below every gap
  called out here.

## DAgger rounds

### dagger_v1 — INVALID (teacher label bug), kept for the record (2026-09-24)

From `bc_v1_s0`, 128 rollouts/round, β 0.3 → 0.15, n=200. Versions `v1_dagger_v1_r1`,
`v1_dagger_v1_r2` (frozen, write-once) carry the bad labels and must not be trained on.

| round | id_easy | id_hard | recovery | kept |
|---|---|---|---|---|
| 0 (bc_v1_s0 re-eval) | 195/200 | 134/200 | 77/200 | — |
| 1 | 176/200 | 86/200 | 82/200 | no (−1.3 SE) |

Round 1 *regressed* with S1 stalls up from 1-17 to 59-66 per set. Cause: `label_chunk`
re-inferred every arm's phase from geometry and re-solved IK from scratch at each label,
so labels disagreed with the expert **on the expert's own trajectory**: 0.12-0.57 mean
abs joint-action error during carry/place and 0.5-0.75 on the gripper. The student was
trained on two conflicting teachers and averaged them. Fixed by the shadowing teacher
(`ScriptedTeacher.observe`, imitation.md §4), which gives exact agreement
(`test_a_shadowing_teacher_labels_like_the_one_driving`, max err < 0.02).

Also found: the round-0 re-eval of the *same* checkpoint gave id_hard 134/200 vs 145/200
in M2.1. GPU outputs differ ~1e-6 with batch size, the cloth sim amplifies that, and
batch composition depends on worker timing. Fixed by padding every policy batch to a
fixed size (`rollout.padded_predict`); evaluations after this commit are deterministic
per seed. **M2.1-M2.3 numbers above predate the fix**: they're valid samples, but a
re-run will not reproduce them exactly.

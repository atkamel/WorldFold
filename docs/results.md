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

### M2.4 — diffusion head vs chunk-MLP, same protocol (2026-09-24)

`diffusion` (7.35M params, 100 train / 10 inference denoising steps) on v1, seed 0, 30k
steps, K=16 / replan 8, n=200, deterministic (padded) eval. The chunk-MLP row is
`bc_v1_s0` re-evaluated under the same padded eval (dagger_v2 round 0).

| policy | id_easy | id_hard | recovery | infer ms/call (batch 16) |
|---|---|---|---|---|
| chunk_mlp | 196/200 = 98.0% [95.0, 99.2] | 139/200 = 69.5% [62.8, 75.5] | 77/200 = 38.5% [32.0, 45.4] | ~1 |
| diffusion | 199/200 = 99.5% [97.2, 99.9] | 160/200 = 80.0% [73.9, 85.0] | 102/200 = 51.0% [44.1, 57.8] | 100 |

Diffusion failure codes: id_easy F1 1; id_hard G1 17, S1 12, F1 11; recovery G1 57, F1 38, S1 3.

Reading: **diffusion wins under shift**, +10.5 pp on id_hard and +12.5 pp on recovery.
The intervals touch on id_hard and are disjoint on recovery. It's a single seed per arm
(chunk-MLP seed spread ~4 pp, M2.1). **Kept** per the M2.4 rule. Cost: 100 ms per
batched call against a 400 ms replan interval (8 steps × 50 ms), which fits the loop,
but that's ~100× the MLP (§8 deployment budget). Next: DAgger from the diffusion
checkpoint, which should combine both gains.

## Offline RL

### M5.1 — student rollout harvest (2026-09-24)

`dagger_v2/round_3` on harvest seeds 400000-400999, 30% knocked off course, Gaussian
chunk noise σ=0.1, replan 8. Frozen as `harvest_v1` (hash `3b5d331bf404`), 2144 s.

| outcome | n |
|---|---|
| success | 811 (81.1% [78.6, 83.4]) |
| S1 stalled | 110 |
| F1 misplaced | 60 |
| G1 grasp | 19 |

Reward variance exists (19% failures across three mechanisms), so the exit is met.

### M5.3 — IQL warm-started from dagger_v2/round_3 — **exit NOT met** (2026-09-24)

Data: `v1_dagger_v2_r3` (v1 + 3 DAgger rounds) + `v1_failures` + `harvest_v1` → 1784
episodes, 28,405 macro transitions (8-step chunks), 1548 success terminals. Reward per
imitation.md §7.1. 40k steps, τ=0.7, twin Q, batch 1024, policy lr 1e-4. n=200.

| policy | id_easy | id_hard | recovery |
|---|---|---|---|
| dagger_v2/round_3 (start) | 194/200 = 97.0% [93.6, 98.6] | 129/200 = 64.5% [57.7, 70.8] | 130/200 = 65.0% [58.2, 71.3] |
| iql_v1 (β=3, raw A) | 187/200 = 93.5% [89.2, 96.2] | 122/200 = 61.0% [54.1, 67.5] | 98/200 = 49.0% [42.2, 55.9] |
| iql_v2 (standardized A, T=1, clip 20) | 188/200 = 94.0% [89.8, 96.5] | 135/200 = 67.5% [60.7, 73.6] | 91/200 = 45.5% [38.7, 52.4] |

Both variants are **worse on recovery** (−16 to −20 pp, intervals disjoint), with S1
stalls up from 47 to 67-76. Diagnosis (critic probed on harvest data):
- The critic ranks *states* correctly: V = 0.94 on successful trajectories, 0.20 on failed.
- It can't rank *actions*: A = Q − V has mean −0.010 and std 0.017 on **both** successful
  and failed trajectories. In a given state the logged actions are near-identical (one
  student, σ=0.1 noise), so there's no action contrast to learn from.
- So advantage weighting is ~uniform, and the policy step reduces to BC over all data.
  Failed episodes are 19% of episodes but ~3× more transitions per episode (stalls run to
  the 250-step cap), so the student imitates stalling. Standardizing A only sharpens noise.

What would change the outcome: action diversity in the harvest (much larger noise, or
several policies), per-episode subsampling so stalls don't dominate, or online
fine-tuning. Parked: DAgger remains the best privileged policy.

## Sensor-only student (Phase 4)

### Vision BC on v1_img (2026-09-24)

`vision` policy (8.61M params): main cam 96² + both wrist cams 64², a CNN per camera,
plus the 48 sensor-available proprio dims (all privileged dims masked; the checkpoint's
`obs_mask` was checked). Expert labels, 30k steps, batch 256, random-shift augmentation.
`v1_img` = v1 replayed with per-episode visual DR. n=200.

| policy | id_easy | id_hard | recovery |
|---|---|---|---|
| state BC (bc_v1_s0, full 139-D) | 196/200 = 98.0% [95.0, 99.2] | 139/200 = 69.5% [62.8, 75.5] | 77/200 = 38.5% [32.0, 45.4] |
| **vision BC** | 194/200 = 97.0% [93.6, 98.6] | **189/200 = 94.5% [90.4, 96.9]** | 43/200 = 21.5% [16.4, 27.7] |

Reading: from cameras, the student **generalizes to shifted cloth far better** than
from the privileged state (+25 pp on id_hard, disjoint intervals). A CNN over an
overhead view is roughly equivariant to where the cloth sits; the state MLP has to learn
that from absolute coordinates. It recovers much worse (−17 pp): BC has no recovery
signal and the knocked-off states look unlike any demo frame. That's the job of
distillation (M4.2), below. Inference: 7.5 ms per batch-16 call.

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

### M3.2 — dagger_v2 (shadowing teacher, deterministic eval) (2026-09-24)

From `bc_v1_s0`; 128 student rollouts/round on DAgger seeds (50k + 1000 r), 30% knocked
off course, β 0.3 → 0.15 → 0.075, labels at replan points, ×4 oversampled, 15k
warm-start steps/round. Eval n=200/set with padded (deterministic) inference. Keep if
score (id_easy + recovery) rises; stop when a round gains < 1 SE.

| round | trained on | id_easy | id_hard | recovery | gain (SE) | kept |
|---|---|---|---|---|---|---|
| 0 | v1 | 196/200 = 98.0% [95.0, 99.2] | 139/200 = 69.5% [62.8, 75.5] | 77/200 = 38.5% [32.0, 45.4] | — | yes |
| 1 | v1_dagger_v2_r1 | 193/200 = 96.5% [93.0, 98.3] | 137/200 = 68.5% [61.8, 74.5] | 112/200 = 56.0% [49.1, 62.7] | +3.1 | yes |
| 2 | v1_dagger_v2_r2 | 198/200 = 99.0% [96.4, 99.7] | 122/200 = 61.0% [54.1, 67.5] | 123/200 = 61.5% [54.6, 68.0] | +1.6 | yes |
| 3 | v1_dagger_v2_r3 | 194/200 = 97.0% [93.6, 98.6] | 129/200 = 64.5% [57.7, 70.8] | 130/200 = 65.0% [58.2, 71.3] | +0.3 | yes |

Final failure codes: {"id_easy": {"S1": 3, "F1": 2, "G1": 1}, "id_hard": {"G1": 22, "S1": 31, "F1": 18}, "recovery": {"S1": 47, "G1": 10, "F1": 13}}.

Reading:
- **Recovery 38.5% → 65.0%** (+26.5 pp, intervals disjoint) with in-distribution held at
  97-99%. **Exit met** (≥80% ID, ≥60% recovery).
- **id_hard is flat to slightly down** (69.5 → 64.5, overlapping intervals). DAgger
  rollouts only use nominal starts, so the shifted-pose states id_hard tests are never
  labelled. The next lever is DAgger rollouts from id_hard-like poses, on a *disjoint*
  seed range.
- The remaining recovery failures are mostly S1 stalls (47): the student reaches a held
  or placed state and never finishes. That's the target for offline RL (M5.3).
- Best: `outputs/imitation/runs/dagger_v2/round_3/final.pt`.

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

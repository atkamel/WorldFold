# Results

> Summary with demo videos, dated 2026-09-24: [reports/2026-09-24-phase1-5.md](reports/2026-09-24-phase1-5.md).

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

## Determinism note (2026-09-24, Phase 5b)

Diffusion inference drew its initial DDIM noise from the global torch RNG, so a row's
action depended on its batch position and call order, i.e. on worker timing. The same
`diff_v1_s0` checkpoint scored id_hard 160/200 (M2.4) and 157/200 (a later run), even with
padded batches. Sampling now uses one fixed-seed noise shared by all rows, so it's a
deterministic function of the observation. **The M2.4 diffusion numbers predate this**;
M5b.1 reports the checkpoint re-evaluated under the fixed sampler.

### M5b.5 — M2.1 / M2.3 checkpoints re-evaluated, deterministic (2026-09-25)

The same checkpoints, re-evaluated at n=200 under the current deterministic eval (padded
batches, fixed seeds, replan 8, 10 workers). Nothing was retrained. The original rows (M2.1,
M2.3) stay as they were; these are the numbers to quote from now on.

| run | id_easy | id_hard | recovery | original (id_easy / id_hard / recovery) |
|---|---|---|---|---|
| bc_v1_s0 | 196/200 = 98.0% [95.0, 99.2] | 139/200 = 69.5% [62.8, 75.5] | 77/200 = 38.5% [32.0, 45.4] | 97.5 / 72.5 / 37.0 |
| bc_v1_s1 | 193/200 = 96.5% [93.0, 98.3] | 138/200 = 69.0% [62.3, 75.0] | 66/200 = 33.0% [26.9, 39.8] | 96.5 / 68.5 / 30.5 |
| abl_proprio_s0 (48) | 186/200 = 93.0% [88.6, 95.8] | 99/200 = 49.5% [42.6, 56.4] | 50/200 = 25.0% [19.5, 31.4] | 92.0 / 48.5 / 26.0 |
| abl_proprio_corners_s0 (74) | 191/200 = 95.5% [91.7, 97.6] | 93/200 = 46.5% [39.7, 53.4] | 44/200 = 22.0% [16.8, 28.2] | 96.0 / 45.0 / 27.0 |

Failure codes: bc_v1_s0 recovery G1 71, F1 50, S1 2; abl_proprio_s0 id_hard G1 63, F1 38;
abl_proprio_corners_s0 id_hard G1 69, F1 23, S1 15.

**Determinism check:** `bc_v1_s0` evaluated a second time (`eval_det_repeat.json`) gives
identical per-set counts and failure codes (`imitation.viz.pick same`: identical). The
same checkpoint also matched its dagger_v2 round-0 evaluation (196 / 139 / 77) from the day
before. Every original row lies inside its re-evaluation's interval, so the M2.1 and M2.3
readings stand: seeds agree, and cloth state is worth ~20-25 pp on id_hard.

## Throughput (Phase 5c)

### M5c.4 — GPU physics feasibility (MuJoCo Warp) — **closed: not viable for this model** (2026-09-25)

Separate env `.venv-warp` (mujoco 3.10.0 + mujoco-warp 3.10.0 + warp-lang 1.14.0; 1.17
fails to compile mujoco-warp's sensor kernels). The pinned `.venv` is untouched. The exact
compiled half-fold model (domain randomization applied, seed 100000) and one expert
episode's physics inputs (ctrl, weld switches, weld offsets per control step) were exported
from the pinned CPU sim (`imitation/gpu_sim/export_episode.py`) and replayed
(`imitation/gpu_sim/warp_check.py`, result `outputs/imitation/gpu_sim/ep_100000/warp_check.json`).

| check | result |
|---|---|
| load the model (1 flex, 121 cloth vertices, 375 dof, 6 weld equalities) | ✅ `put_model` accepts it |
| replay procedure sanity (CPU, same inputs) | ✅ max \|Δqpos\| = 0.0 |
| trajectory match, GPU vs CPU, same inputs | ❌ 0.6 mm after step 1, 2.8 mm at step 10, **> 1 cm from step 15**, max 9.7 cm, final 4.6 cm (172 steps) |
| throughput, CPU 1 thread | 576.5 physics steps/s |
| throughput, GPU, nworld 1 / 16 / 64 / 256 | 21 / 164 / **274** / 198 world-steps/s (47 / 97 / 234 / 1,293 ms per batched step) |

Verdict: **not viable.** At best (nworld=64) the GPU is ~2× *slower than one CPU core* and
~20× slower than the 10-worker CPU pool. Past 64 worlds it degrades: the per-world
constraint solve for 375 dof plus flex contacts is dense (blocked Cholesky), which is what
batches poorly. It also doesn't reproduce the CPU trajectory (float32 + a different
solver path, amplified by the cloth), so every result would need a new baseline even if it
were fast. No cheap follow-up: revisit only with a future mujoco-warp release, and then
against a fresh expert-ceiling baseline. **M5c.5** (batched GPU rendering) is gated on this
and closes with it.

### M5c.3 — GPU busy across a DAgger run with overlapping lanes — **exit not met** (2026-09-25)

`nvidia-smi utilization.gpu` (the share of each sample period in which a kernel ran),
sampled every 5 s over one DAgger round at replan 4 (`dagger_r4`, above). The window came in
two parts because the queue was paused after the rollouts froze.
`imitation.viz.gpu_busy`, files `runs/m5c3_gpu*.csv` / `m5c3_window*.csv`.

| part | what ran | window | samples | mean GPU util | samples > 10% |
|---|---|---|---|---|---|
| 1 | lane A DAgger rollouts; lane B (`diff_v1_s1` eval, CPU) and lane C (`vision_t0_128_s1` training, GPU) overlapping under the CPU slot | 57.0 min | 648 | 32.3% | 63.9% |
| 2 | lane A alone: round-1 training (GPU) + 600 eval episodes | 50.0 min | 561 | 21.7% | 66.7% |
| **whole run** | | 107 min | 1,209 | **27.4%** | 65.2% |

**Exit (> 50% busy) not met.** The GPU does work in two-thirds of the samples but is kernel-active
only ~27% of the time. The loop is CPU-physics-bound: each control step is 100 cloth substeps,
~200 ms of one core (M5c.1). The overlap mechanism (`IMITATION_CPU_SLOT`) works: it kept the
sim at one 10-worker pool across three concurrent queues. Part 1 shows the extra lanes lifting
GPU use (32 vs 22%), but only while there is independent GPU work to run, and the pipeline has
little of it. Filling the GPU needs the physics on it, which M5c.4 ruled out. Known defect: the
slot isn't fair. A lane that releases and re-takes it between evaluation sets beats a lane
polling every 2 s, so lane A waited 37 min behind lane B's whole evaluation.

### M5c.2 — diffusion sampling as one CUDA graph (2026-09-25)

The 10-step DDIM sampler is ~60 tiny kernels; its latency was launch overhead, not
arithmetic. `DiffusionPolicy.predict` now replays the whole sampler as one captured CUDA
graph per batch shape. The timestep schedule became plain ints (indexing with GPU scalars
forced a host sync and blocked capture). Measured on `dagger_diff/round_1`, batch 16,
while the GPU was also training another model:

| path | ms / batched call | actions |
|---|---|---|
| eager (before) | 301.3 | reference |
| CUDA graph | 23.7 | **bit-identical** (`np.array_equal`) |

12.7× lower inference latency. The new eager sampler is also bit-identical to the
previous implementation, so earlier diffusion results stay reproducible. Test:
`test_cuda_graph_diffusion_sampling_matches_eager_exactly`. The rollout-time share of
inference is measured by M5c.1.

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

### Diffusion BC seed variance (seeds 0-2) (2026-09-25)

Same recipe as M2.4 (`diffusion`, v1, 30k steps, batch 1024, replan 8), seeds 1 and 2 trained
on the lazy loader (2,152 / 2,171 s vs 3,263 s for seed 0). n=200, deterministic eval. Seed 0
is its deterministic re-evaluation (M5b.1 round 0).

| seed | id_easy | id_hard | recovery |
|---|---|---|---|
| 0 (`diff_v1_s0`) | 200/200 = 100.0% [98.1, 100.0] | 154/200 = 77.0% [70.7, 82.3] | 111/200 = 55.5% [48.6, 62.2] |
| 1 | 199/200 = 99.5% [97.2, 99.9] | 166/200 = 83.0% [77.2, 87.6] | 124/200 = 62.0% [55.1, 68.4] |
| 2 | 200/200 = 100.0% [98.1, 100.0] | **170/200 = 85.0% [79.4, 89.3]** | 121/200 = 60.5% [53.6, 67.0] |

Failure codes: seed 1 id_hard F1 20, S1 10, G1 4, recovery G1 48, F1 21, S1 7; seed 2 id_hard
F1 16, G1 9, S1 5, recovery G1 43, F1 31, S1 5.

Reading:
- **Seed 0 is the weakest of three**, and every Phase 5b experiment started from it. The
  shifted-set spread across seeds is 8 pp (77-85), about the size of the effects that
  M5b.2 and M5b.6 were chasing. The chunk-MLP spread was ~4 pp (M2.1).
- Seeds 1-2 have **fewer S1 stalls** on id_hard (10 and 5, vs 13 for seed 0 and 33 for
  `dagger_diff/round_1`, all at replan 8). DAgger from seed 0 *added* S1 stalls while it lifted recovery. The
  fine-placement stall is partly a property of the training run, not only of the data.
- Plain BC seed 2 reaches the M5b.2 target (id_hard ≥ 85%) without any shifted-pose data.
  But it trails the DAgger policy on recovery (60.5 vs 72.5-79.5%).
- Consequence for every later comparison: single-seed deltas under ~8 pp on id_hard are not
  evidence. Phase 6+ milestones should be judged on ≥ 2 seeds.

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

### M4.2 — privileged → sensor-only distillation (2026-09-24)

Teacher: `dagger_v2/round_3` (privileged, 97.0 / 64.5 / 65.0%). Student: `vision`
policy. Round 0 trains on `v1_img` with **every step relabelled by the teacher**
(`train --teacher`); rounds 1+ roll the vision student out with cameras rendering on
distill seeds (70k + 1000 r), β 0.3 → 0.15, then retrain on all visited steps relabelled
(student-visited steps ×2). n=200, deterministic eval.

| round | trained on | id_easy | id_hard | recovery | gain (SE) | kept |
|---|---|---|---|---|---|---|
| 0 | v1_img | 194/200 = 97.0% [93.6, 98.6] | 192/200 = 96.0% [92.3, 98.0] | 59/200 = 29.5% [23.6, 36.2] | — | yes |
| 1 | v1_img_distill_v1_r1 | 195/200 = 97.5% [94.3, 98.9] | 187/200 = 93.5% [89.2, 96.2] | 50/200 = 25.0% [19.5, 31.4] | -0.8 | no |
| 2 | v1_img_distill_v1_r2 | 195/200 = 97.5% [94.3, 98.9] | 181/200 = 90.5% [85.6, 93.8] | 63/200 = 31.5% [25.5, 38.2] | +0.5 | yes |

**Exit, with the margin stated:** against its privileged teacher, the sensor-only
student is **within 1 pp in distribution** (97.5 vs 97.0), **26 pp better on shifted
cloth** (90.5 vs 64.5), and **33.5 pp worse on recovery** (31.5 vs 65.0, disjoint
intervals). ID/shift is met; recovery isn't. Distillation lifted recovery only from
vision-BC 21.5% to 29.5-31.5%. The remaining failures are mostly G1 (81 of 137): after a
knock the student misses the re-grasp. The likely causes are 96 px / 64 px resolution
for locating a displaced corner, and only 30% of distill rollouts being perturbed.
Round 0 (97.0 / 96.0 / 29.5) is arguably the better all-rounder. The keep rule scores
id_easy + recovery only, so it kept round 2.

### M4.3 — vision success detector (2026-09-24)

CNN on the `main` frame predicting `folded` (all corners within 5 cm of goal and
grippers open) + fold score. Trained on `v1_img` + `v1_failures_img` (37,468 train /
4,668 val frames, hashed seed split), 8k steps, class-balanced.

- Val frames: **accuracy 94.9%**, fold-score MAE 0.022.
- **Exit — agreement with the sim's success flag** on the final frame of held-out
  student rollouts (the distill rounds' own episodes, never trained on): **235/256 = 91.8% [87.8, 94.6]**.
  The "always success" baseline is 208/256 = 81.2%. Errors: 14 false "folded", 7 missed.

### M4.3 re-run — detector on 128² frames with per-camera normalization (success_v2) (2026-09-25)

Same recipe on the finished Phase 4 path: `v1_img128` + `v1_failures_img128` (the failures
re-rendered at 128² so the two versions don't mix frame sizes; images in both digests),
per-channel normalization fit on the training frames, 8k steps. 4,668 val frames.

- Val frames: **accuracy 95.0%**, fold-score MAE 0.019 (v1: 94.9%, 0.022).
- Agreement with the sim's success flag on the final frame of held-out `distill_v2` round 1-2
  rollouts (never trained on): **237/256 = 92.6% [88.7, 95.2]**. "Always success" baseline:
  170/256 = 66.4%, a harder set than v1's (81.2%), so the margin over the baseline is wider
  (+26 pp vs +11 pp). Errors: 16 false "folded", 3 missed.
  Artifacts: `outputs/imitation/runs/success_v2/`, `success_v2.agree.log`.

### M5b.6 — replan-interval sweep (fine placement / re-grasp) (2026-09-25)

Same checkpoints, evaluated with the chunk re-planned every 8 (default), 4 or 2 steps.
n=200, deterministic. Privileged = `dagger_diff/round_1` (diffusion); vision =
`vision_t0_128` (best camera student, M5b.3 round 0).

| policy | replan | id_easy | id_hard | recovery |
|---|---|---|---|---|
| privileged | 8 | 196/200 = 98.0% [95.0, 99.2] | 144/200 = 72.0% [65.4, 77.8] | 145/200 = 72.5% [65.9, 78.2] |
| privileged | **4** | 199/200 = 99.5% [97.2, 99.9] | **154/200 = 77.0% [70.7, 82.3]** | **159/200 = 79.5% [73.4, 84.5]** |
| privileged | 2 | 198/200 = 99.0% [96.4, 99.7] | 149/200 = 74.5% [68.0, 80.0] | 153/200 = 76.5% [70.2, 81.8] |
| vision | 8 | 190/200 = 95.0% [91.0, 97.3] | 184/200 = 92.0% [87.4, 95.0] | 60/200 = 30.0% [24.1, 36.7] |
| vision | 4 | 192/200 = 96.0% [92.3, 98.0] | 186/200 = 93.0% [88.6, 95.8] | 73/200 = 36.5% [30.1, 43.4] |
| vision | **2** | 188/200 = 94.0% [89.8, 96.5] | 182/200 = 91.0% [86.2, 94.2] | **82/200 = 41.0% [34.4, 47.9]** |

Failure codes, privileged @4: id_hard S1 32, F1 11, G1 3; recovery F1 15, S1 14, G1 11, M1 1.
Vision @2: recovery G1 72, F1 43, S1 3.

Reading:
- **Replanning more often is a real, free lever**: no retraining. The privileged policy gains
  +5 pp id_hard and +7 pp recovery at replan 4 (the recovery gain is outside the intervals'
  overlap only marginally; id_hard intervals overlap). 2 is past the optimum: chunks
  re-planned every 2 steps lose the temporal consistency action chunking is for.
- The vision student's **recovery rises monotonically** with tighter replanning (30 → 36.5 →
  41%) at no cost elsewhere: re-grasping after a knock is closed-loop work.
- **Exit not met**: privileged id_hard peaks at 77% (target 85%). The remaining id_hard
  failures are still S1 fine-placement stalls (32 of 46). Replanning narrows the gap but
  doesn't close it.
- **Adopted as operating points**: privileged replan 4 (99.5 / 77.0 / 79.5), vision replan 2
  (94.0 / 91.0 / 41.0). The vision-vs-privileged recovery gap at these settings is 38.5 pp.

### Vision student seed variance at the operating point (replan 2) (2026-09-25)

`vision_t0_128` recipe (128², per-camera norm, teacher-relabelled `v1_img128`, 30k steps,
batch 256) with seed 1 (1,290 s of training, vs 1,812 s for seed 0 under concurrent load).
n=200, deterministic, replan 2.

| seed | id_easy | id_hard | recovery |
|---|---|---|---|
| 0 (`vision_t0_128`) | 188/200 = 94.0% [89.8, 96.5] | 182/200 = 91.0% [86.2, 94.2] | 82/200 = 41.0% [34.4, 47.9] |
| 1 | 190/200 = 95.0% [91.0, 97.3] | 177/200 = 88.5% [83.3, 92.2] | 69/200 = 34.5% [28.3, 41.3] |

Seed 1 failure codes: recovery G1 78, F1 45, S1 8; id_hard F1 9, G1 8, S1 6.

The shifted-set result reproduces (88.5-91%). Recovery spreads 6.5 pp between seeds, with
overlapping intervals. The camera student's recovery at its operating point is **35-41%**,
and seed 0's 41% is the upper end. Recovery failures stay mostly G1: the re-grasp after a
knock is still the missing skill.

### M5b.4 — offline RL retry with action diversity (2026-09-25)

**Harvest `harvest_v2`** (`febfb9275058`): 1,679 episodes from 4 policies (BC chunk-MLP,
diffusion BC, `dagger_v2/round_3`, `dagger_diff/round_1`) × σ ∈ {0.1, 0.3}, 30% knocked,
stratified to ≥ 60 per failure code. 1,141 success, 271 G1, 145 F1, 122 S1.

**Critic probe** (`imitation.rl.probe`, iql_v3 critics on harvest_v2) vs harvest_v1 (M5.3):

| | harvest_v1 (1 policy, σ 0.1) | harvest_v2 (4 policies × 2 σ) |
|---|---|---|
| V gap, success − failed | 0.72 | 0.51 |
| A spread (std, all transitions) | 0.018 | **0.054** (3×) |
| A gap, success − failed | −0.0005 | +0.003 |

The diversity bought action contrast: advantages spread 3× wider. Per outcome group:
A mean −0.019 to −0.025, std 0.047-0.066.

**iql_v3** (from `dagger_diff/round_1`: diffusion actor, advantage-weighted denoising loss;
standardized A, T=1, clip 20; `max_per_episode` 12; critic *and actor* sampled uniformly over
outcome codes). 27,231 transitions (success 20,451 / G1 3,264 / F1 1,812 / S1 1,704). 40k steps.

| policy | id_easy | id_hard | recovery |
|---|---|---|---|
| dagger_diff/round_1 (start) | 196/200 = 98.0% | 144/200 = 72.0% | 145/200 = 72.5% |
| **iql_v3** | 108/200 = 54.0% [47.1, 60.8] | 48/200 = 24.0% [18.6, 30.4] | 36/200 = 18.0% [13.3, 23.9] |

**Collapse; the cause was my sampling design.** Uniform stratification over the 4 outcome
codes present put **75% failed-episode transitions in every actor batch**, and advantage
weights near 1 can't undo that, so the actor imitated failures (G1 101-120 per set). The
stratification belongs to the critic (it must see failures to value them), not the actor.
Fix: `--actor-strata natural` samples the actor batch separately; advantages are computed on
the actor's own batch (**iql_v4**, below).

### M5b.3 — vision recovery via distillation (distill_v2) — **exit NOT met** (2026-09-25)

Student: `vision` at 128² (per-camera norm, lazy sampler). Teacher: `dagger_diff/round_1`
(98.0 / 72.0 / 72.5), a `PolicyTeacher` labelling on the GPU through the Phase 3 loop
(`imitation.dagger --teacher --relabel`). Round 0 = `v1_img128` with every step relabelled
by the teacher. Rounds: 128 rollouts, **60% knocked**, 30% from shifted poses, β 0.3 →
0.15, student-visited steps ×2. Keep rule on all three sets. n=200, deterministic.

| round | trained on | id_easy | id_hard | recovery | gain (SE, 3 sets) | kept |
|---|---|---|---|---|---|---|
| 0 | `v1_img128` | 190/200 = 95.0% [91.0, 97.3] | 184/200 = 92.0% [87.4, 95.0] | 60/200 = 30.0% [24.1, 36.7] | — | yes |
| 1 | `v1_img128_distill_v2_r1` | 195/200 = 97.5% [94.3, 98.9] | 147/200 = 73.5% [67.0, 79.1] | 70/200 = 35.0% [28.7, 41.8] | −1.8 | no |
| 2 | `v1_img128_distill_v2_r2` | 189/200 = 94.5% [90.4, 96.9] | 136/200 = 68.0% [61.2, 74.1] | 65/200 = 32.5% [26.4, 39.3] | −3.4 | no, stop |

Round 2 failure codes: id_hard S1 42, G1 19, F1 3; recovery G1 71, S1 35, F1 29.

**Margin vs the teacher (best = round 0):** id_easy −3 pp, id_hard **+20 pp**, recovery
**−42.5 pp**. Exit (within 15 pp on recovery) not met.

Finding: **on-policy distillation transferred the teacher's weakness, not its strength.**
Each round pulled the student's id_hard down toward the teacher's own 72% (92 → 73.5 → 68),
the S1 stalls the teacher has on shifted poses (M5b.2), while recovery only moved within
noise (30 → 35 → 32.5). More knocked rollouts (60%) and 128² didn't change that. Recovery
failures stay mostly G1: after a knock the camera student misses the re-grasp. Relabelling
by a teacher that's itself weak off-distribution is a poor fit for the one axis where the
student already beats it. Follow-up: the M5b.6 replan sweep also covers this student
(tighter closed-loop control is the cheap lever for re-grasping), and the best camera
student stays round 0.

### M4.1 closure — vision BC on the finished Phase 4 path (2026-09-25)

Same recipe as the first vision BC (expert labels, 30k steps, batch 256), now with every M4.1
scope item in place: 128² main camera + two 64² wrist cameras from `HalfFoldEnv(obs_mode="dict")`,
per-camera per-channel normalization fit on the training frames, lazy `WindowSampler`.
Dataset `v1_img128` (`532f8a9c08b6`, images in the digest). n=200, deterministic eval.

| policy | id_easy | id_hard | recovery | infer ms (batch 16) |
|---|---|---|---|---|
| vision BC, 96², fixed /255 (Phase 4) | 194/200 = 97.0% [93.6, 98.6] | 189/200 = 94.5% [90.4, 96.9] | 43/200 = 21.5% [16.4, 27.7] | 7.5 |
| **vision BC, 128², per-camera norm, lazy loader** | 189/200 = 94.5% [90.4, 96.9] | 182/200 = 91.0% [86.2, 94.2] | 53/200 = 26.5% [20.9, 33.0] | 9.6 |

No regression outside the intervals (id_easy and id_hard overlap, recovery +5 pp). Training
30k steps took 1,000 s. Failure codes (128²): id_easy F1 11; id_hard F1 13, G1 5; recovery
G1 106, F1 39, S1 2. Recovery is still the gap; M5b.3 targets it.

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

### DAgger at the adopted replan 4 (dagger_r4) — not kept (2026-09-25)

From `dagger_diff/round_1` on its chain `v1_dagger_diff_r1`, one round of 64 rollouts
**re-planned every 4 steps** (β 0.3, 30% knocked, shadowing expert labels at every replan:
1,725 labels) → `v1_dagger_diff_r1_dagger_r4_r1` (`5a3fcacb5fa2`, 576 eps). 15k warm-start
steps, evaluated at replan 4, n=200, deterministic. Round 0 is M5b.6's replan-4 evaluation
of the same checkpoint on the same seeds. The run was paused after the rollouts froze and
resumed with `--resume`, which reused the frozen round.

| round | trained on | id_easy | id_hard | recovery | gain (SE, 3 sets) | kept |
|---|---|---|---|---|---|---|
| 0 @4 | `v1_dagger_diff_r1` | 199/200 = 99.5% [97.2, 99.9] | 154/200 = 77.0% [70.7, 82.3] | 159/200 = 79.5% [73.4, 84.5] | — | yes |
| 1 @4 | `…_dagger_r4_r1` | 198/200 = 99.0% [96.4, 99.7] | 152/200 = 76.0% [69.6, 81.4] | 152/200 = 76.0% [69.6, 81.4] | −0.8 | no |

Failure codes, round 1: id_hard S1 32, F1 10, G1 6; recovery S1 20, F1 16, G1 12.

Reading: labelling at the finer replan interval adds no signal. id_hard failures are
still S1 stalls at the same count (32 vs 32 at round 0). This is the third attempt at the
fine-placement gap by more data of the same kind (M5b.2 shifted poses, M5b.6 replan,
this). **Best privileged policy stays `dagger_diff/round_1` at replan 4.**

### M5b.2 — shifted-pose coverage (dagger_shift) — **exit NOT met** (2026-09-24)

From `dagger_diff/round_1` on its aggregated chain `v1_dagger_diff_r2`. 128 rollouts per
round: 50% from shifted cloth poses (`seeds.shifted_pose`, seeds 500000 + 1000 r, the
id_hard distribution on disjoint seeds) and 30% knocked. The keep rule scores **all three
sets**. n=200, deterministic.

| round | trained on | id_easy | id_hard | recovery | gain (SE, 3 sets) | kept |
|---|---|---|---|---|---|---|
| 0 | `v1_dagger_diff_r2` | 196/200 = 98.0% [95.0, 99.2] | 144/200 = 72.0% [65.4, 77.8] | 145/200 = 72.5% [65.9, 78.2] | — | yes |
| 1 | `…_dagger_shift_r1` | 198/200 = 99.0% [96.4, 99.7] | 122/200 = 61.0% [54.1, 67.5] | 146/200 = 73.0% [66.5, 78.7] | −1.4 | no |
| 2 | `…_dagger_shift_r2` | 195/200 = 97.5% [94.3, 98.9] | 129/200 = 64.5% [57.7, 70.8] | 153/200 = 76.5% [70.2, 81.8] | −0.6 | no, stop |

Round 2 failure codes: id_hard S1 43, G1 23, F1 5; recovery S1 26, F1 12, G1 9.

The round-0 re-evaluation reproduced M5b.1 round 1 exactly (196 / 144 / 145, same failure
codes), which confirms deterministic evaluation end to end.

Diagnosis. Shifted rollouts fail mostly by **S1** (12 of 64 in round 1 vs 2 of 64
nominal). In those episodes the teacher's labels are *not* stationary or contradictory:
they keep driving the arms (mean |joint action| 0.09-0.44). The student moves about half as
much, reaches the goal region, and times out still holding with a corner 4-12 cm off
goal. The expert closes that last centimetre with `QuarterFoldExpert._maybe_retry`
(re-placing by the measured miss after a settle wait). A chunk policy re-planning every 8
steps under-imitates that slow, feedback-driven correction. More shifted rollouts of the
same kind didn't fix it in two rounds. **Best privileged policy stays
`dagger_diff/round_1`** (98.0 / 72.0 / 72.5).

### M5b.1 — DAgger from the diffusion checkpoint (dagger_diff) (2026-09-24)

From `diff_v1_s0` under the deterministic sampler; otherwise the dagger_v2 protocol
(128 rollouts/round, β 0.3 → 0.15, shadowing teacher, 15k warm-start steps, n=200). The
keep rule was still id_easy + recovery.

| round | trained on | id_easy | id_hard | recovery | gain (SE) | kept |
|---|---|---|---|---|---|---|
| 0 | `v1` | 200/200 = 100.0% [98.1, 100.0] | 154/200 = 77.0% [70.7, 82.3] | 111/200 = 55.5% [48.6, 62.2] | — | yes |
| 1 | `v1_dagger_diff_r1` | 196/200 = 98.0% [95.0, 99.2] | 144/200 = 72.0% [65.4, 77.8] | 145/200 = 72.5% [65.9, 78.2] | +3.1 | yes |
| 2 | `v1_dagger_diff_r2` | 199/200 = 99.5% [97.2, 99.9] | 106/200 = 53.0% [46.1, 59.8] | 145/200 = 72.5% [65.9, 78.2] | +0.3 | yes, stop |

Failure codes: round 1 — id_hard S1 33, G1 14, F1 9; recovery F1 19, G1 18, S1 18. Round 2 —
id_hard G1 44, S1 40, F1 9, M1 1.

**Exit met.** Round 1 beats `dagger_v2/round_3` (97.0 / 64.5 / 65.0) by **+1.74 SE** on
id_easy + recovery (+2.38 SE on all three sets). Recovery is +7.5 pp and id_hard +7.5 pp.
Diffusion + DAgger is the new best privileged policy.

**Keep-rule flaw, seen live:** the rule scored id_easy + recovery only, so it kept round 2,
which lost 19 pp on id_hard (72.0 → 53.0, disjoint intervals). DAgger rollouts use nominal
starts only, so each round erodes shifted-pose behaviour. **Round 1 is carried forward**
(the best round on all three sets, `imitation.viz.pick winner --carry-sets`). From M5b.2
on, the keep rule scores all three sets (`--score-sets`).

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

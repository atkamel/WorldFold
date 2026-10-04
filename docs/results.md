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

| 2026-10-02 | half fold | 0-49 (training seeds) | 50 | 48/50 = 96.0% [86.5, 98.9] | 96.7 | — | **I0.1, after merging origin/main (Isaac env, PR #15).** The merge moved constants to `cloth_params.py`, switched the wrappers to accessors, and fixed the `stages` shadowing and the hard-coded one-hot offset. Per-episode rows are identical to M1.5 (JSON equal), failing seeds 24 and 35. MuJoCo is unchanged. |

Artifacts: `outputs/imitation/expert_benchmark.json`, `outputs/imitation/expert_benchmark_m1_1.json`,
`outputs/imitation/expert_benchmark_m1_5.json`, `outputs/imitation/expert_benchmark_i0_1.json`

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

### M5c.1 — where the hours go (2026-09-26)

`imitation.viz.profile`: 20 episodes (seeds 100000-100019), 10 workers, privileged =
`dagger_diff/round_1` (diffusion, CUDA-graph sampler), vision = `vision_t0_128`, replan 8.
Worker columns are summed over workers ÷ (wall × 10). Main-process columns are ÷ wall.
`outputs/imitation/profile.json`.

| loop | ms / step (wall, pool) | physics | rendering | expert label look-ahead | main: plan (inference + GPU labels) |
|---|---|---|---|---|---|
| eval, state policy | 35.7 | 53.4% | — | — | 6.0% |
| eval, vision policy | 41.1 | 50.0% | 7.5% | — | 2.7% |
| DAgger, scripted expert labels | 86.1 | 21.9% | — | **43.0%** | 1.7% |
| DAgger, vision student + policy teacher on GPU | 45.8 | 49.5% | 14.4% | — | 4.8% |

Per-step costs on one core: physics 188-227 ms (100 cloth substeps), rendering 3 cameras
31 ms (66 ms when the policy-teacher loop also renders at label time). Training throughput
(`run.json`): chunk-MLP 99-125 steps/s, diffusion 13.9 steps/s (9.2 before the lazy
loader), vision 16.6-30 steps/s.

Where the hours go:
- **Physics** is the largest single share of every loop that doesn't use the expert. The
  remaining ~40% of worker time is outside `env.step`: pipe round-trips, waiting for the
  next batched plan, and the tail of a 20-episode run where fewer than 10 envs are left
  (episodes take 80-250 steps). Longer runs have a smaller tail.
- **Expert-labelled DAgger is 2.4× slower per step** than evaluation: each label simulates
  the expert ahead on a cloned state, which is twice the cost of the steps actually taken.
  A policy teacher on the GPU removes it (45.8 ms/step).
- **Inference is small**: 1.7-6.0% of wall time in the main process.

### M5c.2 exit check (2026-09-26)

Exit: inference + labels < 5% of rollout time, with identical actions. Actions are
bit-identical (test). Share: vision eval 2.7%, DAgger with GPU policy-teacher labels 4.8%,
expert DAgger 1.7% — met. **State-policy eval with the diffusion head: 6.0% at replan 8,
just over.** It doubles at the adopted replan 4. **Closed ❌ narrowly.** The 10 DDIM steps
are the remaining cost. The follow-up, if Phase 6 needs faster privileged rollouts, is fewer
sampling steps or a distilled one-step head, measured against the same eval sets.

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

**iql_v4** (same data, critic and settings as iql_v3; the actor samples the natural outcome
mix, 75% success transitions). 40k steps. n=200, deterministic, evaluated at replan 8 and at
the adopted replan 4.

| policy | replan | id_easy | id_hard | recovery |
|---|---|---|---|---|
| dagger_diff/round_1 (start) | 8 | 196/200 = 98.0% | 144/200 = 72.0% | 145/200 = 72.5% |
| dagger_diff/round_1 (start) | 4 | 199/200 = 99.5% | 154/200 = 77.0% | 159/200 = 79.5% |
| **iql_v4** | 8 | 178/200 = 89.0% [83.9, 92.6] | 68/200 = 34.0% [27.8, 40.8] | 69/200 = 34.5% [28.3, 41.3] |
| **iql_v4** | 4 | 138/200 = 69.0% [62.3, 75.0] | 37/200 = 18.5% [13.7, 24.5] | 51/200 = 25.5% [20.0, 32.0] |

Failure codes @8: id_hard G1 95, S1 27, F1 8, M1 2; recovery G1 55, F1 43, S1 33.
@4: id_hard G1 94, S1 61, F1 8; recovery G1 58, S1 57, F1 33, M1 1.

**Exit not met; RL is dropped from the plan.** The sampling fix removed the collapse
(iql_v3 54 / 24 / 18) but not the regression: −38 pp id_hard and −38 pp recovery at replan 8.
It is worse at replan 4, where the start policy is best. The cause is the M5.3 one, now
measured on diverse data. The critic separates states (V gap 0.51) but not actions: the
success-minus-failure advantage gap is +0.003 against a spread of 0.054 (probe). So the
AWR weights stay near 1, and the actor clones the harvest mix. That mix includes BC and
dagger_v2 actions and σ = 0.3 noise, which is where the grasp misses (G1 95) come from.
Two harvests, two critics and three actor schemes (M5.3, iql_v3, iql_v4) all end below the
imitation policy. Offline RL doesn't pay on this task at this data scale. The imitation +
DAgger path carries forward; revisit RL only with on-policy fine-tuning or a learned reward
(not in the pre-VLA scope).

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

---

## Isaac Sim runtime, Windows (Phase I, I0.3, 2026-10-03) — new baseline, not comparable to MuJoCo rows

The local stack: `.venv-isaac`, Isaac Sim 5.1.0, LeHome a805ad2 + IsaacLab fork 69f6fa5, torch
2.7.0+cu128, an RTX 5080 Laptop GPU on driver 616.56, and Kit rendering through D3D12 (Vulkan
crashes, see `isaac/INSTALL_REVIEW.md`).

| check | result |
|---|---|
| `isaac/smoke_test.py --mode state` | `SMOKE OK`: contract ok, scripted friction half fold ran (fold score 0.397, mean vertex error 4.9 cm, HalfFoldEnv success no), both wrappers run; **3.4 control steps/s**; 216 s wall incl. Kit start-up |
| `isaac/smoke_test.py --mode hybrid` | `SMOKE OK`: 84² RGB + depth, 6355/7056 depth px valid (0.37–1.47 m); **2.9 control steps/s** |
| `isaac/bench_parallel.py` (60 zero-action steps) | 1 process: 3.32 steps/s, peak 6.4 GB VRAM. 2 processes: 2.32 + 2.65 = **4.97 steps/s aggregate** (1.5×), peak 9.1 GB → `N_ISAAC = 2` |
| determinism | not bit-reproducible. Same seed 0, two runs of the scripted fold: fold score 0.469 vs 0.397 (error max 7.7 vs 12.0 cm). Expected: PhysX doesn't guarantee determinism for cloth, so Isaac gates stay statistical and images are captured at collection |

Artifacts: `outputs/isaac/smoke/20261003-013243/{state,hybrid}.log`, `hybrid_rgb.png`,
`outputs/isaac/runtime.json`.
### Backend switch regression (Phase I, I1.1, 2026-10-03)

MuJoCo, `imitation.data.collect --episodes 8 --workers 8`. The same command at `3861f08` (before
the backend switch) and with the new rollout worker (lazy teacher, ready handshake, NaN guard)
gives content hash **`26d4f1fe9f06` both times**: 8/8 success, 738 steps, including 1 recovery
episode (perturbation → resync path). The `-m slow` suite gives 12 passed. On Isaac,
`tests/imitation/test_isaac_backend.py` gives 3 passed: 2 `EnvPool` Isaac workers reset and
step with finite 139-D obs, the 400-step cap and ±1 cm offsets, then close cleanly. The
sim-free pipeline tests pass under the Isaac venv: 83 passed (numpy 1.26, torch 2.7+cu128).
### Isaac camera rig (Phase I, I1.3, 2026-10-03)

`isaac/rig_check.py`: half fold on Isaac in dict mode, seed 100000, 20 zero-action steps.
- **Frames:** `main` is 3×128×128 and both wrist cameras are 3×64×64, all uint8, with real
  content (std 39 / 33 / 25).
- **Wrist mounts:** each wrist camera's world pose vs the gripper link pose ∘ the so101-nexus
  MJCF `wrist_cam` offset is **0.014 mm / 0.00°** on both arms.
- **Throughput:** dict mode runs at 2.84 control steps/s (vs 3.4 in state mode).
- **No visual DR** in this pass (I-DR).

Artifacts: `outputs/isaac/rig/{rig.json,main.png,left_wrist_cam.png,right_wrist_cam.png}`.
### Isaac evaluation seed sets (Phase I, I1.2, 2026-10-03)

`isaac/reach_check.py` checks reach with no physics. It uses PinchIK on the half_fold_demo pinch
/ above / place targets, for cloth offsets drawn by the id_hard rule (200 offsets per radius).

| r (cm) | reachable | worst grasp (mm) | worst place (mm) |
|---|---|---|---|
| 0.5 | 200/200 | 2.93 | 11.85 |
| 0.75 | 200/200 | 3.78 | 14.64 |
| 1.0 | 200/200 | 4.62 | 17.32 |
| 1.25 | 129/200 | 7.94 | 20.33 |
| 1.5 | 81/200 | 11.51 | 22.79 |
| 2.0 | 48/200 | 19.39 | 27.97 |

A flat 4 mm bound is unreachable even at the nominal pose: the place IK is 5–6 mm off at zero
offset. So reachability is judged against the demo's ±1 cm operating box: grasp ≤ 4.66 mm,
place ≤ 17.6 mm (the place goal tolerance is 5 cm). That gives **R_max = 1.0 cm**, the training
jitter itself.

The Isaac sets:
- **id_easy:** the default reset (±1 cm offset, ≤10° drop tilt).
- **id_hard:** one axis forced to ±1.0 cm, the other U(±1 cm). It is an *edge-of-training-jitter*
  set, **not** an out-of-distribution test. A real OOD axis (yaw / larger tilt) goes to the
  scale-up milestones (IS).
- **recovery:** id_easy starts plus k ∈ [8, 16) random steps at t ∈ [35, 140), which is
  MuJoCo's 15–60 scaled to the ~230-step Isaac fold. This is provisional until the expert's mean
  steps are measured.
- **Training recovery demos:** t ∈ [24, 166).
- **New seed block:** `TUNE_SEED_BASE = 600 000` is reserved for grasp tuning (IG).

The MuJoCo sets are unchanged: `shifted_pose(200000)` is identical, a value pinned in
`test_seeds.py`.
### Isaac scripted expert, grasp as built (Phase I, I2.1, 2026-10-03) — PILOT, no threshold

`IsaacArmExpert` ports `isaac/half_fold_demo.py`'s pinch / arc fold unchanged and runs inside
`QuarterFoldExpert` as the pipeline's ScriptedTeacher. It has the release gate, retry on
measured miss, and resync. Run with `imitation.evaluate --backend isaac --ckpt expert`, 2 Isaac
workers, wilson 95%.

| set | seeds | n | success | fold score | failures | terminations |
|---|---|---|---|---|---|---|
| id_easy | 100000–100019 | 20 | 2/20 = 10% [2.8, 30.1] | 0.609 | G1 ×11, F1 ×3, M1 ×3, S1 ×1 | truncated 15, success 2, cloth_dragged 3 |
| recovery (resync check) | 300000–300009 | 10 | 0/10 = 0% [0.0, 27.8] | 0.348 | G1 ×7, M1 ×2, S1 ×1 | truncated 8, cloth_dragged 2 |

The expert is weak, and the failures are mostly grasp (G1): a corner dropped or released
unplaced. Every arm grasps at some point, but the friction pinch doesn't hold through the carry.
When the retry re-pinches a fallen corner, it can drag the cloth (M1). This is the friction
grasp *as built*, with no tuning in this pass. Grasp reliability is roadmap IG.1–IG.3. The 10%
here is at or below the demo's 1/3 on 3 seeds; the intervals are wide.

The mechanics are what this milestone checks, and they work:
- the phase machine runs pinch → close → arc → place → hold → synchronized release → retreat on
  both arms (`isaac/expert_smoke.py`)
- `resync` after random-action perturbations runs without errors on all 10 recovery episodes

Wall time: 2091 s for 20 id_easy episodes and 1136 s for 10 recovery episodes. That's about
100 s per episode across 2 workers, mostly 400-step truncations.

Artifacts: `outputs/imitation/isaac/pilot/eval_expert{,_recovery}.json` and `.log`.
### Isaac end-to-end micro chain (Phase I, I3.1, 2026-10-03)

`tests/imitation/test_isaac_chain.py` gives **6 passed in 32 min** in `.venv-isaac`. The real
CLIs on the Isaac backend, run on 3 episodes:
1. collect with the 3-camera rig
2. chunk-MLP and diffusion BC, 200 steps each
3. evaluate n = 2 each
4. 1 DAgger round on 2 episodes with takeover labels (p = 0.5)
5. vision BC, 100 steps
6. success detector train + agree

Each stage's artifacts are validated: dataset hashes re-verify, images are present on all 3
cameras, losses are finite, eval n matches, and the DAgger version is chained to its parent with
label shapes [L, K, 12]. This is the reusable regression guard for the Isaac pipeline.
Log: `outputs/isaac/chain/i3_1.log`.
### Isaac pilot run (Phase I, I3.2, 2026-10-03) — PILOT: viability, not performance

This is the whole imitation pipeline run on Isaac through the real CLIs: `scripts/isaac_pilot.ps1`,
2 Isaac workers, about 7.5 h of sim spread over 4 resumable launches. Intervals are Wilson 95%.

**Data.** The expert collected 80 episodes (30% perturbed) with images from the 3-camera rig at
collection time.
- `isaac_pilot` holds **2** successes: 534 steps, hash `a0bbd40bf051`.
- `isaac_pilot_failures` holds the other 78: 30 733 steps, hash `108c622cfcd7`.
- The success rate is 2/80 = 2.5% [0.7, 8.7] (clean 1/59, recovery 1/19). This is the friction
  grasp as built (I2.1); grasp work is IG.

**Training** (1 seed, short budgets) on `isaac_pilot`'s 2 demos. Every trainer fits its data.
Validation loss rises: that's overfitting to 1–2 demos, which is all the data there is.

| policy | steps | train loss first → last | val loss first → last |
|---|---|---|---|
| chunk-MLP | 5000 | 0.0323 → 0.0031 | 0.386 → 0.395 |
| diffusion | 5000 | 0.0366 → 0.0003 | 0.358 → 0.603 |
| vision BC (main 128 + wrists 64) | 3000 | 0.0305 → 0.0056 | 0.318 → 0.327 |

**Closed-loop evaluation on Isaac** (`--backend isaac`):

| policy | set | n | success | fold score | failures | inference (batch 16) |
|---|---|---|---|---|---|---|
| chunk-MLP @8 | id_easy | 20 | 0/20 [0, 16.1] | 0.089 | G1 ×20 | 2.9 ms |
| chunk-MLP @8 | recovery | 10 | 0/10 [0, 27.8] | 0.076 | G1 ×10 | 3.0 ms |
| diffusion @8 | id_easy | 20 | 0/20 [0, 16.1] | 0.087 | G1 ×20 | 11.4 ms (CUDA graph) |
| diffusion @8 | id_hard | 10 | 0/10 [0, 27.8] | 0.093 | G1 ×10 | 11.5 ms |
| diffusion @8 | recovery | 10 | 0/10 [0, 27.8] | 0.109 | G1 ×10 | 11.4 ms |
| vision @2 | id_easy | 10 | 0/10 [0, 27.8] | 0.124 | G1 ×9, M1 ×1 | 7.9 ms |

**DAgger** ran 1 round from the diffusion policy with takeover labels (p = 0.3). It made 16
student rollouts and **208 takeovers**. **204 of them (98.1%) recorded a full 16-step expert
chunk**; 4 were cut short by the episode ending, with no resync or IK errors. That meets I2.2's
pilot bar of ≥ 98%. After retraining on 17 episodes / 195 DAgger labels, round 1 scored 0/20, so
the keep rule kept the init.

**Success detector** (main camera) agrees with the sim on 79/80 episodes = 98.75% [93.3, 99.8]
(1 TP, 0 FP, 1 FN). That's barely above the 97.5% majority-class rate, and it was measured on
the episodes it trained on. This shows the stage runs, not that the detector is good.

**Verdict.** Every stage runs on Isaac and produces valid, hash-verified artifacts:
- collection with cameras
- 3 trainers
- closed-loop evaluation with batched GPU inference, including vision
- DAgger with takeover labels
- the detector

None of these numbers is a performance result. With a 2.5% expert, BC has 2 demos to learn from.
The performance arc (IG grasp reliability, then the IS retrain at n = 200) starts from here.

Artifacts: `outputs/imitation/isaac/pilot/` (eval JSONs, `takeover_labels.json`, logs),
`outputs/imitation/runs/isaac_pilot_*/run.json`, `outputs/imitation/runs/isaac_pilot_detector/agreement.json`.
## Phase W — MuJoCo weld baseline on Isaac

### W1 weld grasp (2026-10-03)

Weld semantics are MuJoCo's:
- hysteresis ±0.3
- every allowed grasp corner within 6 cm attaches on close with its gripperframe offset
- `weld_mask` applies per stage
- `grasp_active` is true while anything is welded

On Isaac a grid vertex pins the particles within 1.2 cm of it (21 particles).

**CPU device (LeHome default): fails.** The only write path is the mesh's USD `points`. The
writes read back exactly (0 mm), but PhysX ignores them mid-simulation: by the next 10 ms substep
the pinned particles were **14 mm (median) to 36 mm** from target, and max tracking error over the
lift was 42 mm. The corner rose only 6.5 cm for a 10 cm lift, carried partly by jaw friction. Each
write also cost 0.15 s. Diagnosed with a per-substep trace (`outputs/isaac/weld_debug.log`).

**GPU pipeline (`device="cuda:0"`): passes** (`isaac/weld_check.py`, `outputs/isaac/weld_cuda/weld.json`).
This is the old 4.5 port's method: pinned particles get zero mass through the particle physics
view, and their positions and velocities are set each substep.

| phase | max tracking error | corner z start → end | s / control step |
|---|---|---|---|
| close | 0.001 mm | 0.4239 → 0.4239 | 0.161 |
| lift 10 cm | 1.92 mm | 0.4269 → 0.5151 | 0.158 |
| carry 10 cm | 2.42 mm | 0.5172 → 0.5144 | 0.165 |
| hold (neutral command) | 0.00 mm | 0.5144 → 0.5144 | 0.165 |
| release | — | 0.5076 → 0.4240 (falls to the table) | 0.154 |

The checks all pass:
- the weld engages, tracks < 3 mm, raises the corner 9 cm, holds on a neutral command and releases
- a masked corner never pins
- the cloth rests on the table (z 0.424)

The GPU path also runs faster than the CPU one (0.16 vs 0.29 s per step). LeHome's reason for the
CPU device (grippers pass through the cloth on CUDA) doesn't matter for the weld, which holds the
cloth by attachment, not by contact.

**W1 close-out checks (2026-10-03):**
- Reset pose on the GPU pipeline: `reset_cloth`'s offset takes effect. With `cloth_pose` (0.05, 0.03),
  the settled centroid moved from (0.0002, −0.1354) to (0.0501, −0.1056), and back again on a zero offset.
  n = 1 seed, which is enough for this check since the reset has no randomness besides the drop tilt.
- Friction defaults are unchanged: `smoke_test.py --mode state` passes, and the scripted half fold
  still gives fold_score 0.407 at seed 0 (3.7 steps/s).
- `grasp_mode="weld"` on the CPU device now raises, and the USD-write path was removed.
### W2 MuJoCo profile on Isaac (2026-10-03)

`IsaacClothFoldEnv(profile="mujoco")`, which is the `isaac_weld` backend in the pipeline. It sets:
- the cloth centred at (0, 0), with MuJoCo's ±2.5 cm jitter and a flat drop
- MuJoCo's arm drives (kp 998.22, kd 2.731 + 0.60 joint damping, 3.35 N m, armature 0.028)
- the dynamics DR, ×U(0.7, 1.3) on cloth mass, cloth–table friction and cloth damping
- the 250-step cap
- the weld grasp on the GPU pipeline
- MuJoCo's eval sets

**Reach of the FoldExpert waypoints** (`isaac/reach_check.py --profile mujoco`):
- Setup: n = 200 per set, seed 0, approach / descend / lift / carry / place per arm.
- A pose counts as reachable when descend is within `GRASP_RADIUS` (6 cm) and every other waypoint is within the
  expert's 2 cm advance distance.
- The same offsets were also solved with MuJoCo's own `solve_ik` on the MuJoCo arm and cloth, as the parity reference.
- MuJoCo misses its own descend waypoint by a median of 7.7 mm (id_easy) and 15.7 mm (id_hard), so a flat 6 mm bar
  would fail MuJoCo itself.

| set | Isaac | MuJoCo (reference) | Isaac Wilson 95% | paired: both / MuJoCo only / Isaac only |
|---|---|---|---|---|
| id_easy (±2.5 cm) | 200/200 | 200/200 | 98.1–100% | 200 / 0 / 0 |
| id_hard (2.5–4 cm) | 122/200 | 130/200 (58.2–71.3%) | 54.1–67.5% | 122 / 8 / 0 |

- The gap is 4.0 pp, within the W2 parity bar of 5 pp.
- Isaac's reachable id_hard set is a subset of MuJoCo's.
- The 8 offsets only MuJoCo reaches come from the Isaac cloth resting 6 mm lower (z 0.424 vs 0.430). It adds about
  2 mm to the median descend error (18.0 vs 15.7 mm).
- The id_hard misses on both simulators are at approach, lift and carry (worst 36 / 25 / 21 mm on Isaac, 34 / 25 /
  21 mm on MuJoCo). The expert passes these anyway with its 45-step patience; MuJoCo's id_hard expert score is 97%.
- Artifacts: `outputs/isaac/reach_mujoco.json`, `outputs/isaac/reach_mujoco_ref.json`.

**Arm drive** (`isaac/profile_check.py`):
- The test: a one-step 0.05 rad command on each left-arm joint, starting from home, measured as the fraction of the
  step reached after one control step.

| joint | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Isaac (MuJoCo profile) | 1.403 | 1.384 | 1.351 | 1.014 | 1.012 |
| MuJoCo | 1.287 | 1.395 | 1.413 | 1.33 | 1.297 |

- Both simulators exceed 1 because the commanded target, not the measured pose, moves 0.05 rad, and the arm sags
  below its target at home. Both then hold the reached pose on the following steps.
- LeHome's drives (kp 17.8) reached ~0.78 in Phase I.

**Dynamics DR** (seeds 5, 5, 6):
- The draws are identical on a repeat seed, differ across seeds, and stay inside the range.
- Particle masses and the particle material's friction and damping read back scaled by the drawn values, to within
  1e-3. Example: seed 6 gives mass ×1.0229, friction ×0.906, damping ×0.921.
- `randomization=False` restores the spawned values: mass ×1, friction 0.5, damping 0.05.
- The material write reaches the solver: a 30 cm drop with damping ×0 vs ×40 ends at mean z 0.6073 vs 0.6197
  after 3 steps.
- Two findings:
  - The cloth's soft reset is parsed at the next physics step and restores the spawned masses. The DR is therefore
    applied after the first settle step.
  - LeHome's `get_friction` returns a tensor on the GPU pipeline.

**Cloth placement:** on a zero offset the centroid is at (0.000, 0.000). An offset of (0.02, −0.03) is applied
exactly, and the cloth rests flat at z 0.424.

**Pipeline plumbing:**
- The `isaac_weld` backend is accepted on every CLI. `eval_set(..., "isaac_weld")` and the recovery perturbation
  equal MuJoCo's.
- `make_env("isaac_weld")` gives profile mujoco, 250 steps and 2.5 cm jitter. The "isaac" backend is unchanged.
- 14 new fast tests.

**Friction profile:** unchanged, but its smoke test is not deterministic run to run. On the friction profile
(CPU pipeline), `smoke_test.py --mode state` at seed 0 gave these fold_scores:
- this tree: 0.371, 0.367, 0.320, and 0.486 under a parallel CPU load
- HEAD 3fb7bc6 (stashed A/B): 0.379, 0.349
- earlier runs: 0.407 twice

The two distributions can't be told apart. **Correction to the W1 close-out note above:** the matching 0.407 values
were coincidence, not evidence of determinism. The phase step counts are identical in every run (above 52, pinch 14,
arc 118); only the cloth outcome varies.

### W3 MuJoCo's expert on Isaac (2026-10-03)

**Setup.**
- `isaac.weld_expert.IsaacFoldExpert` runs MuJoCo's `FoldExpert` phase machine unchanged on the `isaac_weld` env,
  driven by `QuarterFoldExpert` with its retries and release gate.
- Only the sim reads are overridden: `_site`, `_q_now`, `_corner`, `_ik`. The IK is `PinchIK.solve_position`, which
  is `solve_ik` on LeHome's kinematics.
- **MuJoCo stays byte-identical** after splitting FoldExpert's sim reads into methods: `benchmark_expert` at 50
  episodes gives rows equal to M1.5, 48/50 (`outputs/imitation/expert_benchmark_w3.json`).

**How the weld was made to match MuJoCo's.** Each step below was diagnosed with a per-step trace, on tune-block
seeds 600000–600019 (n = 20) unless stated.

1. **Rigid pin, offsets in the gripper frame (W1 as built): 0/3.**
   - The free wrist pitched during lift and swung the held corner 4 cm sideways.
   - FoldExpert's "lift to above the corner" target never converged: each phase ran out its 45-step patience, and
     the corners were released 8–10 cm off goal.
   - In MuJoCo the same corner stays within 2 cm of vertical under the gripper (trace, seeds 100001–2).
   - The fix was to hold offsets in world axes and clamp pinned targets at the table, since the rigid pin had
     pushed the corner 2.7 cm into the table.
2. **World-axis rigid pin: 2/3, but 14/40 corners collapsed after release.** Two separate causes:
   - **The jaw opening.** A ~1.2 rad sweep through the flap hanging under it flung corners up to 9 cm. This
     correlates with the DR friction scale, which also scales cloth-against-jaw friction.
   - **Fix:** in weld mode the jaws stay open physically (the weld holds the cloth). Collapses fell to 8/40.
   - **The fold's stored tension.** A corner held even 1 cm outward stretches the folded 30 cm edge. The
     zero-mass pin holds any tension, so release snapped the corner 9 cm inward (left +x, right −x in every case).
3. **The soft weld.**
   - MuJoCo's weld is an equality constraint at the default solref (0.02 s, damping ratio 1). `sim_main` sets
     none, so the default applies.
   - The Isaac weld now steers the patch at the gripper's velocity plus error / τ, and the patch keeps its mass, so
     the cloth solver can pull it.
   - Stiffness is set by the patch's mass multiple, not by τ: at ×1 the lag was 52 mm for both τ = 0.01 and 0.005.

| `weld_check` (same script on both) | lift lag | carry lag | hold lag | corner rise for a 10 cm lift |
|---|---|---|---|---|
| **MuJoCo weld** (`isaac/weld_check_mujoco.py`) | 23.0 mm | 27.2 mm | 6.7 mm | 9.0 cm |
| Isaac, rigid pin | 1.9 mm | 2.4 mm | 0 | 8.8 cm |
| Isaac, soft τ 0.02, mass ×1 | 64 mm | 65 mm | 63 mm | 5.5 cm |
| **Isaac, soft τ 0.02, mass ×10 (shipped)** | 27 mm | 27 mm | 21 mm | 8.8 cm |
| Isaac, soft τ 0.02, mass ×100 | 8 mm | 8 mm | 6 mm | 9.1 cm |

4. **Overshoot, re-measured on the calibrated weld.**
   - Method: zero overshoot, no retries; each corner's miss read 15 steps after release, or at termination.
   - Result, n = 20 per arm: left mean (0.000, −0.004) m, sd (0.002, 0.011); right (−0.002, −0.002), sd (0.007,
     0.011). One outlier in 40.
   - `OVERSHOOT_ISAAC` stage 0 is set to minus these: left (0.000, +0.004), right (+0.002, +0.002).
   - Unlike MuJoCo's flexcomp (−0.04, −0.03) / (+0.02, −0.03), the Isaac cloth hardly springs back.
   - Earlier estimates came from the uncalibrated welds and are superseded: rigid world-axis about (0, −0.027) m
     with collapses; soft ×1 (±0.019, −0.047).
   - A soft ×1 weld with its own overshoot went 20/20 on the tune seeds with 0 retries. The calibrated weld is
     shipped because it matches MuJoCo's weld, not because it scores higher.
5. **The W1 tracking bar changes.** It is now within 1.5× MuJoCo's 27.2 mm on the same script (it was < 3 mm,
   which only a rigid pin meets). A rigid pin remains available with `weld_tau = None`.

Artifacts: `outputs/isaac/overshoot/*.json`, `outputs/isaac/weld_mass_{10,100}/`, `outputs/isaac/weld_tau_*/`,
`outputs/isaac/weld_mujoco_ref.json`.

**W3 expert gate** (`scripts/w3_gate.ps1`):
- Setup: `imitation.evaluate --backend isaac_weld --ckpt expert`, n = 100 per set on MuJoCo's eval sets (seeds
  100000+, 200000+, 300000+), 2 Isaac workers, calibrated soft weld, `OVERSHOOT_ISAAC` as above.

| set | Isaac (MuJoCo profile) | Wilson 95% | bar | MuJoCo expert (M1.7) |
|---|---|---|---|---|
| id_easy | **100/100** | 96.3–100% | ≥ 95 ✅ | 100% |
| id_hard (2.5–4 cm) | **100/100** | 96.3–100% | ≥ 90 ✅ | 97% |
| recovery (knock t 15–60, 8–16 steps) | **81/100** | 72.2–87.5% | ≥ 90 ❌ | 96% |

- id_hard reaches 100% even though W2's static reach was 122/200. The expert passes waypoints the IK misses on its
  45-step patience, and the 6 cm weld radius catches the corner.
- **Recovery failures** (19, all truncated at 250 steps): S1 stalled 9, G1 dropped / not re-grasped 6, F1 placed
  off target 4. Every episode grasped at least once.
- **Diagnosis** from a replay of recovery seeds 300000–300019 with the same knocks: 17/20.
  - The knock's random gripper commands release the weld mid-carry.
  - The re-grasp, from the side, leaves the corner's horizontal offset frozen, so lift and place run out their
    45-step patience.
  - After a disturbed fold, corners settle 4.5–8 cm from goal (success radius 5 cm). Each retry costs about 50
    steps, and two retries don't converge before the cap.
- **Tried and reverted:** aiming the gripper so the held corner, not the gripper, reaches FoldExpert's targets
  (subtracting the held offset): 16/20 on the same seeds, no gain.
- W3 is **not met** on recovery. It stays open, pending a decision (status.md).

Artifacts: `outputs/imitation/isaac_weld/w3/eval_expert_{id_easy,id_hard,recovery}.json` and their logs.

**W3 decision (2026-10-03, user):** recovery 81/100 is accepted as the Isaac expert's recovery ceiling, and W4 runs
on this expert. `verify W3` records the bar as 80, with the planned 90 noted. id_easy and id_hard meet their bars.

## Track G — friction grasp on Isaac (LeHome profile)

### IG.1 grasp bench, baseline as built (2026-10-03)

`isaac/grasp_bench.py` runs the friction expert (`IsaacArmExpert` in `QuarterFoldExpert`, the pipeline's
ScriptedTeacher) on `IsaacClothFoldEnv(profile="lehome")`, CPU device, with **retries off**, so each arm's first
attempt is scored. Per-step traces (phase, gripperframe site, corner, jaw angle, anchor) are kept per seed and the
flags are computed from them by `isaac/grasp_metrics.py`:
- **acquired:** when the expert enters carry, the corner is ≥ 2 cm above its rest height and within 5 cm of the site.
- **held:** acquired, and the corner stays within 5 cm of the site until the jaw opens. A held corner rides 2.5–3.8 cm
  from the site (the fixed fingertip), up between the pads; a dropped one jumps to ≥ 6 cm.
- **placed:** 15 steps after the jaw opens, the corner is within 5 cm of its goal (the env's test).
- **released:** held, and from the jaw opening to the end of the retreat the corner moves < 3 cm horizontally and ends
  < 2.5 cm above its rest height.
- **anchor drift:** the largest displacement of the arm's anchor corner (cloth_0 / cloth_110) over the episode.

Rates are unconditional (out of all episodes). Knobs all as built (particle friction 0.5, pad lining 1.5 / 3 mm,
LeHome gripper drive, closed target −0.1 rad; pinch inset 5 mm, pinch height 1 cm). Tune seeds 600000–600019, n = 20
(seeds 600000–1 come from the bench's smoke run, same code and config). Wilson 95%.

| arm | acquired | held | placed | released | anchor drift mean / max |
|---|---|---|---|---|---|
| left | 20/20 [83.9, 100] | 18/20 [69.9, 97.2] | 14/20 [48.1, 85.5] | 12/20 [38.7, 78.1] | 4.0 / 5.0 cm |
| right | 11/20 [34.2, 74.2] | 1/20 [0.9, 23.6] | 0/20 [0, 16.1] | 0/20 [0, 16.1] | 5.1 / 7.5 cm |

- **The right arm is the problem.** Its corner gets pushed out of the pinch as the jaw closes: in the gripper frame the
  corner is displaced up to 3.4 cm sideways (vs < 1 cm on the left) and lifts into the pads. 9/20 corners never
  rise; 10 of the 11 acquired fall out during the carry (steps 82–127, mostly while the arm descends toward the goal).
- **Left:** it holds 18/20, but 6 placements miss by 5.5–8.4 cm and 8 releases move the corner 3–6 cm: the corner is
  let go about 5 cm above the table (it rides 2.5 cm above the site, and the site stops ~2.4 cm above the table at
  place), and it falls outward.
- **Anchor drift** is 3–7.5 cm, well under the 20 cm drag limit; most of it happens at close, when the pinch pulls the
  cloth toward the arms.
- **Speed:** 115–350 s per episode (0.7–2 control steps/s) with the W4 pilot's 2 Isaac workers sharing the GPU, so an
  n = 20 trial takes about 50 min and an n = 100 block about 4 h.

Artifacts: `outputs/isaac/grasp/ig1_baseline/{rows.jsonl,summary.json,bench.log}` (traces regenerate, not committed).

### IG.2 reliability loop — attempt log (2026-10-03, one knob per attempt)

Each attempt changes one knob from the best config so far and runs the bench on tune seeds 600000–600019 (n = 20,
paired with the IG.1 baseline), Wilson 95%. A knob is kept only if it beats its parent. Rates as left / right.

| # | change (from parent) | acquired | held | placed | released | anchor drift max | verdict |
|---|---|---|---|---|---|---|---|
| 0 | IG.1 baseline (as built) | 20 / 11 | 18 / 1 | 14 / 0 | 12 / 0 | 5.0 / 7.5 cm | parent |
| 1 | particle friction 0.5 → 1.5 | 20 / 20 | **2 / 0** | 3 / 5 | 1 / 0 | 1.8 / 6.0 cm | rejected |
| 2 | pinch inset 5 → 15 mm (expert) | 17 / 7 | **1 / 0** | 0 / 0 | 0 / 0 | 23.5 / 6.2 cm | rejected |
| 3 | closed jaw target −0.1 → +0.05 rad | 20 / 20 | **20 / 14** | 9 / 8 | 2 / 1 | 4.3 / 5.1 cm | **kept** (new parent) |
| 4 | #3 + place height 1.1 → −1.0 cm (expert) | 20 / 20 | 19 / 14 | 9 / 5 | 5 / 2 | 5.1 / 28.0 cm | not kept |
| 5 | closed jaw target +0.05 → +0.08 rad | 20 / 19 | **0 / 0** | 0 / 0 | 0 / 0 | 5.4 / 11.9 cm | rejected |
| 6 | closed jaw target +0.05 → +0.02 rad | 20 / 20 | 19 / 17 | 5 / 1 | 4 / 2 | 7.2 / 13.3 cm | not kept (held within noise of #3, placement worse) |
| 7 | #3 + jaw opens at 0.1 rad / step (`jaw_open_rate`, ~11 steps instead of ~3) | 20 / 20 | 20 / 14 | 10 / 9 | 2 / 0 | 6.7 / 12.5 cm | not kept (no release gain) |
| 8 | #3 + placement overshoot from #3's settled miss: left (−0.1, +5.2) cm, right (+1.3, +5.0) cm (`OVERSHOOT_FRICTION`) | 20 / 20 | 20 / 12 | **18 / 11*** | 10 / 3 | 5.9 / 5.2 cm | **kept** (new parent); 11/20 episodes succeed with retries off |

| 9 | #8 + pinch height 1.0 → 0.5 cm (expert) | 20 / 20 | **20 / 19** | **19 / 18*** | 6 / 4 | 7.7 / 5.1 cm | **kept** (new parent); 17/20 episodes succeed with retries off |

\* Placement is read at the episode end from #8 on; see the metric note below.

- **#1:** the higher friction fixes the right arm's pinch (acquired 20/20 [83.9, 100] vs 11/20 [34.2, 74.2]), and the
  cloth barely slides on the table (anchor drift mean 0.8 / 1.4 cm). But it loses the carry on both arms: held
  2/20 [2.8, 30.1] and 0/20 [0, 16.1]. The corner slides down out of the fingers near the top of the arc (steps
  97–144). The particle material's friction is also the cloth–table friction, so the table now holds the rest of
  the sheet and the arc pulls the corner out.

- **#2:** a deeper bite (the fixed finger 15 mm in from the corner instead of 5) is worse on every flag: left held 1/20 [0.9, 23.6], right acquired 7/20 [18.1, 56.7]. One episode dragged the cloth (an arc that ran along the table). The pinch works best right at the edge.
- **#3 (kept):**
  - Why: at −0.1 rad the jaw passes through the fixed pad (the articulation doesn't collide its own adjacent links),
    so the pads overlap and squeeze the particles out. At +0.05 the jaw stops short of the pad and pinches the cloth.
  - Held: left 20/20 [83.9, 100] (was 18/20 [69.9, 97.2]); right 14/20 [48.1, 85.5] (was 1/20 [0.9, 23.6]).
    Acquired is 20/20 on both arms. 3/20 episodes succeed, with retries off.
  - What fails now is release and placement: placed 9/20 [25.8, 65.8] and 8/20 [21.9, 61.3], released 2/20 [2.8, 30.1]
    and 1/20 [0.9, 23.6].
  - The held corner rides about 3.6 cm above the fingertip, and the site stops about 2.4 cm above the table at place.
    So the corner is let go about 6 cm up, and as the jaw opens it falls 3–5 cm outward (−y, away from the fold line).
- **#4:** the place target 2.1 cm lower didn't move the site down meaningfully, and releases stayed within noise:
  5/20 [11.2, 46.9] and 2/20 [2.8, 30.1]. Right-arm placement fell to 5/20 [11.2, 46.9], and one episode dragged the
  cloth (right anchor 28 cm). Not kept.
- **#5 (sweep around #3):** +0.08 rad lifts every corner (acquired 20/20, 19/20), but no corner is held through the
  carry: held 0/20 [0, 16.1] on both arms. The working window of the closed target is narrow. −0.1 squeezes the
  cloth out, +0.08 pinches too lightly, and +0.05 holds.
- **#6:**
  - Held: 19/20 [76.4, 99.1] and 17/20 [64.0, 94.8], against #3's 20/20 and 14/20 (34 vs 36 of 40 arms).
  - Placements are worse: 5/20 [11.2, 46.9] and 1/20 [0.9, 23.6].
  - Settled miss (corner − goal, 15 steps after release, held episodes): left (+4.1, −4.4) cm, sd (1.9, 1.9);
    right (−4.5, −5.9) cm, sd (1.9, 1.8). With #3: left (+0.1, −5.2), sd (1.3, 1.2); right (−1.3, −5.0), sd (1.0, 2.0).
  - #3 stays the parent: it holds as well, and its spring-back is tighter, which is what an overshoot can correct.
- **#7:** opening the jaw over about 11 steps changes nothing that matters (released 2/20 [2.8, 30.1] and 0/20
  [0, 16.1]; placed 10/20 and 9/20, within noise of #3). The trace confirms the jaw ramps at 0.1 rad per step.
- **What "released" is measuring (analysis on attempts 0, 3, 6, 7).**
  - Every held corner is let go: by the end of the retreat it lies back on the cloth (< 2.5 cm above rest).
    - #3: 20/20 left, 14/14 held on the right.
    - #7: 20/20 and 14/14 (13 of 13 were scored when this was written).
  - What fails the strict 3 cm test is the horizontal move after opening: median 4.2 / 4.6 cm in #3, 3.8 / 4.8 cm in
    #7, max 7.4 cm. The corner is let go about 5 cm above the table and the fold's flap springs outward; on MuJoCo the
    same spring-back is 3–4 cm and is absorbed by `OVERSHOOT`.
  - So with the current hold height, the strict "released" flag is a spring-back test, not a test of the jaw letting
    go. It moves only if the corner is held lower (it rides 3.6 cm above the fingertip, up between the pads). A wider
    or slower opening doesn't change it. Placement can be corrected with an overshoot (#8).
- **#8 (kept):** the overshoot is minus #3's mean settled miss, as W3 did for the weld.
  - Placed: left 18/20 [69.9, 97.2], right 11/20 [34.2, 74.2].
  - Released: 10/20 [29.9, 70.1] and 3/20 [5.2, 36.0]. Releasing nearer the fold line also shortens the outward slide.
  - 11/20 bench episodes end in the env's own success, with retries off. The I2.1 expert had 2/20 with retries.
  - The open problem is now the right arm's hold: 12/20 [38.7, 78.1] here, 14/20 in #3. Its drops repeat on the same
    seeds (600001, 010, 013, 014, 018), late in the carry (steps 127–169, while the arm descends to the goal), and
    often after a weak lift (corner rise 3 cm against 7 cm).
- **Metric note (2026-10-04), placement read time.**
  - "placed" was read 15 steps after the jaw opened. A released corner keeps sliding for 20+ steps, so that read
    disagreed with the env's own success test: #8 seed 600017 scored not placed at that read, yet ended in env
    success.
  - From #8 on, placement is read at the episode end: the env's success, or SETTLE_WAIT steps after both arms
    retreated. That is the env's own judging time. `isaac/grasp_metrics.py` changed, and every run was rescored from
    its traces (`--summarize --rescore`).
  - Placed under the new read, left / right: baseline 18 / 0 (was 14 / 0), #1 4 / 5, #2 0 / 0, #3 9 / 8, #4 9 / 5,
    #5 0 / 0, #6 6 / 6, #7 10 / 8, #8 18 / 11. The other flags are unchanged. The rows above keep the earlier
    read, except #8's.
- **#9 (kept):** the fingertip goes 5 mm lower at the pinch, so the pads bite the cloth closer to the table.
  - Left / right, Wilson 95%:
    - acquired 20/20 [83.9, 100] on both arms
    - held 20/20 [83.9, 100] and 19/20 [76.4, 99.1] (the right arm's was 12/20 in #8)
    - placed 19/20 [76.4, 99.1] and 18/20 [69.9, 97.2]
  - 17/20 episodes end in the env's success with retries off. The seeds that dropped before (600001, 013, 014, 018)
    now hold; 600010 still drops.
  - Settled miss with #8's overshoot: left (+1.1, +0.1) cm, sd (2.1, 2.1); right (−1.6, 0.0) cm, sd (1.2, 2.2).
    The overshoot is centred.
  - Strict "released" is 6/20 [14.5, 51.9] and 4/20 [8.1, 41.6]. Every held corner is let go (back on the cloth after
    the retreat: 20/20 and 19/20).
  - The release move (median 5.1 / 5.3 cm) is now by design. The overshoot releases each corner about 4 cm inside its
    goal, and the spring-back carries it onto the goal.
- **IG.2 status at #9: plateau on the bar as defined.** Acquired, held and placed are at or near their bars on the tune
  seeds (n = 20, point estimates 95–100%). Strict "released" (< 3 cm horizontal move after the jaw opens) can't be met
  together with an overshoot placement, since that placement relies on the move. A decision on the release metric is
  needed before the n = 100 blocks (see status.md).

- **Definition change: "released" (approved by the user 2026-10-04).**
  - New "released": the corner was held, and by the end of the expert's retreat it is back on the cloth (within
    2.5 cm of its rest height, so not carried up by the gripper). Where it lands is scored by "placed". The old
    test is kept as `released_strict` (also needs a horizontal move < 3 cm from the jaw opening to the end of the
    retreat).
  - Why: with an overshoot placement the move is by design (the spring-back carries the corner onto the goal), so
    strict "released" could not be met together with the placement bar. The flags are recomputed from the stored
    traces (`--summarize --rescore`); no run was repeated. n = 20, seeds 600000-600019 (baseline and #3 onward; the
    earlier attempts' rows are as logged above), Wilson 95%.

  | # | released, new (left / right) | released_strict = old definition (left / right) |
  |---|---|---|
  | 0 baseline | 17 / 1 | 12 / 0 |
  | 1 particle friction 1.5 | 1 / 0 | 1 / 0 |
  | 2 inset 15 mm | 1 / 0 | 0 / 0 |
  | 3 closed +0.05 | 20 / 14 | 2 / 1 |
  | 4 place low | 19 / 14 | 5 / 2 |
  | 5 closed +0.08 | 0 / 0 | 0 / 0 |
  | 6 closed +0.02 | 19 / 17 | 4 / 2 |
  | 7 jaw rate 0.1 | 20 / 14 | 2 / 0 |
  | 8 overshoot | 20 / 12 | 10 / 3 |
  | 9 pinch 0.5 cm | **20 / 19** [83.9, 100] / [76.4, 99.1] | 6 / 4 [14.5, 51.9] / [8.1, 41.6] |

  Under the new definition released equals held in every attempt except the baseline left arm (held 18, released
  17: one held corner was carried up by the retreat) and #1 left (held 2, released 1): a held corner is, in practice,
  always let go.

Artifacts: `outputs/isaac/grasp/t{1..4}_*/{rows.jsonl,summary.json,bench.log}`.

Weld path unchanged by the opt-in knobs: `tests/imitation/test_isaac_weld.py` + `test_isaac_profile.py` in
`.venv-isaac` on this branch, 17 passed (`outputs/isaac/grasp/weld_regression.log`).

**Naming (2026-10-04):** the Isaac profile `"mujoco"` is renamed `"weld"`. It runs entirely in Isaac and only
carries over the MuJoCo setup's settings. Earlier entries and artifacts above keep the old name. MuJoCo itself is
now a frozen reference: no new MuJoCo runs (user decision).

### W4 pilot on the weld baseline (2026-10-03/04) — PILOT, artifacts in `outputs/imitation/isaac_weld/pilot/`

- **Collection:** `isaac_weld_pilot`, 80 episodes on the W3 expert (recovery 81 version). 76 succeeded (6,681 steps;
  hash `667f6215`), 4 failures went to `isaac_weld_pilot_failures`. The Phase I pilot had 2/80 with the friction
  grasp.
- **Training:** chunk-MLP 5k steps; diffusion 5k steps; vision 3k steps at 128² main plus 64² wrist cameras; all
  seed 0.

| policy | set | n | success | Wilson 95% |
|---|---|---|---|---|
| chunk-MLP | id_easy | 20 | 19 | 76.4–99.1% |
| chunk-MLP | recovery | 10 | 0 | 0–27.8% |
| diffusion | id_easy | 20 | **20** | 83.9–100% |
| diffusion | id_hard | 10 | 8 | 49.0–94.3% |
| diffusion | recovery | 10 | 1 | 1.8–40.4% |
| vision (replan 2) | id_easy | 10 | 10 | 72.2–100% |

- W4's bar (diffusion id_easy Wilson lower bound > 0) is met. Recovery is weak from BC at this scale, as on MuJoCo,
  which is what DAgger is for.
- The DAgger round and the detector were cut off by the 2 h job limit and are pending.


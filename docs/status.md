# Status

**Updated:** 2026-09-26 · **Branch:** `feature/imitation` · **Phase:** 6 next (everything before the VLA is closed)

One-screen answer to "where are we". Update at the end of **every work pass** (see
`CLAUDE.md`), and add a line to the pass log at the bottom. Full plan in
[roadmap.md](roadmap.md); spec in [imitation.md](imitation.md); numbers in
[results.md](results.md).

---

## Where we are

**Every milestone before the VLA is closed** (✅ met or ❌ closed with evidence; tracker
below). The half-fold demo is rendered from pipeline-trained policies at their operating
points: `docs/reports/media/half_fold_privileged.mp4` (4/4) and `half_fold_sensor.mp4` (3/4).
Dated report: [reports/2026-09-25-phase5b.md](reports/2026-09-25-phase5b.md), with a shareable
copy at https://claude.ai/artifact/GWfhYiGusezV6rWqpziQQi.

| policy | id_easy | id_hard | recovery | where |
|---|---|---|---|---|
| expert (ceiling) | 100% | 97% | 96% | M1.7 |
| BC chunk-MLP | 98.0 | 69.5 | 38.5 | M2.1, M5b.5 |
| BC diffusion, seeds 0 / 1 / 2 | 100 / 99.5 / 100 | 77.0 / 83.0 / 85.0 | 55.5 / 62.0 / 60.5 | M2.4 + seeds |
| **DAgger diffusion, replan 4 (privileged)** | 99.5 | 77.0 | **79.5** | M5b.1, M5b.6 |
| IQL (iql_v4, replan 8) | 89.0 | 34.0 | 34.5 | M5b.4 ❌, RL dropped |
| vision BC 128² | 94.5 | 91.0 | 26.5 | M4.1 |
| **vision student, replan 2 (sensor-only)** | 94.0 | **91.0** | 41.0 (seed 1: 34.5) | M5b.3, M5b.6 |

What's still weak, carried into Phase 6: the last-centimetre S1 stall on shifted poses,
sensor-only recovery (35-41%, re-grasping), and seed variance (8 pp on id_hard) that is as
large as most effects measured. All in `results.md`.

## Next action

**Phase 6, M6.1: fold variants with text instructions** (roadmap). Before starting, decide
how the expert's `_maybe_retry` (reads the miss from sim vertices) behaves for VLA-specified
folds. Judge Phase 6 milestones on ≥ 2 seeds. Run long jobs as visible tasks, ≤ 10 workers,
one sim pool at a time.

## Pre-VLA milestone tracker

Everything above Phase 6 must be ✅ or ❌-closed-with-evidence before the VLA starts
(roadmap rule). Updated every pass.

| milestone | state | closes via | where it runs |
|---|---|---|---|
| M4.1 image plumbing | ✅ 2026-09-25 | vision BC on the finished path, 94.5 / 91.0 / 26.5 | done |
| M4.2 distillation margin | ❌ closed, margin stated (−3 / +20 / −42.5 pp) | — | done |
| M5.3 offline RL | ❌ closed: below the imitation policy 3× (M5.3, iql_v3, iql_v4); RL dropped | — | done |
| M5b.1 diffusion DAgger | ✅ +1.74 SE over dagger_v2 | — | done |
| M5b.2 shifted poses | ❌ closed, diagnosed (S1 stalls) | — | done |
| M5b.3 vision recovery | ❌ closed: distillation transfers teacher weakness | — | done |
| M5b.4 offline RL retry | ❌ closed 2026-09-26: iql_v4 89.0/34.0/34.5 @8, 69.0/18.5/25.5 @4; RL dropped | — | done |
| M5b.5 hygiene / determinism | ✅ 2026-09-25: repeat identical | — | done |
| M5b.6 fine placement | ❌ closed, operating points adopted (privileged replan 4, vision replan 2); DAgger at replan 4 not kept | — | done |
| M5c.1 profile | ✅ 2026-09-26: physics ~50%, expert label look-ahead 43% of DAgger, inference 1.7-6% | — | done |
| M5c.2 GPU inference wins | ❌ closed narrowly: 12.7×, bit-identical; share 6.0% for diffusion state eval (< 5% elsewhere) | — | done |
| M5c.3 overlap GPU/CPU | ❌ closed: GPU kernel-active 27% across a DAgger run (target > 50%) | — | done |
| M5c.4 GPU physics feasibility | ❌ closed: slower than 1 CPU core, drifts > 1 cm by step 15 | — | done |
| M5c.5 batched GPU rendering | ❌ closed (gate M5c.4 failed) | — | done |

## Environment

| | |
|---|---|
| pins | `imitation/requirements.txt` — mujoco **3.10.0**, so101-nexus **0.4.8**, numpy 2.5.1, gymnasium 1.3.0 |
| torch | 2.13.0+cu130, **CUDA available** |
| tests | 96 fast + 10 slow, all passing (`pytest -m "not slow"` / `-m slow`) |

⚠️ `cloth_fold_rl/requirements.txt` pins mujoco 3.11.0 / so101-nexus 0.5.1 for its own
committed checkpoint. Do not "unify" these without re-running the expert benchmark — cloth
grasping is contact-dominated and MuJoCo minors change results.

## Frozen datasets

| version | episodes | source | hash | notes |
|---|---|---|---|---|
| v1 | 384 (+16 in `v1_failures`) | expert, seeds 0-399, 30% perturbed | `769dd372719c` | successes only; 38,136 steps; failures hash `699a1dc71c71` |
| v1_img / v1_failures_img | 384 / 16 | v1 replayed + cameras | `769dd372719c` / `699a1dc71c71` | same hash as source: froze before images entered the digest |
| v1_dagger_v2_r1..r3 | +128 each | DAgger (shadowing teacher) | see manifests | parent chain on v1 |
| harvest_v1 | 1000 | dagger_v2/round_3, σ=0.1 | `3b5d331bf404` | 81% success |
| v1_img_distill_v1_r1..r2 | +128 each | vision student rollouts, images | see manifests | parent chain on v1_img |
| v1_dagger_diff_r1 / r2 | 512 / 640 | diffusion DAgger (M5b.1) | `51dad5a14683` / `aac732c40b20` | r1 is the privileged best's data |
| …_dagger_shift_r1 / r2 | 768 / 896 | 50% shifted-pose rollouts (M5b.2) | `6fb649e243f8` / `fe75f9ac7af5` | not kept |
| v1_dagger_diff_r1_dagger_r4_r1 | 576 | DAgger at replan 4 | `5a3fcacb5fa2` | not kept |
| harvest_v2 | 1,679 | 4 policies × σ {0.1, 0.3}, ≥ 60 per failure code | `febfb9275058` | 68% success |
| v1_img128 / v1_failures_img128 | 384 / 16 | v1 replayed, 128² main + 64² wrists | `532f8a9c08b6` / `3c1f9e376161` | images in the digest |
| v1_img128_distill_v2_r1 / r2 | 512 / 640 | camera-student rollouts, teacher-relabelled | `b87a7e4ef895` / `ab63f57fe382` | not kept |

Collection log: `outputs/imitation/collect_v1_m1_8.log`. The earlier failed attempt is
kept as `outputs/imitation/collect_v1.log`.

## Best checkpoint

- privileged: `outputs/imitation/runs/dagger_diff/round_1/final.pt` at **replan 4** (99.5 / 77.0 / 79.5)
- sensor-only: `outputs/imitation/runs/vision_t0_128/final.pt` at **replan 2** (94.0 / 91.0 / 41.0)
- success detector: `outputs/imitation/runs/success_v2/detector.pt` (128², 92.6% agreement)
- best plain BC on id_hard: `outputs/imitation/runs/diff_v1_s2/final.pt` (100 / 85.0 / 60.5, replan 8)

Weights are gitignored; each run's `run.json` / `history.json` is committed.

## Open blockers and known defects

Ordered by what they block. Each is a roadmap milestone.

| # | defect | blocks | milestone |
|---|---|---|---|
| 12 | `imitation.cpu_slot` isn't fair: a lane that releases and re-takes it between eval sets starves a lane polling every 2 s (lane A waited 37 min) | parallel queues | — |
| 11 | `v1_dagger_v1_r1`/`_r2` frozen with bad teacher labels — never train on them | DAgger | — |
| 10 | `cloth_angles/tasks.py::QUARTER` diverged from `quarter_fold_env.STAGES` (3 ways) | world-model work only (parked) | M7.3 |

## Parked

- **World model** (`cloth_angles/`) — off the critical path by decision. Its task
  definition has diverged and must be reconciled or deleted before any result from it
  counts.
- **VLA** — enters at Phase 6 as a subgoal proposer, not an action labeler. MolmoAct emits
  8×6 absolute joint degrees, single arm; the env takes 12-D normalized dual-arm deltas.
- **PPO** — `cloth_fold_rl/train.py` and `bc.py` are superseded and deleted in M7.2. The
  unrelated `MuJoCoTouch-v1` PPO baseline stays.

## Pass log

Newest first. One line per work pass: date · what changed · commit.

- 2026-09-26 · **All pre-VLA milestones closed.** M5b.4 ❌ + RL dropped (iql_v4 89/34/34.5 @8); M5c.1 ✅ (expert label look-ahead 43% of DAgger); M5c.2 ❌ narrowly (6.0% inference share for diffusion eval); M5c.3 ❌ (GPU 27%); DAgger at replan 4 not kept; diffusion seeds 77-85% id_hard, vision seed 1 recovery 34.5%; demos at operating points (4/4, 3/4); dated report 2026-09-25-phase5b.md; page republished · 96 fast + 10 slow pass · fb33f4f
- 2026-09-25 · M5b.5 ✅ (determinism repeat identical; M2.1/M2.3 re-evals inside original intervals); detector v2 on 128² (92.6% agreement); M5c.1 profile ran; pipeline.md Data section; expert label look-ahead now timed in profiles; `imitation.viz.gpu_busy`; diffusion seeds 1-2 + vision seed 1 trained; final queue **paused** at 15:16 (resume script) · 96 fast tests pass · ac0a45e
- 2026-09-25 · M5b.6 closed: replan sweep — privileged best at replan 4 (99.5/77.0/79.5, +5 id_hard, +7 recovery), vision recovery 30→41% at replan 2; id_hard target 85% not reached; operating points adopted · 111241e
- 2026-09-25 · M5b.4: harvest_v2 gives 3× advantage spread; iql_v3 collapsed to 54/24/18 because uniform outcome stratification fed the actor 75% failures; fix = separate actor sampling (`--actor-strata natural`), iql_v4 training (final attempt before dropping RL) · e7f2523
- 2026-09-25 · M5c.4 ❌ closed: MuJoCo Warp 3.10 loads the cloth model but runs ≤274 world-steps/s (CPU 1 thread 576) and drifts >1 cm from the CPU trajectory by step 15; M5c.5 closed with it; tools `imitation/gpu_sim/` · 358bc3b
- 2026-09-25 · M5b.3 ❌ closed (distill_v2: best round 0 = 95/92/30; rounds transfer the teacher's id_hard weakness, not its recovery) → M4.2 closed with margin; harvest_v2 frozen (1679 eps); IQL made diffusion-aware (per-sample losses) after the queue crashed on it; remaining work relaunched as visible task bdbxibw36; mujoco-warp env (approved) installing as task bbxccvf4m · fee819f
- 2026-09-25 · M5c.3 mechanism: `imitation.cpu_slot` — rollouts hold one cross-process CPU slot when IMITATION_CPU_SLOT is set, so parallel lanes overlap GPU training with simulation without exceeding the worker cap; 94 fast + 10 slow green · 5024dd0
- 2026-09-25 · M5c.1 profiling hooks + `imitation.viz.profile`; M5c.2 CUDA-graph diffusion sampler (301→24 ms, bit-identical; int timestep schedule, identical to before); 92 fast + 10 slow green · 57cd487
- 2026-09-25 · Pre-VLA tracker added to status.md; M4.1 closed ✅ (vision BC 128² on the finished path: 94.5/91.0/26.5); M5b.6 fine-placement added as M5b.2 follow-up; closure rule written into roadmap; queue restarted as a visible task (bpbn1o3yx) after the previous session's processes ended · df44dc0
- 2026-09-25 · CPU relief: policy-teacher labels moved from every worker's CPU to one batched GPU call; workers only simulate; runs capped at 10 workers, one CPU job at a time (phase5b_seq.sh, resumed distill + harvest); harvest resume keeps code counts; on-GPU frames → training at 94-96% GPU · dc8e9c0
- 2026-09-25 · Roadmap Phase 5c (GPU throughput) added: training faster, but rollouts/eval are CPU-physics-bound with the GPU at 2-17%; profile → inference/compile → overlap → GPU physics feasibility · fa83335
- 2026-09-24 · GPU acceleration: teacher targets cached once per dataset on GPU (226→160 s / 2k steps), image frames on-GPU when they fit; Phase 5b split into 3 parallel tracks (vision / RL+re-eval / figures) · f6dfcdd
- 2026-09-24 · M5b.2 not met: shifted-pose DAgger 2 rounds, id_hard 61/64.5% vs 72% start; diagnosed as fine-placement stalls; best privileged stays dagger_diff/round_1; M5b.3 running · d05343a
- 2026-09-24 · Pushed `feature/imitation` through M5b.1 (0cccd91) + committed the finished run/eval logs (not the live `dagger_shift`); PR to main opened; M5b.2 still running; 90 fast green · this commit
- 2026-09-24 · M5b.1 done: diffusion DAgger round 1 = 98.0/72.0/72.5 (+1.74 SE over dagger_v2); round 2 kept by the old rule but lost id_hard, so round 1 carried forward (`pick --carry-sets`); M5b.2 running · 0cccd91
- 2026-09-24 · imitation.md §2.3 exact index map (base env 141 → 139), cameras, consumers; figure generator `imitation.viz.report_figures`; diffusion sampling made deterministic (fixed initial noise; same ckpt scored 160 then 157/200); M5b.1 restarted on it · 6d8f101
- 2026-09-24 · Phase 4/5 scope gaps closed in code (A0): dict obs + 128² main cam, lazy WindowSampler, per-camera normalizers, PolicyTeacher in the Phase 3 loop (distill.py retired), stratified multi-policy harvest, unstable tagging + per-episode subsampling, critic probe, shifted-pose seeds + --score-sets; M4.1 re-marked ◐; 87 fast green · 820b5e2
- 2026-09-24 · Dated report `reports/2026-09-24-phase1-5.md` + demo videos committed + shareable page; roadmap Phase 5b added (M5b.1-5b.5); next action M5b.1; 79 fast green · f75d218
- 2026-09-24 · Phase 4 done: vision distill (97.5/90.5/31.5, recovery margin not met), detector 91.8% agreement; half-fold demos rendered (DAgger + vision, 4/4 each); status rewritten; 90 fast + 10 slow green · 3552e37
- 2026-09-24 · M3.2 done: dagger_v2 3 rounds, recovery 38.5→65.0%, ID 97%, id_hard flat; detector trained (95% val frame acc) · 4e22261
- 2026-09-24 · Phase 4/5 code: vision success detector, RL transitions + IQL + harvest, vision-aware demo, image-inclusive version hash; vision BC trained (val 0.066), v1_failures_img rendered; driver `runs/phase3to5.sh` chained after dagger_v2; 79 fast green · b72fe71
- 2026-09-24 · Found + fixed DAgger teacher label bug (shadowing teacher) and eval nondeterminism (padded predict); dagger_v1 marked invalid; Phase 4 plumbing (rig, image store, v1_img, vision policy, distill loop); 75 fast + 9 slow green · 91254d9
- 2026-09-24 · M2.1 (BC 2 seeds, n=200) + M2.2 (mechanistic histograms) + M2.3 (obs ablation) done; docs/pipeline.md added; DAgger + diffusion launched · 5b02a77
- 2026-09-24 · M2.2 code (mechanistic failure codes + `perturbed_failures`, collision-free eval versions), M2.3 plumbing (`--obs-subset`, `obs_mask` buffer), M3.1 done (hashed val split, expert-only dense targets, terminal-only padding, SE keep/stop at n=200, `--resume`, clamp logging), `imitation/demo.py`; 71 fast green · 33067c6
- 2026-09-23 · M1.8 done, Phase 1 complete: froze v1 (384 eps, `769dd372719c`) + v1_failures (16); clean 271/279, recovery 113/121 · 212469d
- 2026-09-23 · M1.7 gate passed: expert id_easy 100/100, id_hard 97/100, recovery 96/100; check_resync 92/94/93 of 100 · 08eeda3
- 2026-09-23 · M1.6 done: 20-ep smoke chain green (collect 19+1 fail split, train 600 steps, evaluate n=14×3, dagger 1 round froze smoke_dagger_r1 with 332 labels); no code change needed · a0f92ce
- 2026-09-23 · M1.5 done: cached IK scratch (no speedup: 81-83 s → 82-93 s, identical hash); fixed cross-episode IK-rng leak that made collection nondeterministic; benchmark rows identical 48/50; 90 fast + 10 slow green · c95c431
- 2026-09-23 · M1.4 done: failures → `<version>_failures` in collect, `bc_episodes` filter in train (`--allow-failures`); 28-ep all-perturbed collect split 25/3; 61 fast green · c64e9ce
- 2026-09-23 · M1.3 done: `terminated`/`truncated`/`discount` arrays in schema + validator (only-last-step, exactly-one, discount mask); `unstable` → truncated; 14-ep real collect validates; 90 fast + 10 slow green · 1fb0e0e
- 2026-09-23 · M1.2 done: streaming/resumable `DatasetWriter` + `collect --resume`, seed-named npzs, guard on unfrozen versions; real 20-ep collect killed at 60 s kept 18, resume finished 20/20 unique, hashes verify; 59 fast green · fa31797
- 2026-09-23 · M1.1 done: 139-D obs (per-stage goals, no one-hot, +stage/settle), `cloth_offset_xy` recorded, goals in snapshot; 63 tests green; expert benchmark unchanged 48/50 · e847847
- 2026-09-23 · Sim verified against docs before M1.1: pins, 57+2 tests, benchmark 48/50 reproduced, all §2.2 defects confirmed · no code change
- 2026-09-23 · Phase 0: imitation + DAgger pipeline committed, env pinned, tracking docs · a5aa63f

# Status

**Updated:** 2026-09-25 · **Branch:** `feature/imitation` · **Phase:** 5b/5c final queue (**paused** 15:16)

One-screen answer to "where are we". Update at the end of **every work pass** (see
`CLAUDE.md`), and add a line to the pass log at the bottom. Full plan in
[roadmap.md](roadmap.md); spec in [imitation.md](imitation.md); numbers in
[results.md](results.md).

---

## Where we are

Phases 1-5 have run end to end. The half-fold demo is rendered from pipeline-trained
policies (`outputs/imitation/demo/half_fold_dagger.mp4`, `half_fold_vision.mp4`).

| policy | id_easy | id_hard | recovery | where |
|---|---|---|---|---|
| expert (ceiling) | 100% | 97% | 96% | M1.7 |
| BC chunk-MLP | 98.0 | 69.5 | 38.5 | M2.1 |
| BC diffusion | 99.5 | 80.0 | 51.0 | M2.4 |
| **DAgger (privileged)** | 97.0 | 64.5 | **65.0** | M3.2 |
| IQL from DAgger | 94.0 | 67.5 | 45.5 | M5.3 ❌ |
| vision BC | 97.0 | 94.5 | 21.5 | M4 |
| **vision distilled** | 97.5 | **90.5** | 31.5 | M4.2 |

Not met: M5.3 (IQL worse than DAgger: the critic can't rank actions on single-policy
data) and the recovery half of M4.2 (the vision student is 33 pp behind its teacher).
All in `results.md`.

## Next action

**PAUSED 2026-09-25 15:16 (laptop closed); nothing is running.** Resume with a visible task:

    sh outputs/imitation/runs/phase5b_resume.sh

It runs sequentially (≤10 workers):
1. DAgger r4 round 1 (`--resume` reuses the frozen rollouts `v1_dagger_diff_r1_dagger_r4_r1`, so only the train and eval steps run).
2. Evaluations: `diff_v1_s2`, `vision_t0_128_s1` at replan 2, `iql_v4` at replan 8 and 4.
3. Demos at replan 4 / 2.
4. M5c.1 profile re-run (the label look-ahead is now timed).
5. M5c.3 part 2.

Done before the pause and not yet in results.md:
- `diff_v1_s1` eval, 99.5 / 83.0 / 62.0 (`runs/diff_v1_s1.eval.log`).
- M5c.3 part 1, 57 min over the DAgger rollouts with overlapping lanes: GPU mean 32%, busy (>10%) in 64% of 5 s samples (`runs/m5c3_gpu.csv`, `m5c3_window.csv`).

Then:
- Record these results.
- Close M5b.4 and M5c.1-3.
- Write `docs/reports/2026-09-2x-phase5b.md` and republish the page (same URL).

## Pre-VLA milestone tracker

Everything above Phase 6 must be ✅ or ❌-closed-with-evidence before the VLA starts
(roadmap rule). Updated every pass.

| milestone | state | closes via | where it runs |
|---|---|---|---|
| M4.1 image plumbing | ✅ 2026-09-25 | vision BC on the finished path, 94.5 / 91.0 / 26.5 | done |
| M4.2 distillation margin | ❌ closed, margin stated (−3 / +20 / −42.5 pp) | — | done |
| M5.3 offline RL | ❌ (v1) | M5b.4 result, or drop RL with evidence | queue step 2 |
| M5b.1 diffusion DAgger | ✅ | — | done |
| M5b.2 shifted poses | ❌ closed, diagnosed | follow-up M5b.6 | — |
| M5b.3 vision recovery | ❌ closed: distillation transfers teacher weakness | follow-up in M5b.6 sweep | done |
| M5b.4 offline RL retry | ◐ iql_v3 collapsed (54/24/18: actor fed 75% failures — sampling bug); corrected iql_v4 training | iql_v4 eval (replan 8 and 4) vs the privileged best, else drop RL | trained; eval in `phase5b_resume.sh` |
| M5b.5 hygiene / determinism | ✅ 2026-09-25: repeat identical; M2.1/M2.3 re-evals inside original intervals | — | done |
| M5b.6 fine placement | ❌ closed: replan 4 → privileged 99.5/77.0/79.5, vision replan 2 recovery 41%; id_hard < 85% | adopted as operating points | done |
| M5c.1 profile | ◐ instrumented (`rollout(stats=)`, `imitation.viz.profile`) | profile ran (eval ~36-41 ms/step wall, physics ~50%, inference ≤7%); re-run with label timing | `phase5b_resume.sh` |
| M5c.2 GPU inference wins | ◐ CUDA-graph sampler done: 12.7× faster, bit-identical | share of rollout time from M5c.1 | `phase5b_resume.sh` profile |
| M5c.3 overlap GPU/CPU | ◐ `imitation.cpu_slot` (cross-process CPU lock around rollouts) done + tested | measure GPU busy % across a DAgger run | part 1 done (busy 64%, mean 32%); part 2 in `phase5b_resume.sh` |
| M5c.4 GPU physics feasibility | ❌ closed: slower than 1 CPU core, trajectory drifts > 1 cm by step 15 | — | done |
| M5c.5 batched GPU rendering | ❌ closed (gate M5c.4 failed) | — | done |

## Environment

| | |
|---|---|
| pins | `imitation/requirements.txt` — mujoco **3.10.0**, so101-nexus **0.4.8**, numpy 2.5.1, gymnasium 1.3.0 |
| torch | 2.13.0+cu130, **CUDA available** |
| tests | 96 fast + 10 slow (`pytest -m "not slow"` / `-m slow`) |

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

Collection log: `outputs/imitation/collect_v1_m1_8.log`. The earlier failed attempt is
kept as `outputs/imitation/collect_v1.log`.

## Best checkpoint

- privileged: `outputs/imitation/runs/dagger_v2/round_3/final.pt` (97.0 / 64.5 / 65.0)
- sensor-only: `outputs/imitation/runs/distill_v1/round_2/final.pt` (97.5 / 90.5 / 31.5)
- success detector: `outputs/imitation/runs/success_v1/detector.pt` (91.8% agreement)

Weights are gitignored; each run's `run.json` / `history.json` is committed.

## Open blockers and known defects

Ordered by what they block. Each is a roadmap milestone.

| # | defect | blocks | milestone |
|---|---|---|---|
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

- 2026-09-25 · M5b.5 ✅ (determinism repeat identical; M2.1/M2.3 re-evals inside original intervals); detector v2 on 128² (92.6% agreement); M5c.1 profile ran; pipeline.md Data section; expert label look-ahead now timed in profiles; `imitation.viz.gpu_busy`; diffusion seeds 1-2 + vision seed 1 trained; final queue **paused** at 15:16 (resume script) · 96 fast tests pass · COMMIT
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

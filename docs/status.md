# Status

**Updated:** 2026-10-23 · **Branch:** `chore/cleanup` (to be merged into `main`) · **Phase:** Q, the quarter fold on
Isaac (revised plan of 2026-10-22, steps 1 to 3)

One-screen answer to "where are we". Update it at the end of every work pass (see `CLAUDE.md`) and add a line to the
pass log. The plan is in [roadmap.md](roadmap.md), every number in [results.md](results.md), the re-evaluation and
the revised plan in [reviews/2026-10-22-quarter-fold-reevaluation.md](reviews/2026-10-22-quarter-fold-reevaluation.md).

---

## Where we are

**Goal (user, 2026-10-14):** an RL policy in Isaac Sim that does a quarter fold with the friction gripper (no weld),
with at least one recorded clean demo. Bar: mean ≤ 2.0 cm and edge ≤ 3.0 cm on gated episodes (flat on the table,
released, still), user 2026-10-22.

- **Half fold** reaches the half-fold bar (1.0 / 1.5 cm) in about 1 of 64 episodes, with the scripted base plus one
  touch-up (seed 930005, 0.95 / 1.45 cm) and with the RL policy p8 (seed 980008, 0.43 / 1.13 cm). Videos in
  `docs/reports/media/isaac_half_fold_pass_*.mp4`. RL shows no average gain over the scripted base (p8 hunt 2.65 cm,
  base hunt 2.62 cm).
- **Quarter fold** has not passed. Every one-arm fold 2 tried failed. The first two-arm runs on the `quarter2` layout
  (n = 2, 2026-10-23) went badly: fold 1 broke in one, and three of four fold-2 grasps slipped.
- **Gentle release** (user, 2026-10-22) is the place-servo default and has not run in Isaac yet. The jaw opens
  0.02 rad per step to 0.3 rad with the arm still, the arm lifts 2 mm per step, and the jaw opens fully 5 cm up.
  It adds about 30 steps per release, so window runs need `WORLDFOLD_WINDOW_STEPS=400`.
- **Repo cleanup** (user, 2026-10-22/23): retired code deleted, the Isaac code split into `isaac/sim`, `isaac/experts`,
  `isaac/rl` and `isaac/tools`, `mujuco/` renamed `mujoco_sim/`, run outputs and videos untracked. Tag
  `pre-cleanup-2026-10-22` has everything as it was. No WATcloud jobs are queued (the user cancelled 711774-711776).

The pre-VLA milestones are all closed (✅ or ❌ with evidence, 2026-09-26). Best policies: DAgger diffusion at replan 4
(privileged, 99.5 / 77.0 / 79.5 on id_easy / id_hard / recovery) and the vision student at replan 2 (sensor-only,
94.0 / 91.0 / 41.0). Tracker in [roadmap.md](roadmap.md), report in [reports/2026-09-25-phase5b.md](reports/2026-09-25-phase5b.md).

## Next action

1. Merge `chore/cleanup` (the WATcloud check, job 712165, ran the new layout end to end).
2. Find why `quarter2` folds fail: read the per-arm grasp traces of 712165, then run 8 seeds with the gentle release
   off and on. If the tilted pinches (A0 39°, S 34.5°) are what slips, try the slide grasp (`GRASP=slide`) on those
   two. Target: fold 2 landing within 3 cm on gated episodes, with video.
3. The two cheap checks, rerun with the gentle release: adhesion 0 vs 0.1 on the half fold, and the one-arm fold 2
   with no drag and the idle arm parked.
4. Then RL (residual or corrector) on fold 2 from fold-1 snapshots, judged on settled gated paired episodes against
   the scripted base.

## Environment

| | |
|---|---|
| pins | `imitation/requirements.txt`: mujoco **3.10.0**, so101-nexus **0.4.8**, numpy 2.5.1, gymnasium 1.3.0. The root `requirements.txt` adds stable-baselines3, pyyaml and matplotlib |
| Isaac | Docker image from `isaac/Dockerfile` on WATcloud (`isaac/watcloud.sbatch`): Isaac Sim 5.1.0, LeHome a805ad2, IsaacLab fork 69f6fa5. On Windows, `.venv-isaac` from the hashed locks in `isaac/` (Kit on D3D12) |
| tests | 226 fast on macOS 2026-10-23, 1 failing there (#15), plus 12 slow, all passing (`pytest -m "not slow"` / `-m slow`) |

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
| isaac_v1_friction / _failures | 171 / 29 | friction expert (F3b defaults), seeds 0–199, 50% knock_arm, state-only | `71a993b12a3c` / `be7f7e50083b` | Phase F4; clean 93/101, bumped 78/99 |

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
| 16 | WATcloud jobs rebuild the image each time (~10 min on trpro, 29 min on thor 2026-10-20): `docker save` kills the job (memory); rsync to WATcloud fails mid-protocol since 2026-10-15 (code goes over `tar \| ssh`) | long Isaac jobs | Q |
| 15 | `tests/imitation/test_seeds.py::test_mujoco_eval_sets_unchanged` fails on macOS arm64: `shifted_pose(200000)` differs from the Windows-pinned value by 3.5e-18 (1 ulp) | fast suite on the Mac | — |
| 13 | `cloth_fold_rl/fold_env.py` puts `mujoco_sim/` first on `sys.path`; `mujoco_sim/tests` then shadows the root `tests` namespace for later-collected test modules that do `from tests.imitation...` (worked around in `test_backend.py` with lazy imports) | test collection order | — |
| 12 | `imitation.cpu_slot` isn't fair: a lane that releases and re-takes it between eval sets starves a lane polling every 2 s (lane A waited 37 min) | parallel queues | — |
| 11 | `v1_dagger_v1_r1`/`_r2` frozen with bad teacher labels — never train on them | DAgger | — |
| 10 | `cloth_angles/tasks.py::QUARTER` diverged from `quarter_fold_env.STAGES` (3 ways) | world-model work only (parked) | M7.3 |

## Parked

- **World model** (`cloth_angles/`) is off the critical path by decision. Its task definition has diverged (#10) and
  must be reconciled or deleted before any result from it counts.
- **VLA** enters at Phase 6 as a subgoal proposer, not an action labeler.
- **F4 DAgger on `isaac_friction`** waits behind Phase Q.
- **Roy's grasp port** is paused. Its slide grasp stays in `isaac/experts/fold_expert.py` (`GRASP=slide`) as the
  fallback for steep pinches.

## Pass log

Newest first. One line per work pass: date · what changed · commit. Older lines are in git history.

- 2026-10-23 · **Cleanup check (WATcloud 712165):** new layout runs in Isaac end to end; first `quarter2` folds, n = 2: fold 1 broke in 1/2, fold-2 grasps held 1/4, both endings invalid (results.md) · (this commit)
- 2026-10-23 · **Repo cleanup:** run outputs and videos untracked (17 showcase videos kept), retired code deleted (weld, once-per-fold RL, Roy profiles, old quarter profiles, one-off probes, IQL, verifier, early PPO/MolmoAct), `isaac/` split into subpackages, `mujuco/` → `mujoco_sim/`, docs rewritten; fast suite 226 passed + #15, slow 12 passed · (this commit)
- 2026-10-22 · **Gentle release:** jaw opens slowly part way, slow straight lift, full open above (env `jaw_open` cap for the binary gripper); isaac tests 72, fast suite 273 passed + #15 · (this commit)
- 2026-10-22 · **Revised plan, steps 1-3 started:** gates + re-score (fa4c1ce); `quarter2` layout + two-arm fold 2 (28752bb); WATcloud jobs queued; fast suite 268 passed + #15 · (this commit)
- 2026-10-22 · **Re-evaluation:** independent audit + Roy + research; corrections in results.md; revised plan in reviews/ · (this commit)
- 2026-10-22 · **RL half fold at the bar:** p8 v12 hunt 1/64, seed 980008 0.43/1.13 cm with video, no touch-up · (this commit)
- 2026-10-22 · **Recorded pass + RL corrector:** scripted hunt 1/64 at the bar with video (930005); rounds don't help the edge; p7 cancelled (degraded); corrector built (62 isaac tests), c1 launched · b663cf2
- 2026-10-22 · **Touch-up + first pass:** sweep 3 (touch-up, 1/31 at the bar: 0.85/1.17 cm), p8 and video hunt launched · (this commit)
- 2026-10-22 · **Sweeps 1-2, p6 done:** best base latch + g-fold 1.2 (2.69 cm); p6 plateaued at 2.77; p7 + demo hunt launched · (this commit)
- 2026-10-21 · **Brainstorm + base fixes:** rigid top-layer shift (3.36 → 1.66 cm when removed), lifted corners, 3 base knobs, sweep command; p6, A/Bs and sweep 1 running · 22e82ae
- 2026-10-21 · **Error structure + p5:** base vs v1 demos (v1 −0.52 cm paired), error = fold line forming short, p5 launched from v1 · 050b9f1
- 2026-10-21 · **p4 learning:** 2.81 vs 3.22 cm at 10k steps; cold-scene restore found (warm-up added); demo command · 3959c8e
- 2026-10-21 · **p3 read, p4 launched:** bounded residual, scored step cap, bottom-shift charge, critic warm-up · (this commit)
- 2026-10-21 · **Snapshot frame bug:** p2's corruption detail (all particles 2.5 m off) pinned p1/p2 on cross-copy restores of world-frame snapshots; fixed, p3 launched (711562) · (this commit)
- 2026-10-20 · **p1 blow-up handled:** table floor, check-recover (not reproduced), park + corrupt-scene restart; p2 launched (711487) · (this commit)
- 2026-10-20 · **R1 + launch:** restore verified (sim nondeterministic), throughput 22 steps/s, drop no longer a failure, base A/B rejected, training run p1 launched (711482) · 1483199
- 2026-10-20 · **Closed-loop RL R0-R3:** R0 baselines (closed loop 3.61 cm vs open 6.10), s3e release failure found, snapshot/restore, window env, RLPD learner (subagent), driver; isaac/tests 52 passed, fast suite 253 passed + #15 · 99da9a4
- 2026-10-20 · **RL redesign + stack test:** plan `superpowers/plans/2026-10-20-closed-loop-rl.md` (3 research agents); `make_quarter_fold_env(fold2="stack")`, `quarter_fold_run.py --fold2/--overshoot` + per-arm stack score, `WORLDFOLD_EXPERT_PARAMS` bare words and passed into the container. Fast suite 233 passed, 1 failing (#15) · (this commit)
- 2026-10-19/20 · **Roy grasp port, steps 1–3 (paused by the user):** box pads, roy profile (on lehome), soft pinch IK, slide grasp, side camera + bench video, WATcloud GPU-by-UUID. Benches: pads + our pinch lift but don't hold (0/20); roy pinch 0/40; slide 1/40, 0/40, 0/20 — jaw faced half a turn round (IK blind to antiparallel axes), fixed with `jaw_weight`, re-run pending WATcloud. 30 isaac tests pass · (this commit)
- 2026-10-19 · **Plan:** port Roy's grasp into our env instead of merging his branch (user): `superpowers/plans/2026-10-19-roy-grasp-port.md`, 5 steps with tests and pass bars. No code · (this commit)
- 2026-10-18 · **Diagnosis, no code:** the square quarter fold looked infeasible because layout_search capped pinch tilt at 16.5° (our corner pinch); Roy's LeHome layout needs 30–46° and works with his slide-and-press grasp, plus a drag between steps. Grasp/env differences vs Roy inventoried. Tests not run (no code change) · (this commit)
- 2026-10-14/15 · **Phase Q overnight:** layout search (reach, tilt, collision meshes), quarter profiles, position gripper, alignment score, fold-move PPO (untested on Isaac), WATcloud container fixes; 20 Isaac episodes on new layouts, 0 quarter folds; no collision-free two-arm layout at 24 cm. Fast tests: 221 passed, 1 failed (#15) · (this commit)
- 2026-10-14 · **Quarter fold RL: survey and plan, no code.** Plan `superpowers/plans/2026-10-14-quarter-fold-rl.md`. Found: stage 1 out of reach on the Isaac layout (37 / 40.5 cm vs 32 cm), gripper binary only, no whole-cloth alignment number for the friction expert, WATcloud unreachable (#16), Modal CLI works. Fast tests on macOS: 205 passed, 1 failed (#15, 1-ulp platform diff), 6 skipped · (this commit)

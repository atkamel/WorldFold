# Status

**Updated:** 2026-11-08 · **Branch:** `fix/half-fold-audit` · **Phase:** Q, the quarter fold on
Isaac (revised plan of 2026-10-22), half-fold expert fixes first

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
- **Audit (2026-11-07, results.md).** MuJoCo's weld half fold never met the bar either. It scores 2.43 cm with
  0/97 passing, so it only passed the 5 cm corner test. Isaac's friction expert scores 3.32 cm. The Isaac expert has
  verified defects: the pinch lands outside the corner, the heights treat the site as the fingertip, the place can't
  finish, and hold re-aims the jaw. The largest term in both sims is a rigid shift of the top layer of about 2.2 cm,
  set before release. Removing it takes Isaac to 1.35 cm.
- **Quarter fold** has not passed. Every one-arm fold 2 tried failed. The first two-arm runs on the `quarter2` layout
  (n = 2, 2026-10-23) went badly: fold 1 broke in one, and three of four fold-2 grasps slipped.
- **Gentle release** (user, 2026-10-22) is the place-servo default. The jaw opens 0.02 rad per step to 0.3 rad with
  the arm still, the arm lifts 2 mm per step, and the jaw opens fully 5 cm up. It adds about 30 steps per release, so
  window runs need `WORLDFOLD_WINDOW_STEPS=400`. On 20 scripted half folds (job 738561) 35 of 40 arm releases were
  clean and no corner was left up at the end (results.md); not A/B'd against the old release yet.
- **Repo cleanup** (user, 2026-10-22/23): retired code deleted, the Isaac code split into `isaac/sim`, `isaac/experts`,
  `isaac/rl` and `isaac/tools`, `mujuco/` renamed `mujoco_sim/`, run outputs and videos untracked. Tag
  `pre-cleanup-2026-10-22` has everything as it was. No WATcloud jobs are queued (the user cancelled 711774-711776).
- **World model on Isaac** (PR #20, user 2026-11-05, parked 2026-11-07): the pipeline works end to end in Isaac on
  the half fold (play + scripted data with RGB/depth frames, 119,764 transitions; model + learned grasp head; reward
  check 1.5e-6; live predictor; evaluator). The model beats persistence on held-out windows (task, grasped, 10 steps:
  24.3 vs 39.1 mm), the grasp head predicts releases but not slips, and imagination raised its return by letting go
  of the cloth. Parked because folding fails on grasp physics the model can't predict (results.md, 2026-11-06).

The pre-VLA milestones are all closed (✅ or ❌ with evidence, 2026-09-26). Best policies: DAgger diffusion at replan 4
(privileged, 99.5 / 77.0 / 79.5 on id_easy / id_hard / recovery) and the vision student at replan 2 (sensor-only,
94.0 / 91.0 / 41.0). Tracker in [roadmap.md](roadmap.md), report in [reports/2026-09-25-phase5b.md](reports/2026-09-25-phase5b.md).

## Next action

1. Half fold, batch S then batch B of the plan
   ([superpowers/plans/2026-11-08-overnight-half-fold.md](superpowers/plans/2026-11-08-overnight-half-fold.md)). Batch S
   (user, 2026-11-08) tries a stiffer cloth on the audit base r0 (kept jaw, latch + g-fold 1.2, 2.07 / 3.65 cm, 1/32):
   bend-only springs up and down, bend + shear, and a 51x51 mesh at 6 mm, plus a drape probe that reads back the springs
   and measures 8 cm of overhang, for matching to the real cloth. No cloth is promoted without a repeat and a drape match.
   Batch B (the 17 mm floor under `PLACE_ALONG` 3-4.5 cm, `CARRY_FLOOR`) stays on r0's cloth. Launch commands in the plan
   for B. S launched 2026-11-09 08:55 as 743600-743602, the drape probe as 743603.
2. `quarter2` fold 2 rerun with traces. The right arm missed the stack S because S sat past an upright pinch's reach
   and the IK weights pushed it onto a flipped wrist (results.md). `TWO_IK` (fold 2's right arm at orientation weight
   0.005, jaw weight 0.02) reaches 24/24 offline within 5 mm, untested in Isaac. Run the 742655 job again with it
   (command in the plan). Target: fold 2 landing within 3 cm on gated episodes, with video.
3. The two cheap checks, rerun with the gentle release: adhesion 0 vs 0.1 on the half fold, and the one-arm fold 2
   with no drag and the idle arm parked.
4. Then RL (residual or corrector) on fold 2 from fold-1 snapshots, judged on settled gated paired episodes against
   the scripted base.

## Environment

| | |
|---|---|
| pins | `imitation/requirements.txt`: mujoco **3.10.0**, so101-nexus **0.4.8**, numpy 2.5.1, gymnasium 1.3.0. The root `requirements.txt` adds stable-baselines3, pyyaml and matplotlib |
| Isaac | Docker image from `isaac/Dockerfile` on WATcloud (`isaac/watcloud.sbatch`): Isaac Sim 5.1.0, LeHome a805ad2, IsaacLab fork 69f6fa5. On Windows, `.venv-isaac` from the hashed locks in `isaac/` (Kit on D3D12) |
| tests | macOS 2026-11-08: fast 276 passed, 5 skipped (#15 deselected), isaac 99 passed; macOS 2026-10-29: slow 12 passed, 8 skipped (Isaac runtime unavailable) |

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

## Parked

- **World model on Isaac** (2026-11-07, user): resume once fold 2 works and RL is limited by Isaac's speed. First fixes
  then: split the imagined return by reward term to find what pays without a grasp, and soften the grasp head's
  change weighting (it over-predicts onsets). Data: `outputs/cloth_angles/isaac_wm_{task,play1,play2}` (frames only
  on WATcloud, about 6 GB); model `outputs/cloth_angles/isaac_wm_model/state_l1_seed0.pt`.
- **VLA** enters at Phase 6 as a subgoal proposer, not an action labeler.
- **F4 DAgger on `isaac_friction`** waits behind Phase Q.
- **Roy's grasp port** is paused. Its slide grasp stays in `isaac/experts/fold_expert.py` (`GRASP=slide`) as the
  fallback for steep pinches.

## Pass log

Newest first. One line per work pass: date · what changed · commit. Older lines are in git history.

- 2026-11-09 · **WATcloud back (08:51), batch S launched:** 743600-743602 (9 cloth configs on r0) and the drape probe 743603; the outage (blocker 21, 22:55 on 11-07 to 08:51 on 11-09) closed; no code change · (this commit)
- 2026-11-08 · **Stiffer cloth, batch S (user):** cloth knobs `bend_stiffness`, `shear_stiffness`, `spring_damping`, `cloth_subdiv` (FRICTION_GRASP, onto LeHome's garment_config); `isaac/tools/drape_probe.py` (8 cm overhang droop, spring / mass readback); batch S designed and reviewed (opus: springs not saturated, bend-only configs, hold promotion until a real-cloth match); round 2a fold-shape diagnostics in results.md; WATcloud still down; fast 276 passed + 5 skipped (#15 deselected), isaac 99 passed · (this commit)
- 2026-11-08 · **Overnight run blocked by the WATcloud outage:** no batch ran (blocker 21); before it, `CARRY_FLOOR` (carry and place above a 17 mm floor, lower before release), `quarter2` fold 2's right-arm IK fix (`TWO_IK`, `_carry_w`) and arm joints in the traces landed in a528405, batch B reviewed; launch commands for batch B and the fold-2 rerun in the plan; fast 268 passed + 5 skipped (#15 deselected), isaac 91 passed · (this commit)
- 2026-11-07 · **Audit closed, overnight run started (user):** round 2 read (results.md); the kept jaw is permanent (knob gone), the stall exit deleted, the 10 mm pinch, 17 mm floor and 5 cm shift cap rejected; defects 17-20 closed (19 and 20 fixed, 17 and 18 measured and left as they were); `PLACE_ALONG` / `PLACE_INWARD` added for batch A; overnight plan written; isaac tests 86 passed · (this commit)
- 2026-11-08 · **Audit round 1 read, round 2 launched:** the kept jaw halves the edge; LeHome's jaw actuator doubled carry drops and is reverted (MuJoCo's armature stays on the jaw); floor, shift, pinch and adhesion results in results.md; `quarter2` traces: fold 2's right arm never reaches the stack, the gentle release made no difference to fold 1; round 2 742669 / 742670 and the reset check 742671 submitted; isaac tests 86 passed · (this commit)
- 2026-11-07 · **Audit fixes + sweep (user):** hold keeps the pinch's jaw (`KEEP_JAW`), the place ends once the corner is over its goal and the site stops coming down (`PLACE_STALL_EXIT`), the jaw gets LeHome's own actuator under its drive (no MuJoCo armature), resets zero the cloth's particle velocities, an optional layer-shift correction in hold (`SHIFT_GAIN`, off), pinch comments corrected (the inset's sign is intended; the site lands ~7 mm further out than planned), demo rows log reset flatness, the site's offset at the close, the sheet's move before the carry and the corner's height in the jaws; WATcloud 742650 / 742651 submitted; isaac tests 86 passed, fast 263 passed + 5 skipped (#15 deselected) · (this commit)
- 2026-11-07 · **Half-fold audit, no code (user):** three opus reviewers plus lead re-checks. MuJoCo's weld half fold scores 2.43 cm on the alignment score with 0/97 passing (lead re-run 1.97-2.18 cm), so it only ever passed the 5 cm test. Isaac's friction expert scores 3.32 cm. Defects 17-20 added; a 2.2 cm rigid top-layer shift is the largest term in both sims; the 2026-11-05 "mid-carry slips" corrected (flag artefacts in place / hold); tests not run (docs only) · (this commit)
- 2026-11-07 · **Quarter-fold traces:** 712165 reviewed (no per-step traces existed; seed 1's fold-2 corners both left the jaws mid-carry by the expert's own rule, seed 0's fold 1 left the sheet 10.5 cm up); `quarter_batch` writes a per-seed trace (stage, cloth, per arm phase / carried vertices / jaw / command / tilt / site / corner / drop); nothing run in Isaac; fast 257 passed + #15 · (this commit)
- 2026-11-07 · **World model full data, then parked (user):** 738561-63 done (260 episodes, 119,764 transitions); model beats persistence on every held-out group, grasp head predicts releases not slips, reward check 1.5e-6, imagination exploits (return up, grasp rate 0.22 → 0.01); release check on the gentle release (35/40 clean); merged main (#21) into `isaac`; fast 256 passed + #15 · (this commit)
- 2026-11-05 · **World model smoke on Isaac:** collection (738554), live predictor (738556) and evaluator (738557) ran, python exit 0; train, benchmark, reward check (1.3e-6), clone + imagination ran locally on the smoke store (results.md); full collection submitted as 738561-63; no code change · (this commit)
- 2026-11-05 · **World model phase B (sim-free):** learned grasp head (change-weighted BCE, trained with the model, grasp metrics in the benchmark), imagination on HALF with the head's soft flags and per-joint bias logging, FoldReward check vs env rewards (<1e-4 on a scripted fold), Isaac evaluator with a shared release + settle finish before the gate; weld rule (`gripper_transition`, `weld_corners`, `Task.grasp_radius`) deleted; fast 256 passed + #15 · (this commit)
- 2026-11-05 · **World model phase A (PR #20 continued):** single-corner goal out of reach on friction (45-52 cm vs 44.5 cm); code moved to `isaac/world_model/` (two-arm adapter, play + half-fold collector with RGB/depth frames, online runner); union trainer + restored 1/5/10-step benchmark in `cloth_angles/`; HALF/QUARTER built from STAGES (#10 closed, M7.3); rig main + wrist cameras render depth; fast 240 passed + #15, slow 12 passed / 8 skipped · (this commit)
- 2026-11-05 · **Review, no code:** PR #20 never ran in Isaac (train script import error, random-action collector, runner scores only, mislabelled corner metric, no imagination path); quarter fold: fold 2 scripted only, RL wired for the half fold only; PR tests 7 passed, quarter fast tests 35 passed · (this commit)
- 2026-11-05 · **SO-101 hardware:** real-arm scripts from `lerobot-env` into `hardware/so101/` (servo limit calibration, idle pose, web control panel with 3D twin, phone teleop); no tests touched · (this commit)
- 2026-10-29 · **Sync Isaac branch with main:** merged `origin/main` (`24515e0`), retained the branch's world-model tools and updated them for `isaac.sim`; fast 226 passed / 1 known float failure / 5 skipped, slow 12 passed / 8 skipped · `24515e0`
- 2026-10-29 · **Online world-model integration:** added Isaac dataset trainer and pre-step predictor/scorer, plus a synthetic train-load-predict check; focused check passed, fast 227 passed / 1 known float failure / 5 skipped. Real Isaac execution remains blocked by missing runtime/data and unreachable WATcloud · (this commit)
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

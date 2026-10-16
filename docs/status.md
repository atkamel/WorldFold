# Status

**Updated:** 2026-10-15 · **Branch:** `feature/isaac-imitation` · **Phase:** Q rethink (R0–R3, square fold-over only); Q0 ✅ Q1 ✅ R0 ✅ R2 ⏸ paused (user, 2026-10-15)

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

**2026-10-02:** `origin/main` brought a new simulator, `isaac/`. It is the cloth-fold env on
LeHome's Isaac Sim 5.1 stack, with a friction grasp and no weld. Phase I (roadmap) makes this
pipeline run on it, at pilot scale and locally on Windows. Grasp tuning and the full-scale
retrain come later (IG, IS).

I0.1 ✅: main is merged into `feature/imitation`, and MuJoCo is unchanged (expert benchmark
rows identical to M1.5). Phase I work continues on `feature/isaac-imitation`.

## Next action

**2026-10-15, Phase Q rethink** (user: square only, i.e. the fold-over with both arms on the same short edge; fold 1 must
be accurate; no weld anywhere; add a reset skill and a safe-failure benchmark). Plan: the approved plan
`ok-i-want-to-noble-fairy.md` (Claude plans folder), roadmap rows R0–R3.
- **R0 ✅ (diagnostic):** `WORLDFOLD_QF_START=clean` (`isaac/diag_clean_fold.py`, temporary) writes a perfect half
  fold, 14.8 cm wide with a 0.5 cm split, and starts at stage 1.
  - The fold-over from it scored **0/16**, so fold 2 is broken on its own (results.md R0).
  - On 15/16 seeds the right arm holds v5 while the left arm hovers over the v0+v10 stack, near its own base, and
    every descent drifts 7–9 cm off.
- **PAUSED (user, 2026-10-15)** while tracing the reversed layout with a 2.5 cm inward edge pinch (the run was stopped
  before any result). Nothing of this project is running. To resume:
  - `WORLDFOLD_QF_SECOND=reversed_square WORLDFOLD_QF_START=clean`
  - knobs `APPROACH_CRUISE_MAX=0.50,APPROACH_CHAIN=1,STAGE1_ARC_HEIGHT=0.08,STAGE1_ARC_ORIENT_W=0.01,STAGE1_PLACE_SHORT=0.055,STAGE1_RETREAT_TOOL=1,STAGE1_PINCH_INSET=-0.025`
  - `isaac/recovery_replay.py --task quarter_fold --no-knock --max-steps 1200 --trace-dir ...`
- **R2 reversed fold 1 ◐ (user 2026-10-15):** `reversed_square` (near half onto far half, the strip away from the
  bases). Offline, every right-arm square-up target is near-vertical.
  - From the clean start, the left arm's edge pinch on the far strip stalls 1.7 cm high and the closing jaw swings the
    arm 4–7 cm, so the rough fold fails 0/3 (results.md R2 reversed). The square-up was not reached.
  - Reversed fold 1 by the expert is not built yet.
- **R2 drag-only square-up ✗ (user 2026-10-15):** `left_fold_push` (push moves, left anchor at the fold line). 0/3 in
  traces. The 110 goal is 8 cm from the right base, so the push path stalls or swings, and a closed-jaw descent hooked
  the cloth.
  - Every R2 failure traces to fold 1 laying the strip on the arms' base line (results.md R2 drag-only).
  - Waiting on the user's layout call: fold 1 the other way, or the cloth started farther from the arms.
- **R2 square-up ◐ (user 2026-10-15):** `WORLDFOLD_QF_SECOND=left_fold_square` runs the rough left fold, then the right
  arm pulls the corner stack onto 110 and the crease end onto 115. From the clean start, n = 16: **1/16** (Wilson
  1–28).
  - The rough fold completes on 11/16; the stack square-up passes on 3/11; the crease square-up on 1/3.
  - The bottleneck is pulling the top two layers off a four-layer stack: G2 slips, or it drags the lower layers.
  - Next mechanisms (results.md): a lower pinch on the flap corner, the idle left arm anchoring, a drag-only pull.
- **R2 option 1 ◐ (user 2026-10-15: the left arm folds, the right anchors):** `WORLDFOLD_QF_SECOND=left_fold`.
  - The grip, the carry and the anchor's lift-off all work; the midpoint lands on target.
  - One success in replay (660001); n = 16 from the clean start: **0/16**, fold score 0.88.
  - A single mid-edge pinch lets the 15 cm edge collapse, so the corners land 5–10 cm inward of the square (results.md
    R2 option 1). Waiting on the user's call; options are in the pass log.
- **R2 (earlier, the two-arm fold-over):** the left arm's stage-1 grasp is fixed (knobs `APPROACH_CRUISE_MAX=0.50`,
  `APPROACH_CHAIN=1`, `STAGE1_PINCH_INSET=-0.025`). Both arms grasp on 13/16, and the left holds through the carry.
  - The right arm cannot pinch the west edge top-down (≥ 35° tilt; 65° at v5), and its grip slips at every lift.
  - Both arms on the west short edge is a reach limit of LeHome's layout (results.md R2). The task definition needs
    the user's call; options are in the pass log.
- **Earlier next step (user's call):**
  - R1: fold-1 accuracy. The user must confirm the bar first (proposed: width ≥ 13, split ≤ 2, crease shift ≤ 2,
    drift ≤ 2 cm).
  - R2: the left arm's descent onto the stack, iterated from the clean start, which is cheap and fold-1 independent.
    Its first check is IK reach at table height near the left base vs contact with the right arm.
- **State:** nothing running.

**Earlier (2026-10-14):**

**2026-10-14, Phase Q PAUSED** (user: "wrap up [Q1] and pause, I want to rethink the quarter fold process").
- **State:** nothing is running. No Isaac, training or thermal-guard processes, and no pause flag.
- **Plan:** [superpowers/plans/2026-10-13-quarter-fold.md](superpowers/plans/2026-10-13-quarter-fold.md); roadmap Phase Q.

**Closed:**
- **Q0 ✅ benchmark v2:**
  - Per-episode rows and per-stage metrics.
  - `sel_` / `final_` seed blocks.
  - Paired McNemar and bootstrap: `python -m imitation.stats a.json b.json [--flips]`.
  - Paired DAgger keep rule; G2 / G3 codes.
  - **Isaac noise floor:** rates reproduce (70/70, 44/45), but 24–25% of seeds flip run to run. At n = 100 the paired
    SE is ≈ 7 pp per set.
- **Q1 ✅ loss v2:**
  - The diffusion gripper head is kept: `sel_id_hard` 86 vs 70, `sel_knock_arm` 60 vs 44; pooled +16 pp, p = 0.001.
    S1 stalls go to 0.
  - Horizon and phase weighting are not kept (+1 pp, p = 0.89).
  - Val-loss checkpoint selection is rejected (`best_val` at 4k: 0/200).
  - A first head design (ε loss without the gripper dims) corrupted the joints; it was fixed and has a regression test.

**What the quarter-fold work found** (Q2 / Q3, results.md):
- **Second fold reach:** reachable with multi-seed IK. The single-seed "unreachable" was an IK artefact.
- **The bimanual fold-over is 0/16** after 6 mechanism attempts (A–F), overshoot and carry-speed knobs, and a
  straighten stage.
  - The root cause is fold 1 on this cloth: it wrinkles the strip to ~7 of 15 cm (crease 4–5 cm toward the arms,
    stack 5–6 cm split, `isaac/fold1_summary.py`).
  - So the second fold's pinch points end up 6–9 cm apart and the grippers collide.
  - A fully sequential fold-over is impossible by cloth geometry.
- **The parallel second fold** (each arm folds its crease end onto its own near corner) is **7/16 untuned**; slips
  at the crease bend are the remaining failure. It is now the default on LeHome backends (`WORLDFOLD_QF_SECOND`).

**Built and ready when the quarter fold resumes:**
- `--task quarter_fold` everywhere, and obs v2 (157-D, quarter fold only).
- Stage options: `STAGES_FOLD_OVER`, `STAGES_FOLD_OVER_STRAIGHTEN`, `STAGES_PARALLEL`.
- Expert changes (stage ≥ 1 only): multi-seed IK, lift gate, staggered approach, Cartesian approach, stage-start goals.
- QF0 knobs; the quarter-fold knock window (35–420).
- Tools: `isaac/recovery_replay.py --task --stop-at-stage --trace-dir`, `isaac/reach_check_qf.py`, `isaac/reach_map.py`.
- `scripts/q_retrain.ps1` is the Q4 fast-track pipeline. It is written but has not been run.

**Next:** the user's rethink of the quarter-fold process. Open questions for it:
- Which second fold.
- Whether to change fold 1 (e.g. a different first-fold motion) or the cloth/pad physics.
- Whether shift/straighten should become separate distilled skills.
- Q3 gate bars need the user's OK before any gate run.

**2026-10-08, F4 finals (BC only, 3.5 h plan):** privileged diffusion BC on `isaac_v1_friction` scored id_easy
81 / id_hard 71 / knock_arm 35 (n = 100), joint_noise 26 and overshoot 4 (n = 50). The expert scored 92 / 94 / 75 on the same sets. Stalls (S1) dominate the
clean and shifted sets; slips (G2) dominate the bumped and noise sets.

**Next:**
- Run DAgger, the stage that was cut. Resume command:
  `powershell -ExecutionPolicy Bypass -File scripts4_retrain.ps1 -NoVision -FinalN 100 -NoiseN 50 -Episodes 200 -BCSteps 20000`
- Test a best-val checkpoint (~6k steps).
- Then Adam's comparison row (`scripts/adam_baseline.ps1`).

**Phase F (user, 2026-10-07): replace the weld with a physical grasp on LeHome's physics, retrain, raise both
benchmarks.** Plan: [superpowers/plans/2026-10-07-physical-grasp.md](superpowers/plans/2026-10-07-physical-grasp.md);
roadmap Phase F (F0–F6). Rules: friction + adhesion ≤ 0.3, no attachments, pins or welds. The anchor comparison is
dropped (user, 2026-10-07). The comparison row is now Adam's as-built friction grasp (`scripts/adam_baseline.ps1`).

- **Found and fixed:**
  - **Slips:** `MUJOCO_ARM_DRIVE` was applied to the gripper too, at kp 998 and 3.35 N m. With LeHome's gripper drive
    (kp 17.8 / kd 0.6 / 10 N m) the corners stop slipping.
  - **Adhesion:** 0.1, LeHome's own value, is now the default.
  - **Overshoot:** the mean-miss correction regressed placement, so it was reverted to #9's.
  - **Reach:** at LeHome's cloth position the pinch can fold only about ±1.5 cm of cloth offset. MuJoCo's id_hard is
    0/200 reachable, so `isaac_friction` runs LeHome's own task sets (1 cm jitter, the Isaac id_hard ring and knock).
    **The weld (W5) numbers are on MuJoCo's sets, so the two are not directly comparable.**
- **Where the grasp stands** (F2 attempt M, n = 40, 1 cm jitter):
  - held 40/40 on both arms, placed 40/40 and 38/40, 36/40 env successes.
  - The friction profile's defaults are now: LeHome's gripper drive, closed jaw +0.05, adhesion 0.1, pad friction
    2.0, and the expert's `ALIGN_TOL` of 8 mm.
  - Gate 1 (config I, n = 300) held the right arm 93% (`f2_gate1_*`). The cause: the closing jaw swept an off-centre
    pinch. ALIGN_TOL fixes it.
- **F1 ✅** (`verify F1` PASS).
- **Fast path** (user, 2026-10-07 21:10):
  - F2 gate 2 was stopped; F2 closes on config M plus F3's full-task grasp and slip rates.
  - check_resync is informational.
  - F4 runs one BC seed, with DAgger selection on id_hard + recovery.
  - 2 × 8 envs per process.
- **F3 (n = 100 per set):** the expert scores 94 / 94 / **27** (id_easy / id_hard / recovery). **F4 is on hold**:
  the expert caps every student's recovery.
- **Recovery diagnosis** (3 traced knocked episodes, `outputs/isaac/recovery_friction/base.jsonl`):
  - Re-grasp descents take 30–85 steps (clean: 4–5).
  - Re-grasped corners slip.
  - The 400-step cap runs out.
  - The cause fits a pinch planned at table height into a corner lying on a fold.
- **Recovery A/B, done:** neither REGRASP (26/100) nor PINCH_FOLLOW_Z (29/100) helps. M1 = 20% is the knock itself
  dragging the cloth, so ~80% is the practical ceiling.
- **F4 paused at 01:45, 2026-10-08** (user: "we're getting ahead of ourselves if the expert is failing at
  recovery").
  - The partial collection (281/400, base expert) was discarded.
  - Its journals, the failure episodes and the logs are archived in `outputs/imitation/isaac_friction/f4_aborted/` as
    diagnosis evidence.
- **F3b result, as code defaults:** RESYNC_IK_HOME, DESCEND_RETRY = 2 and CART_ARRIVE (a Cartesian descend-arrival
  test). Tune sets, n = 50:

  | set | success |
  |---|---|
  | id_easy | 96 |
  | knock_arm | 84 |
  | joint_noise | 74 |
  | overshoot | 86 |
  | drop | 42 |

  - Drop is stuck after five attempts. The forced double drop leaves a crumpled half-fold
    (`docs/reports/media/f3b_drop_cart_expert.mp4`).
  - **The user skipped the forced-drop set entirely (2026-10-08).** It is not in the gate, the F4 data, the selection
    or the finals.
  - The servo expert (`isaac/servo_expert.py`) was not adopted.
- **In flight (user: done within 3.5 h of 15:00, 2026-10-08):** `scripts/f4_retrain.ps1 -NoVision -FinalN 100 -NoiseN 50
  -Episodes 200 -BCSteps 20000 -SkipDagger -NoDemos`. The F3 gate passed at 14:52.
  - **Stages:**
    - 200 state-only demos (50% arm-bumped), about 26 s per episode;
    - diffusion BC, 20k steps;
    - privileged finals: n = 100 on id_easy, id_hard and knock_arm; n = 50 on joint_noise and overshoot.
  - **ETA:** ~18:30.
  - **Cut for time:** DAgger, the vision student, the demo videos and n = 200 finals. Each can be added afterwards by
    re-running the script without the switch.
  - **Log:** `outputs/imitation/isaac_friction/f4/f4.log`. To resume, re-run the same command.
- **Afterwards, in idle Isaac windows:**
  - check_resync;
  - **Adam's as-built row:** `scripts/adam_baseline.ps1`, n = 50 per set. It is one CPU Isaac process, so it runs only
    while at most one other Isaac process does.
- **Heat cap** (keep under 94 °C):
  - ≤ 2 Isaac processes; `scripts/thermal_guard.py` alongside.
  - **8 envs per process allowed since 2026-10-07** (user). It was 4.
  - With cameras, fall back to 6 if VRAM overflows (16 GB).
- **Watch logs with `scripts/watch_logs.sh`, not `tail -F`.** A tail holds a Windows file handle. That blocks renames
  and the launcher's final `== exit` line.

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
| Isaac venv | `.venv-isaac` (gitignored): py3.11.9, Isaac Sim 5.1.0, torch 2.7.0+cu128, LeHome a805ad2 + IsaacLab fork 69f6fa5. Hashed locks are in `isaac/requirements-*-windows.lock`; Kit uses D3D12 (Vulkan crashes on driver 616.56); `N_ISAAC = 2` |
| tests | 236 fast (+ Isaac-only weld / profile / chain tests in .venv-isaac) + 14 slow, all passing (`pytest -m "not slow"` / `-m slow`); testpaths now include `isaac/tests`, `cloth_fold_rl/tests` |

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
| 13 | `cloth_fold_rl/fold_env.py` puts `mujuco/` first on `sys.path`; `mujuco/tests` then shadows the root `tests` namespace for later-collected test modules that do `from tests.imitation...` (worked around in `test_backend.py` with lazy imports) | test collection order | — |
| 12 | `imitation.cpu_slot` isn't fair: a lane that releases and re-takes it between eval sets starves a lane polling every 2 s (lane A waited 37 min) | parallel queues | — |
| 11 | `v1_dagger_v1_r1`/`_r2` frozen with bad teacher labels — never train on them | DAgger | — |
| 10 | `cloth_angles/tasks.py::QUARTER` diverged from `quarter_fold_env.STAGES` (3 ways), and from `STAGES_FOLD_OVER` (Q2). Left as is in Q2: the parked world model's own tests encode its version | world-model work only (parked) | M7.3 |

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

- 2026-10-15 · **Paused (user)** before the reversed-layout inward-pinch trace finished; docs only · see git log
- 2026-10-15 · **R2 reversed fold 1** (`reversed_square` stages, clean start follows stage 0's direction, `CLOSE_AT_ARRIVAL` knob): offline reach good for the square-up, but from the clean start the left edge pinch stalls high and swings on close, so the rough fold is 0/3. Tests: not-slow 251 passed / 5 skipped · see git log
- 2026-10-15 · **R2 drag-only square-up** (`left_fold_push`, `Move.kind="push"`, `set_push`, rough-fold role fix): 0/3 traces. Targets on the strip's near edge sit 4–8 cm from the arm bases (fold 1 puts the strip on the base line); a layout call goes to the user. Tests: not-slow 250 passed / 5 skipped · see git log
- 2026-10-15 · **R2 square-up stages** (`left_fold_square`, idle-arm parking, per-move `tol`, `SHORT_MOVE`, `WORLDFOLD_QF_MAX_STEPS`): n = 16 from the clean start **1/16** (rough fold 11/16, stack square-up 3/11, crease 1/3). The bottleneck is the stack square-up (G2 / drag). Tests: not-slow 249 passed / 5 skipped · see git log
- 2026-10-15 · **R2 option 1 (left fold + right anchor)**: `STAGES_LEFT_FOLD`, anchor moves, square `Stage.checks`, knobs D–H. The carry and anchor work, with 1 replay success (660001), but n = 16 is 0/16: the single mid-edge grip lets the corners collapse 5–10 cm inward. Next options for the user: a right-arm square-up stage, a two-grip left fold, or relaxed checks. Tests: not-slow 247 passed / 5 skipped · see git log
- 2026-10-15 · **R2 attempts A–C** from the clean start: the left arm's stage-1 grasp is fixed (cruise cap + chained IK + inward pinch: both arms grasp 13/16, left holds through the carry). The right arm's top-down reach ends ~0.33 m from its base (offline: ≥ 35° tilt anywhere on the west edge), so a two-arm west-edge fold-over is a layout limit; waiting on the user's task call. Tests: not-slow 243 passed / 5 skipped · see git log
- 2026-10-15 · **R0: clean half-fold diagnostic** (`isaac/diag_clean_fold.py`, `WORLDFOLD_QF_START=clean`, `scripts/r0_clean_fold.ps1`). Start geometry 14.8 cm wide / 0.5 cm split; fold-over from it 0/16 (G1@1 15: left arm never closes on the stack). Tests: not-slow 243 passed / 5 skipped; slow 14 passed / 34 skipped; isaac 36 passed / 1 skipped · see git log
- 2026-10-14 · **Q1 closed, Phase Q paused (user).**
  - Gripper head kept: +16 pp, p = 0.001.
  - Weighting not kept; val-loss selection rejected.
  - Fold-1 geometry study (overshoot, speed, straighten: no gain).
  - Parallel second fold 7/16, now the default.
  - `verify Q0` / `Q1` PASS.
  - All jobs stopped.
  - Fast tests: 238 passed, 5 skipped. 2b716e7
- 2026-10-14 · **Q3 fold-over iterations A–F.**
  - Multi-seed IK, lift gate, stage-start goals, staggered approach, Cartesian approach, wait-in-place. Sequential
    fold ruled out.
  - Q0 `sel_id_hard` 70/100 (first noise-floor set).
  - Fast 236 passed (1 flaky failure in 4 runs, not reproduced). d63b15a
- 2026-10-13/14 · **Phase Q start.**
  - Plan approved.
  - Q0 benchmark v2 and Q1 loss v2 code; both Q1 arms trained.
  - Q2:
    - `--task` plumbing (sonnet, merged by the lead);
    - reach: the "unreachable" second fold was a single-seed IK artefact;
    - the user chose a bimanual fold-over; its stages, expert placement and obs v2 were built.
  - Q0 noise floor crashed on a GPU overlap; re-queued.
  - Fast 233 passed, 5 skipped; slow 14 passed (34 Isaac-only skipped). c035eab
- 2026-10-08 · **Phase F report:** `docs/reports/2026-10-08-physical-grasp.md` (+ HTML, shareable https://claude.ai/artifact/Jj6aJ5aTKBNQEFzLkU9uuF), 4 videos (student clean 2/3, shifted 2/3, arm bump 1/3; expert arm bump 3/3) · tests unchanged (207 fast) · 43a4d8b
- 2026-10-08 · **F4 finals, BC only:** id_easy 81, id_hard 71, knock_arm 35 (n = 100), joint_noise 26, overshoot 4 (n = 50). Docs. Tests: 207 passed, 5 skipped (not slow) · 5f7e931
- 2026-10-07/08 · **Phase F: physical grasp through F3; F4 paused (expert first).**
  - Track G merged. The friction profile has: LeHome gripper drive, jaw +0.05, adhesion 0.1, pad friction 2.0,
    ALIGN_TOL 8 mm, and LeHome's task sets (pinch reach).
  - F1 ✅ (honesty).
  - F2 config M: held 40/40 per arm.
  - F3 expert: 94 / 94 / 27.
  - REGRASP and PINCH_FOLLOW_Z: no recovery gain.
  - The anchor comparison was dropped; Adam's as-built row was added.
  - F4 paused by the user; its partial data was discarded.
  - Fast tests: 198 passed. (this commit)
- 2026-10-07 · **Repo state and PR prep.**
  - Fetched: 0 behind `origin/main` (still `ce02b44`, PR #15). W5 ran on that sim.
  - New remote `feat/isaac-half-fold` (Ruby Zhou) overlaps; trial merge conflicts in 9 files.
  - Change log `docs/changelog/2026-10-07-feature-isaac-imitation.md`. (this commit)
- 2026-10-07 · **W5 done.**
  - Privileged 99.0 / 97.5 / 54.5, sensor-only 99.0 / 82.5 / 31.0 (n = 200); detector 98.0%.
  - 6 benchmark demos; dated report; `verify W5` PASS.
  - `evaluate --resume` (per-set saves).
  - Fast 187 passed, 4 skipped; slow 12 passed. (this commit)
- 2026-10-06 · **W5 privileged phase done; heat cap.**
  - BC s0 100/85/44, s1 98/73/38; DAgger r1 100/88/49 (n = 100).
  - Vision student trained; privileged demos ×3.
  - Thermal guard plus a 2 × 4 cap (`4153e19`); DAgger resume (`607ebf6`); demo `--set` (`32ff854`).
  - Fast suite 187 passed, 4 skipped. (this commit)
- 2026-10-06 · **W5 fast track started.** W3b closed (attempt D 58/80, plateau accepted); track G paused;
  `scripts/isaac_retrain.ps1` + `verify W5`; vectorised cameras smoke-checked; blocker 14 removed (not reproduced).
  Fast suite 183 passed, 4 skipped. (this commit)
- 2026-10-04 · **W3b recovery plateau and parallel tracks.**
  - W3b attempts A–C: 30, 34, 31 of 40 against a 35/40 baseline. A 300-step cap: 31/40. Pooled 81.7%; the user
    chose to keep working.
  - Z1: no transfer. V0 go; V implemented by an agent (pending merge).
  - Track G: IG.2 "released" redefinition approved by the user; now on Sonnet.
  - (this commit)
- 2026-10-04 · **V vectorised Isaac env** (`feature/isaac-vec`, background agent). B scene copies run in one Kit
  process, with `IsaacClothFoldBatch`/`IsaacSubEnv` and a lockstep multi-env worker (`imitation/lockstep.py`).
  - Parity: B=4 40/40 vs B=1 40/40 (Wilson 91.2–100 each).
  - Throughput: 1.25/1.66/2.15× episodes/h at B = 2/4/8, so kept; GPU memory flat.
  - `vec_check` passes; 16 fast lockstep tests pass.
  - Isaac tests (`-m isaac`, .venv-isaac): test_isaac_vec + test_isaac_weld + test_isaac_profile, 24 passed. The
    B = 1 worker path was also exercised by the B = 1 parity run.
  - Fast tests: lockstep + backend, 38 passed. The full fast suite is blocked by defect 14.
  - Commits b46a6d8 … this pass's final commit on `feature/isaac-vec`.

- 2026-10-04 · **W4 ✅ weld-baseline pilot.**
  - Collect 76/80; diffusion id_easy 20/20 (Wilson 83.9–100); DAgger round runs; detector 80/80 (in-sample).
  - W4 verifier check added (shares I3.2's pilot checks); `isaac/recovery_replay.py` and `w3_gate.ps1 -Sets` added for W3b.
  - `verify W4` and I3.2 PASS · (this commit)

- 2026-10-04 · **track G · IG.2 attempts 5–9; plateau, decision needed on the "released" metric.**
  - Best config (#9): closed jaw target +0.05 rad, `OVERSHOOT_FRICTION` left (−0.1, +5.2) cm / right (+1.3, +5.0) cm,
    pinch height 0.5 cm.
  - n = 20, tune 600000–19, left / right: acquired 20/20 / 20/20; held 20/20 [83.9, 100] / 19/20 [76.4, 99.1];
    placed 19/20 [76.4, 99.1] / 18/20 [69.9, 97.2]; strict released 6/20 / 4/20. 17/20 env successes with retries off.
  - Strict "released" (< 3 cm move after opening) measures the spring-back the overshoot relies on. Under a let-go
    definition (the held corner is back on the cloth after the retreat) it is 20/20 / 19/20.
  - Rejected: closed +0.08 (held 0 / 0), +0.02 (no gain), slow jaw opening (no gain).
  - Metric fix: placement is read at the episode end (the env's own success time); all runs rescored.
  - Next, pending the decision: n = 100 blocks A 600000–099, B 600100–199, fresh 600200–299, then ship #9 as the
    lehome defaults.
  - Tests: 172 fast pass (4 skipped) · (this commit)

- 2026-10-04 · **track G · IG.2 in progress (4 attempts, n = 20 each on tune 600000–19).**
  - Kept: the closed jaw target −0.1 → +0.05 rad, since at −0.1 the jaw overlaps the fixed pad and squeezes the
    cloth out. Held left 20/20 [83.9, 100], right 14/20 [48.1, 85.5] (baseline 18 / 1).
  - Rejected: particle friction 1.5 (held 2 / 0), pinch inset 15 mm (1 / 0), place height −1 cm (no gain).
  - Open: release and placement (released 2/20 and 1/20). The corner is let go about 6 cm up and falls 3–5 cm outward.
  - Weld regression: `test_isaac_weld` + `test_isaac_profile` 17/17 in .venv-isaac. Tests: 172 fast pass (4 skipped) · (this commit)

- 2026-10-03 · **track G · IG.1 ✅ friction-grasp bench + baseline** (branch `feature/isaac-grasp`).
  - `isaac/grasp_bench.py` (resumable, per-step traces) + sim-free `isaac/grasp_metrics.py`; `scripts/grasp_bench.ps1`.
  - Opt-in friction knobs on the "lehome" profile only (`isaac_env.FRICTION_GRASP`, all None = as built); the
    `IsaacArmExpert` pinch geometry is now class attributes. The weld path and `PROFILES["mujoco"]` are untouched.
  - Baseline n = 20 (tune 600000–19): left acquired / held / placed / released 20 / 18 / 14 / 12, right 11 / 1 / 0 / 0.
    The right jaw pushes its corner sideways out of the pinch.
  - `verify IG.1` and `I3.3` PASS (I3.3 now skips the IG checks); IG.2 / IG.3 checks added. Tests: 172 fast pass (4 skipped); weld-path Isaac tests pending the next free Isaac slot · (this commit)

- 2026-10-04 · **Phase W revised and the profile renamed `mujoco` → `weld`.**
  - New roadmap rows W3b, Z1 and V; W5 slimmed. W4 pilot results are in results.md.
  - Tests: 167 fast pass; Isaac weld + profile 17/17; `verify W1–W3` PASS · (this commit)

- 2026-10-03 · **W3 closed with an accepted exception.** Recovery 81 accepted by the user; the W3 verifier bar is set to 80 with the decision recorded; W4 pilot launched · (this commit)

- 2026-10-03 · **W3 gate run: 100 / 100 / 81 (recovery below its 90 bar).**
  - The weld is calibrated to MuJoCo's on the same script: soft (τ 0.02 s, mass ×10), world-axis offsets, table
    clamp, jaws open. Lag 27 vs 23–27 mm, rise 8.8 vs 9.0 cm.
  - `IsaacFoldExpert` = FoldExpert on Isaac accessors. MuJoCo benchmark rows are identical to M1.5.
  - Overshoot re-measured: < 5 mm.
  - W1's tracking bar is now MuJoCo parity. `scripts/isaac_pilot.ps1 -Backend` is added for W4.
  - Tests: 167 fast pass; Isaac `test_isaac_weld` + `test_isaac_profile` 17/17; `verify W1` W2 PASS, W3 FAIL (recovery) · (this commit)

- 2026-10-03 · **W2 ✅ MuJoCo profile on Isaac (`isaac_weld` backend).**
  - Reach of the FoldExpert waypoints is at parity with MuJoCo's own arm: 200/200 and 122/200 vs 200/200 and
    130/200 (gap 4.0 pp; Isaac's set is a subset of MuJoCo's).
  - `profile_check` 9/9: arm step response matches MuJoCo; DR is deterministic, applied and read back.
  - Found and fixed: the soft reset restores masses at the next step, so DR is applied after it.
  - Found: the friction smoke is nondeterministic run to run (A/B vs HEAD: 0.371/0.367 vs 0.379/0.349).
  - Sonnet subagent: pipeline plumbing.
  - Tests: 167 fast pass (4 skipped), 12 slow pass; Isaac `test_isaac_profile` 10/10, `test_isaac_weld` 7/7; `verify W2` and `I3.3` PASS · (this commit)

- 2026-10-03 · **W1 ✅ weld grasp on Isaac.**
  - Uses MuJoCo semantics on the GPU pipeline: zero-mass pins, with positions and velocities set every substep.
  - The CPU USD-write path was ruled out (14–36 mm drift per substep) and removed; weld on CPU now raises.
  - A Sonnet review found no bugs. The GPU reset offset was probed and applies.
  - Friction smoke test unchanged (fold_score 0.407). `verify W1` PASS; `test_isaac_weld` 7/7; 153 fast pass · (this commit; code in 487f4c1)

- 2026-10-03 · **Phase I closed (viability pass).** The whole pipeline runs on Isaac through the real CLIs:
  - I2.1: the Isaac expert, with the friction grasp as built, runs as the teacher. Pilot: 2/20 id_easy, 0/10 recovery.
  - I2.2: DAgger takeover labels, 204/208 full chunks (98.1%).
  - I3.1: e2e micro chain, 6/6.
  - I3.2: the pilot run. 80-episode collection (2 successes), chunk-MLP / diffusion / vision BC, closed-loop evals (all 0%: 2 demos), a DAgger round, a detector (79/80, majority 97.5%) and 3 demo videos.
  - I3.3: `docs/imitation.md` §10 backends, `docs/pipeline.md` §12.
  - `verify --all` PASS in both venvs; 153 fast + 12 slow (+10 Isaac-only slow, skipped in .venv) in `.venv` · (this commit)

- 2026-10-03 · **I1.1 ✅ backend switch.** `make_env(backend="mujoco"|"isaac")` and `--backend` on collect/evaluate/dagger (check_resync, benchmark_expert and demo refuse isaac until I2.1/I3.2). The rollout worker:
  - builds its teacher lazily on Isaac only
  - ticks Kit while idle and starts Isaac workers serially (ready handshake)
  - has a NaN guard and `os._exit` on close/EOF
  - the parent now closes the child's pipe end, so a dead worker raises instead of hanging (Sonnet review finding)

  MuJoCo collection hash is identical before and after (`26d4f1fe9f06`); 2 Isaac `EnvPool` workers give 139-D finite obs. `tests/` is now a regular package: draccus ships a stray top-level `tests` in the Isaac venv. 148 fast + 12 slow pass; 3 Isaac backend + 83 sim-free pass in `.venv-isaac` · (this commit)

- 2026-10-03 · **I0.3 ✅ (gate passed): Isaac Sim runs locally on Windows.** `isaac/smoke_test.py` passes in state and hybrid modes. 3.4 steps/s for one process; two processes give 4.97 aggregate at 9.1 GB VRAM peak, so `N_ISAAC = 2`. Not bit-deterministic for cloth (as expected).
  - Fixes on the way, each isolated and logged in `isaac/INSTALL_REVIEW.md`:
    - Kit's Vulkan path crashes in `omni.hydra.rtx.plugin` on driver 616.56, so it now uses D3D12 (`--/app/vulkan=false`).
    - LeHome's lock was re-resolved for Windows.
    - `h5py` pinned to 3.14.0, to match Kit's HDF5 1.14.6 DLL.
    - `isaaclab_tasks`/`isaaclab_assets` editables and `omegaconf` added.
    - The pipe deadlock in the throughput benchmark was fixed.
  - Footprint outside `.venv-isaac`: Kit logs and caches in `~\.nvidia-omniverse` and `%LOCALAPPDATA%\ov` (~560 MB), documented with uninstall steps.
  - I1.1 in progress: `make_env(backend=)`, `imitation/isaac_runtime.py`, and an Isaac-aware rollout worker (lazy teacher, Kit keep-alive, serial start, NaN guard, `os._exit`). MuJoCo imports are lazy.
  - 138 fast pass, 3 Isaac-only skipped · (this commit)

- 2026-10-03 · **I0.4 ✅** verifier `python -m imitation.verify` (Sonnet subagent, re-verified: 12 tests, I0.1/I0.2/I0.4 PASS). I0.3 lock fixes: LeHome's lock was never resolved for Windows (`isaacsim` pins `pywin32==306`, `networkx==3.3`, `filelock 3.13.1`, `fsspec 2024.6.1`), so `isaac/requirements-lehome-windows.lock` is re-resolved from LeHome's pyproject, seeded with its lock (4 versions changed + 15 Windows-only packages, all hashed). Hashed locks install with `--no-deps`. Install running · (this commit)

- 2026-10-02 · **I0.3 in progress** (paused at the usage limit):
  - `.venv-isaac` was created (Python 3.11.9 + hash-pinned uv 0.12.22).
  - Hashed locks are committed: `isaac/requirements-{lehome,isaaclab,torch-cu128}-windows.lock`, plus `isaac/setup_windows.ps1` and `env_windows.ps1`.
  - `start_app` honours `WORLDFOLD_KIT_ARGS`.
  - The install was started in the background (log: `outputs/isaac/setup.log`).
  - Resume: check the log ends with `== done`; if not, re-run `powershell -File isaac\setup_windows.ps1` (idempotent). Then run the I0.3 smoke tests and the scope check (`outputs/isaac/scope_before.txt` is the before-listing).
  - · (this commit)

- 2026-10-02 · **I0.2 ✅ install review:**
  - Both LeHome repos were fetched at their pinned SHAs and reviewed by two Sonnet subagents, then spot-checked by the lead. Verdict: SAFE.
  - The lock's isaacsim hashes match the official pypi.nvidia.com index.
  - Telemetry and the extension registry will be off. CPU torch gets swapped for cu128. `/Assets` is a junction.
  - The user OK'd the artifact list (~8 GB) and accepted the NVIDIA EULA.
  - `isaac/INSTALL_REVIEW.md` · (this commit)

- 2026-10-02 · **I0.1 ✅ pulled origin/main (Isaac Sim env, PR #15)** into `feature/imitation`. Resolved `quarter_fold_env.py`: main's `base_env=` + accessors and our 139-D obs. Fixed the silent-breakers: `stages` shadowing (it would have made the half fold 2-stage), the hard-coded one-hot offset, and the isaac test stub. Added Phase I to the roadmap, the subagent policy to CLAUDE.md and `docs/subagents.md`. 116 fast + 12 slow pass; expert benchmark 48/50, rows identical to M1.5 · (this commit)

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

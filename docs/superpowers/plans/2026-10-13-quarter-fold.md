# Phase Q — Robust quarter fold on Isaac friction (fast track)

## Context
Phase F gave a physical friction grasp. On it the expert scores 92 / 94 / 75 and the BC-only
student 81 / 71 / 35 on the **half** fold. The end goal is a **quarter fold**: fold in half,
release, then fold in half again. It needs the full pipeline (collect → BC → DAgger → vision →
detector → finals) and a side-by-side comparison. The user also wants a better benchmark and
better loss functions first, because the current ones hide or cause failures.

Surveys of the code (this session) found:
- **Task.** `cloth_fold_rl/quarter_fold_env.py` already defines both stages (`STAGES`, `QuarterFoldEnv`).
  - Stage 2: the right arm carries the two-layer stack (corners 0+10) onto corner 110. The left
    arm carries **vertex 5** onto vertex 115.
  - Vertex 5 is the midpoint of the west short edge, the end of the first fold's crease
    (`CLOTH_5 = N//2`, `quarter_fold_env.py:63`).
- **Everything downstream stops at stage 0.** That covers `make_env` (`imitation/tasks/half_fold.py`),
  `isaac_env.make_half_fold_env`, and collect's hard-coded `task: "half_fold"`.
- **The Isaac friction expert is wrong for stage 2.**
  - `IsaacArmExpert` mirrors the place point in y only (`isaac/fold_expert.py:171,231`), and the
    stage-2 overshoot is 0.
  - The pinch is planned at table height, which is wrong for a two-layer stack.
  - Stage-2 reach on Isaac's shifted cloth was never checked.
  - `grasp_active` counts one held layer as a grasp.
- **Observation.** The 139-D obs shows only the four corners, so vertex 5 and its goal are invisible
  to the policy. The stage feature (dim 137) has never been non-zero in training.
- **Benchmark.**
  - DAgger selection and the finals share seeds.
  - `gain_in_se` treats evals on identical seeds as independent.
  - No per-episode rows or per-stage metrics are kept.
  - Gates use point estimates.
  - Isaac repeat-run noise is not measured.
  - G2 is missing from `FAILURE_CODES`.
  - `--resume` ignores the backend.
- **Losses.**
  - All 12 dims and 16 chunk steps get equal weight (L1 or ε-MSE).
  - The gripper is regressed into a ±0.3 hysteresis band.
  - Actions are not normalized, so fine placement deltas are drowned out (the S1 stall).
  - The saved checkpoint is always `final.pt`. F4's val loss was best at about 6k of 20k steps.

Decisions (user, 2026-10-13): **Isaac friction** backend; **fast-track** budget (200 eps, 1 seed,
1 DAgger round, n = 100 finals). Obs default below: **extend the obs**. Vertex 5 was explained above.

## Order of work
Q0 (benchmark) and Q1 (losses) are mostly sim-free and come first. They are A/B'd on the existing
half-fold data (`isaac_v1_friction`), so every quarter-fold number is measured with the new
tooling.

### Q0 — Benchmark v2 (sim-free code; delegate the code to sonnet, the lead reviews)
1. **Per-episode rows** in the eval JSON (`imitation/evaluate.py` `summarize` + `on_set`). Each row
   holds: seed, success, `stage_reached`, per-stage success, fold_score, steps, failure code + the
   stage it happened in, per-arm grasp/slip.
2. **Per-stage summary:** stage-1 rate, stage-2 rate conditional on stage 1, and time-to-success
   censored at the cap (report the median including failures, not the mean over successes only).
3. **Disjoint seed blocks** in `imitation/seeds.py`.
   - A `sel_*` block (DAgger keep/stop, checkpoint pick, loss A/Bs) and a `final_*` block (reported
     once) per set, on new bases.
   - Quarter-fold sets get their own bases.
   - Extend `tests/imitation/test_seeds.py` for disjointness.
   - Change the `benchmark_expert` / `check_resync` defaults to held-out seeds.
4. **Paired statistics** (new `imitation/stats.py`): exact McNemar + paired bootstrap CI on the
   success difference over shared seeds, plus Wilson per policy.
   - Replace `dagger.gain_in_se` with the paired test.
   - Move `wilson` here from `verify.py` and re-export it.
   - Gates report the point estimate **and** the Wilson LB. The bars are set before the run, in
     the roadmap row.
5. **Isaac noise floor:** re-run one checkpoint twice on the same 50 sel seeds and record the
   per-seed flip rate. A difference smaller than the flip rate is reported as noise. This is the
   one sim step in Q0 (about 20 min).
6. **Fixes:**
   - Add G2 to `FAILURE_CODES` and to imitation.md §5.2.
   - Add a new **G3: partial stack** code (only one layer moved).
   - `--resume` checks the backend.
   - The default out path is tagged with backend and task.

### Q1 — Loss v2 (scoped code + tests to sonnet; design and A/B calls stay with the lead)
All changes are opt-in flags on `imitation/train.py` (`--loss-cfg` JSON). The baseline stays
bit-identical when the flags are off (test).
1. **Gripper head:** a 3-way classification (close / hold / open, mapped onto the env's ±0.3
   hysteresis) with CE loss.
   - Chunk-MLP and vision: separate logits.
   - Diffusion: the head sits on the obs encoder. Diffusion runs on the 10 joint dims only, and
     `predict` emits ±1 / 0 from the argmax.
2. **Per-dim action normalization** (mean/std of the joint deltas, stored with the normalizer) so
   small placement corrections carry weight. This matters most for diffusion.
3. **Horizon weighting:** weight 1 for the executed steps (< replan) and 0.5 for the rest of the
   chunk.
4. **Phase weighting:** upweight samples near a goal (carried corner < 5 cm from goal) and near
   gripper transitions, ×3 by default. Both are derived from obs/actions already stored, so there
   is no schema change for the half-fold A/B.
5. **Checkpoint selection:** save every 2k steps plus best-val. Pick by a short rollout eval on
   the `sel` seeds (n = 50 id_hard + knock_arm), not by val loss.
6. **A/B protocol** on `isaac_v1_friction` half fold, diffusion seed 0, 20k steps:
   - Arms: baseline vs +gripper vs +all.
   - Eval: sel id_hard + knock_arm, n = 100, paired McNemar.
   - Keep a change if its paired gain is above the Q0 noise floor and nothing regresses
     beyond it.
   - About 3 trainings (GPU, cheap) and 6 eval sets (about 4 h Isaac).

### Q2 — Quarter-fold task on Isaac friction (lead designs; sonnet implements against spec)
1. **Task wiring:**
   - `imitation/tasks/quarter_fold.py` (`make_env` / `make_env_batch`, `--task quarter_fold` on
     collect, evaluate, dagger, demo).
   - `isaac_env.make_quarter_fold_env`.
   - The task is recorded in dataset metadata.
   - The step cap is set from the expert's measured p95 (start at 800).
2. **Observation v2** in `imitation/spec.py`: replace the four fixed corner-goal slots with a
   per-arm **carried point + its goal** slot (position, goal, delta), and add a stage one-hot.
   - Half and quarter fold use the same layout. Nothing is hard-coded; sizes come from `spec.py`.
   - Update imitation.md §2.3.
   - This is a new dataset version. Old checkpoints stay readable through the 139-D path, which is
     kept.
3. **Expert stage-2 fixes** (`isaac/fold_expert.py`):
   - Reflect the place point across the stage's fold line (general, not y-only).
   - Pinch the stack at its measured height (`PINCH_FOLLOW_Z` on for stage 2).
   - Calibrate the stage-2 overshoot on tune seeds.
   - Check stage-2 reach for both arms on the Isaac cloth centre before anything else (IK-only
     script, n = 200 poses).
   - If the cross-table reach fails, swap the arm roles for stage 2 in `STAGES`. The user is told
     before this happens.
4. `grasp_active` / honesty: for a stack move, require both layers to follow (G3 otherwise).
5. **Quarter-fold eval sets** (LeHome style):
   - `qf_id_easy`, `qf_id_hard`.
   - `qf_knock_s1` and `qf_knock_s2`: an arm knock during stage 1 / stage 2.
   - `qf_disturbed_mid`: the half-fold result is jittered before stage 2.
6. **Reconcile `cloth_angles/tasks.py::QUARTER`** with `STAGES`, by deleting it or importing from
   `STAGES` (closes defect #10 / M7.3).

### Q3 — Quarter-fold expert gate (lead; one mechanism per attempt, tune seeds n = 40, paired)
- **Proposed bars**, confirmed by the user before the run, n = 100 on `final`:
  - stage-1 ≥ 92;
  - qf_id_easy ≥ 85, qf_id_hard ≥ 80;
  - qf_knock_s1 / s2 ≥ 60;
  - G2 + G3 ≤ 8%.
- A plateau goes to the user (as in F3b).

### Q4 — Fast-track pipeline + comparison (one script, `scripts/q_retrain.ps1`, modelled on `f4_retrain.ps1`)
1. **Collect:** `isaac_qf_v1`, 200 eps, 50% knocked (half in s1, half in s2). Successes become
   demos; failures go to `_failures`.
2. **BC diffusion, seed 0:** loss v2 (whatever Q1 kept) **and** baseline loss. Train on GPU, select
   on `sel`.
3. **DAgger:** 1 round (takeover p = 0.6) from the better BC. Keep/stop by paired test on `sel`.
4. **Vision student**, 1 round (distillation from the privileged best).
5. **Detector:** stage-aware labels (stage-1 done, quarter done), with agreement on held-out
   episodes.
6. **Finals, n = 100 on the `final` blocks**, all policies on identical seeds:
   - Rows: expert, BC-baseline-loss, BC-loss-v2, DAgger, vision.
   - Paired McNemar between adjacent rows.
   - Per-stage breakdown and failure codes by stage.
7. **Demos:** 3 privileged and 3 sensor videos.
8. **Heat cap and logs:**
   - ≤ 2 Isaac procs × 8 envs (6 with cameras), `thermal_guard.py` alongside.
   - Each stage runs as a visible background task; watch logs with `scripts/watch_logs.sh`.
   - Expected wall time is about 1 day.

### Q5 — Report
Write `docs/reports/<date>-quarter-fold.md` plus an HTML artifact with the comparison table and the
Q0/Q1 findings. Add roadmap Phase Q rows (Q0–Q5) with exits and `verify Q0..Q5` checks in
`imitation/verify.py`.

## Critical files
- `imitation/evaluate.py`, `imitation/seeds.py`, new `imitation/stats.py`, `imitation/dagger.py`
  (`gain_in_se`), `imitation/verify.py`
- `imitation/train.py`, `imitation/policies/{chunk_mlp,diffusion,vision,common}.py`,
  `imitation/data/{dataset,loader}.py`
- `imitation/spec.py`, `cloth_fold_rl/quarter_fold_env.py` (`_observe`, `_set_goals`), new
  `imitation/tasks/quarter_fold.py`, `imitation/data/collect.py`
- `isaac/fold_expert.py`, `isaac/isaac_env.py`, `imitation/isaac_runtime.py`
- `scripts/f4_retrain.ps1` → `scripts/q_retrain.ps1`
- Docs: `docs/{status,roadmap,results,imitation,subagents}.md`

Reuse: `wilson` (verify.py), `rollout.py` perturbation draws, `EnvPool`/lockstep worker,
`SuccessDetector`, `pick --carry-sets`, f4_retrain staging and resume.

## Verification
- `.venv/Scripts/python.exe -m pytest -m "not slow"` after each of Q0, Q1 and Q2, plus `-m slow`
  after the env/sim changes. Isaac tests run in `.venv-isaac` (`-m isaac`).
- **New tests:**
  - seed-block disjointness;
  - McNemar against a hand-computed table;
  - loss-v2 flags off ⇒ identical loss and gradients to baseline;
  - gripper class mapping matches the env hysteresis;
  - obs v2 shape from spec, with vertex 5's goal present in stage 2;
  - reflection placement for a stage-2 move;
  - quarter-fold env reaches stage 1 with a scripted expert (Isaac, slow).
- `python -m imitation.verify Q0 … Q5`. The lead re-runs every verify itself (CLAUDE.md).
- MuJoCo half-fold expert benchmark rows are unchanged (regression guard on the obs and env edits).
- Each pass ends with status/roadmap/results/subagents updates and a commit, per CLAUDE.md.

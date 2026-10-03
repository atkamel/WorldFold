# Phase W — MuJoCo weld baseline transferred to Isaac, plus a parallel friction-grasp track

## Context

Phase I closed: the whole imitation pipeline runs on Isaac Sim (`feature/isaac-imitation`,
`verify --all` PASS in both venvs). But its baseline used Isaac's friction grasp, which succeeds
2/80, so every policy trained on 2 demos and scored 0%.

You want two things:
1. **The first prototype on the easier grasp.** That means MuJoCo's *weld* grasp (a corner within
   6 cm attaches when the gripper closes), with the MuJoCo setup fully transferred into Isaac, then
   retrained.
2. **Friction-grasp work in parallel**, as a separate task.

**Your decisions**
- Pilot first, then the full MuJoCo-scale retrain if the expert gate passes.
- The GPU is split: weld track 2 Isaac workers, grasp track 1 (about 12 GB VRAM total).
- The grasp track runs as a **separate session** on its own worktree and branch.

## What is left overall

| track | items | status |
|---|---|---|
| **W: weld baseline (this plan)** | W1 weld grasp, W2 MuJoCo scene/arm/DR profile, W3 MuJoCo expert on Isaac + gate, W4 pilot, W5 full keepers retrain + report | next |
| **G: friction grasp (parallel session)** | IG.1 grasp bench, IG.2 reliability loop (bars already in roadmap), IG.3 friction-expert gate; afterwards optionally the IS retrain on friction | starts with W |
| I-DR | visual DR in the Isaac rig. Dynamics DR moves into W2 | after W5 |
| Phase 6 / 7 | VLA subgoals (M6.1–6.3); deploy + cleanup (M7.1–7.3) | after W5 |
| Housekeeping | push `feature/imitation` / `feature/isaac-imitation` + PRs (I ask first); defect #13 (`fold_env` sys.path hack); untracked pilot stage logs | anytime |

## How MuJoCo got 100/97/96, and what Isaac must copy (from exploration)

- **Grasp.** A weld on every allowed vertex within `GRASP_RADIUS = 0.06` while the gripper is
  closed (`mujuco/sim_main.py:489-517`):
  - hysteresis ±0.3
  - the offset is captured in the gripper frame at engage
  - `weld_mask` is applied per stage
  - `grasp_active` means a weld is engaged
- **Expert.** `cloth_fold_rl/expert.py` FoldExpert:
  - position-only IK with 12 restarts and a free wrist
  - targets: corner +6 cm → +0.5 cm → lift to table +12 cm → carry → goal +2 cm → hold →
    release → retreat
  - each phase advances within 2 cm, or after 45 steps
  - `QuarterFoldExpert` adds the overshoot, 2 retries and the release gate (it already accepts
    `expert_cls`)
- **Scene:**
  - cloth centre (0,0), jitter ±2.5 cm, no drop tilt
  - dynamics DR: mass, table friction and damping each ×U(0.7,1.3)
  - episode cap 250
  - MuJoCo eval sets: id_hard 2.5–4 cm, recovery knock t 15–60
- **Arm.** MuJoCo position actuators: kp 998, kv 2.73. LeHome's SO101 drives are kp 17.8, kv 0.60,
  about 56× softer, which is why the Isaac arm lags. The kinematics are identical (<0.3 mm), and
  `PinchIK` uses the same `gripper_frame_link` site, so position-only IK with
  `orientation_weight=0` reaches what MuJoCo reached.

---

## Track W milestones

Each milestone uses the existing self-verification protocol:
- tests first
- `python -m imitation.verify <ID>` writes committed evidence
- results.md before any other doc, then status, roadmap, then commit
- MuJoCo regression: `.venv` fast + slow green and the expert benchmark rows identical

The weld mode and the MuJoCo profile are **opt-in**. Isaac's friction defaults stay untouched for
track G and for the Modal path.

### W1 — Weld grasp in Isaac (`grasp_mode="weld"`)

**Files:**
- `isaac/isaac_env.py`: `IsaacClothFoldEnv(grasp_mode="friction"|"weld")`
- `isaac/lab_scene.py`: per-substep pin hook in `SceneEnv._apply_action`
- `tests/imitation/test_isaac_weld.py` plus a script `isaac/weld_check.py`

**Grabbing.** Port `_try_pin` / `_unpin_all` from `origin/feat/isaac-sim:isaac/isaac_env.py:461-519`:
- same hysteresis and `weld_mask` as MuJoCo
- a grid vertex within 6 cm is grabbed on close, and later ones can still join while closed
- the offset is captured in the gripperframe

**Pinning.** The CPU device has no per-particle mass, so instead, each physics substep:
1. Write the pinned particles' USD `points` (target = gripper pose ∘ offset, converted to cloth-local
   with `cloth.inverse_transform_points`).
2. Write their `velocities` (set to the gripper point velocity).
3. Pin the **patch** of particles within 1.5 cm of the grid vertex, each with its own offset. That
   avoids single-particle tearing.

**Release.** Stop writing and zero the released velocities. `grasp_active` = anything pinned.

**Verify:** `isaac/weld_check.py`, run by the test. It closes on `cloth_10`, lifts 10 cm and carries.
- tracking error ‖vertex − (gripper ∘ offset)‖ < 3 mm on every step
- the corner rises with the gripper
- after opening, it falls and `grasp_active` goes false
- a vertex outside the mask never pins; a command of 0 holds the state
- the cost per control step is recorded

### W2 — MuJoCo profile for the Isaac task (`profile="mujoco"`)

**Files:**
- `imitation/isaac_runtime.py`: a profile table
- `isaac/isaac_env.py` / `lab_scene.py`: opt-in arguments
- `imitation/tasks/half_fold.py`
- `imitation/seeds.py`

**The profile:**
- cloth centre (0,0), jitter 0.025, drop tilt 0
- arm drives overridden in `_robot()` to stiffness ≈ 998 and damping ≈ 2.73, with MuJoCo's torque
  limit
- dynamics DR at reset: particle mass, table friction and particle damping ×U(0.7,1.3), recorded in
  `_domain_params`
- episode cap 250
- `eval_set(..., backend="isaac", profile="mujoco")` returns the **MuJoCo** sets
- grasp mode weld

**Verify:**
- `isaac/reach_check.py --profile mujoco`: position-only IK on the MuJoCo pose distribution,
  200/200 within the 6 cm weld radius
- an arm tracking test: a 0.05 rad step command reaches ≥ 90% of the target within one control step
  (today it reaches ~78%)
- DR values are drawn per seed, deterministic, and recorded
- `smoke_test.py` friction defaults are unchanged

### W3 — MuJoCo expert on Isaac and the expert gate (M1.7 analogue)

**Files:**
- `cloth_fold_rl/expert.py`: factor FoldExpert's phase machine out from its MuJoCo IK and state
  reads into an `ExpertBackend` (MuJoCo: `solve_ik` + `data`; Isaac: `PinchIK` with
  `orientation_weight=0`, restarts and the 0.1 rad step cap, plus env accessors)
- `imitation/teachers/scripted.py`: pick by profile; the weld profile uses FoldExpert
- `isaac/fold_expert.py`: stays the friction-pinch expert for track G

**MuJoCo must stay byte-identical:** benchmark rows equal to m1_5, and an identical collection hash.

**OVERSHOOT** is re-measured on the Isaac cloth: the spring-back over 20 weld episodes goes into a
per-profile table.

**Verify:**
- `evaluate --backend isaac --ckpt expert` at n = 100 on the MuJoCo sets
- **id_easy ≥ 95, id_hard ≥ 90, recovery ≥ 90**; 92–94 extends to n = 200
- if it misses, iterate on the weld or expert (not on friction), logging each attempt in results.md

### W4 — Pilot on the weld baseline (cheap check that learning works)

**Run:** `scripts/isaac_pilot.ps1 -Profile mujoco`, which takes a version suffix `_weld`:
- collect 80, chunk-MLP + diffusion + vision, evals, 1 DAgger round with takeover labels, detector,
  demos
- 2 workers, resumable

**Verify:** `imitation.verify W4`:
- every artifact is valid (as I3.2)
- the expert's collection success rate is recorded
- diffusion id_easy success is above 0 with Wilson LB > 0. That's a sanity check that learning
  happens; there is no target.

### W5 — Full keepers retrain at MuJoCo scale

**Steps:**
1. Freeze `isaac_v1_weld`: 400 episodes, 30% perturbed, images at collection.
2. BC: chunk-MLP ×2 + diffusion ×2 seeds, n = 200 × 3 sets.
3. Replan sweep 8/4/2 on the best diffusion seed.
4. Diffusion DAgger: 2 rounds with takeover labels, reusing the init eval.
5. Vision BC ×2 seeds at replan 2/4.
6. Detector ≥ 90% on held-out episodes.
7. Demos.
8. A dated report with a **labelled MuJoCo-vs-Isaac side-by-side**.

**How it runs:** resumable scripts per stage (`scripts/isaac_retrain.ps1`), 2 workers, about 2–4
days of laptop sim, relaunched every 2 h by the background-task limit.

**Verify:** `imitation.verify W5.x`, using the MuJoCo-phase rules: n ≥ 200, Wilson intervals, gains
in SE, ≥ 2 seeds.

## Track G — friction grasp (separate session)

I spawn a session chip (`spawn_task`) with a self-contained brief:
- **Branch:** worktree `feature/isaac-grasp` off `feature/isaac-imitation`.
- **Compute:** `WORLDFOLD_N_ISAAC=1`, a new env override honoured by `imitation/isaac_runtime.py`
  and `resolve_workers`.
- **Milestones:** IG.1 grasp bench, IG.2 reliability loop and IG.3 friction-expert gate, with the
  bars already in `docs/roadmap.md`.
- **Constraints:** friction mode only; adhesion 0; friction ≤ 2.0; no attachments.
- **Logging:** its own pass-log lines and subagent log.
- **Merge:** back into `feature/isaac-imitation` only after IG.2's bar is met.
- **Conflicts:** it touches friction knobs and `isaac/fold_expert.py`. Track W touches
  weld/profile code paths, so conflicts stay small.

## Compute and process rules (updated with your choice)

- **Concurrency:** 3 concurrent Isaac processes are allowed (W 2 + G 1, about 12 GB VRAM). This
  relaxes "one sim pool at a time" for Isaac; I'll update the `cpu-heat-budget` memory after
  approval.
- **Monitors** poll without holding the log open (no `tail -f`), and logs are UTF-8.
- **Subagents** follow the `CLAUDE.md` policy: Haiku for mechanical work, Sonnet for scoped code
  and reviews. W1 and W3 get a Sonnet code review before closing.

## Critical files

- **Modified:**
  - `isaac/isaac_env.py`, `isaac/lab_scene.py` (weld, profile, drives, DR; all opt-in)
  - `cloth_fold_rl/expert.py` (backend split, MuJoCo byte-identical)
  - `cloth_fold_rl/quarter_fold_expert.py` (per-profile overshoot)
  - `imitation/isaac_runtime.py`, `imitation/tasks/half_fold.py`, `imitation/seeds.py`,
    `imitation/teachers/scripted.py`, `imitation/verify.py` (W checks)
  - `scripts/isaac_pilot.ps1` (profile)
  - docs: roadmap Phase W, status, results, imitation §10
- **New:** `isaac/weld_check.py`, `tests/imitation/test_isaac_weld.py`,
  `tests/imitation/test_isaac_profile.py`, `scripts/isaac_retrain.ps1`
- **Reused:**
  - old weld: `origin/feat/isaac-sim:isaac/isaac_env.py:461-519`
  - `PinchIK` (`isaac/pinch.py`)
  - `QuarterFoldExpert(expert_cls=…)`
  - takeover DAgger labels
  - `EnvPool` Isaac lifecycle
  - the pilot driver, the verifier, `isaac/reach_check.py`

## Verification (end to end)

- `python -m imitation.verify --all` passes in both venvs, covering I* and W*.
- The MuJoCo expert benchmark rows are identical and the collection hash is unchanged.
- `.venv` fast + slow are green; `-m isaac` is green in `.venv-isaac`.
- The W3 gate meets MuJoCo-level ceilings, and W5 numbers are in results.md with n = 200 and
  Wilson intervals.
- I watch the demos via frames.

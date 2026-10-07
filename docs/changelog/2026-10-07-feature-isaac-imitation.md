# Change log — `feature/isaac-imitation` (as of 2026-10-07)

**Base:** `origin/main` @ `ce02b44` (PR #15, atkamel: WorldFold cloth env on LeHome's Isaac Sim 5.1 stack).
**Branch state:** 0 behind, 90 ahead of `origin/main` (fetched 2026-10-07). 133 files outside `outputs/` and media
changed, +18.6k / −0.1k lines.

## Repo state on 2026-10-07

- **`origin/main`:** hasn't moved since PR #15, so this branch is not behind it.
- **PR #15's simulator is what this branch runs.**
  - `main` was merged in at `88a7b1f` (2026-10-02).
  - Every Isaac result here comes from `isaac/isaac_env.py` and `isaac/lab_scene.py`: the LeHome SO101 arms, the
    garment cloth and the table.
  - This branch adds about 600 lines to those two files: the weld grasp profile, the vectorised scene copies, and
    the camera rig.
- **The branch also carries the whole MuJoCo imitation pipeline** (Phases 1–5c, previously only on
  `feature/imitation`), because `main` has never had `imitation/`.
- **Merged in:** `feature/isaac-vec` (milestone V), at `129d1b9`.
- **Not included:**
  - `feature/isaac-grasp` (track G, friction grasp): paused by the user on 2026-10-06; its best config is `6c8c5af`.
  - `origin/feat/isaac-half-fold` (Ruby Zhou, 2026-10-03 → 10-06): a parallel Isaac half-fold pipeline (anchor
    grasp, BC 9/10 held-out). It branched from the same `ce02b44` plus `feature/imitation`.
    - A trial merge conflicts in 9 files: `.gitignore`, `cloth_fold_rl/quarter_fold_env.py`, `docs/results.md`,
      `docs/status.md`, `imitation/data/collect.py`, `imitation/demo.py`, `imitation/rollout.py`,
      `imitation/seeds.py`, `imitation/teachers/__init__.py`.
    - It also touches `isaac/isaac_env.py` and `isaac/lab_scene.py`.
    - The two need reconciling before either lands on `main`.

## What changed, by phase

### Imitation pipeline (Phases 1–5c, MuJoCo), from `feature/imitation`
- **`imitation/`:**
  - data schema, store and loaders
  - scripted and policy teachers
  - chunk-MLP, diffusion and vision policies
  - training, evaluation and DAgger
  - offline RL (IQL; dropped)
  - success detector
  - profiling and viz tools, and the milestone verifier (`imitation/verify.py`)
- **Docs:** `docs/imitation.md`, `docs/pipeline.md`, `docs/roadmap.md`, `docs/results.md`; reports
  `docs/reports/2026-09-24-phase1-5.md` and `2026-09-25-phase5b.md`.

### Phase I: the pipeline on Isaac (2026-10-02 → 10-03)
- **Install:** the LeHome stack install is reviewed (`isaac/INSTALL_REVIEW.md`). Hashed Windows locks are in
  `isaac/requirements-*-windows.lock`. Kit runs on D3D12 (`isaac/env_windows.ps1`).
- **Backend switch:** `make_env(backend=...)`, an Isaac-aware `EnvPool` worker, and `--backend` on every CLI. The
  MuJoCo collection hash is unchanged.
- **Isaac pieces:** the camera rig (main 128² plus wrist cameras 64²), eval seed sets, a reachability check, the
  Isaac arm expert, DAgger *takeover* labels (Isaac has no sim snapshot), and the demo renderer.
- **Pilot:** every stage ran on Isaac. The friction grasp as built was the blocker (expert 2/20).

### Phase W: weld baseline on Isaac (2026-10-03 → 10-04)
- **W1:** MuJoCo-semantics weld grasp on Isaac (GPU pipeline, zero-mass pins).
- **W2:** `isaac_weld` profile: MuJoCo drives, cloth, domain randomisation, a 250-step cap, the MuJoCo eval sets.
- **W3:** MuJoCo's FoldExpert on Isaac. Gate 100 / 100 / 81 (recovery accepted below its bar).
- **W3b:** recovery-tuning attempts A–D (`isaac/recovery_replay.py`). Plateau at 81.7% pooled; closed by the
  user's decision.
- **W4:** weld pilot (diffusion 20/20 on id_easy).
- **Z1:** the MuJoCo checkpoints don't transfer zero-shot (0–1/20), so W5 trained from scratch.
- **V:** vectorised Isaac env, B scene copies per process (`IsaacClothFoldBatch`, `imitation/lockstep.py`).
  Parity is 40/40 vs 40/40 and throughput 2.15× at B = 8. Opt-in with `WORLDFOLD_ISAAC_ENVS_PER_PROC`.

### W5: full retrain on Isaac (2026-10-06 → 10-07), this pass
- **Driver:** `scripts/isaac_retrain.ps1`. It is resumable stage by stage (skips finished stages), relaunched at
  the 2 h job limit, with STEP / DONE / FAIL markers for monitors.
- **Pipeline changes:**
  - `imitation/dagger.py` `--resume` reuses a round's trained checkpoint and its saved eval.
  - `imitation/evaluate.py` saves each set as it finishes; `--resume` skips finished sets.
  - `imitation/demo.py` `--set` plays an eval set's shifted poses and knock, labelled on screen.
- **Heat cap** (user: keep the laptop under 94 °C):
  - `scripts/thermal_guard.py` pauses at a GPU temperature of 90 °C and resumes at 84 °C.
  - Rollouts and training pause cooperatively (`imitation/thermal.py`); workers keep ticking Kit, so nothing is
    suspended.
  - The driver is capped at 2 Isaac processes × 4 envs.
- **Verify:** `imitation/verify.py` gains a W5 check, which passes.
- **Tests:** `tests/imitation/test_thermal.py`, plus a thermal-pause case in `test_lockstep.py`. Fast suite 187
  passed (4 skipped); slow suite 12 passed.
- **Data:** `isaac_v1_weld` (384 expert episodes + 16 failures, hash `468089355dd7`) and the DAgger round-1
  version (hash `32fb84b8c1c2`).
- **Results** (n = 200 per set; id_easy / id_hard / recovery):
  - privileged policy, DAgger round 1 at replan 4: **99.0 / 97.5 / 54.5**
  - sensor-only student at replan 2: **99.0 / 82.5 / 31.0**
  - success detector: agrees with the sim on 98.0% of episodes
- **Report:** [docs/reports/2026-10-06-isaac-w5.md](../reports/2026-10-06-isaac-w5.md), with 6 benchmark videos in
  `docs/reports/media/half_fold_isaac_*`.

## Known open items
1. **Recovery** is the gap (54.5 / 31.0); the expert ceiling is 81.7%.
2. **The weld grasp is a stand-in.** Track G (friction grasp) is paused.
3. **The overlap with `feat/isaac-half-fold`** needs a decision on which Isaac half-fold path lands on `main`.
4. **Untracked run logs** under `outputs/` are deliberately left out of git.

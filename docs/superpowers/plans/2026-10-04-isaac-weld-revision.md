# Phase W revision (2026-10-04): finish W4, raise expert recovery, zero-shot MuJoCo models, vectorise Isaac, slim W5

## Context

Phase W so far:
- W1, W2 and W3 are done. The `isaac_weld` backend is MuJoCo's setup on Isaac with a weld calibrated to MuJoCo's.
- MuJoCo's expert gate on Isaac: 100 / 100 / 81. You accepted recovery 81 for now.
- The W4 pilot shows learning works: diffusion 20/20 on id_easy, vision 10/10. Its last stages (DAgger, detector)
  were cut off by the 2-hour job limit.

Your direction (2026-10-04):
- Adopt my recommendations: zero-shot test of the MuJoCo checkpoints, finish W4, vectorise Isaac before W5, and
  slim W5.
- Execute while track G (friction grasp) runs in parallel.
- Update the docs.
- **Keep the cloth at 101×101.**
- **The scripted expert must reach ≥ 90% on recovery.** Spend time on it; 85+ is acceptable if it plateaus early.

Track G is already running: a background agent in its own worktree, 1 Isaac process. I'll review its results and
merge only with your approval.

**MuJoCo's role (your decision, 2026-10-04): Isaac only, with a rename.**
- All new work runs in Isaac only.
- The MuJoCo code stays in the repo untouched, as a frozen reference: no new MuJoCo runs, and no new byte-identical
  regression checks. The existing tests stay.
- The profile name `"mujoco"` is renamed to `"weld"`.
- Z1 stays: it reuses old trained weights inside Isaac.

### R — rename the profile (first step, small)
- `PROFILES["mujoco"]` → `PROFILES["weld"]` in `isaac/isaac_env.py`.
- `ISAAC_PROFILES = {"isaac": "lehome", "isaac_weld": "weld"}` in `imitation/isaac_runtime.py`.
- `profile="mujoco"` call sites: `isaac/profile_check.py`, `isaac/reach_check.py --profile`, the `_prof` checks,
  `imitation/teachers/scripted.py` (`profile == "weld"`), the tests (`test_isaac_profile.py`,
  `test_isaac_weld_backend.py`) and the `verify` evidence text.
- Comments that say "MuJoCo profile" become "weld profile (settings carried over from the MuJoCo setup)".
- Recorded artifacts and results.md entries stay as written (append-only); a note records the rename.
- **Verify:** fast suite; `test_isaac_weld` + `test_isaac_profile` in .venv-isaac; `verify W1 W2 W3` PASS.

## Compute plan

- Isaac: track W ≤ 2 processes, track G 1 (16 GB GPU, already at 15.6 GB).
- MuJoCo: ≤ 10 workers, one pool at a time.
- Jobs over 2 h are resumable scripts, relaunched visibly with a polling Monitor.

Order, to stay within W's 2 Isaac slots:
0. **R rename** (minutes, no sim beyond the regression tests).
1. **W4 tail** (2 slots, about 1 h).
2. **Z1 zero-shot** (1 slot) running alongside **W3b recovery work** (1 slot).
3. **V vectorisation**: development needs 1 slot, the benchmark needs 2.
4. **W5 slim**, run on the vectorised env.

## Milestones (added to docs/roadmap.md Phase W)

### W4 (finish) — pilot tail
- **Run:** relaunch `scripts/isaac_pilot.ps1 -Backend isaac_weld`. It skips finished stages; the DAgger stage
  uses `--resume`.
- **Exit:**
  - all artifacts valid: dagger round_1, detector, agreement
  - diffusion id_easy Wilson lower bound > 0 (already met: 20/20)
- **Verify:** add a W4 check to `imitation/verify.py` (artifacts + bar), modelled on `_i32`.

### W3b — expert recovery ≥ 90 (85 if it plateaus)
- **Iterate on the tune block only:** recovery-style knocks on seeds 600_000+. The existing replay is
  `recovery_trace.py`; move it into the repo as `isaac/recovery_replay.py`.
- **Measure** n = 40 per attempt with Wilson intervals, and log every attempt in `docs/results.md`.
- **Candidates**, cheapest first. All are Isaac-expert-only (`isaac/weld_expert.py` / `isaac/isaac_env.py`
  weld path); the frozen MuJoCo code is not touched.
  1. A faster retry: re-approach from a lower height and skip the full approach waypoint.
  2. A per-phase patience scaled to Isaac's slower steps (MuJoCo's 45).
  3. Re-grasp handling after a knock drop: compensate the held offset only on re-grasps, where the offset
     exceeds 2 cm.
  4. A retry trigger using the settled miss with SUCCESS_DIST hysteresis.
- **Never relax:** the 250-step cap (MuJoCo parity) and the eval sets.
- **Plateau rule:** stop when two consecutive attempts gain < 2 pp. Accept ≥ 85; record a plateau below 85 and
  ask you.
- **Gate:** `scripts/w3_gate.ps1 -Sets recovery`. Add a `-Sets` parameter. n = 100 on the MuJoCo recovery eval
  set, then id_easy and id_hard re-run to confirm no regression.
- **Bars:** `verify W3` goes back to 90, with an 85 floor if plateau is recorded.
- **Collection:** W5 collects only after W3b. W4's pilot data keeps the old expert and is labelled as such.

### Z1 — zero-shot MuJoCo checkpoints on isaac_weld
- **Run:** `imitation.evaluate --backend isaac_weld` on the MuJoCo best checkpoints: diffusion DAgger at replan 4,
  best plain BC, and the vision student at replan 2. n = 20 on id_easy, id_hard and recovery, 1 worker.
- **Fallback:** if a checkpoint won't load in torch 2.7, add the smallest compatibility shim (e.g. weights-only
  load), documented.
- **Decision rule for W5:**
  - success ≥ 50% on id_easy → W5 fine-tunes from it (`imitation.train` init-from-checkpoint)
  - otherwise W5 trains from scratch
  - record either way
- **What exploration found:**
  - Checkpoints load unchanged: there is no backend check, `torch.load(weights_only=False)` is used, and the
    cameras match by name and size.
  - Fine-tuning works via `train.py --init-from` and `dagger.py --init`, but both keep the MuJoCo normalizer.
  - If W5 fine-tunes, add a `--refit-normalizer` option to `imitation/train.py` (around the normalizer block at
    `train.py:97-111`). Compare fine-tune with and without refit on a small run before W5.

### V — vectorised Isaac env (many cloth envs per process, 101×101 kept)

**What exploration found** (applies to the GPU weld profile only; the CPU friction profile gains nothing):
- **LeHome's `GarmentObject` is single-cloth.** It is one prim at a fixed path, with its own particle system and
  material. The way around it: spawn B cloth meshes under `/World/envs/env_i/cloth` and read them all through one
  regex `ClothPrim`. Isaac Sim's `ClothPrim` supports `(B, N, 3)` positions, velocities and masses with per-cloth
  indices.
- **The scene is single-env.** Arms, table, floor and cameras use absolute `/World/...` paths. They move to env
  namespaces with `clone_environments()` / `filter_collisions()` and `replicate_physics=False`.
- **env 0 is hard-coded** in `lab_scene.py` (`targets (1, 12)`, `site_pose [0, b]`, `_drive_pins`,
  `set_pinned_masses`, `particle_positions()[0]`) and in the `isaac_env.py` readers. All need an env index.
- **Per-env DR needs per-env particle materials.** With one shared material, only mass can vary per env.
- **Reset is per-env.** It writes one env's particles and joints by index while the others keep stepping (no
  global `lab.reset()`).
- **Workers need a step barrier.** The rollout driver is event-driven per env, so a worker hosting B sub-envs
  buffers step commands until all B have one. `label` and `resync` never step physics.
- **No precedent.** There's no particle-cloth example with num_envs > 1 in IsaacLab or LeHome, and Isaac Sim logs
  a deprecation warning for the particle-cloth API. That makes V0's go/no-go probe essential before investing.
- **Payoff on memory.** Each Kit process costs 3–6 GB of VRAM. Two processes reached only 1.5× aggregate
  throughput.
- **V0, feasibility probe:** one SceneEnv with num_envs = B (2, then 4) and B cloths under
  `/World/envs/env_i`. Check:
  - each cloth's particle view and its per-cloth masses / positions
  - per-env reset offset
  - per-env weld pins
  - camera rig per env (or state-only)
  
  Measure s/step versus B at 101×101. If GarmentObject can't be replicated, build the cloths directly via the
  particle-cloth API.
- **V1, batched env:** `IsaacClothFoldEnv` is batched internally. `B` sub-envs expose the same per-env gym API
  through a thin `SubEnv` view, so teachers and `HalfFoldEnv` stay unchanged. All per-env state gets a leading
  B axis: joint targets, gripper, pins, particles, DR, step count. Physics steps once for all B.
- **V2, EnvPool:** a worker hosts B sub-envs. The driver sees B× the slots, so the worker protocol stays as is.
  `WORLDFOLD_ISAAC_ENVS_PER_PROC` defaults to 1, so nothing changes until it is enabled.
- **Exit:**
  - Parity: the expert on isaac_weld with B = 4 matches B = 1 within Wilson CIs (id_easy n = 40).
  - DR is deterministic per seed regardless of slot.
  - Throughput ≥ 3× episodes/hour at equal VRAM. If < 3×, record the measured gain and keep it if > 1.5×.
  - The `test_isaac_weld` / profile tests pass.
- **Ownership:** design and verification stay with the lead. Scoped pieces go to sonnet subagents, logged in
  docs/subagents.md.

### W5 (slimmed) — Isaac weld retrain
- **Dataset:** `isaac_v1_weld`, 400 episodes, 30% perturbed, images at collection, collected on the W3b expert
  with the vectorised env.
- **Train:** diffusion seed 0 plus a check seed 1. Fine-tune from the MuJoCo checkpoint if Z1 says so. Vision BC
  at 1 seed. **Chunk-MLP dropped.**
- **Then:** diffusion DAgger, 2 rounds with takeover labels; detector ≥ 90%; n = 200 × 3 evals with Wilson; demo
  videos; a dated report with a MuJoCo-vs-Isaac side-by-side.
- **Exit:** `verify W5`, using the n / Wilson / SE rules from the MuJoCo phase (seed variance stated with 2
  seeds).

## Track G (parallel, unchanged brief)
- Friction-grasp IG.1–IG.3 on `feature/isaac-grasp`, 1 Isaac process.
- **Status at plan time:**
  - IG.1 is done (38b45d9).
  - IG.2 has had 4 attempts. A closed-jaw target of +0.05 rad fixed holding (left 20/20, right 14/20). Placed is
    5–9/20 and released 1–5/20 against bars of 95–98.
  - The agent paused for plan mode.
- **On approval:**
  - Resume the same agent: log trials 3–4, commit, then continue the IG.2 release knobs (jaw close sweep, open
    dwell, slower opening, calibrated place offset).
  - Rename-related edits are merged into its branch first, to avoid conflicts.
  - I delete its stray `outputs/isaac/grasp/smoke/bench.log` (untracked, written by its script).
- **Compute:** track G is compute-bound (30–50 min per n = 20 trial while W uses 2 slots). V would also free VRAM
  for G later.
- When it reports, I re-run its verify checks myself and summarise. Merging into `feature/isaac-imitation` needs
  your approval.

## Docs (every pass, CLAUDE.md)
- `docs/roadmap.md`: Phase W table gains W3b, Z1 and V; W4 and W5 rows updated (W5 slimmed, chunk-MLP dropped,
  cloth 101×101 fixed).
- `docs/status.md`: Next action, pass log with commit hashes, test counts.
- `docs/results.md`: every number first.
- `docs/subagents.md`: every delegation, including the track G agent.
- `docs/imitation.md` §10: Isaac env-per-process setting, the fine-tune decision.
- Copy this revision to `docs/superpowers/plans/2026-10-04-isaac-weld-revision.md`.

## Critical files
- **Modify:**
  - `isaac/weld_expert.py` (W3b)
  - `isaac/isaac_env.py`, `isaac/lab_scene.py` (V; weld path only for W3b)
  - `imitation/rollout.py`, `imitation/isaac_runtime.py` (V2)
  - `imitation/verify.py` (W4, Z1, V, W5 checks; W3 bar)
  - `scripts/w3_gate.ps1` (`-Sets`)
  - the W5 driver script `scripts/isaac_retrain.ps1` (new)
- **Reuse:**
  - `isaac/overshoot_measure.py`, the `recovery_trace` logic, `imitation.evaluate`
  - `imitation.train` / `imitation.dagger --init`
  - the `EnvPool` lifecycle
  - `isaac/profile_check.py` and `isaac/weld_check.py` as regression tests for V

## Verification
- For each milestone: `python -m imitation.verify <ID>` PASS in both venvs, with evidence committed.
- Fast suite in `.venv`; `-m isaac` weld / profile tests in `.venv-isaac`.
- MuJoCo expert benchmark rows stay identical to M1.5 after any FoldExpert-side change.
- Numbers are in results.md with n, seeds and Wilson intervals before any other doc.

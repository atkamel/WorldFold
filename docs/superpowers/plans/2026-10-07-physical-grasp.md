# Phase F — a physical grasp on LeHome's Isaac physics, then retrain and raise the benchmarks

## Context

The W5 report (`docs/reports/2026-10-07-isaac-half-fold.md`) shows 99 / 97.5 / 54.5 for the privileged policy and 99 / 82.5 / 31 for the sensor-only student. Those numbers come from the `weld` profile, where the grasp is a cheat:
- `_try_weld` pins the cloth particles once the gripper site is within 3 cm of the corner.
- `_targets()` forces both jaws open in weld mode.

So the video shows an arm hovering near the cloth with its jaws open while the cloth follows it. The goal is a grasp in which the jaws close and only contact forces move the cloth (LeHome's friction + adhesion), then a retrain on it and a push on both benchmarks.

### What the research found

**Ruby's `origin/feat/isaac-half-fold` (e188194):**
- Her "anchor grasp" is another kinematic attachment: a 4 mm kinematic sphere, attached to the cloth with `PhysxPhysicsAttachment` and teleported every substep.
- It engages purely on distance (6 cm radius, no contact check).
- Her friction-grasp numbers were 5.9% for the expert and 1/10 for BC. All 204 slips she recorded happened with the jaws shut, at a median of 4 cm into the lift.
- **Worth borrowing:** the Markov expert in `isaac/half_fold_expert.py`:
  - lift only once the jaws are shut and still (jaw < 0.25, speed < 0.05);
  - a hold check (`HOLD_RADIUS` 4 cm) that sends a miss back to re-grasp;
  - replanning when the corner moves more than 1 cm;
  - `CARRY_SPEED`;
  - `PINCH_OFFSET` 5 mm.
- Per the user, the anchor is also kept as a comparison profile.

**Our paused track G (`feature/isaac-grasp` @ `6c8c5af`, worktree `.claude/worktrees/agent-a55d8fa30a92a92f8`):**
- Friction only, n = 20. Left / right: acquired 20/20, held 20/19, placed 19/18, released (new definition) 20/19.
- Expert env success 17/20 with retries off.
- The fixes that got there:
  - closed-jaw target +0.05 instead of −0.1 (at −0.1 the jaw overlaps the pad and squeezes the cloth out);
  - pinch height 0.5 cm;
  - a per-arm friction overshoot.
- Never gated at n = 100. It runs on the CPU, single env, and is behind V (the vectorised env) and the `_Copy` refactor.
- Caveat: the closed-jaw +0.05 is not actually a default in `FRICTION_GRASP`; it is still all `None` there.

**Web (LeHome official repo, NVIDIA forums, DexGarmentLab):**
- LeHome grasps with friction plus adhesion, with no attachments. Its defaults are friction 0.5, adhesion 0.1, `adhesion_offset_scale` 0, gravity_scale 2 and particle_mass 1e-2.
- We currently override adhesion to 0 and gravity scale to 1.
- NVIDIA staff's fixes for cloth slipping or penetrating:
  - convex-hull finger colliders (not SDF);
  - lower gripper gains (stiffness about 100);
  - 240–360 Hz physics (we run 100 Hz);
  - friction set on both the cloth and the pads;
  - stop the jaws at a gap of about 2 × rest offset rather than squeezing through.
- LeHome's winning team used DAgger-style intervention data plus advantage-weighted RL on a VLA.

### User decisions (2026-10-07)

- **Grasp rules:** LeHome-style. Friction plus adhesion, adhesion ≤ 0.3 with a release test. Pad, jaw-gap, drive-gain and physics-rate changes are allowed. No attachments, pins, welds or kinematic writes to particles.
- **Ruby's branch:** port her expert ideas and also port the anchor, as a third comparison profile.
- **Benchmark push:** friction profile only. Weld results are frozen as the reference.

---

## Milestones (Phase F, added to `docs/roadmap.md`)

Each milestone closes on `python -m imitation.verify F<n>` plus committed evidence. Numbers go to `results.md` first, with n, seeds, dataset hash and Wilson intervals. The heat cap holds throughout:
- ≤ 2 Isaac processes × 4 envs;
- `scripts/thermal_guard.py` at 90 / 84 °C;
- long evals run with `--resume`.

### F0 — Bring track G forward onto the vectorised GPU env (gate)

Steps:
1. Merge `feature/isaac-grasp` into `feature/isaac-imitation`.
   - Resolve the `lab_scene.py` conflicts against V / `_Copy`.
   - The new files `isaac/grasp_bench.py`, `isaac/grasp_metrics.py`, `scripts/grasp_bench.ps1`, `tests/imitation/test_grasp_metrics.py` and the IG.1/IG.2 checks in `verify.py` come over as they are.
   - Bring the IG results over into `results.md`.
2. Add a new `PROFILES["friction"]` in `isaac/isaac_env.py`.
   - It is the weld profile's setup (cloth pose and jitter, arm drive `MUJOCO_ARM_DRIVE`, dynamics DR, cap 250, MuJoCo eval sets, `cuda:0`, vectorised) with `grasp_mode="friction"`.
   - It uses track G's #9 knobs written as real defaults: closed jaw +0.05, pinch 0.5 cm, `OVERSHOOT_FRICTION`.
   - The weld code is unreachable in this profile.
3. Gate: the grasp bench on the friction profile at B = 4 on the GPU reproduces #9 within its interval (n = 20 per arm). The friction grasp has only ever run on the CPU pipeline, so if the GPU particle contact behaves differently, record it and re-tune in F2.

**Exit:** the bench is within interval of #9, or a documented delta plus a re-tune task. Fast tests green.

### F1 — Grasp honesty and observability

Steps:
1. Add `isaac/honesty_check.py` (verifier F1), which asserts on the friction profile:
   - no pins: `_pinned` stays empty and `set_pinned_masses` is never called;
   - no `PhysxPhysicsAttachment` or AutoAttachment prims on the stage;
   - jaw targets are not forced open;
   - and, every step while the gripper holds the cloth: the jaws are physically closed and the corner is within the jaw pocket.
2. Redefine `grasp_active()` on friction to be physical:
   - the jaw is closed and settled;
   - the corner vertex is inside the pad box (`PAD_*` geometry);
   - the corner's z tracks the gripper once lifted.
   - The current version is only a proxy based on the closed command and distance.
   - This feeds `failure_code` (G1 / slip), the expert, and the obs grasp flags. Obs sizes are unchanged and still read from `imitation/spec.py`.
3. Add a slip code to `imitation/evaluate.py::failure_code`: G2 means the corner was lost while the jaws were shut, which separates slips from misses.
4. Add a demo overlay: jaw gap, the contact/held flag, the slip event.

**Exit:** the honesty check passes on 20 expert episodes and fails on the weld profile (a negative control).

### F2 — Grasp reliability loop (IG.2 continued, under the LeHome-style rules)

- One knob per iteration on the tuning seeds 600k+. The order is cheapest and most likely first:
  1. adhesion 0 → 0.1 (LeHome's default) → 0.2 → 0.3, each with the release test;
  2. `adhesion_offset_scale` 0 → 0.5;
  3. gravity scale 1 vs 2 (LeHome's default);
  4. the closed-jaw gap, a sweep around +0.05;
  5. gripper drive stiffness and effort, lowered toward about 100;
  6. pad friction and thickness, convex-hull pad colliders;
  7. physics rate 100 → 200 Hz (`PHYSICS_DT` / `n_substeps`; measure the throughput cost);
  8. particle friction ≤ 2.0.
- Each knob is kept only if held or placed gains more than 1 SE without released regressing.
- Ruby's lift rule (lift only once the jaw is settled) and a slower `CARRY_SPEED` are tried in the expert in the same loop.

**Exit (the existing IG.2 bar):** per arm, n = 100 on two blocks plus a fresh block, with acquired / held / released ≥ 98 and placed ≥ 95. The plateau rule: ≥ 95 / 90 if two iterations in a row gain < 2 pp, and that is recorded.

### F3 — Expert on the physical grasp, with Ruby's Markov ideas (IG.3)

- Extend `IsaacFoldExpert` (`isaac/weld_expert.py`, which subclasses `cloth_fold_rl/expert.py::FoldExpert`), or add a friction subclass, with:
  - a jaw-settle wait before lift;
  - a physical hold check, where a miss or slip leads to open, lift 3 cm and re-approach the corner's current position (this is the re-grasp that limited weld recovery to 81.7%);
  - replanning on corner moves > 1 cm;
  - a carry-speed limit;
  - an approach along the corner's outward diagonal (from `feat/quarter-fold-physical-grasp`) if F2 shows pushing misses.
- Keep `imitation/teachers/scripted.py` and the DAgger takeover path working with the new phases.
- Also port Ruby's anchor as `PROFILES["anchor"]`.
  - Port `_spawn_anchors`, `_update_anchor` and `_drive_anchors` from `origin/feat/isaac-half-fold`, adapted to the vectorised `_Copy`.
  - The expert gate is run on it only as a comparison row, not for training.

**Exit (IG.3):** expert n = 100 with id_easy ≥ 95, id_hard ≥ 90, recovery ≥ 90 (≥ 85 under the plateau rule), and check_resync ≥ 90. Weld, anchor and friction expert rows go side by side in `results.md`.

### F4 — Retrain on the friction profile

The same pipeline as W5, via `scripts/isaac_retrain.ps1` with a `-Profile friction` switch:
1. Collect `isaac_v1_friction`: 400 episodes. The knocked share goes up from 30% to 50%, because recovery is the gap.
2. Diffusion BC × 2 seeds, 30k steps. Warm-start from the W5 weld checkpoints is tried only if a 20-episode zero-shot check gives id_easy ≥ 50%, as in Z1.
3. DAgger, 2 rounds, takeover p = 0.6. Keep a round only if it gains on the SE rule.
4. Replan sweep {2, 4, 8}.
5. Vision student, round 0 recipe.
6. Detector.

**Exit:** n = 200 × 3 sets with Wilson intervals, and side by side with weld W5.

### F5 — Raising both benchmarks (A/B, one change at a time, SE rules, n = 200)

**Privileged teacher** (target: recovery ≥ 75, id_hard ≥ 95):
1. Recovery-weighted data. Upweight takeover and intervention frames ×3 in training (the LeHome-winner and HG-DAgger idea).
2. More DAgger rounds until two rounds in a row gain < 1 SE.
3. A seed ensemble at inference: average the denoised chunks from 2 seeds. Its 12 pp seed variance is the biggest noise source.

**Student** (target: id_hard ≥ 90, recovery ≥ 45):
1. Student DAgger: student rollouts, relabelled by the privileged teacher, 2 rounds.
   - On MuJoCo this transferred the teacher's id_hard weakness.
   - On Isaac the teacher is strong on id_hard (97.5), so it is worth re-testing.
2. Image augmentation: random crop ±4 px, colour jitter.
3. Recovery-weighted frames, as for the teacher.
4. An auxiliary head that predicts corner positions from images, trained on privileged labels (the "privileged state only in training" idea). This targets the re-grasp failures: 115 of the student's 138 recovery failures were missed re-grasps.

**Exit:** the best configuration of each, and a ❌ with the measured margin for any target missed.

### F6 — Report, demos, close-out

- Six demo videos on the friction profile, plus one weld vs friction side-by-side clip showing the jaws closing on the cloth.
- A dated report `docs/reports/2026-10-xx-isaac-physical-grasp.md` plus an HTML page.
- `verify F0..F6` PASS.

---

## Critical files

| file | change |
|---|---|
| `isaac/isaac_env.py` | `PROFILES["friction"]` and `["anchor"]`; knob defaults; physical `grasp_active` |
| `isaac/lab_scene.py` | merge track G knobs with V / `_Copy`; pad colliders; adhesion; physics rate; anchor spawn |
| `isaac/grasp_bench.py`, `isaac/grasp_metrics.py` | from `feature/isaac-grasp` |
| `isaac/weld_expert.py` (or a new `isaac/friction_expert.py`) | Markov re-grasp logic |
| `imitation/evaluate.py` | G2 slip code |
| `imitation/verify.py` | F0–F6 checks (`@check`), IG.1/IG.2 from the grasp branch |
| `imitation/teachers/scripted.py` | the friction overshoot and new phases |
| `scripts/isaac_retrain.ps1` | `-Profile` switch |
| docs | `roadmap.md` (Phase F), `status.md`, `results.md`, `imitation.md` §10 (new profile), `subagents.md` |

To reuse rather than rewrite:
- the `imitation/verify.py` helpers (`wilson`, `eval_counts`, `results_md_has`);
- `imitation/seeds.py` eval sets (tune block 600k);
- `isaac/pinch.py::PinchIK`;
- the `evaluate --resume` path;
- the `scripts/thermal_guard.py` lockstep vectorised worker (`imitation/lockstep.py`).

## Delegation (per CLAUDE.md)

| work | model |
|---|---|
| F0 merge-conflict resolution; F2 sweep plumbing; F4 run orchestration scripts | Sonnet |
| doc table updates; log sweeps | Haiku |
| the `grasp_active` redefinition; expert re-grasp design; F5 experiment selection; final verification of every milestone | lead |

Every delegation is logged in `docs/subagents.md`.

## Verification

- `.venv/Scripts/python.exe -m pytest -m "not slow"` and `-m slow`, plus the Isaac tests in `.venv-isaac` (`-m isaac`), after F0, F1 and F3.
- `python -m imitation.verify F0` … `F6`; the lead re-runs each one personally.
- Expert benchmark rows for the MuJoCo and weld profiles are unchanged after the merge (regression check).
- Demo frames are inspected visually: jaws closed on the cloth, no hovering.
- Every pass ends with the status.md / roadmap / results updates and one commit, per CLAUDE.md.

## Risks

- **GPU particle contact may differ from the CPU results of track G.** F0 gates this first.
- **200 Hz physics roughly halves throughput.** F2 measures it before adopting.
- **Adhesion can make the cloth stick on release.** The released metric gates it.
- **A physical recovery ceiling could still sit below 90.** The plateau rule and the user's call apply, as in W3b.

# Phase F — pause F4, fix the expert first (re-plan, 2026-10-08 ~01:40)

## Context

The user: "pause F4 first … we're getting ahead of ourselves if the IK expert is failing at recovery."

They are right. Every F4 stage learns from the scripted expert:
- BC learns from its demos;
- DAgger learns from its takeover labels;
- the vision student learns from the privileged teacher.

### Where the expert stands (F3, n = 100 per set)

| set | success | failures |
|---|---|---|
| id_easy | 94 | 5 G2 slips, 1 drag |
| id_hard | 94 | 6 G2 slips |
| recovery | **27** | 53 G2, 20 M1 |

- **REGRASP** (26) and **PINCH_FOLLOW_Z** (29) did not help. Both changes were guesses.

### Offline evidence (82 F4 collection episodes so far)

- **Every post-knock corner loss happens with the jaw closed** (50 of 50).
- **Re-grasps are slow:** median 45 steps after the knock.
- **Re-grasped corners are lost again** 2–160 steps later.
- **Some `cloth_dragged` endings come from the expert's own re-carry,** not the knock. DRAG_LIMIT is 20 cm of anchor
  drift.
- **4 of the first 28 clean episodes truncated at 400 steps.**
- **Best reading:** after a knock the re-grasp pinches a fold near the displaced corner, not the corner tip. This is
  unconfirmed: there are no per-step phase or position traces.

## Plan

### 1. Pause F4 now

- Stop `scripts/f4_retrain.ps1` and its collect workers. Collect is at 82/400 and unfrozen.
- **Discard the partial `isaac_v1_friction` and `_failures`.** They are demos from an expert we are about to change.
  - The write-once rule covers frozen versions only.
  - Archive the failures' journal to `outputs/imitation/isaac_friction/f4_aborted/` first, as diagnosis evidence.
- Record the pause in status.md and the pass log.

### 1b. Redefine the perturbation benchmarks (user, 2026-10-08: "choose something more realistic")

**Why the current knock is unrealistic.**
- Today a knock is k = 8–15 steps of uniform-random actions on all 12 dims, including both gripper commands (dims 5
  and 11). Each step, each jaw is commanded open (> 0.3) or closed (< −0.3) about 35% of the time.
- So a knock randomly drops a held corner, and randomly clamps on cloth while the arm jerks around.
- About 20% of episodes end right there as `cloth_dragged` (the anchored half moved more than 20 cm). No policy can
  recover from that.
- A real disturbance does not toggle the gripper at random.

**"Gripper-free knock", what it means.**
- The same 8–15 steps of random joint deltas on the 5 arm joints. Each jaw keeps the command it had when the knock
  began.
- A held corner stays held unless the jerk physically pulls it out: a bump to the arm.
- An open jaw stays open: a bump while reaching.
- The physics, not a random gripper click, decides whether the corner is lost.

**Proposed suite** (replaces the single `recovery` set as the gated recovery metric; n = 100 each; seeds are disjoint
blocks inside the recovery range; the onset is drawn in (35, 140) as today):

| set | what happens | real-world analogue | what it tests |
|---|---|---|---|
| `knock_arm` | k = 8–15 random arm-joint deltas, jaw command held | someone bumps the arm | disturbance rejection, re-plan from a displaced pose |
| `drop` | while a corner is held, the jaw is forced open for 3 steps; then the policy continues | the cloth slips out of the grip | **a true re-grasp from zero on a displaced corner** (the user's "just another grasp") |
| `joint_noise` | Gaussian noise (σ = 0.15 of the max joint delta) added to every executed arm action, whole episode | servo noise, backlash | closed-loop robustness |
| `overshoot` | executed arm deltas scaled by a per-episode gain U(1.2, 1.4) | miscalibrated servo gains, overshoot | tolerance to dynamics mismatch |

- **The legacy `recovery` set** (the random all-dims knock) is still run and reported as a continuity row against the
  MuJoCo and weld reports. It is no longer gated.
- **Training data** (F4): collection spreads its 50% perturbed episodes across the four kinds.
  - For `joint_noise` and `overshoot`, the recorded label is the expert's clean action and the executed action is the
    noisy one (DART-style noise injection).
  - That teaches the student to correct its own drift without extra DAgger rounds.
- **Implementation:**
  - `imitation.rollout.Perturbation` gains `kind` and parameters.
  - One pure function, `apply_perturbation(kind, action, rng, ...)`, is used both for teacher steps (in the worker:
    `step` carries the perturbation) and for policy steps (in the parent's queue).
  - `imitation.seeds.eval_set` gains the four sets.
  - `imitation.data.collect` gets a perturbation mix.
  - Sim-free tests: determinism per seed, the jaw held under `knock_arm`, labels clean under noise.
  - Spec changes go in `docs/imitation.md` §4 (eval sets).

### 2. F3b — expert recovery and slips, evidence first (new roadmap row, before F4)

**a. Instrumentation.**
- `isaac/recovery_replay.py` (friction backend, already added) gains a per-step trace file per episode, `--trace-dir`.
  Each step records, per arm:
  - phase, jaw angle;
  - gripper site, corner position, corner–jaw distance;
  - which grid vertices lie within 2 cm of the jaw;
  - the corner's height and whether cloth lies above it (a fold).
- `isaac/grasp_bench.py` traces already carry most of this; reuse its `record()` fields.

**b. Diagnosis run.**
- 20 knocked episodes on tune seeds 610000–019, plus the 4 clean truncations' seeds re-run with traces.
- 2 Isaac processes in parallel (single env each), about 25 min.
- Classify each failure by mechanism:
  - re-grasped a fold, not the corner;
  - corner folded under the cloth;
  - re-carry drag;
  - slow descend / timeout;
  - slip on a good grip.
- Numbers go in results.md.

**c. Fix loop** (W3b-style):
- One mechanism-targeted change per attempt.
- Recovery replay on tune seeds, n = 40, paired against the base on the same seeds (2 lanes, about 35 min each).
- Kept only on a gain of 1 SE or more, with the id_easy spot check (n = 20) not regressing.
- **Candidate fixes, in order:** the evidence picks which.
  1. **Corner-follows check:** after the lift starts, if the corner hasn't risen with the jaw (rise < 50% of the jaw's
     after 5 steps, or the corner–jaw distance exceeds HOLD_DIST), open, back off 3 cm, re-plan on the corner's current
     pose and re-approach. Grabbing a fold becomes a quick retry instead of a carried failure.
  2. **Fold-aware pinch target:** if cloth lies above the corner, aim the pinch at the topmost corner-adjacent particle,
     or flatten first with a short drag-out.
  3. **Gentler re-carry after resync:** a lower arc and a slower carry (`CARRY_SPEED`) to stop expert-caused drags.
  4. **Resync descend patience:** cap descend steps after a resync; the 30–85-step descends burn the 400-step budget.

**d. Exit.** F3 re-run on id_easy, id_hard and the four perturbation sets (n = 100 each, 2 × 8):
- id_easy ≥ 95, id_hard ≥ 90, G2 ≤ 5% on both;
- **each perturbation set ≥ 60%**: the user's floor, "get it above 60% first", with ≥ 80% the aim on `drop` and
  `knock_arm`;
- the legacy `recovery` set is reported, not gated;
- plateau rule: two attempts in a row gaining < 2 pp ends the loop, and the user decides.
- The fix loop (c) runs on tune-seed versions of `drop` and `knock_arm` first: they are the re-grasp tests.

### 3. Then F4 from scratch

- `scripts/f4_retrain.ps1` (fast path, 2 × 8, cameras at 2 × 6) with the fixed expert.
- About 13–14 h, since collection proved to be about 48 s per episode with cameras.

## Time

| step | wall time |
|---|---|
| pause + archive | 5 min |
| perturbation suite (code + tests, no sim) | ~1 h, in parallel with the diagnosis run |
| instrumentation | 30–45 min (code + tests, no sim) |
| diagnosis run | ~25 min |
| fix attempts | ~35–45 min each, 2–5 expected: 1.5–4 h |
| F3 re-gate | ~1.5 h |

So F4 restarts in about 4–7 h, depending on how many attempts the fixes take.

## Files

| file | change |
|---|---|
| `isaac/recovery_replay.py` | `--trace-dir` per-step traces; expert knobs via `WORLDFOLD_EXPERT_PARAMS` (already read on import) |
| `isaac/fold_expert.py` | the kept fixes, each a knob defaulting off until kept: `FOLLOW_CHECK`, fold-aware pinch, resync descend cap |
| `docs/roadmap.md` | new F3b row; F4 marked paused |
| `docs/status.md`, `docs/results.md` | pause, diagnosis, attempt log |
| `imitation/rollout.py`, `imitation/seeds.py`, `imitation/data/collect.py` | perturbation kinds, the four eval sets, the collection mix, DART labels |
| `docs/imitation.md` | eval-set spec |
| `imitation/verify.py` | F3: each perturbation set ≥ 60%; legacy recovery reported only |

## Verification

- Fast tests after each code change.
- Every attempt's paired numbers go in results.md: n, seeds, Wilson.
- F3 re-gate artifacts are committed, then `verify F3`.

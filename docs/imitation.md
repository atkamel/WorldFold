# Imitation pipeline — design doc

Source of truth for the `imitation/` package: the task, the observation and action
spec, the episode schema, the metric definitions, the seed discipline, and the
deployment budget. Modules cite this file by section.

Companions: [roadmap.md](roadmap.md) (what we are building, in order),
[status.md](status.md) (where we are right now), [results.md](results.md) (every
number we have measured).

---

## 1. Task

**Half fold** — stage 0 of the two-stage quarter fold, which is the largest unit the
scripted expert does reliably enough to bootstrap from.

`imitation.tasks.HalfFoldEnv` subclasses `cloth_fold_rl.quarter_fold_env.QuarterFoldEnv`
and sets `stages = STAGES[:1]`, so the episode ends in **success** when stage 0 settles
rather than advancing to stage 1. Max 250 steps (`HALF_FOLD_MAX_STEPS`), vs 400 for the
full quarter fold.

Stage 0 is two simultaneous arm moves, folding the cloth about y = 0:

| arm | carries vertex | goal | anchors held down |
|---|---|---|---|
| `left_` | `CLOTH_10` (10) | start position of `CLOTH_0` | (0, 0) |
| `right_` | `CLOTH_120` (120) | start position of `CLOTH_110` | (110, 110) |

Cloth is an 11×11 flexcomp grid (121 vertices, `cloth_0..cloth_120`, row-major
`ix*11 + iy`), 3 cm spacing. Grasping is a **weld cheat**: closing a gripper activates an
equality constraint to any eligible vertex within `GRASP_RADIUS = 0.06` of the gripper
site. `weld_mask` restricts which vertices each arm may weld per stage, because vertex 10
appears in both arms' `GRASP_CORNERS`.

Control: `control_dt = 0.05` (20 Hz), 100 physics substeps per control step.

## 2. Observation

`Box(139,) float32` (`imitation.spec.OBS_DIM`) — `QuarterFoldEnv._observe` takes the
`StateOnlyWrapper` (`mujuco/sim_main.py:627`) flattening of three keys, drops the task
one-hot and appends two wrapper-state dims:

| block | dims | contents |
|---|---|---|
| `proprio` | 50 | per arm ×2: 5 joint pos, 5 joint vel, 1 gripper ctrl, 3 EE pos, 4 EE quat, 6 EE vel (ang+lin), 1 grasp flag |
| `cloth_state` | 69 | 4 corner pos (12), 4 corner linvel (12), 9 sampled vertices (27), CoM (3), z min/max/mean (3), 4 corner-to-goal (12) |
| `task` | 18 | mean corner progress (1), goal keypoints 4×3 (12), per-corner progress (4), time-left (1) |
| wrapper | 2 | `stage / n_stages`, `settle_steps / SETTLE_STEPS` |

Corner order throughout is `corner_idx = [0, 10, 110, 120]`. Goal keypoints are set per
stage by `QuarterFoldEnv._set_goals`: a carried corner's slot holds its move's goal, every
other slot holds that corner's position at the stage start. Corner-to-goal, progress and
the mean-progress scalar are all measured against these goals.

### 2.1 Sensor-available vs privileged

This split determines how much of Phase 4 (sensor-only student) is real work:

- **Sensor-available: 48 of 50 `proprio` dims.** Joint pos/vel from encoders, gripper
  command, EE pose/vel from forward kinematics. The two exceptions are the per-arm
  `grasp_active` flags, which are MuJoCo weld queries with no direct sensor analogue
  (substitute: gripper position + current).
- **Privileged: `cloth_state` (69) + `task` (18) + wrapper (2) = 89 dims.** Cloth vertex positions and
  velocities read straight from the simulator.
- **Reward, success and termination are privileged and stay that way.** They are labels,
  not policy inputs. Hardware evaluation needs a separate vision-based success detector.

### 2.2 Observation defects (fixed in M1.1)

All three were fixed in M1.1, before any dataset was frozen. Kept for the record; they
describe the 141-D observation, which no frozen data uses. `cloth_offset_xy` is also now
recorded in `domain_params` (it was drawn by the wrapper and bypassed the base env's record).

1. **The goal channels describe the wrong fold, permuted.** `sim_main.py:463-469` sets
   `_goal_corners = corners0[[3, 2, 1, 0]]` — each corner's diagonal-opposite start
   position — **unconditionally, ignoring `_task_id`**. The half fold's real goals are
   present but in swapped channels:

   | needed | lives in channel |
   |---|---|
   | goal for left arm's v10 = start(v0) | index 3 (v120's slot) |
   | goal for right arm's v120 = start(v110) | index 1 (v10's slot) |

   So the policy must learn a permutation to use its own goal signal. Affects the 12
   `goal keypoints` dims and the 12 `corner-to-goal` dims in `cloth_state`.
2. **~5 dims are dead.** The 4-dim task one-hot is constant (`HalfFoldEnv` pins task 0),
   and the `stage` scalar plus 4 `progress` dims are measured against the diagonal goal
   from defect 1, so they sit near 0 for the whole episode.
3. **Termination is not observable.** `QuarterFoldEnv.stage` and `_settle_steps` (0-20,
   `quarter_fold_env.py:232`) are wrapper state absent from the observation. The
   observation therefore cannot express "2 steps from the success bonus", which makes the
   reward and termination **non-Markovian w.r.t. the observation** — a hard blocker for
   offline RL (Phase 5), and the reason M1.1 must land before any dataset is frozen.

## 3. Action

`Box(-1, 1, (12,)) float32`, expanded to the base env's 14-D at
`quarter_fold_env.py:213`:

```
[0:5]  left joint deltas    [5]  left gripper
[6:11] right joint deltas   [11] right gripper
```

Joint deltas are scaled by `JOINT_DELTA_SCALE = 0.05` rad/step and applied as
`ctrl = qpos + delta * scale`, clipped to actuator range. Gripper uses hysteresis:
`< -0.3` closes, `> 0.3` opens, otherwise holds.

Policies emit **chunks**: K = 16 actions, re-planned every 8 steps. The executed chunk is
open-loop for those 8 steps, which is also the macro-action the offline-RL stage should
treat as its action (Phase 5).

## 4. Episode schema

Defined and documented in `imitation/data/schema.py` — see that module's docstring for the
per-episode array list, which is authoritative. Properties that matter:

- **One npz per episode**, episode boundaries preserved (unlike the older flat
  `cloth_fold_rl/collect_demos.py` `demos.npz`).
- **Write-once, content-hashed.** `freeze()` writes `manifest.json` with a sha256 per
  episode; `load_dataset` verifies every hash. A version lists its parent's episodes plus
  its own, so DAgger aggregation never copies or mutates earlier data.
- **Streaming, resumable.** Each episode is written as it finishes (`<source>_s<seed>.npz`,
  atomic rename) and journaled in `journal.jsonl` until `freeze()`; `collect --resume`
  continues a killed run, and a non-empty unfrozen version is refused without it.
- **Transition flags.** `terminated` (success, `cloth_dragged`), `truncated` (step cap, and
  `unstable` — a solver blow-up is not an MDP outcome, so it must not zero the bootstrap)
  and `discount` (0 only at a true terminal) are stored per step, last step only.
- **Failures are kept**, tagged by `termination_reason`, in their own version
  `<version>_failures` (collect default since M1.4). `train` additionally drops any failed
  *expert* episode via `bc_episodes` unless `--allow-failures`; DAgger episodes are kept
  regardless of outcome because their teacher labels are valid (see §5.3).
- `actor` per step records who chose the executed action (teacher / student /
  perturbation); `labels` holds teacher chunks at the steps DAgger queried.

Auditing a result means: `run.json` (git commit, dataset hash, config, checkpoint hash,
written by `imitation/train.py:36`) + the dataset `manifest.json`. The docs index these;
they do not restate them.

## 5. Metrics

### 5.1 Success and fold score

- **Success:** every move in the stage is simultaneously *not grasped* and *placed* (all
  carried vertices within `SUCCESS_DIST = 0.05` m of goal) for 20 consecutive steps
  (1.0 s). Releasing is part of the task — a held corner never counts.
- **Fold score:** fraction of the total required carry distance covered, `fold_score()` in
  `quarter_fold_env.py:168`. Continuous progress measure; reported as the mean final value.
- **Failure termination:** `anchor_drift > 0.20` m → `cloth_dragged`; `|qacc| > QACC_LIMIT`
  → `unstable` (a solver blow-up, not a task failure — see §7); step cap → `truncated`.

### 5.2 Failure codes

From `imitation/evaluate.py:27`:

| code | meaning |
|---|---|
| G1 | missed or unstable grasp (an arm never grasped, or dropped its unplaced corner) |
| M1 | motion error: anchor corner dragged, or sim unstable |
| F1 | fold placed with poor alignment (released off target) |
| S1 | stalled: timed out still holding or hovering |
| R1 | no recovery after a disturbance |

**Known defect:** `failure_code` returns R1 for *any* failed perturbed episode before the
mechanistic checks run, and every `recovery` episode is perturbed — so that set's
histogram is uniformly R1 and carries no information. Report the mechanistic code plus a
separate `perturbed` flag (roadmap M2.2).

### 5.3 Statistical discipline

Non-negotiable, because the differences being chased are a few percentage points:

- **Eval n ≥ 200** per set. Eval needs no teacher labeling, so it is cheap; n = 48 gives
  ≈7 pp standard error, which is wider than every effect of interest.
- **Report Wilson intervals**, not bare rates.
- **≥ 2 training seeds per config.** Otherwise "diffusion beats MLP" is indistinguishable
  from initialization noise.
- **Express improvement thresholds in standard errors**, not raw percentage points.
- Every reported number goes in [results.md](results.md) with its run id, dataset version
  and n.

## 6. Seed discipline and evaluation sets

Disjoint ranges, so training data never shares a cloth start with evaluation
(`imitation/seeds.py`):

| range | use |
|---|---|
| 0 .. | expert demos (training) |
| 50 000 + 1000·round .. | DAgger rollouts |
| 100 000 .. | `id_easy` |
| 200 000 .. | `id_hard` |
| 300 000 .. | `recovery` |

- **`id_easy`** — same start distribution as the demos.
- **`id_hard`** — cloth offset on a ring outside the training jitter (≥2.5 cm, up to 4 cm
  in one axis). Note this is an **out-of-distribution pose test only**: mass, friction and
  damping randomization (±30%) is unchanged, so it is not an OOD-dynamics test.
- **`recovery`** — `id_easy` starts, knocked off course mid-episode (random joint
  perturbation of 8-16 steps, starting between steps 15 and 60).

**Any number measured on seeds 0-49 is a training-set number** and cannot be compared to a
student. The existing 48/50 expert benchmark is one of these; establishing the expert
ceiling on the three eval sets is roadmap M1.7.

There is currently **no generalization axis beyond pose and dynamics** — no held-out fold
variant, cloth size or stiffness. Roadmap M6.1 adds one.

## 7. Reward

`r = Φ(s') − Φ(s) − CTRL_COST·‖a‖²` (`quarter_fold_env.py:230`), a potential difference.
Usable as an offline-RL label with four caveats, all of which Phase 5 must handle:

1. **Exploitable discontinuities.** Φ jumps by `GRASP_BONUS = 3.0` on grasp and
   `RELEASE_BONUS = 3.0` on release-while-placed, against `SUCCESS_BONUS = 20.0` and a
   carry term of ≈2.4. With gripper hysteresis, a flickering grasp emits ±3.0 reward spikes
   per toggle. A Q-learner will find that.
2. **Shaping bias.** Potential-based shaping is policy-invariant only at γ = 1.
3. **`unstable` is a simulator artifact.** The −5 penalty fires on solver divergence, not
   on the robot doing something bad, and those transitions are the most likely to relabel
   under a MuJoCo version change. Tag them separately.
4. **Markov w.r.t. the observation since M1.1** — `stage` and `settle_steps` are observed
   (§2.2 defect 3).

Recommendation on record: learn the critic on the **sparse** signal (success + small time
penalty), using the shaped reward only to warm-start.

## 8. Deployment budget

Declared here up front so architecture choices respect it throughout, rather than being
audited after the fact. The deployed policy runs next to the robot with no VLA, no world
model and no offline-RL machinery in the stack.

| property | budget |
|---|---|
| observation | RGB-D + proprioception only — no simulator state |
| control rate | ≥ 20 Hz sustained (matches `control_dt = 0.05`) |
| inference latency | < 25 ms per chunk re-plan, single consumer GPU |
| parameters | ≤ 50 M |
| dependencies | no 20 GB VLM checkpoint at runtime |

Diffusion policies must meet the latency budget with batched DDIM sampling (~10 steps) or
they do not ship, regardless of success rate.

## 9. Reproducibility

- Pins: `imitation/requirements.txt`. **mujoco 3.10.0 / so101-nexus 0.4.8** — the versions
  the expert benchmark was measured on. This deliberately differs from
  `cloth_fold_rl/requirements.txt` (3.11.0 / 0.5.1, which its committed checkpoint was
  verified against). Cloth grasping is contact-dominated, so MuJoCo minor versions change
  results; upgrading invalidates the benchmark and every dataset collected under it.
- Tests: `pytest -m "not slow"` for the fast suite, `-m slow` for the MuJoCo ones
  (snapshot/restore determinism, teacher labeling side-effect freedom).
- Datasets are gitignored and reproducible from seed + pinned env; the **manifest hash in
  status.md** is what makes a run auditable.

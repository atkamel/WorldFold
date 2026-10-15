# Quarter fold on Isaac, driven by an RL policy (plan, 2026-10-14)

**Status:** proposed, waiting for the user's sign-off. Nothing below has been built.

**Decisions (user, 2026-10-14):** alignment bar mean ≤ 1.0 cm, edges ≤ 1.5 cm; RL at the level of fold moves;
the user fixes WATcloud access for the long jobs. Geometry: keep the 30 cm square cloth and search arm layouts
first ("probably just the layout"). By hand, arms on the west and east sides of the cloth miss the second fold's
straight-down reach by about 2 cm (the arm beside a stage-0 edge must sit within 17 cm of x = 0 to reach both
ends of a 30 cm sideways carry, and the cloth edge is at 15 cm plus the base's footprint). The probe searches
base poses, tilted placing (39 cm) and a one-arm second fold before any change to the cloth. If none passes,
shrink the cloth to the largest size that does (likely 26-28 cm), still a square fold (user). Development waits
for the user's review of this doc.

**Goal (user, 2026-10-14):** an RL policy that does a two-stage quarter fold on the Isaac env with a physical
(friction) grasp. Good enough for at least one clean recorded demo in which the cloth's edges and body line up
near perfectly. Modal for fast iteration, WATcloud for long jobs, few tokens.

## 1. Where the Isaac env stands

| area | state | where |
|---|---|---|
| sim | Isaac Sim 5.1 + LeHome SO101 arms, 101×101 particle cloth (30 cm), friction grasp, no weld | `isaac/isaac_env.py`, `isaac/README.md` |
| task on Isaac | half fold only (`HalfFoldEnv`, stage 0). The quarter fold's stage 1 has never run on Isaac | `isaac/weld_expert.py:19-20` |
| friction expert | half fold id_easy 92-96 / id_hard 94 / knock_arm 75-84 | `docs/status.md` (F3, F3b) |
| best student | diffusion BC on `isaac_v1_friction`: 81 / 71 / 35; DAgger never ran | `docs/results.md` (F4) |
| fold quality | success is corner-based (5 cm). No whole-cloth alignment has been measured for the friction expert. The older scripted demo scored a mean vertex error of 3.2-4.9 cm: the near edge slides ~4.5 cm, and the far edge sags between the two held corners | `isaac/README.md` (Grasp) |
| gripper | binary with hysteresis: command < -0.3 closes to a fixed target (+0.05 rad), > 0.3 opens to 1.0 rad, anything between holds. `jaw_open_rate` can slow the opening. No continuous jaw control | `isaac/isaac_env.py:532-545` |
| speed | 3.4-5 control steps/s per env; vectorised B = 8 gives 2.15× episodes/h; Modal A10G smoke 4.3 steps/s | `docs/results.md`, commit `0ea87a7` |
| cloud | Modal image builds and the smoke test passes (A10G, driver 580). The pipeline has never run on Modal or WATcloud. WATcloud's login node times out on port 22 from this machine (2026-10-14) | `isaac/modal_isaac.py`, `isaac/watcloud.sbatch` |

## 2. The blocker: stage 1 is out of reach

Pointing straight down, an SO101 reaches the table 8-32 cm from its base (39 cm with the fingers tilted 35°, but
the tilted pinch didn't hold the cloth). The arm bases sit at (±0.22, -0.30); the Isaac cloth is centred at
(0, -0.135). `QuarterFoldEnv`'s stage 1 asks:

- right arm: grasp the west stack `cloth_0` at (-0.15, -0.285). That's 37 cm from its base.
- left arm: carry `cloth_5` onto `cloth_115` at (0.15, -0.135). That's 40.5 cm from its base.

A 30 cm sideways carry is longer than any one arm's reach band at any depth, with the bases 44 cm apart. Mo
Naqi's MuJoCo port (`origin/feat/quarter-fold-physical-grasp`) hit the same wall and shipped a one-stage task
instead. The geometry has to change before a perpendicular quarter fold can run.

Options:

- **A. Perpendicular ("square") quarter fold on a quarter-fold-only layout.** Smaller cloth (~20 cm) and arm
  bases moved in (~±0.15). The back-of-envelope reach works for both folds. The arms may collide in stage 1, so a
  no-physics IK and clearance probe decides the numbers first. It gets its own profile; the half-fold pipeline
  and its results stay as they are.
- **B. Same-direction quarter fold on today's layout.** Fold north onto south, then fold the crease edge onto
  the near edge, giving a 30 × 7.5 cm strip with four layers. Both carries are radial, which is the motion the
  tuned half fold already does. Lowest risk, but the result isn't a square.

Superseded by the user's decision at the top: a layout search on the 30 cm cloth, shrinking only if it fails.

## 3. Success condition (needs the user's numbers)

A **layer-alignment** score on the 11 × 11 grid vertices, read after the last release and a 1 s settle. Each
vertex belongs to a group of four that the two folds should stack on one spot (two for B's strip). Score =
spread of each group in XY. Proposed bar:

- every group (body): mean spread ≤ 1.0 cm
- outer-edge groups: max spread ≤ 1.5 cm
- anchors not dragged more than the env's `DRAG_LIMIT`; the stack held flat on its own after release

This ignores a rigid slide of the whole cloth, which "lined up" doesn't care about. For scale, the old scripted
half fold was 3-5 cm off on a stricter, start-pose-referenced metric. The friction expert has never been scored
on any whole-cloth metric, so step 1 measures that first.

## 4. Gripper: continuous jaw target

New `gripper_mode="position"` on `IsaacClothFoldEnv`: the gripper action in [-1, 1] maps linearly to a jaw
target between the joint's closed limit (-0.1745 rad) and open (1.0 rad), with an optional per-step rate limit
(today's `jaw_open_rate`, generalised to both directions). The jaws meet at about -0.015 rad, so a target below
that squeezes: LeHome's drive (kp 17.8 N m/rad, 10 N m cap) turns the gap into grip force. That's how the real
STS3215 servo grips as well. `grasp_active` becomes "target below a closed threshold, jaw settled, corner within
`HOLD_DIST`". Observation size doesn't change: the proprio slot that holds the commanded gripper value now holds
the continuous target.

`binary` stays the default, because the frozen `isaac_v1_friction` data and every trained policy emit ±1 values
that the new mapping would read differently.

## 5. RL formulation

Isaac runs at ~9 env-steps/s per A10G with B = 8. A quarter fold is ~450 steps, so one container does ~70
episodes/h. PPO from scratch on joint deltas got 0% on MuJoCo's single corner with far more samples. IQL and
world-model imagination were dropped for this task too (`docs/world_model.md`: 0/20 on the quarter fold). So the
RL acts at the level of the fold, not the joint:

- **Recommended: primitive-level RL.** One decision per stage (plus an optional touch-up decision). The policy
  is a small MLP that reads the cloth state (grid keypoints, stage, gripper state) and outputs per-arm
  parameters: grasp offset, jaw squeeze level, lift height, arc bulge, outward edge tension during the carry,
  place offset and height, and jaw opening rate at release. Existing IK pinch and waypoint code
  (`isaac/pinch.py`, `isaac/fold_expert.py`) executes them, using the continuous gripper. Trained with PPO
  (Gaussian policy whose mean starts at the scripted values), reward = drop in layer-alignment error per
  stage plus a success bonus. A few thousand episodes over a night is enough for ~20 dims per decision.
  Because it is state-conditioned, it adapts to how the cloth landed after the drop and after stage 0.
- **Stretch: joint-level residual RL.** Per-step 14-D action = scripted action + a learned bounded residual,
  PPO. Closer to end-to-end, but it needs 10-100× the samples. Only if A converges early.

**No BC stage.** At the level of fold moves the scripted fold already is a policy: the network outputs offsets
from the scripted values, starting at zero, so it begins exactly at the scripted fold and PPO moves it from
there. BC earns its place only when the starting policy is a joint-level network, which can't be written by
hand. It comes back if a joint-level network is wanted as the final product: roll out the trained fold-move
policy as the teacher, then the existing collect → BC → DAgger pipeline. That doesn't fit in one night.

## 6. Work order (each step has a check)

| # | step | check | where it runs |
|---|---|---|---|
| 1 | stage-1 reach + arm-clearance search for the 30 cm cloth (base poses, tilted placing, one-arm variant) | every pinch / place within reach_check's tolerances, arms ≥ 2 cm apart along the path; else shrink | Modal (no physics) |
| 2 | `gripper_mode="position"` + layer-alignment metric (both sim-free code) | existing tests unchanged (binary path); new mapping and `grasp_active` tests; metric tests on synthetic perfect / shifted folds | local |
| 3 | **second-fold feasibility, first thing on Isaac**: scripted half fold then scripted second fold in one episode, on the new layout, failures sorted by cause (table in §7) | n = 10: ≥ 3 finish stage 1; first fold re-checked on the new layout; alignment logged as the baseline | Modal |
| 4 | primitive-level PPO | learning curve on alignment; strict success rate at n ≥ 20 with Wilson interval | WATcloud (or Modal under a cap) |
| 5 | demo video (main camera + an overhead camera) from the trained policy, report, docs | video shows aligned edges; the score is printed on the frames | Modal |

Step 3 is the gate everything after it depends on, so it runs as early as the code allows (~2-3 h in).

Rules for the night: no goalpost moves; if the bar isn't met, the report gives the best score and video as they
are. Jobs run detached and write one summary JSON, so polling reads a few lines.

## 7. If it doesn't work

Time boxes assume a ~10 h night. Each fallback is tried in order; every switch goes into the pass log.

| what fails | sign | next |
|---|---|---|
| no 30 cm layout reaches | layout search finds nothing | shrink the cloth to the largest size that passes (decided) |
| the scripted second fold won't complete | under 3/10 episodes finish stage 1 after ~3 h of tuning | by cause, below |
| folds complete but RL stalls above the bar | flat learning curve; the score's per-group breakdown says where the error is | (1) correction moves: after each fold, up to 2 extra pick-and-place moves where the policy picks the point and where it goes, as the expert's retries already do for corners; (2) grasp points along the edge, not only at the corners, to cut the sag; (3) CEM / CMA-ES in place of PPO if returns are too noisy for policy gradients |
| too slow or too noisy to learn | < 1,000 episodes by midnight, or seed-to-seed spread larger than the gains | train the second fold from saved half-folded states (if the env can restore a cloth state; unchecked), more WATcloud GPUs |
| nothing meets the bar by morning | deadline | stop and report the best score, its video and the per-group errors. The bar doesn't move without the user. Easier physics (e.g. cloth-table friction) is an option for the user to decide, since it changes the task, not the policy |

**The second fold, by cause.** It differs from the tuned half fold in two ways: it carries sideways, and its
corner is a two-layer stack.

| cause | what it looks like | responses, in order |
|---|---|---|
| reach / collision | IK misses, arms touch | caught by step 1 before any physics; one-arm variant; shrink |
| the pinch takes only the top layer | the layers part at the stack corner | squeeze harder (continuous gripper); pinch lower so the jaw passes under both layers; grasp the crease end, where the fold joins the layers and they can't part; adhesion within the Phase F limit (≤ 0.3) |
| the carry drags the sheet or undoes the first fold | the near layer slides or lifts | higher lift, a lay-down arc, slower carry; the free arm presses the sheet near the fold line |
| placement misses | corners land > 5 cm off or spring back | overshoot and release rate (the half fold's fix) |
| none of the above helps | | (1) swap the fold order: the sideways fold first, on single-layer corners with the tuned grasp, then the two-layer corners go on the well-tuned straight-out carry; (2) a wide CEM search over the fold-move parameters in place of hand tuning; (3) shrink the cloth further |
| still nothing by ~6 h in | | stop the quarter-fold work and report what failed and why, with videos. The morning's choices are the user's: the strip fold (its second fold grabs the crease, where the layers can't part), more cloth physics, or a different grasp |

Every evaluation episode is recorded as it runs. The cloth sim isn't bit-reproducible, so a passing episode
has to be captured live; a replay could come out differently.

## 8. Leftover notes that bear on this

- Ruby's `quarter-fold-bc` (MuJoCo): a history-dependent phase-machine expert can't be cloned; a Markov
  (state-only) expert fixed it (expert 81%, BC+DAgger r4 61% held-out). Stage-1 stack placement was the main
  failure. Primitive decisions that depend only on the current state follow the same lesson.
- Phase F tuning that carries over: LeHome's gripper drive (the arm drive on the gripper caused slips), closed
  jaw +0.05 rad (-0.1 squeezed the cloth out), adhesion 0.1, pad friction 2.0, ALIGN_TOL 8 mm, and the
  mean-miss overshoot correction that made placement worse.
- `ROY-vla-teacher`: grabs closed 7-14 cm off target until the aim point was frozen during the grab; left-arm
  grabs near the base stop 2-4 cm high (reach).
- Isaac Sim 5.1 segfaults on NVIDIA's 610 driver (some Modal L40S hosts); WATcloud needs a ≥ 16 GB GPU on the
  580 driver; the WATcloud Docker image has never been built.

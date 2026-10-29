# Physical grasp on Isaac Sim: 2026-10-08 (Phase F, the weld replaced by friction)

**Dated:** 8 Oct 2026 (runs 7–8 Oct) · **Branch:** `feature/isaac-imitation`
· **Sim:** Isaac Sim 5.1 (LeHome stack), `isaac_friction` profile
· **Eval:** held-out seeds, deterministic inference, Wilson 95% intervals.
· **Page with videos inline:** [2026-10-08-physical-grasp.html](2026-10-08-physical-grasp.html) (open locally) · shareable copy https://claude.ai/artifact/Jj6aJ5aTKBNQEFzLkU9uuF
· **Previous report:** [2026-10-07-isaac-half-fold.md](2026-10-07-isaac-half-fold.md), the weld grasp (W5).

The W5 policies folded the cloth with a **weld**: the cloth particles under the jaw were pinned to the gripper while
the jaws hovered open. Phase F replaces it with LeHome's own physics, a friction-plus-adhesion pinch with no pins, welds
or attachments. On that physics it re-tunes the grasp and fixes the expert's recovery. The pipeline is then retrained
from scratch. Every number here is copied from [results.md](../results.md).

## Where it stands

| | clean | shifted | arm bump | joint noise | overshoot |
|---|---|---|---|---|---|
| **Scripted expert** (privileged, n = 100) | 92 [85.0, 95.9] | 94 [87.5, 97.2] | 75 [65.7, 82.5] | 74* | 86* |
| **Diffusion BC student** (privileged state, replan 4) | **81** [72.2, 87.5] | **71** [61.5, 79.0] | **35** [26.4, 44.7] | **26** [15.9, 39.6] | **4** [1.1, 13.5] |

\* The expert's joint-noise and overshoot rates are from tune sets (n = 50). Student n = 100 / 100 / 100 / 50 / 50.

- **The grasp is physical now, and checked.** On every step of 20 expert episodes, the honesty check looked for
  pinned particles, attachment prims, a jaw forced open while the corner rides, and an open-jaw carry of more than 5
  steps. It found **0** of each, with 18/20 successes. The weld, run as a negative control, flags 251 pins and 47-step
  open-jaw carries.
- **The expert is good on clean and shifted starts** (92 / 94) **and recovers from a mid-carry arm bump 75% of the
  time** (n = 100). On the tune seeds the F3b fixes lifted it from 70% to 84%. Under the old all-dims knock,
  which toggled the jaws, the expert had scored 27%.
- **The student is a first cut.** It got 200 demos and 20k BC steps, with no DAgger, a single seed and no vision, to
  fit a 3.5 h budget. It folds 4 of 5 clean starts. It is weak whenever it leaves the expert's distribution: the
  arm bump, noise and overshoot sets.

**Not comparable with W5 one to one:** the weld numbers (99 / 97.5 / 54.5) are on MuJoCo's task sets. The friction
pinch can only reach about ±1.5 cm of cloth offset at LeHome's cloth position, so `isaac_friction` runs LeHome's own
sets (1 cm jitter, a nearer shifted ring). The comparison worth making is the expert against the student on the
same seeds, as in the table above.

## How the grasp was made real

| step | what changed | effect (grasp bench, both arms) |
|---|---|---|
| F0 | the friction profile on the GPU pipeline, cloth at LeHome's position | held 8/20 · 10/20: corners slip out of shut jaws |
| F2 A | **the gripper used the arm's drive** (kp 998, 3.35 N m). Switched to LeHome's gripper drive (kp 17.8, 10 N m) | slips stop on the left arm |
| F2 | jaw closed target +0.05, adhesion 0.1 (LeHome's value) | held 99 / 89–96 at n = 100 |
| F2 M | **pad friction 2.0 + `ALIGN_TOL` 8 mm**: re-descend if the pinch is off before closing (the right jaw was sweeping its corner sideways) | **held 40/40 · 40/40**, placed 40 / 38 |
| F1 | honesty check | 0 pins, 0 attachments, 0 open-jaw carries |

## How the expert learned to recover (F3b)

The perturbation suite was redefined with the user to be realistic:
- **arm bump**: the arm joints are knocked mid-carry; the gripper is left alone.
- **joint noise**: σ 0.15 on the joints.
- **overshoot**: the action gain is scaled ×U(1.2, 1.4).
- **forced drop**: tried, then skipped by decision. A double drop crumples the cloth beyond a grasp-the-corner
  strategy.

Traces of the failing recoveries showed the arm *could* reach the dropped corner: the gripper got within 1–5 cm,
7–20 cm from the base. The open-loop expert's joint-space "arrived" test never fired near the base, so it parked
and timed out. Three changes are now the code defaults:
- `RESYNC_IK_HOME`: plan from the home seed after a disturbance.
- `DESCEND_RETRY`: back up and re-approach instead of closing off-corner.
- `CART_ARRIVE`: the descend arrives when the gripper is within 2 cm of the pinch point and still.

Tune-set results: clean 96, arm bump 84, overshoot 86, joint noise 74.

## Video evidence

Three episodes per video: the first three seeds of each held-out set, rendered from the evaluation loop in real time.
Lines on screen show each arm's grasp state. Failures are shown as they happened; per-episode outcomes are in the
`.json` next to each video.

| video | outcome |
|---|---|
| [half_fold_friction_privileged_id_easy.mp4](media/half_fold_friction_privileged_id_easy.mp4) | **Student · clean · 2/3**: seeds 100000–2 · 100000 times out at 400 steps (fold 0.77) · 100001 / 100002 fold in 167 / 180 steps |
| [half_fold_friction_privileged_id_hard.mp4](media/half_fold_friction_privileged_id_hard.mp4) | **Student · shifted · 2/3**: seeds 200000–2 · 167 / 154 steps for 200000 / 200002 · 200001 times out at 400 (fold 0.76) |
| [half_fold_friction_privileged_knock_arm.mp4](media/half_fold_friction_privileged_knock_arm.mp4) | **Student · arm bump · 1/3**: seeds 310000–2 · 310001 recovers in 179 steps · 310000 and 310002 time out (fold 0.74, 0.38) |
| [half_fold_friction_expert_knock_arm.mp4](media/half_fold_friction_expert_knock_arm.mp4) | **Expert · arm bump · 3/3**: same seeds 310000–2 · recovers in 183 / 175 / 185 steps · fold 0.94 / 0.87 / 0.91 |

## How the student was trained (F4, 3.5 h plan)

| stage | setting | result |
|---|---|---|
| expert demos | 200 seeds, 50% arm-bumped, state-only, 2 × 8 envs, 39 min | 171 kept (31,117 steps) `isaac_v1_friction` `71a993b12a3c`; 29 failures `be7f7e50083b` |
| diffusion BC | 20k steps, seed 0, 14 min | val loss best ~6k steps (0.021), final 0.042 |
| DAgger, vision student, detector | **cut** for time | — |
| finals | replan 4, 2 × 8 envs, ~2 h | table above |

## What fails, and what fixes it

1. **Stalls (S1) on clean and shifted starts** (14 and 23 of 100). The policy parks near the cloth and never commits
   to the descent: classic covariate shift. **DAgger** is the targeted fix; on Isaac weld it moved shifted from 85 to
   88, and on MuJoCo recovery from 60 to 79.5.
2. **Slips (G2) under arm bump, noise and overshoot** (27 / 32 / 47). The demos never show an off-course gripper, so
   the student closes on a misplaced jaw. The fix is DAgger rollouts on those kinds, or noise-injected demos
   (DART-style).
3. **Overfitting.** The 6k-step checkpoint had half the final validation loss. It is a cheap test before any
   retraining.
4. **The expert's ceiling** on clean starts is 92–96%. Its remaining failures are slips with the jaw shut.

**Next:** test the best-val checkpoint (~20 min). Then DAgger on arm bump, noise and overshoot (about 2–3 h). Then
Adam's as-built comparison row (`scripts/adam_baseline.ps1`).

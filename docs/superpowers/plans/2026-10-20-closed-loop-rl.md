# Closed-loop residual RL for precise folds (plan, 2026-10-20)

**Goal:** an RL policy that lines the folded layers up to the demo bar (mean ≤ 1.0 cm, edges ≤ 1.5 cm,
`cloth_fold_rl.fold_alignment`), first on the half fold, then on fold 2 of the quarter fold. It replaces the fold-level
PPO of `isaac/fold_rl.py` (plan 2026-10-14 §5), which stays in the repo as a record.

**Status:** signed off 2026-10-20 with two changes: no weld anywhere (the user: a gripper fold is possible because a
person can do it in teleop, so the policy has a reachable goal; R0 is the gripper baseline only), and training starts
with dynamics randomization off. Half fold on the GPU `friction` profile (old layout).

## Why fold_rl failed

Run strip_r2 (WATcloud, 10 h): 461 episodes, 19 PPO versions, mean alignment flat at 9 to 10 cm, best single episode
2.1 cm, exploration std stuck at 0.25. The design caused most of that:

| defect | consequence |
|---|---|
| The policy picks 18 move parameters once per fold from 9 keypoints, then the scripted fold runs blind | It can shift the average landing point. It can't correct an episode's own miss, which is what a 1 cm bar needs. |
| Each corner is carried to its partner vertex's **start** position (`QuarterFoldEnv.goal`) | The bottom layer slides 6 to 14 cm during the carry (strip runs), so the target itself is wrong by that much. |
| 2 decisions per episode | 922 learning signals in 10 h, each a noisy terminal score. |
| On-policy PPO | Every episode is used for one update and thrown away. |
| Exploration noise of 0.25 per parameter (about ±2 cm on placement) on top of ~2 cm natural scatter, never annealed | The noise was as large as the effect being learned. |
| No fixed-seed evaluation | Versions were compared on different cloth draws. |

## What the evidence says (research 2026-10-20, three read-only agents, see subagents.md)

- **Residual RL on a working controller** learns contact-rich precision tasks in about 10⁴ steps (Johannink et al.
  2019, arXiv 1812.03201; Silver et al. 2018, arXiv 1812.06298): a bounded correction added to the controller's
  command, starting at zero.
- **Off-policy with prior data and a high update-to-data ratio** is the standard answer when steps are expensive:
  RLPD (Ball et al. 2023, arXiv 2302.02948), SERL (Luo et al. 2024, arXiv 2401.16013), REDQ (arXiv 2101.05982).
  Reported budgets for precise placement and insertion are 2·10⁴ to 1.5·10⁵ env steps.
- **Sub-centimetre fold alignment** in the literature comes from closed-loop, multi-step execution (Foldsformer,
  arXiv 2301.03003: 3 to 8 mm particle error on quarter-fold-like SoftGym tasks). No one-shot open-loop method gets
  there.
- **This repo:** closed-loop per-step PPO solved the MuJoCo single-corner weld fold at 100% (`docs/world_model.md`).
  Offline IQL failed (M5b.4) because scripted data had almost no action diversity, so prior data here must carry
  injected noise.
- **Throughput:** Isaac Lab has no first-class multi-env path for legacy particle cloth, but our scene copies
  (milestone V, GPU pipeline only) gave 2.15× episodes/h at B = 8. The LeHome-winning team (reported in arXiv
  2606.27163; not checked by the lead) ran RL on 3 to 5 single-env Isaac processes per machine at ~30 s per
  episode. The GPU pipeline also exposes the particle positions and velocities for writing
  (`lab_scene._drive_pins`), which makes saving and restoring a mid-fold state possible. The CPU device
  (`lehome`, `roy`) has no such view.

## Design

**Base controller.** The proven open-loop pinch of `IsaacArmExpert` (friction profile, 40/40 held) does the approach,
close and lift. From the carry on, `IsaacServoExpert`'s corner steering takes over (F3b; its grasp servo was dropped
untuned, its carry and place were never measured on their own): each step the gripper target moves the **corner**
toward its next waypoint, so spring-back and lag need no overshoot table. The place target is the partner vertex's
**current** position, re-read every step, with the corner pressed onto the bottom layer before release. Retries are
off inside the learning window (strip5's partner-chasing failed through retries re-grasping a springing corner).

**Learning window.** From the start of the carry (corner lifted and held) to 15 steps after both releases, about 120
control steps per fold. The grasp is outside the window: it is already measured on its own (friction profile 40/40
held; roy slide 16/20).

**Policy.** One policy for both arms, acting every control step (0.05 s).

| | content | dims |
|---|---|---|
| observation | per arm: carried corner minus partner's current position, gripper site minus carried corner, gripper velocity, jaw angle, accumulated residual, phase one-hot, time in phase | ~2 × 20 |
| | cloth: XY offset of 8 matched layer pairs (corners, carried-edge midpoint, side-edge quarter points), drift of 4 bottom-layer vertices since the window began | ~36 |
| | previous action | 8 |
| action | per arm: Cartesian residual on the base's gripper target, ≤ 4 mm per step, accumulated offset clipped to ±6 cm | 2 × 3 |
| | per arm: release vote; the arm opens once its vote is positive and both arms are in hold, or after a hold cap (the friction profile's gripper is binary) | 2 × 1 |

The residual is added to the steered gripper target just before the servo's IK solve (`_toward`), so the phase
machine and IK stay unchanged.

**Reward.** Potential-based shaping on the alignment, so the optimum is unchanged
(Φ = −mean over matched pairs of log(d_xy + 0.5 cm)), with r_t = Φ(s_t+1) − Φ(s_t). The log scale makes 3 → 1 cm pay
as much as 30 → 10 cm. On top: −0.01‖a‖², a penalty on bottom-layer drift per step, and at the end of the window
−log(mean / 1 cm) − log(edge / 1.5 cm) plus 10 if the bar is met. Dropping the corner before the place phase, an
unstable cloth or `cloth_dragged` costs −10 and ends the episode.

**Learner.** SAC with RLPD's settings, written here in torch (no new pins): an ensemble of 10 critics with LayerNorm,
2 sampled for the target (REDQ), 10 gradient updates per env step, batches drawn half from the prior buffer and half
from the online buffer, actor's last layer zero-initialised so step 0 is the base controller, a low entropy target.
Prior data: 200 base episodes with an Ornstein-Uhlenbeck residual (σ 3 mm per step), so the critics see varied actions.

**Episodes and throughput.**
- Snapshots: the scripted fold runs to the start of the window once per seed and saves the state (particle positions
  and velocities, joint positions, velocities and targets, jaw, expert and env-wrapper state). RL episodes restore a
  snapshot and run only the window, skipping ~40% of each fold.
- Profile `friction` (GPU, old layout, where the half fold already places 18/20) with B scene copies per process and 2
  processes per GPU. One long SLURM job per run so the image is built once.
- Expected: 3.5 to 9 env steps/s per GPU (V measurements and the 10-14 plan's A10G estimate), so 10⁵ steps in 3 to 8 h.
  R2 measures it.
- Fallback if restore or the GPU copies don't hold up: full episodes on N single-env processes (the LeHome team's
  setup).

**Evaluation.** 20 held-out seeds as fixed snapshots, deterministic policy, every 5,000 env steps, written to
`learner.jsonl`: mean and edge alignment, pass rate with a Wilson interval, drops. Best version gets a full episode
with video (grasp included).

## Steps

| # | what | test | pass bar |
|---|---|---|---|
| R0 | **Gripper baseline.** Half fold on `friction` with the closed-loop base, DR off, n = 20 (and the open-loop expert on the same seeds for comparison) | alignment per episode, split into bias and scatter | none: it is R4's reference |
| R1 | **Snapshot and restore** on the GPU pipeline | sim-free: the snapshot round-trips. Isaac: replaying the scripted actions from a snapshot ends within 3 mm mean alignment of the original run | ≥ 9/10 seeds |
| R2 | **Window env**: the learning window as an env over B copies, observation, action and reward as above | sim-free: observation layout, residual bounds, shaping sums to Φ_T − Φ_0. Isaac: zero residual reproduces R0's friction numbers (n = 20); throughput logged | within R0's Wilson interval |
| R3 | **Learner** (SAC + RLPD settings) and the prior-data collector | sim-free: solves a toy 1-D placement task through the same interfaces | toy solved |
| R4 | **Half-fold RL run**, WATcloud overnight | learning curve on the fixed eval seeds | mean alignment below R0's friction base by > 2 SE; ≥ 1 eval episode at the bar, recorded with video |
| R5 | **Fold 2 of the quarter fold**: the same window on fold 2, starting from fold-1 snapshots, on the layout the Roy-port work settles (roy on the GPU pipeline, or the friction layout re-searched with the slide grasp) | as R4 | one recorded quarter-fold episode at the bar |

**Stop rules.** R2's parity failing means the window env changed the physics; fix that before any learning. If R4 is flat after 5·10⁴ steps, look at the critic's value
error and the eval curve before changing anything else.

**Effort.** R0 to R3: about 2.5 days of code and checks. R4: one overnight run, then iterations. R5 depends on the
layout work.

## Decisions (user, 2026-10-20)

1. Half fold on the old layout, GPU `friction` profile: yes.
2. Dynamics randomization off for training first: yes.
3. No weld ceiling: R0 is the gripper baseline.

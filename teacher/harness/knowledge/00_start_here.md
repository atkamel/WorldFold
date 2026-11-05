# Start here: where we stopped (2026-11-05)

Nothing is assigned. This is where the work is, so you can take it from here without redoing anything.

## Where we stopped
**We finished the research on Oct 20, and the next step was writing the stage-1 RL design. That design doesn't exist yet.**
No RL has run on the teacher. So the next thing anyone needs to do is design and build:

> **Stage 1: SGS-style neatness RL on Ilia Larchenko's `lehome_sim` π0.5, in its own LeHome Isaac world.**

The method is in `rl_recipe.md`. The decisions below are still open.

## Why this, in one paragraph
Everything since September came down to this. The teacher (1st of 62 in the LeHome 2026 sim round) already folds real garment
shapes, often ugly. Distilling it didn't help because the teacher itself wasn't good enough. Roy's mouse teleop only works with
sticky cloth, so those demos are invalid. Training from zero costs billions of steps. The teacher stops when LeHome's weak
check passes, so it was never asked to tidy. And some of its folds are already neat. So: use RL to make it do that more.

## Open decisions
1. **Neatness pass line.** `fold_score` exists, the line doesn't. Roy calibrates it by eye. Until then no target is real.
2. **Learner.** Not decided.
3. **What December shows.** LeHome's physics or realistic physics? Unresolved.

## What's measured
All in the teacher's own LeHome world unless stated. Data gathered 2026-10-19 and 20 on one GCP L4.

**Folding**
- LeHome's own check: 18/20 passed. It tests 6 vertices and nothing about flatness, so this doesn't mean good folds.
- Neatness, using `teacher/mujoco_live/neat_metric.py` (`fold_score`, resists crumple cheats):

| | Roy's folds (60, sticky-cloth game) | Teacher (20) |
|---|---|---|
| Best | 28.8 | 32.7 |
| Score ≥ 15 | 17 | 6 |
| Median | 11.3 | 12.9 |
| Median rectangle fit | 0.69 | 0.81 |
| Median flatness | 0.71 | 0.56 |

The two sets ran in different sims and only the metric matches, so this is rough. **The teacher's trousers look bad on video
even where the score is decent,** so the score needs calibrating before training on it.

Data and videos are in `neat_data/`: every score, picture sheets, `teacher_top3_3d.mp4`, `top3_teacher_vs_roy.mp4`.

**Grasping** (`neat_data/GraspTest_report.pdf`, `RunA_report.pdf`)

| Test | Pinched then lifted ≥ 4 cm, of attempts |
|---|---|
| Full folds, 20 episodes | 78% |
| Restored mid-fold, cloth as it was | 51% |
| Restored, cloth moved 3 cm and turned 15° | 27% |
| Same move, after 5 policy steps | 16% |

Part of the grasp is trajectory memory: the jaw closes where the cloth used to be, and doesn't re-aim once committed.
By garment (full folds / restored / moved): shorts 100/93/55, trousers 89/38/23, long tops 80/67/4, T-shirts 65/10/20. RL has
something to amplify on shorts, long tops and trousers, and very little on-demand on T-shirts (interpretation). Grasps usually
take 2 to 5 layers.

**Realistic physics (for later stages)**
- LeHome's cloth is about 3,000× too heavy (147 kg-class), which is why its grasps "work".
- Teacher in corrected physics: 3/8 passed (earlier run, older settings).
- Arm fit against NVIDIA's real SO-101 data: 0.284° held-out, vs 0.385° for NVIDIA's parameters in the same sim
  (`arm_sysid/`, `so101_fitted_rl.json`).
- Cloth fit (Newton, free-hanging real recordings): held-out 0.615 vs clothilde 0.778 (`cloth_sysid/`,
  `results/newton/cloth_fitted_rl.json`). Isaac port: 28% deviation with bend/shear ×100 and 64 iterations, against 209% stock.
- Isaac's jaw-cloth contact holds 2 to 10× too weakly and doesn't really squeeze. `oracle/attach_grasp.py` is the candidate fix
  (attach + slip above 2·μ·N); its Isaac test is still pending.

**Dead ends, don't retry without something new**
- Mouse-teleop demos: only work with sticky cloth, and sticky demos are invalid.
- Dream-1 zero-shot on SO-101: 0/6 from rest, 2/6 lifted from a ready pose, none held 2 s (`dream_test/`).
- Training from scratch: published step counts (10^8 to 10^9) are out of reach for cloth.
- Teacher in our MuJoCo world: 0/12. It's a specialist in its own world.

## The road after stage 1
1. **Stage 1: neatness RL in the teacher's own world.** If it works, it's most of the December result, assuming LeHome's sim
   counts as the deliverable.
2. **Stage 2: Run B.** The teacher in realistic physics, to see whether its hands or its planning break.
3. **Stage 3: continued RL of the stage-1 result in realistic physics.** Not distillation, because copied actions break when
   physics changes.
4. **Real SO-101s,** with heavy visual and physics randomization. Then a small real fine-tune.

The larger hypothesis behind this (subskills, ADEPT, a brain) and the other open lines are in `bigger_bet.md`.

## What you have
| You need | Where |
|---|---|
| The method | `rl_recipe.md` |
| The teacher running, and your half-completed states | `run_it.md` |
| The neatness score | `teacher/mujoco_live/neat_metric.py`, `teacher/neat/` |
| Save and restore mid-fold states | `teacher/grasp_test_patch.py`, `teacher/select_pregrasp.py` |
| Per-grasp logging | `teacher/grasp_log_patch.py`, `teacher/oracle/grasp_log.py` |
| Designing and red-teaming the score | `research_good_rl_tasks.md` |
| Mistakes already paid for | `team_lessons.md` |

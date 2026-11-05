# Start here: where we are (as of 2026-11-04)

## Not square one
The teacher, Ilia Larchenko's `lehome_sim` π0.5 (1st of 62 in the LeHome 2026 sim round), already folds real garment
shapes in Isaac Sim. The folds are often ugly. Everything below is about closing that gap. We are on Isaac, and the plan
is RL on top of the teacher.

## What's measured
All in the teacher's own LeHome world unless stated. Data was gathered 2026-10-19 and 20 on one GCP L4.

**Folding**
- LeHome's own check: 18/20 passed. That check is weak, so this doesn't mean they were good folds.
- Neatness, using `teacher/mujoco_live/neat_metric.py` (`fold_score`, resists crumple cheats):

| | Roy's folds (60, sticky-cloth game) | Teacher (20) |
|---|---|---|
| Best | 28.8 | 32.7 |
| Score ≥ 15 | 17 | 6 |
| Median | 11.3 | 12.9 |
| Median rectangle fit | 0.69 | 0.81 |
| Median flatness | 0.71 | 0.56 |

The two sets ran in different sims and only the metric matches, so this is rough. The teacher stops at LeHome's success
check, so it never learned to tidy up. Its trousers look bad on video, whatever the score says.

**Grasping**

| Test | Pinched then lifted ≥ 4 cm, of attempts |
|---|---|
| Full folds, 20 episodes | 78% |
| Restored mid-fold, cloth as it was | 51% |
| Restored, cloth moved 3 cm and turned 15° | 27% |
| Same move, after 5 policy steps | 16% |

Part of the grasp is trajectory memory: the jaw closes where the cloth used to be, and doesn't re-aim once committed.
By garment (full folds / restored / moved): shorts 100/93/55, trousers 89/38/23, long tops 80/67/4, T-shirts 65/10/20.
Grasps usually take 2 to 5 layers, and single-layer grabs are rare in full folds (2/65, 2/67).

**Realistic physics**
- LeHome's cloth is about 3,000× too heavy (147 kg-class), which is why its grasps "work".
- Teacher in corrected physics: 3/8 passed (earlier run, older settings).
- Arm fit against NVIDIA's real SO-101 data: 0.284° held-out, vs 0.385° for NVIDIA's parameters in the same sim
  (`arm_sysid/`, `so101_fitted_rl.json`).
- Cloth fit (Newton, free-hanging real recordings): held-out 0.615 vs clothilde 0.778 (`cloth_sysid/`,
  `results/newton/cloth_fitted_rl.json`).
- Isaac's jaw-cloth contact holds 2 to 10× too weakly and doesn't really squeeze. `oracle/attach_grasp.py` is the
  candidate fix (attach + slip above 2·μ·N).

**Dead ends, don't retry without something new**
- Mouse-teleop demos: only work with sticky cloth, and sticky demos are invalid.
- Dream-1 zero-shot on SO-101: 0/6 from rest, 2/6 lifted from a ready pose, none held 2 s (`dream_test/`). This only
  tests zero-shot, not fine-tuned.
- Training from scratch: published step counts (10^8 to 10^9) are out of reach for cloth.

## The plan
1. **Stage 1: neatness RL in the teacher's own LeHome world.** Episodes don't stop at LeHome's check. Reward: pass/fail
   on neatness. **If this works, it's most of the December result.**
2. **Stage 2: Run B.** The teacher in realistic physics, to see whether its hands or its planning break.
3. **Stage 3: continued RL of the stage-1 weights in realistic physics.** Not distillation, because copied actions
   break when the physics changes.
4. **Real SO-101s,** with heavy visual and physics randomization.

**Other pathways, open to claim:**
- Keep the teacher as the brain and swap in an RL-trained grasp and release skill where it fails.
- Train skills bottom-up (pinch, release, drag), chain them with a high-level model.
- Real demos with leader-follower arms, then fine-tune.

**Open question:** is December's deliverable in LeHome's sim or in realistic physics?

## What to read next
- `rl_recipe.md`: what we're copying from the research.
- `run_it.md`: how to get the teacher running.
- `team_lessons.md`: mistakes already paid for.

# Stage 1 design: neat folds with SGS-style RL and Advantage Guidance

**Status:** draft, 2026-11-05. Not built, not run. Roy hasn't signed off, and the decisions in section 11 are open.

Labels: **measured** = we ran it. **published** = from the paper. **proposed** = my design choice or estimate, untested.

## 1. Goal
Make Ilia Larchenko's `lehome_sim` π0.5 fold T-shirts and trousers neatly, more often, on garments it never trained on, in
its own LeHome physics. This is Pathway A: top-down, whole fold first. Success is also Roy's precedent for subskill RL later.

**Not in stage 1:** realistic physics (Run B), real robots, world models, subskill training.

## 2. The recipe, piece by piece
| Roy's recipe | In this design | Source |
|---|---|---|
| Almost-complete states | Bank of the teacher's own half-completed folds, mixing all stages | AlphaNIST, OmniReset; 1,049 states measured |
| SGS | Sample starts near 50% success, per group of states (section 6) | AlphaNIST q ∝ ε + √(p(1−p)); grouping is proposed |
| Binary reward | Neat or not, once, at episode end | AlphaNIST; pass line from Roy |
| KataGo efficiency | Critic also predicts the neatness sub-scores; cheap checks most of the time, full score at the end | KataGo 1.65× in Go; untested on cloth |
| One shared policy | All garments together, with no garment-specific training | AlphaNIST 98.2 vs 81.4 |
| Pretrained VLA | The teacher stays in charge | Roy's bet, and the difference from AlphaNIST |
| **PPO → Advantage Guidance** | Replaces the learner, because PPO needs far more steps than cloth allows | arXiv 2610.19783 |
| Safeguards | Keep best weights, retest every round, roll back on a drop | ADEPT plan |

## 3. The model
Today, at each denoising step (`pi_modified.py:1738`, measured from code):

    v = v_uncond + cfg * (v_cond - v_uncond)

Proposed:

    v = v_teacher + w * [ v_residual(y=+1) - v_residual(y=-1) ]

- **`v_teacher`:** the teacher's existing output, CFG included, weights frozen.
- **`v_residual`:** a new small flow network, conditioned on advantage `y`, trained with a plain supervised loss. The paper's
  demo-task actor is an MLP of 1024 × 8. Inputs: noised action, time, teacher's image embedding, joint state, `y`.
- **`w`:** one guidance weight. `w = 0` gives back the teacher exactly, so we can always compare and fall back.

**Decision for Roy:** this keeps the teacher's weights frozen, which is not Route A as we defined it. It isn't Route B either,
because there's no hand-off: the residual acts at every denoising step and the teacher stays in control. If this stalls and
the critic works, the next step is to unfreeze part of the action expert. That needs the memory work in section 8.

## 4. Critic
It answers: from this state, how much better is this action chunk than usual, for ending neat?
- **Reward:** 1 if the final fold is neat, otherwise 0. Published setting: γ = 0.999 on demo tasks.
- **Method:** IQL, as in the paper, working on action chunks.
- **Inputs (proposed):** a critic only runs in training, so it can read true cloth points from the sim as well as the teacher's
  embedding. That avoids having to judge neatness from cameras.
- **Extra heads:** predict rect, flat, solid and layers (the KataGo piece).
- **Advantage labels:** per state, normalized over the dataset action plus K imagined actions. Paper uses K = 7.

**Imagined actions:** `sample_actions(num_candidates=N)` runs the expensive prefix once per request and samples N chunks. So
K = 7 should cost far less than 7 full calls (read from code, not timed).

## 5. Episodes and reward
- **Start states:** restored half-completed folds, plus flat starts.
- **Don't stop at LeHome's check:** `stop_on_success` comes from the task dict, default True (`remote_loop.py:258,312`). Episodes
  get a time budget instead. Line 275 says the server normally stops sooner, so that needs checking too.
- **Score only a finished state (proposed):** jaws open and clear, cloth still, then `fold_score`. Stops holding a neat shape in
  the air from counting.
- **Randomize:** garment pose and rotation, 5 seen garments per type, unseen held out, mild visual changes.
- **Score timing (measured):** `fold_score` took 109 ms on the laptop for a 663-point shirt. LeHome's garments are far larger, so
  run it at episode end only, until timed on them.

## 6. SGS at our speed
At about 210 restored episodes/hour (measured) across 1,049 states, each state is tried once every 5 hours, so per-state success
can't be estimated. **Proposed:** track success per group, garment × fold stage. 20 garments × 4 stages is 80 groups, or about 2.6
tries per group per hour. Keep sampling floor ε so no group is dropped.

## 7. Steps, each with a gate
Pass lines marked * are my proposals. Change them freely.

| # | Step | Passes when | If it fails |
|---|---|---|---|
| 0 | **Calibrate the score.** Roy marks neat or not on the 20 teacher and 60 of his folds, by video. Try to cheat it | Score separates Roy's labels, including trousers, at one line per garment type | Fix the score first. Nothing else is meaningful until then |
| 1 | **Can it tidy?** 20 folds, not stopping at the check | Neatness holds or rises after the check | If it wrecks folds: end at a time limit. If it never improves: RL has less to amplify, and we rethink |
| 2 | **Collect data.** About 300 folds plus restored episodes, cached | Enough neat and non-neat folds in every garment type | Adjust the line, collect more |
| 3 | **Test the critic.** Train offline. Then restore about 49 states, try 8 chunks each, and compare the critic's ranking with real outcomes | Predicts held-out neatness (AUC ≥ 0.75*), and ranks actions better than chance | **Stop.** No critic, no Advantage Guidance |
| 4 | **Train the residual.** Offline from cache | Paper's checks: asking for a higher advantage gets actions the critic rates higher, and the advantage effect exceeds the noise effect | Check residual inputs; try K and w |
| 5 | **Test it.** Unseen garments, 40 folds each for teacher alone and with residual, same seeds | Neat rate up (p < 0.05*), LeHome passes down no more than 5 points*, and **Roy agrees on video** | Roll back |
| 6 | **Close the loop.** New folds with the improved policy, SGS choosing starts, back to step 3 | Each round beats the last on held-out garments | Keep best round |

**Metric cheats to test at step 0:** crumple, flatten without folding, sticky cloth.

## 8. Compute (estimates from measured rates)
L4 about $0.94/hour, about 48 full folds/hour, about 12,600 policy steps/hour (5,245 steps in 25 min, measured).

| Step | L4 time | Cost |
|---|---|---|
| 0 | none, laptop | $0 |
| 1 | about 25–60 min, longer episodes | about $1 |
| 2 | about 6 hours | about $6 |
| 3, states test | about 2 hours | about $2 |
| 3–4, training | **not measured** | unknown until timed |
| 5 | about 3–4 hours | about $3–4 |

300 folds is about 79,000 policy steps, or roughly 16,000–26,000 decisions at 3–5 actions per chunk. The paper's demo tasks used
10,000–30,000 transitions, on far simpler tasks. Training cost is not included, because it hasn't been measured.

**Memory (proposed):** teacher inference measured 16 GB on a 24 GB L4. Cache embeddings and imagined actions during collection, then
train with the teacher unloaded. The teacher only needs loading again for testing. Fitting teacher + residual at test time is untested.

**Storage:** Ilia's eval can write about 0.8–1.6 GB of pkl per episode (measured). Store embeddings, joints, actions, cloth points at
chunk boundaries, and a few videos, not every frame.

## 9. Code
| Need | Where |
|---|---|
| Add residual to sampling | `src/lehome_solution/models/pi_modified.py:1738`, inside the JIT-compiled loop |
| Get imagined chunks | `sample_actions(num_candidates=...)`, `pi_modified.py:1438` |
| Don't stop at the check | `remote_loop.py:258,312`; example `scripts/dagger_collect.py:976` |
| Save and restore states | `teacher/grasp_test_patch.py`, `teacher/select_pregrasp.py` |
| Score | `teacher/mujoco_live/neat_metric.py`, `teacher/neat/teacher_end_states.py` |
| Advantage Guidance | `zhouzypaul/advantage-guidance`, MIT, Python, `agents/` |

Line numbers are from `lehome_solution` at `ca73828`. Check the copy in the Docker image. Use env-gated patches, as our existing
ones do. Published code first, no from-scratch RL.

## 10. Risks
| Risk | How we find out |
|---|---|
| Critic can't rank actions, the biggest one | Step 3, before building the residual |
| The paper's method was never tried on a VLA, on images, or on cloth | Steps 3–5 |
| Residual on a frozen teacher isn't enough, since the paper trains its own base | Step 4 alignment checks |
| Teacher can't tidy, or grasp limits the result (T-shirts on demand lift at 10–20%) | Steps 1 and 5, by garment |
| Binary signal too rare in hard groups | Step 2 neat rates by group |
| Offline only; the paper leaves online to future work | Step 6 |
| Score fits Roy's eye on our data only | Step 5, unseen garments |

## 11. Decisions Roy needs to make
1. **Frozen teacher + residual first?** Or go straight to changing the teacher's weights (Route A)?
2. **The neatness pass line,** at step 0. Nothing later means anything without it.
3. **What December shows:** LeHome physics or realistic physics?

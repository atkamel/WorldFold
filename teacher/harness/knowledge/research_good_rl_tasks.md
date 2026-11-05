# "How to make a good RL environment/task", Eli Gottlieb (X article)

Our own notes on the article, written 2026-10-20.

## The article
- **A task has four parts:**
  - **prompt** (the request);
  - **world** (the data and dynamics the agent acts in, through fixed tools);
  - **verifier** (how "good" is defined: a script, one model call, or a grading agent);
  - **intent** (the explicit skill to improve, or the failure mode to fix).
- **Making a task is constraint satisfaction across these parts.** The alignment axes:
  - **Prompt ↔ verifier:** the prompt must give a capable model a fair chance to earn the reward.
  - **Verifier ↔ intent:** the verifier is the only source of optimization pressure. Whatever satisfies it is reinforced, including reward hacks.
  - **Prompt ↔ intent:** training prompts should match what the model will face in use.
  - **World:** worlds range from real to synthetic. A world is sufficient if and only if it reliably elicits the target failure. A world bug is not a legitimate failure, and a right answer that skips necessary steps means the world isn't rich enough. The closer to real, the less fiddling.
- **RL reinforces whatever earns reward,** intended or not.
- **Good tasks come from revision loops:**
  - **Pipeline:** author → eval → audit traces → revise, keep or drop.
  - **Stronger pipeline:** solver agents + a separate verifier agent + **hacker agents** that look for undesirable solutions that still pass + a reviewer.
  - **Calibrating the verifier is almost always needed:** map edge cases and regression-test against ground-truth labels.
  - **Making tasks harder** must stay on meaningful axes.
- **Verifiability spectrum:** from math to creative writing. Most real work is in between, where the human expert is the ultimate verifier.
- **Task factory:**
  - **Inner loop:** the recipe.
  - **Outer loop:** update the recipe as understanding sharpens.
  - **Explicit criteria:** decompose human judgment into criteria, each with a guard.
  - **Compile the expert into cheaper components over time:** human gate → agent → cheap classifier.

## Mapping to WorldFold neatness RL
- **Intent:** neat whole folds (Q1: amplify the teacher's occasional neat folds).
- **World:** LeHome Isaac, the teacher's own physics, for stage 1.
- **Prompt:** garment + start state from the reset bank.
- **Verifier:** the neatness score + pass line.
- **Expert:** Roy.
- **Do before RL:**
  1. **Calibrate the verifier** on a labeled set (Roy's folds, the teacher's folds, deliberate bad folds) until it agrees with Roy.
  2. **Red-team the verifier:**
     - crumpling (blocked by `neat_metric`);
     - flatten-without-fold;
     - sticky-physics cheats.
  3. **Accept only real failures:** stage 1 is proof only; realistic physics comes later.
  4. **Make starting states look like real use:** varied garments and poses.
  5. **Keep Roy as the outer loop:** review RL's best and worst outputs, and fix the verifier when he disagrees.
- **Contrast with AlphaNIST:** its verifier was trivial (1 mm, binary). Neatness is mid-spectrum, so verifier calibration is the extra step we need.

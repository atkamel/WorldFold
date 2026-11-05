# The bigger bet: subskills, chaining, a brain

Neat-fold RL is the first step of a larger idea. If it works, Roy reads it as precedent for subskill RL: "if i RL and it
genuinely gets good at neat folding through sgs style rl, then we'll have precedence to believe in subskill RL so that it
can generalize to much harder clothes." My note on that: it would prove the recipe, not generalization.

## The idea
Roy's words: AlphaNIST on "skills (grasp, scoop, pick, place, drag), then chain with adept, then a hierarchical model knows
which subskill to pick at which point, then small teleop fine-tune for neat folds."

- **Fold from flat first.** Crumpled comes later.
- **Neatness is emergent.** "It's how well the entire subskills are chained that leads to neatness", so a subskill doesn't
  have its own neatness score.
- **Brain:** a WAM, VLA or similar at about 20 Hz choosing skills, over a 100+ Hz actor.
- **ADEPT** to chain without skill rot: freeze a skill once it passes, retest after every update, roll back if it drops.
- **Baby analogy:** whole-hand grasp comes first, neat pinch late, and deliberate release and two-hand roles later still. For
  cloth that maps to grabbing a bunched wad, then pinching one layer at an edge. Single-layer grabs are rare in our data
  (2/65, 2/67).

## How we got to "RL the whole fold first"
1. Roy's Q1 (Oct 20): can plain RL on whole folds, with a neatness reward, fix ugly folds without touching sub-skills?
2. Q2, only if not: measure and train the sub-skills.
3. The teacher sometimes folds neatly already, and it never tidies after LeHome's check. So Q1 has something to amplify.

## What research says about chaining (Oct 17-18, published, our summaries)
- **Naive chaining fails.** Plain sequencing scored 0-16%, and methods that fix the hand-off scored 46-56%.
- **Errors multiply.** 5 calls at 90% each is 59%; at 97% each it's 86%.
- **Widening a skill's start states can hurt it.** Gu 2022: pick success fell from 95% to 45%.
- **Composition needs training coverage.** CompoSuite: 8-18% on unseen combinations with 56 of 256 trained, 88% with 224.
- **RL isn't a bad generalizer, but it follows training diversity.** RL4VLA: unseen objects 71% vs 45%. CoinRun: 99% train vs
  67% test with 100 levels.
- **"Where" outputs should be maps.** FlingBot's per-pixel map got 93% on unseen shirts; regressed coordinates gained about 0.
- **Flat T-shirt:** 4-7 steps. Most cloth papers script the fold order and learn only a greedy one-step choice. We found
  nothing with RL skills under an RL brain for cloth. That's not proof it doesn't exist.
- **WAMs:** no out-of-box cloth proof. Fine-tuned π0.5 ties or beats WAMs in 4 of 9 cloth head-to-heads, and we found no
  SO-101 WAM cloth result.
- **No held-out-object tests found** for AlphaNIST, SGS or OmniReset.
- **Throughput if skills are the unit (estimate from Newton):** about 8-10k skill calls per 3.5 h on an L4.

## Other lines nobody owns
- **Teacher as brain, RL-trained grasp and release** where it fails. It's unknown whether the frozen teacher would use a
  better skill, so that needs a design.
- **Bottom-up skills** (pinch, release, drag) with a high-level model.
- **Run B:** the teacher in realistic physics, to see whether its hands or its planning break. Parked until stage 1.
- **Real leader-follower demos, then fine-tune.** Ilia's real-robot result needed about a week of DAgger: his zero-shot failed
  and he took 2nd in the real final.
- **Dream-1:** zero-shot failed on SO-101. Fine-tuned is untested.

## World model work (not in the plan, but it exists)
Roy's `ROY-belief-planner` branch has an RSSM with delta decoding, a CEM planner and a diffusion belief front-end, on an old
MuJoCo square sheet.
- Prediction error on moving cloth: 0.0141, against 0.0350 for repeating the last state.
- Carries planned in imagination: 0/10, against 10/10 scripted.
- Draft PR #20 is a separate one-step state predictor, not this.

More on this, and on SuperMimic, in `research_nov5.md`.

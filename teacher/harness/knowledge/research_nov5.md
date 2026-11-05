# What we learned on Nov 5

Labels: **measured** = we ran it. **published** = from a paper. **agent** = from two search agents' reports, not rechecked by
hand. **estimate** = worked out from measured speeds. **Roy** = Roy's stated view.

## Two different A's, don't mix them up
- **Pathway A vs B:** A is top-down, whole-fold neatness RL first. B is bottom-up: skills, then ADEPT chain, then a brain. **Roy
  chose A first.** His words: it's a bet, children learn bottom-up, but top-down is easier.
- **Route A vs B:** A is RL fine-tuning the teacher's own weights. B is a separate policy bolted on. On Oct 19 Roy said Route A is
  what he meant by SGS RL, and he leaned against bolt-on. On Oct 18 he also doubted the weights needed retraining: his view
  was that the teacher already has the skills and needs to use them right (paraphrased).

`stage1_design.md` proposes a third shape: teacher frozen, with a small residual network acting at every denoising step. It isn't
Route A, because the weights don't change. It isn't Route B, because there's no hand-off. Roy hasn't decided.

## Learner candidates
**DSRL is rejected** (Roy): it's "no were close to what we want to do". It has no half-completed states or SGS. Small DSRL-style papers such
as RARM (base 2/30, binary 7/30, RoboMeter 9/30, RARM 25/30 at 90 rollouts) don't change his view that binary beats dense with a
good setup, as in AlphaNIST (98.2 vs 17.8, shared policy).

**Advantage Guidance is the candidate in the design, and isn't decided.** Zhou and Levine, [arXiv 2610.19783](https://arxiv.org/abs/2610.19783),
code at `zhouzypaul/advantage-guidance` (MIT). Paper read in full (published):
- Trains a policy with plain supervised flow matching, conditioned on advantage, with no gradient through a critic.
- **Three parts:** action imagination (K = 7 actions from a BC policy give extra labels on the same state), a residual flow
  (v_π = v_π̂ + v_θ), and contrastive guidance when sampling: v = v_π̂ + w[v_θ(y+) − v_θ(y−)], with y± = ±1.
- **Critic:** IQL on action chunks. Advantages are normalized per state. γ = 0.99 or 0.999.
- **Results:** 26 sparse-reward tasks, about 70 vs EDP about 62, CFGRL about 50, RCP-A about 49 (read from a chart). On OGBench,
  large-N best-of-N sampling slightly beats it. On OCBench, it wins.
- **Data sizes:** OGBench 1M transitions, OCBench 10,000 to 30,000. MLPs of 512 × 4 and 1024 × 8.
- **Limits:** offline only, with online left as future work. It appears to be state-based, and wasn't tested on a VLA, on images or
  on cloth.

**Why it fits our teacher (from reading Ilia's code):** the teacher already has a good/neutral advantage token, an AdaRMS vector and
CFG, which is closest to CFGRL. The line is `v_t = v_uncond + cs * (v_cond - v_uncond)` at `pi_modified.py:1738`. `sample_actions` at
line 1438 takes `num_candidates`, and runs the expensive prefix once for N candidates (read, not timed). It also has success and
completion heads.

## Fine-tune vs bolt-on (agent)
**Fine-tuning a VLA**
- **Larchenko:** fine-tuned everything but SigLIP, on 12,500 kept episodes and about 300k steps, on an H200, with rollouts on an RTX
  PRO 6000. He restored saved states (success replay, hard mining, semi-success) and had a per-garment curriculum. In the sim config
  it's commented out, with no ablation. He called plain binary "too sparse". His episode total stays 0/1, with checkpoints.
- **RECAP:** over 90% after one 600-episode round on real T-shirts.
- **πRL:** 40.1 to 90.9 on 8 H100s.
- **Task-collapse paper:** sparse PPO on a 450M SmolVLA action expert gained nothing.
- **Memory (openpi):** inference over 8 GB, LoRA over 22.5 GB, full over 70 GB. Ilia's repo has no LoRA config. Teacher inference
  measured 16 GB on a 24 GB L4, so LoRA and inference won't share an L4.

**Bolting on a separate policy**
- We found no evidence for a hand-off on cloth.
- Chained π0 scored 41.4 vs 54.9 end-to-end. A better grasp lowered predicted downstream success from 0.64 to 0.32.
- RLT zip tie went from about 27 to about 87%.
- Nobody has measured how a VLA recovers on states it rarely saw.

## What raw AlphaNIST would cost on cloth (estimate)
On an L4 at 2.3 shirt-seconds per second, 25 Hz:

| AlphaNIST steps | Time | Cost |
|---|---|---|
| 0.8 billion | about 160 days | about $3,600 |
| 3.3 billion | about 1.8 years | about $15,000 |
| 51 billion | about 28 years | about $232,000 |

So raw AlphaNIST on cloth is out, and the teacher has to do most of the work.

**Roy's lean:** he doubts the cost of RL is worth it, and thinks BC may be the better way to a good baseline. Once there's a baseline,
port it with `stats-worldfold`. Not decided.

## SuperMimic (published, paper read, parked)
Liao, Abbeel, Sreenath, [arXiv 2610.28480](https://arxiv.org/abs/2610.28480).
- A learned MLP world model plus backprop through time with a critic, H = 32, k = 8.
- 1 H100 for 3 h vs 128 L40S for 168 h. More than 10× sim speed, about 64× fewer imagined steps.
- Proprioceptive policy. Dense tracking reward, credited for stability. States "no terrain or object interaction".
- Its reset sampling (1 s bins: 50% uniform, 40% softmax on low return, 10% past weight) resembles grouped SGS.
- **Our read:** not for cloth now. Worth another look alongside `stats-worldfold` after a baseline.

## World models we already have
- **`stats-worldfold` / `ROY-belief-planner`, Roy's, Aug (same code):** RSSM on the old MuJoCo square sheet, with a CEM planner and a
  diffusion perception front-end. The key fix was predicting the per-step change. Error on moving cloth went from 0.1116 to 0.0141,
  against 0.0350 for repeating the last state. The planner controlled the carry only, and finished 0/10 vs 10/10 scripted (fold score
  0.514 vs 0.797). Latent KL sat at its floor. No garments, no Isaac.
- **PR #20, apun16, draft:** a residual MLP one-step predictor on 121 exact cloth points, not a Dreamer or JEPA model. It doesn't choose
  actions, and has no data or live Isaac run yet.
- **Adam's `docs/world_model.md`:** an actor trained on imagined rollouts took BC from 17.5 to 87.5%, then 95%, on a one-corner MuJoCo fold.
  Main has the world model marked as parked.

## VR teleop (Roy is looking at this, untested)
- Roy's looking at a used Quest 2 at $169, against $400+ for a Quest 3. It has two screens for depth and two tracked controllers,
  which is enough physically. Ask the seller for both controllers and a factory reset.
- **Connecting it:** NVIDIA CloudXR lists Quest 3 and Pico 4 Ultra, not Quest 2. ALVR + SteamVR needs a real GPU on the PC. The likely
  route is our own teleop through the Quest browser (WebXR). **Not built or tested.**
- **Still a blocker:** a headset doesn't fix the grip. Sticky-cloth demos are invalid, and whether attach-and-slip grasps
  (`teacher/oracle/attach_grasp.py`) work under teleop is untested.

## Elsewhere in the repo
Mo merged `hardware/so101/` to `main` on Nov 5: real SO-101 calibration, idle pose, control panel and phone teleop. It isn't on this branch.

## Other research
Everything else from Roy's screenshots is in `research_screenshots.md`: MIT blind policy, Bhatt, Foresight, AdaDex, video planners,
3DPWM, Tactile Curiosity, RARA, Isaac Lab 3.0. Most of those are authors' claims, with the papers unread. Two pieces are cut off or
unread: the rest of PrismaX's π0.5 how-to, and what Isaac Lab 3.0 means by "RL post-training".

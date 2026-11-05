# Research screenshots, transcribed

Text notes from X posts Roy collected. Numbers are the authors' claims unless a paper was read. Since these notes were written,
the Advantage Guidance and SuperMimic papers have been read: see `research_nov5.md`. The screenshots themselves aren't in this repo.

# Screenshot notes, batch of 2026-10-19 (19 WhatsApp images in Downloads)

Read one by one by Claude, no agents.

## Topic A: blind precise assembly, MIT (13 images, many duplicates)
Younghyo Park and Antonia Bronars, advised by @kushalk_ and @pulkitology. arXiv 2610.23849, project page younghyopark.me/blind-proprioc..., code github.com/younghyopark/mjfit.
- 7.36.45 PM (6): thread start. Only joint positions as input: no camera, no tactile, no F/T, no torque, no object poses.
- 7.36.43 PM: Bronars' version of the same announcement.
- 7.36.45 PM (5): [2/n] the "last mile" of assembly. A generalist VLA or an agent gets the part "roughly there", then their policy takes over. Every episode starts roughly there, and training is RL in simulation.
- 7.36.45 PM (4): [3/n] near-perfect sim success, same as with a wrist camera. Training curves are plotted to 300 million environment steps. [4/n] The peg policy works zero-shot in the real world on 8 pegs, 3 of them unseen.
- 7.36.45 PM (2) and (3): [5/n] a linear probe on the latent decodes the insertion phase on the real robot. In distribution, blind vs visual: peg 98 vs 90%, gear >99 vs 98%, nut >99 vs 95%.
- 7.36.45 PM (1): [6/n] on unseen parts, blind vs visual: peg 85 vs 39%, gear >99 vs 95%, nut 99 vs 63%. Their explanation: joint readings are compact, plus heavy randomization of parts, poses, grasps and contact.
- 7.36.44 PM (6) and 7.36.45 PM: [7/n] going blind is slower: median 0.76 s vs 0.44 s with vision. A scripted spiral search takes 1.47 s.
- 7.36.44 PM (4) and (5): [8/n] mjfit, convex decompositions for tight-clearance rigid parts, ~10x faster than SDF. Not relevant to cloth.
- 7.36.44 PM (3): TL;DR repost.
- 7.36.44 PM: Bronars 9/, the links.

## Topic B: Aditya Bhatt, "Blind Dexterity" (1 image)
- 7.36.44 PM (1): "Searching" only emerges from blind RL, and teleop BC policies don't learn it. Distilling an expert teacher "who already knows where things are" into a blind student works poorly: "expert behaviors are undistillable into blind students! You have to train them blind with RL." aditya.bhatts.org/BlindDexterity.

## Topic C: first user report on Dream-1 (1 image)
- 7.36.44 PM (2): Robert Lukoszko, 3 h after launch, ran Dream's own "Dream YAM Sim" (two arms, three cameras, MuJoCo + Dream-1-Beta, joint control, 48-step chunks). "Pick up banana and put in the plate" worked, but it got stuck 3 times and repeated actions twice. Asked to flip a bottle, it looped and never picked it up. The run log shows 1,766 actions in 37 requests, about 1 min of motion, about $0.05 at $3/h.

## Topic D: Foresight failure detection (3 images)
- 7.36.46 PM (1), (2), (3), via @vigram_void: UMich/Princeton/UVA, haoranzhangumich.github.io/Forsight_web.
  - Method: a frozen V-JEPA 2 encoder plus an action-conditioned predictor. A causal transformer turns the predicted future latents into a failure score, trained from final success/failure labels only. Conformal calibration sets the alarm threshold.
  - Results: BEHAVIOR-1K balanced accuracy 0.78 vs 0.64. Real ACT rollouts, ROC-AUC 0.93 vs 0.89.
  - Limits: trained on ACT and tested on pi0.5 it drops to 0.56, and it takes 184 ms per check on an H200.
  - Quote: "A robot can miss a grasp and still show a perfectly ordinary camera frame."

## Topic E: not robotics (1 image)
- 7.36.46 PM: Nathan Lambert on models making slides and plans, "slop smell", his colloquium slides tool.

---

# Batch of 2026-11-05 (56 WhatsApp images, sent in chat; posts dated about Oct 19-22, a few older)

Transcribed by Claude, no agents. X screenshots only: no paper in this batch has been opened, so every number is the author's claim. Long threads are transcribed closely, not always word for word. Some are cut off in the screenshot.

## 1. Advantage Guidance: Paul Zhou and Sergey Levine, UC Berkeley (about 15 images, 2 duplicates)
Links: advantage-guidance.github.io, arxiv.org/abs/2610.19783, github.com/zhouzypaul/adv... (cut off). Posted 10/20/26, Levine quote 10/21/26.
- Zhou: "Last year I trained the advantage-conditioned VLA for π*0.6 during my internship @physical_int. Now, I have a much improved recipe." A "performant RL method with stable advantage-conditioned supervised training and contrastive guidance sampling".
- RL can improve beyond the data but is unstable and hyperparameter-sensitive. Supervised learning is stable but limited by data quality. "Can policy training be as simple and stable as SL, but as performant as RL?"
- Naive advantage conditioning usually doesn't work well. Reason 1: an advantage-conditioned policy doesn't maximize; it generates actions close to the queried value and penalizes actions above it. Reason 2, "advantage alignment problem": asked for a high-advantage action, naive policies often ignore the conditioning and regress to the unconditional policy (shown on 20 OGBench tasks across 4 domains).
- Three ingredients: (1) action imagination: augment the data with actions sampled from a BC policy, giving counterfactual action/advantage supervision on the same state; (2) residual parameterization: conditional policy = unconditional + residual flow, v_pi(x,t,s,y) = v_psi(x,t,s) + v_theta(x,t,s,y), encouraging the residual to attend to the conditioning; (3) contrastive guidance when sampling: v_cg = v_psi + w[v_theta(y+) - v_theta(y-)], which they show samples the KL-regularized advantage-maximizing action under an exponential-family assumption.
- Results: 26 sparse-reward tasks from OGBench (random play-style data) and OCBench (smooth human-like demos). Aggregate bars, read by eye: RCP-R ~11, FAWR ~43, CRR ~45, FQL ~46, RCP-A ~49, CFGRL ~50, QGF ~58, EDP ~62, AG ~70.
- "test-time best-of-N sampling used to be the strongest RL method I know... On OCBench... Advantage Guidance beats best-of-N!!" Compared up to N=32.
- Replies: Daniel Lenton: normalising over 8 actions per state caps advantage at sqrt(7)≈2.65, about where the plain conditional policy falls off in Fig 15; Zhou: that's why contrastive guidance is essential. On LLMs Zhou: works as policy extraction but a good value/advantage function is much harder there. Vanraj Narula: is the guidance mainly keeping actions on-support? (unanswered in view).
- Levine: "simply extracting policies with supervised learning conditioned on the advantage can get state of the art results in RL, as long as we get a few important details right!"

## 2. SuperMimic: Qiayuan Liao, with Pieter Abbeel's quote (4 images)
supermimic.github.io. "reduces GPU-hours by more than 1000x. Across 129,514 motions, SuperMimic surpasses the state of the art (128 GPUs, 7 days) with 1 GPU in 3 hours." Abbeel: "(1) World model much faster than regular sim (2) Backprop through time collects way more signal than policy gradients". Figures: "Only the world model passes the gradient back", the simulator breaks the gradient; "PPO estimates the gradient from samples", SuperMimic computes it.

## 3. AdaDex: Yuxuan Kuang and Jiashun Wang (about 10 images), yuxuank.com/AdaDex/, 10/22/26
- From one short hand-object clip per skill, learns an adaptable skill: functional grasping, reposing, in-hand reorientation, throw-and-catch.
- Hybrid Dexterous Mimicking: one policy, PPO on the union of rollouts from two kinds of massively parallel env. TrackEnv: starts along the human reference, dense tracking reward, supplies the human prior. GenEnv: large initial noise, goals far from the reference, sparse object success plus dense wrist guidance, no finger terms, supplies coverage. "No discriminator, no phase variable, no distillation stage, and no reference needed at test time."
- On unseen motions: ManipTrans 7.3% tracking progress, SimToolReal 13.7%, AdaDex 80.0%. Casual RGB(D) video: 80.5% progress on 9 (cut off) vs 31.7% ManipTrans, 25.0 mm hand error.
- "System 0": coding agent (Codex with GPT-6 Astra) writes sparse wrist and object waypoints at low frequency; skills execute at high frequency. Isaac to MuJoCo zero-shot, real rollouts at 30 Hz. Hands: Arti-MANO, XHand, Wuji, Sharpa, xArm6+XHand.
- Wang: sparse low-dimensional goals as the planner-to-control interface. Earlier related work: "Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework", 6/18/26.

## 4. Video planners vs world models: Hyunwoo Ryu, Vincent Sitzmann (2 images), 10/20/26
- Ryu: with expert data you don't need an action-conditioned world model. Offline Minecraft: video planner + IDM beats Dreamer 4, ~2x on iron pickaxes, ~8x on diamonds. VERA (Lester Li, Evan Kim, Xingjian): video planner + IDM beats DreamZero on tasks needing reasoning. WMs and WAMs learn through action, "a tiny embodiment-specific channel"; video is universal.
- Sitzmann: "go where the data is". Video planning is the easy path, "albeit expensive"; preferable to a world model even with action labels, because action labels are scarce.

## 5. 3D Point World Models: Fuxin Li, arxiv.org/pdf/2607.00148, github.com/pvskand/3dpwm (1 image)
Partial point clouds aren't enough for physics. Completes geometry first, then imagines with robot action, up to 15 s rollouts, improving on ParticleFormer and PointWorld. Sim-trained dynamics and completion transferred to real. Learned on Coffee and MugCleanup, planned unseen CoffeePreparation at 45%.

## 6. Tactile Curiosity Drives Robot Interaction, CoRL 2026 (2 images)
Iten, Proshkin, Sukhija, Coros, Krause, Abbeel, Sferrazza. kiten.me/projects/tacex. RL from scratch with no task reward: only touch-curious agents learn to grasp, and their data supports offline pick-and-place. Steering a frozen VLA that never saw touch: touch as observation helps, plus curiosity is best, especially contact-rich tasks.

## 7. openpi / π0.5 Fine-Tuning: How-To and Tips, PrismaX (1 image, article cut off)
Fine-tuning behind their IROS 2026 demo. 668 episodes (about 10 hours) of YAM pick and place, then large-scale sim ablation. "a number of training settings have a profound impact on checkpoint performance". Data blend proprietary. The tips themselves aren't in the screenshot.

## 8. RARA, UVA, via vigram (1 image), shua-kang.github.io/RARA_robot/
Fine-tuning on a few demos can hurt visual features. Retrieves similar pretraining demos, aligns stages, anchors features to pretrained ones. RoboTwin OOD 47.3 to 58.3; RoboCasa GR00T N1.7 44.3 to 55.5. Random pretraining data 34.9, random stage pairs 40.6, both worse than plain. Needs related skills in the pretraining corpus; visual shifts in the same robot.

## 9. Isaac Lab 3.0 release, NVIDIA Robotics (2 images)
Modular architecture, multi-physics and multi-rendering, Newton integration, OVRTX renderer, "RL post-training", LEAPP deployment.

## 10. Others
- Discrete action heads, zeji yi, Max Simchowitz: simplest discrete policies on par or better than continuous on bimanual tasks.
- DEX-X, via Leo Kharon: replays human video motion in sim to synthesize touch, RL teacher, distills to a point-cloud student.
- AR-WAM, HKUST, via Robots Digest: asks whether the manipulation policy should never see language (cut off).
- Eka: Vision-Force-Action model, 1500 picks/hour. "Force is the language of manipulation. Reinforcement learning is how dexterous manipulation scales."
- MPC vs differentiable RL for arm tracking, C. Zhang, via Codex (about 5 images): both errors negligible under perfect modelling, so hardware is the bottleneck. First-order gets nm, zeroth-order µm. Quazi: RL struggles at isotropic tracking over a large workspace. John Rood: "negative results are the first thing a summary drops".
- Dream-1 launch, Yasu (2 images): "It won't work properly at all unless it's something like an instruction that feels like it was totally pulled from a dataset".
- You Jiacheng on MIT blind paper: tactile won't help blind assembly; it learns an impulse-like behaviour to identify contacts.
- mjfit, Younghyo Park, Lucky Iyinbor (2 images): already noted in the Oct 19 batch; not relevant to cloth.
- AMD PhysX port, Binh Pham and replies (4 images): PhysX on Radeon; Genesis and MJX mentioned for AMD. Not relevant to us.
- LidarVerse: driving. Not relevant. One non-research post (Mecka/thirdspace).

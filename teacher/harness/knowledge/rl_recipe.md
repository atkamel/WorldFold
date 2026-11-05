# The RL recipe

Roy's name for it: **"sgs-alphanist-ketago-almost complete states".** Numbers are **published** unless marked. The rest is
**interpretation**. Not built, not tested on cloth.

## AlphaNIST ([arXiv 2610.13354](https://arxiv.org/abs/2610.13354)) and SGS ([arXiv 2610.12465](https://arxiv.org/abs/2610.12465))
It worked on a real robot.

**Recipe:** a reset bank (almost-complete and mid-task states) + SGS sampling + a binary reward + plain PPO.
- **Reset bank:** cached states across all tasks: part lying anywhere, in the gripper, mid-carry, almost mated, half
  assembled. It isn't a backward sequence, and the policy isn't told the stage or task.
- **SGS sampling:** weight q ∝ ε + √(p(1−p)), where p is the recent success rate of that start state. It trains mostly on
  states it passes about half the time, which builds the curriculum automatically.
- **Binary reward:** under 1 mm, plus a control cost that carries no progress information. Episodes also end if a joint
  exceeds its real speed or torque limit.
- **Network:** SimBa. Franka at 25 Hz, 32 steps per env per iteration, 4K envs per GPU.

**Results**

| Comparison | Outcome |
|---|---|
| Shared policy, binary reward | 98.2% |
| Separate policies, binary | 81.4% |
| Shared, dense reward | 17.8% |
| Separate, dense | 78.3% |
| GPU-hours, shared vs separate | 410 vs 16.1K |
| Envs: 1K / 4K / 8K / 256K | 38.6 / 78.9 / 86.9 / 97.4% |
| 256K, uniform sampling | 84.3% |

- Most of the gain comes by about 8K envs, and 4K is one GPU. Reward, sampling and diversity matter more than scale.
- **On a real UR5e** (SGS, DAgger to RGB cameras, about 2 days on 4 L40S): gear 94%, rod 61%, nut 37.5%.

## What cloth changes
- AlphaNIST's score is trivial: 1 mm, pass/fail. Neatness isn't. **Calibrate the score against a human first, and try to
  cheat it** (crumple, flatten without folding, sticky physics). See `research_good_rl_tasks.md`.
- Cloth sim is far slower, and their step counts don't transfer (their cheapest working point is about 0.8 billion steps). So
  we **adapt the teacher, never train from scratch**. Precedents: πRL took π0.5 from 77.1 to 98.3% on LIBERO, AlphaGo did
  supervised then RL, RL-100 reached 250/250 on towels, and HIL-SERL took 1 to 2.5 hours on real robots.

## Stage 1 as planned (not built, not tested)
- **Start states:** half-completed folds, from the teacher's own mid-fold snapshots (1,049 from 20 folds, every 5 steps).
- **Reward:** binary on neatness, at a line calibrated with Roy.
- **Episodes:** don't end at LeHome's success check, so tidying can be learned.
- **Sampling:** SGS, near 50% success.
- **KataGo-style help:** predict the neatness sub-scores (rect, flat, layers) as extra critic targets. In KataGo, auxiliary
  targets made training 1.65× faster. Dense prediction helps, dense reward hurts.
- **Randomize:** garment pose and rotation, LeHome's 5 seen garments per type, unseen held out, mild visual changes.
- **Learner:** not decided.

## Rules for RL code
- Build on published code (Isaac Lab `rsl_rl`, UWLab/OmniReset, LeRobot HIL-SERL, Ilia's LeHome solution). No from-scratch RL
  algorithms, no Stable-Baselines3.
- Our part is the honest physics, observations, score, start states, and plugging them in.
- Freeze a skill once it passes, retest after every update, roll back if it drops.

## Further reading
- `research_curriculum_katago.md`: mid-state resets, and KataGo's compute efficiency.
- `research_good_rl_tasks.md`: how to design and red-team the verifier.

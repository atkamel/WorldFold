# The RL recipe, and why

Numbers here are **published** unless marked. The rest of the reasoning is **interpretation**.

## AlphaNIST ([arXiv 2610.13354](https://arxiv.org/abs/2610.13354)) and SGS ([arXiv 2610.12465](https://arxiv.org/abs/2610.12465))
Same NVIDIA team, same recipe. It worked on a real robot.

**Recipe:** a reset bank (mid-task states) + SGS sampling + a binary reward + plain PPO.
- Sampling weight q ∝ ε + √(p(1−p)), where p is the success rate. So it trains mostly on states it passes about half the
  time, which builds the curriculum automatically.
- Binary reward: under 1 mm, plus a control cost.
- Network: SimBa. Franka at 25 Hz, 32 steps per env per iteration, 4K envs per GPU.

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
- AlphaNIST's score is trivial: 1 mm, pass/fail. Neatness isn't. **Calibrate the score against a human first, and try
  to cheat it** (crumple, flatten without folding, sticky physics). See `research_good_rl_tasks.md`.
- Cloth sim is far slower. Whole-shirt PPO at AlphaNIST scale is out, so we **adapt the teacher, never train from
  scratch**. Precedents: πRL took π0.5 from 77.1 to 98.3% on LIBERO, AlphaGo did supervised then RL, and HIL-SERL took
  1 to 2.5 hours on real robots.

## Proposed for stage 1 (not built, not tested)
- **Reset bank:** the teacher's own mid-fold snapshots (1,049 from 20 folds, every 5 steps).
- **Reward:** pass/fail on neatness, at a line calibrated with Roy.
- **Episodes:** don't end at LeHome's success check, so tidying can be learned.
- **Sampling:** near 50% success, per state.
- **Extra prediction heads:** predict the neatness sub-scores (rect, flat, layers) as extra critic targets. In KataGo,
  auxiliary targets made training 1.65× faster. Dense prediction helps, dense reward hurts.
- **Method, likely DSRL** ([nakamotoo/dsrl_pi0](https://github.com/nakamotoo/dsrl_pi0)): steer the starting noise with a
  small SAC network, teacher frozen. That fits on an L4 (24 GB), which full π0.5 fine-tuning (>70 GB) doesn't. It needs
  adapting to Ilia's inference (CFG, inpainting, candidates).
- **Randomize:** garment pose and rotation, LeHome's 5 seen garments per type, unseen held out, mild visual changes
  (Ilia's augmentor).

## Rules for RL code
- Build on published code (Isaac Lab `rsl_rl`, UWLab/OmniReset, LeRobot HIL-SERL, Ilia's LeHome solution, DSRL). No
  from-scratch RL algorithms, no Stable-Baselines3.
- Our part is the honest physics, observations, score, start states, and plugging them in.
- Freeze a skill once it passes, retest after every update, roll back if it drops.

## Further reading
- `research_curriculum_katago.md`: mid-state resets, and KataGo's compute efficiency.
- `research_good_rl_tasks.md`: how to design and red-team the verifier.

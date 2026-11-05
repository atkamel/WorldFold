# The RL recipe: what we're copying, and where it stops fitting

Numbers are **published** unless marked **measured**. Anything else is **interpretation**. Not built, not tested on cloth.

Roy's name for it: **"sgs-alphanist-ketago-almost complete states".** It's several pieces together, not just a reward type.

## 1. Who did what
| Piece | Source | What it decides |
|---|---|---|
| Reset bank (almost-complete states) | OmniReset, used by both papers below | Which starting states exist |
| SGS, Success-Guided Sampling | "A Balanced Data Diet", [arXiv 2610.12465](https://arxiv.org/abs/2610.12465). Zhang, Guaman Castro, Yin, Dagnino, Gupta, Scalise, Boots (University of Washington, NVIDIA) | Which starting state to practice next |
| AlphaNIST | [arXiv 2610.13354](https://arxiv.org/abs/2610.13354). Zhang, Akinola, Scalise, Guaman Castro, Yin and others (NVIDIA, University of Washington) | Combines bank + sampling + binary reward + shared policy + scale |
| KataGo | Wu 2019, [arXiv 1902.10565](https://arxiv.org/abs/1902.10565) | Getting the same strength from far less compute |
| Curriculum Mining | [Matteo's blog](https://the-puzzler.github.io/blog/curriculum-mining/) | Why mid-state resets work |

## 2. The published recipe, exactly
**Reset bank.** Cached states across every task: part loose in the workspace, being picked up, in transport, near mating,
partly assembled. "These reset families define regions of the training distribution, not a prescribed sequence", and the
policy gets no stage label. It isn't worked backward from the end.

**Sampling.**
- AlphaNIST: pick entry i with q_i ∝ ε + √(p_i(1−p_i)), where p_i is its success rate over the last L tries from that entry.
- SGS: keep the last H = 100 outcomes per configuration. Score w = (p+ε)^(κt) · (1−p+ε)^(κ(1−t)), then softmax over
  log(w+ε) at temperature T = 2.
- SGS manipulation settings: N = 32,768 configurations, t = 0.5, κ = 1, ε = 1e-4. It reports little sensitivity to these.
- Training ends up mostly on states it passes about half the time, so nobody has to design a curriculum.

**Reward.** r = 1 once when the part is within 1 mm of assembled, then the episode ends, minus c(‖a‖² + ‖q̇‖² + ‖τ‖²). That
penalty keeps motion smooth and carries no progress information. Episodes also end at horizon T, or when a joint exceeds
its real speed or torque limit.

**Learner.** Plain PPO, SimBa actor and critic, 25 Hz, 32 steps per env per iteration, 5 epochs, 4 minibatches. 4K envs per
GPU. One policy for all tasks, with no task or stage input.

**Results**
| Comparison | Outcome |
|---|---|
| Shared policy, binary reward | 98.2% |
| Separate policies, binary | 81.4% |
| Shared, dense reward | 17.8% |
| Separate, dense | 78.3% |
| GPU-hours, shared vs separate | 410 vs 16.1K |
| Envs at 25k iterations: 1K / 4K / 8K / 256K | 38.6 / 78.9 / 86.9 / 97.4% |
| Sim steps at those scales | 0.8B / 3.3B / 6.6B / 210B |
| 256K, uniform sampling instead of SGS | 84.3% |

**On a real UR5e** (SGS teacher, DAgger to an RGB student, about 2 days on 4 L40S, zero-shot): gear 94%, rod 61%, nut 37.5%.

## 3. KataGo and Curriculum Mining
- **Dense prediction helps, dense reward hurts.** Predicting ownership and score as extra targets made training 1.65×
  faster in Go. For us: extra critic heads predicting the neatness sub-scores.
- **Cheap most of the time, expensive sometimes.** Playout cap randomization was worth 1.37×. For us (interpretation): a
  cheap neatness proxy on most steps, full `fold_score` occasionally.
- **Mid-state resets** shorten credit assignment and mine a curriculum of states that are currently solvable.

Full notes: `research_curriculum_katago.md`.

## 4. Where it stops fitting cloth
**We can't run PPO at their scale.** Their cheapest working point is about 0.8 billion steps.

| Measured on one GCP L4 (about $1/hour), 2 workers, teacher's own physics | |
|---|---|
| Full folds | 20 in 25 min, about 48/hour |
| Restored mid-fold episodes (60 policy steps) | 60 in 17 min, about 210/hour |

**So per-state SGS doesn't work as published (interpretation).** With 1,049 states at 210 episodes/hour, each state gets
tried about once every 5 hours. SGS keeps 100 outcomes per configuration. Options nobody has tried:
- Track success per group, like garment × fold stage, instead of per state. 20 garments × 4 stages is 80 groups.
- Use far fewer states.
- Find more throughput first. We've never profiled for that; `teacher/README.md` has the old speedups.

**Dropping PPO for off-policy learning (interpretation).** Precedent for adapting a trained policy instead of starting from
zero: AlphaGo's RL stage took about 1 day on 50 GPUs, πRL took π0.5 from 77.1 to 98.3% on LIBERO, and RL-100 took towel
folding to 250/250.

## 5. The reward question is open
Binary won AlphaNIST at billions of steps. A much smaller test points the other way:

**RARM** ([arXiv 2606.22027](https://arxiv.org/abs/2606.22027), Yang, Wang, Jing and others) ran DSRL on a frozen π0, on real
bimanual clothes folding. Base policy 2/30. After 90 rollouts: binary reward 7/30, RoboMeter reward 9/30, RARM's learned
progress reward 25/30.

It doesn't contradict AlphaNIST, because it's a different setting. As far as we read, it had no reset bank and no SGS, and
it ran for 90 rollouts. **Nobody has tested DSRL + SGS + reset bank on neatness.** Binary may need the bank and sampling to
work at our budget, or may need more help. Test it, don't assume it.

Another published pattern, from Larchenko and π\*0.6: keep the return tied to the final outcome and use dense signals only
for credit assignment.

## 6. Cheapest experiments first
Ranking from `reports/Cloth fold quality metrics.md`, section "a cost ladder". Costs there are inference, not measured.

| Rung | What | Trains anything? |
|---|---|---|
| 1 | Sweep CFG, noise temperature, N and execution length, using neatness as the bandit's reward | No |
| 2 | Let episodes continue past LeHome's check; rank episodes by neatness | No |
| 3 | Best-of-N with a scorer, or look-ahead: roll each candidate forward in sim and score it (untested) | Little |
| 4 | **DSRL**: small network steers the teacher's input noise | Small MLPs |
| 5 | Residual policy (PLD, about 250k steps) | Small MLP |
| 6 | Reward-weighted imitation (SARM: T-shirts 8% to 83%, LoRA, 2× RTX 4090) | LoRA |
| 8 | Full RECAP rerun as Ilia did it: about 12,500 episodes, H200 | Too expensive as published |

**Why rung 2 matters first:** if the teacher can't tidy when allowed to continue, RL has less to amplify. Letting it
continue has no published evidence and hasn't been run.

**DSRL caveats:** its noise is the same variable that best-of-N samples and temperature scales, so it replaces those rather
than stacking. Nobody has tested it on a policy already using CFG and best-of-N. Published gains start from weak bases, and
ours already passes LeHome's check 70-93% of the time, so the open question is moving neatness at fixed success.

DSRL also fits our GPU: full π0.5 fine-tuning needs >70 GB and LoRA >22.5 GB (openpi's figures), against 24 GB on an L4
where inference alone measured 16 GB.

## 7. Where to plug in
Line numbers are from `IliaLarchenko/lehome_solution` at `ca73828`. On the VM it's at `/opt/lehome_solution`, so check it
matches.

| Need | Where |
|---|---|
| Don't stop at LeHome's check | `lehome-challenge/scripts/utils/remote_loop.py:258,312`: `stop_on_success` comes from the task dict and defaults True. `scripts/dagger_collect.py:976` sets it False. The comment at line 275 says the server normally stops sooner, so check that too |
| Step limit | `scripts/run_eval.py:635`, `--max_steps`, default 600 |
| Steer the noise (DSRL) | `src/lehome_solution/models/pi_modified.py`: `sample_actions` (1438) only draws noise `if noise is None` (about 1607), then scales by temperature |
| Inference knobs | Set per garment; Ilia's table: executed 5/5/3/3, guidance 7/7/9/7, temperature 0.9/0.7/0.7/0.7, candidates 2/3/3/3 (long top, short top, long pant, short pant) |
| Save and restore states | Ours: `teacher/grasp_test_patch.py`. His: `capture_physics_state` and `restore_physics_state` in `remote_loop.py` |
| Neatness score | `teacher/mujoco_live/neat_metric.py`, `fold_score(x, tris, flat_area_m2, thickness)` |
| Score a run's end states | `teacher/neat/teacher_end_states.py` |
| Layer spacing for the score | 5 mm in LeHome (`H` in `teacher_end_states.py`) |

Not measured: how long `fold_score` takes per call. Measure that before using it inside training.

## 8. The score has to come first
AlphaNIST's check is trivial: 1 mm. Neatness isn't, and a verifier is the only pressure RL feels (`research_good_rl_tasks.md`).
- **`fold_score` exists,** and resists crumpling.
- **The pass line doesn't.** 15 appears in our tables, but that's only a tally. Roy decides what counts as neat.
- **The score disagrees with Roy's eye on trousers.** See the data in `00_start_here.md`.
- **Alternative:** the gated composite in `reports/Cloth fold quality metrics.md`, not implemented. It compares vertices to
  an ideal fold target, and multiplies by LeHome's check so half-folds earn nothing.
- **Calibrate on labelled folds, then try to cheat it:** crumple, flatten without folding, sticky physics.

## 9. Rules for RL code
- Build on published code: [DSRL](https://github.com/nakamotoo/dsrl_pi0), RLinf's DSRL, Isaac Lab `rsl_rl`, UWLab/OmniReset,
  LeRobot HIL-SERL, Ilia's solution. No from-scratch RL algorithms, no Stable-Baselines3.
- Our part is the honest physics, observations, score, start states, and plugging them in.
- Keep the best weights, retest after every update, roll back if it drops. Ilia rolled back 4 times.
- With 15 trials the error is about ±13 points at 50% success. Differences of 10-20 points at that size are noise.
- Hold out unseen garments, and watch videos.

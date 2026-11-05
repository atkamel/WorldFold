# Curriculum Mining blog + compute-efficient AlphaGo-level training (notes 2026-10-20)

## Blog: https://the-puzzler.github.io/blog/curriculum-mining/ (Matteo, code: github.com/the-puzzler/selfplay)
- **Setup:** pure self-play PPO, no search, terminal ±1 reward. Same 1.83M-parameter AZNet as Pgx's Gumbel AlphaZero baselines.
- **Results,** score vs the published baselines over 10,240 games: Othello 0.671, Hex 11×11 0.835, Go 9×9 0.817.
- **Compute:** about 1.05B env steps (8,000 iterations × 4,096 envs × 32 steps). The baseline cost is 865M (Othello/Hex) or 1.73B (Go), counting 33 interactions per frame (32 MCTS simulations + the move). Go 9×9 is beaten at 0.61× the baseline's interactions.
- **Ingredient 1, random legal mid-game resets** (up to 40–50 random plies, inspired by OmniReset): shortens credit assignment and mines a curriculum of currently-solvable states. Stepping stones cannot be planned (Stanley & Lehman).
- **Ingredient 2, a ring of 12 frozen checkpoints:** a new one every 100 iterations, half the envs play against the bank, and the loss applies only to the learner's moves. This stops self-play drift.
- **Author's limits:** reference networks, not SOTA; greedy raw policies on both sides; finite games.

## Compute-efficient AlphaGo-level training
- **AlphaGo Zero:** about 4.9M self-play games to beat AlphaGo Lee (DeepMind). DeepMind's main Go run was about 41 TPU-years (KataGo paper's figure).
- **ELF OpenGo:** 2,000 V100s × 13–14 days ≈ 74 GPU-years.
- **KataGo (Wu 2019, arXiv 1902.10565):** about 27 V100s × 19 days ≈ 1.4 GPU-years, 4.2M games, 241M samples. It surpasses ELF, about 50× less compute.
  - **Ablation slowdowns:**

    | Removed | Slowdown |
    |---|---|
    | Auxiliary ownership + score targets | 1.65× |
    | Global pooling | 1.60× |
    | Game-specific features | 1.55× |
    | Playout cap randomization | 1.37× |
    | Auxiliary opponent-policy target | 1.30× |
    | Forced playouts + policy target pruning | 1.25× |

    Their product is about 9.1×, an underestimate per the paper.
  - **Evaluation:** opponents sampled by p(1−p).
- **Gumbel AlphaZero/MuZero** (Danihelka et al., ICLR 2022): policy improvement with few simulations. MiniZero reports strong 9×9 Go at only 2 simulations.

## Takeaways for cloth RL
1. **Mid-state resets are the strongest shared trick** (blog, OmniReset/AlphaNIST, SGS). Use the teacher's harvest snapshots (1,049 states, every 5 steps) as the reset bank.
2. **Dense prediction helps, dense reward hurts.** Keep the reward binary on neatness, and add auxiliary critic heads that predict the neatness sub-scores (KataGo 1.65×).
3. **Run the cheap proxy most of the time and the full scorer occasionally** (playout-cap-style).
4. **Sample near 50% success** (SGS, AlphaNIST, KataGo evaluation).
5. **From-scratch step counts (10^8–10^9) are out of reach for cloth.** Adapt the teacher instead.

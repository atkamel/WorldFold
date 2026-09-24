# Quarter-fold behavior-cloning workflow

The quarter-fold policy uses two actors: stage 0 performs the simultaneous
two-arm half fold and stage 1 parks the left arm while the right arm folds the
stack. The environment's live stage selects the actor at execution time.

The policy is cloned from `MarkovQuarterExpert`
(`cloth_fold_rl/markov_quarter_expert.py`, collector flag `--expert markov`)
and then improved with DAgger rounds. Cloning `QuarterFoldExpert` directly does
not work. That expert is a phase machine whose actions depend on episode history
the policy cannot see: patience timeouts, IK targets cached when a phase began,
and hidden stage-1 retry offsets. On states the policy reaches, its labels
contradict each other, and each DAgger round made the clone worse. The Markov
expert reads its phase from the state each step and solves IK from the current
joints, so its action is a function of the state. It folds about as often as
the phase machine (81/100 on seeds 60000-60099).

## Results

Success is the full quarter fold settling, on 100 episodes with domain
randomization. The Markov expert scores 81% on seeds 60000-60099.

| checkpoint (`outputs/cloth_angles/`) | data | seeds 60000+ | held-out seeds 200000+ |
|---|---|---|---|
| `quarter_policy_markov` | clean Markov-expert demos | 16% (50 eps) | |
| `quarter_policy_markov_d1_long` | + DAgger r1 (beta 0.3) | 38% (50 eps) | |
| `quarter_policy_markov_d2` | + r2 (beta 0.1) | 37% | |
| `quarter_policy_markov_d3` | + r3 (beta 0) | 62% | |
| `quarter_policy_markov_d4` | + r4 (beta 0) | 69% | 61% |

Checkpoints were chosen on seeds 60000+, so the held-out column is the honest
estimate. Almost all remaining failures are in stage 1: the two-layer stack
springs back after release (`stage1_unsettled_or_timeout`), or it is never
lowered to the goal (`stage1_no_placement`).

## 1. Collect clean expert demonstrations

Use a new output directory for every collection run:

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/collect_fold_state_episodes.py \
  --task quarter --expert markov \
  --output outputs/cloth_angles/quarter_markov_expert \
  --kinds expert --per-kind 300 --test-per-kind 30 \
  --workers 10 --seed-base 110000
```

The Markov expert solves IK every step and takes about 2 minutes per episode.
The collector checkpoints `manifest.json` after every completed episode. If a
long run is interrupted, repeat the same command with `--resume`.

## 2. Audit the recording

```bash
.venv/bin/python scripts/audit_fold_dataset.py \
  outputs/cloth_angles/quarter_markov_expert --task quarter
```

## 3. Train the staged clone

```bash
OMP_NUM_THREADS=8 .venv/bin/python scripts/train_quarter_bc.py \
  --data outputs/cloth_angles/quarter_markov_expert [more datasets ...] \
  --output outputs/cloth_angles/quarter_policy_markov \
  --steps 5000
```

`--data` takes several datasets. The trainer learns from each episode's
`labels` (the expert's action) where they differ from the executed action. It
drops plain expert episodes that failed (`--include-failures` keeps them). It
keeps every DAgger episode, because their labels are the expert's corrections.
Normalization comes from training episodes only. Grasp, gripper-change and
stage-boundary transitions are oversampled. The best checkpoint by held-out
action loss is kept. Once DAgger data is included, use `--steps 25000
--hidden-dim 512` or more.

## 4. Evaluate in the real simulator

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/evaluate_quarter_policy.py \
  --checkpoint outputs/cloth_angles/quarter_policy_markov/bc_staged.pt \
  --episodes 100 --seed-base 60000 --workers 10
```

The evaluator reports grasp, stage-0 completion, stage-1 stack grasp,
placement and final success rates, and a failure category for every failed
episode.

## 5. DAgger rounds

The latest policy drives. The Markov expert runs alongside and records what it
would have done as the episode's `labels`. `--beta` is the fraction of steps
on which the expert's action is executed instead.

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/collect_fold_state_episodes.py \
  --task quarter --expert markov --kinds dagger \
  --actor-checkpoint outputs/cloth_angles/quarter_policy_markov/bc_staged.pt \
  --beta 0.3 \
  --output outputs/cloth_angles/quarter_markov_dagger_r1 \
  --per-kind 150 --test-per-kind 15 --workers 10 --seed-base 120000
```

Retrain on the clean demos plus every DAgger round, then evaluate and collect
the next round from the new checkpoint. The results above used beta 0.3, 0.1,
0, 0 with 150, 200, 250, 250 episodes.

Only mix datasets labelled by the same expert. The `expert_noisy`,
`expert_release` and `dagger` kinds also store expert labels.

## Video

```bash
.venv/bin/python -m cloth_fold_rl.record_video --task quarter --policy bc \
  --checkpoint outputs/cloth_angles/quarter_policy_markov_d4/bc_staged.pt \
  --episodes 3 --out outputs/videos/quarter_fold_bc_d4.mp4
```

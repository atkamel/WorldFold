# Quarter-fold behavior-cloning workflow

The quarter-fold policy uses two actors: stage 0 performs the simultaneous
two-arm half fold and stage 1 parks the left arm while the right arm folds the
stack. The environment's live stage selects the actor at execution time.

## 1. Collect clean expert demonstrations

Use a new output directory for every collection run:

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/collect_fold_state_episodes.py \
  --task quarter \
  --output outputs/cloth_angles/quarter_expert_v1 \
  --kinds expert \
  --per-kind 300 \
  --test-per-kind 30 \
  --workers 10 \
  --seed-base 40000
```

The collector checkpoints `manifest.json` after every completed episode. If a
long run is interrupted, repeat the same command with `--resume`; completed
`(kind, seed)` pairs are retained and only missing episodes are scheduled.

## 2. Audit the recording

```bash
.venv/bin/python scripts/audit_fold_dataset.py \
  outputs/cloth_angles/quarter_expert_v1 --task quarter
```

Do not train until the audit reports zero errors and the summary contains a
useful number of stage-1 episodes, grasp events, and successful episodes.

## 3. Train the staged clone

```bash
OMP_NUM_THREADS=4 .venv/bin/python scripts/train_quarter_bc.py \
  --data outputs/cloth_angles/quarter_expert_v1 \
  --output outputs/cloth_angles/quarter_policy_v1 \
  --steps 5000 \
  --gripper-weight 3
```

The trainer computes normalization from training episodes only, trains each
stage for the full number of updates, oversamples grasp, gripper-change and
stage-boundary transitions, and restores the best checkpoint according to
held-out action loss. The result is `quarter_policy_v1/bc_staged.pt`.

## 4. Evaluate in the real simulator

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/evaluate_quarter_policy.py \
  --checkpoint outputs/cloth_angles/quarter_policy_v1/bc_staged.pt \
  --episodes 50 \
  --seed-base 60000 \
  --workers 10
```

The evaluator reports separate grasp, stage-0 completion, stage-1 stack grasp,
placement and final success rates. It also assigns every failed episode a
failure category so the next data-collection round can target the actual
weakness instead of merely adding more of the same demonstrations.

## 5. Add recovery data only after measuring failures

If the policy fails after small deviations, collect a separate recovery set:

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/collect_fold_state_episodes.py \
  --task quarter \
  --output outputs/cloth_angles/quarter_recovery_v1 \
  --kinds expert_noisy expert_release \
  --per-kind 150 \
  --test-per-kind 15 \
  --workers 10 \
  --seed-base 70000
```

Keep the clean demonstrations as the main dataset. Recovery-data mixing and
expert-labelled policy rollouts are follow-up steps if the clean staged clone
still has distribution-shift failures.

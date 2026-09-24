#!/bin/sh
# Phase 5b driver (runs after dagger_diff = M5b.1). Each step reads the previous one's
# history.json, so the chain needs no hand edits. One heavy job at a time, 14 workers.
set -e
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
pick() { $PY -m imitation.viz.pick "$@"; }

# ---- M5b.1 winner: dagger_diff best vs dagger_v2/round_3 on id_easy + recovery ----
W1=$(pick winner --a $R/dagger_diff/history.json --b $R/dagger_v2/history.json --sets id_easy recovery)
W1_DATA=$(pick winnerdata --a $R/dagger_diff/history.json --b $R/dagger_v2/history.json --sets id_easy recovery)
echo "M5b.1 winner: $W1 (dataset $W1_DATA)"

# ---- M5b.2 shifted-pose coverage ----
$PY -m imitation.dagger --init $W1 --dataset $W1_DATA --out $R/dagger_shift --rounds 3 --episodes 128 \
    --shift-fraction 0.5 --score-sets id_easy id_hard recovery --workers 14 > $R/dagger_shift.out 2>&1
W2=$(pick best --history $R/dagger_shift/history.json)
echo "M5b.2 best: $W2 | $(grep -E 'round [0-9]:' $R/dagger_shift.out | tr '\n' '|')"

# ---- M5b.3 vision on the new path: 128^2, per-camera norm, lazy loader; then distill ----
[ -f outputs/imitation/datasets/v1_img128/manifest.json ] || { echo "v1_img128 missing"; exit 1; }
[ -f outputs/imitation/datasets/v1_failures_img128/manifest.json ] ||     $PY -m imitation.vision.replay --version v1_failures --out v1_failures_img128 --workers 14 > $R/replay_fail128.log 2>&1
$PY -m imitation.train --policy vision --dataset v1_img128 --run $R/vision_bc_128 --batch 256 > $R/vision_bc_128.log 2>&1
$PY -m imitation.evaluate --ckpt $R/vision_bc_128/final.pt --n 200 --workers 14 > $R/vision_bc_128.eval.log 2>&1
echo "M4.1 vision BC (128, new path): $(grep success $R/vision_bc_128.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"
$PY -m imitation.train --policy vision --dataset v1_img128 --run $R/vision_t0_128 --batch 256 --teacher $W2 \
    > $R/vision_t0_128.log 2>&1
$PY -m imitation.dagger --init $R/vision_t0_128/final.pt --dataset v1_img128 --out $R/distill_v2 --teacher $W2 \
    --relabel --rounds 3 --episodes 128 --recovery-fraction 0.6 --shift-fraction 0.3 --dagger-weight 2 \
    --score-sets id_easy id_hard recovery --workers 14 > $R/distill_v2.out 2>&1
V2=$(pick best --history $R/distill_v2/history.json)
echo "M5b.3 best vision: $V2 | $(grep -E 'round [0-9]:' $R/distill_v2.out | tr '\n' '|')"

# ---- M5b.4 offline RL with action diversity ----
$PY -m imitation.rl.harvest --ckpts $R/bc_v1_s0/final.pt $R/diff_v1_s0/final.pt $R/dagger_v2/round_3/final.pt $W2 \
    --noise 0.1 0.3 --episodes 1600 --min-per-code 60 --max-episodes 2400 --version harvest_v2 --workers 14 \
    > $R/harvest_v2.log 2>&1
echo "M5b.4 harvest: $(tail -2 $R/harvest_v2.log | tr '\n' ' ')"
W2_DATA=$(pick lastdata --history $R/dagger_shift/history.json)
$PY -m imitation.rl.iql --init $W2 --versions $W2_DATA v1_failures harvest_v2 --out $R/iql_v3 \
    --adv-norm --beta 1.0 --adv-clip 20 --max-per-episode 12 --strata uniform > $R/iql_v3.log 2>&1
$PY -m imitation.rl.probe --run $R/iql_v3 --versions harvest_v2 > $R/iql_v3.probe.log 2>&1
$PY -m imitation.evaluate --ckpt $R/iql_v3/final.pt --n 200 --workers 14 > $R/iql_v3.eval.log 2>&1
echo "M5b.4 IQL: $(grep success $R/iql_v3.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"

# ---- M5b.5 re-evaluate M2.1-M2.3 checkpoints under deterministic eval (+ repeat check) ----
for run in bc_v1_s0 bc_v1_s1 abl_proprio_s0 abl_proprio_corners_s0; do
  $PY -m imitation.evaluate --ckpt $R/$run/final.pt --n 200 --workers 14 --out $R/$run/eval_det.json > $R/$run.eval_det.log 2>&1
done
$PY -m imitation.evaluate --ckpt $R/bc_v1_s0/final.pt --n 200 --workers 14 --out $R/bc_v1_s0/eval_det_repeat.json > $R/bc_v1_s0.eval_det_repeat.log 2>&1
echo "M5b.5 re-evaluated: $(for r in bc_v1_s0 bc_v1_s1 abl_proprio_s0 abl_proprio_corners_s0; do grep -h success $R/$r.eval_det.log | tr -s ' ' | cut -c1-30 | tr '\n' ' '; echo '|'; done)"
echo "determinism: $(pick same --a $R/bc_v1_s0/eval_det.json --b $R/bc_v1_s0/eval_det_repeat.json)"

# ---- detector on the 128 frames, agreement on held-out distill_v2 rollouts; figures; demos ----
$PY -m imitation.vision.success train --versions v1_img128 v1_failures_img128 --out $R/success_v2 > $R/success_v2.log 2>&1 || true
$PY -m imitation.vision.success agree --ckpt $R/success_v2/detector.pt \
    --versions $(for d in outputs/imitation/datasets/v1_img128_distill_v2_r*; do [ -f $d/manifest.json ] && basename $d; done) \
    > $R/success_v2.agree.log 2>&1 || true
$PY -m imitation.viz.report_figures --images v1_img128 --student harvest_v2 --out docs/reports/media > $R/figures.log 2>&1
$PY -m imitation.demo --ckpt $W2 --seeds 100000 100001 100002 100003 --out docs/reports/media/half_fold_privileged.mp4 > $R/demo_priv.log 2>&1
$PY -m imitation.demo --ckpt $V2 --seeds 100000 100001 100002 100003 --out docs/reports/media/half_fold_sensor.mp4 > $R/demo_sensor.log 2>&1
echo "demos: $(tail -1 $R/demo_priv.log) | $(tail -1 $R/demo_sensor.log)"
echo PHASE5B_DONE

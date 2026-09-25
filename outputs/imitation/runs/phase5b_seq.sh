#!/bin/sh
# Phase 5b, sequential and CPU-capped (<= 10 sim workers at a time; teacher labels and all
# learning on the GPU). Resumes the distillation and the harvest where they were stopped.
set -e
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
W2=outputs/imitation/runs/dagger_diff/round_1/final.pt
N=10

# M5b.3 -- distillation (resumes: round 0 evaluated, round 1 data frozen)
$PY -m imitation.dagger --init $R/vision_t0_128/final.pt --dataset v1_img128 --out $R/distill_v2 --teacher $W2 \
    --relabel --rounds 3 --episodes 128 --recovery-fraction 0.6 --shift-fraction 0.3 --dagger-weight 2 \
    --score-sets id_easy id_hard recovery --workers $N --resume >> $R/distill_v2.out 2>&1
V2=$($PY -m imitation.viz.pick best --history $R/distill_v2/history.json)
echo "M5b.3 best vision: $V2 | $(grep -E 'round [0-9]:' $R/distill_v2.out | tr '\n' '|')"

# M5b.4 -- harvest (resumes from its journal), IQL on GPU, probe, eval
$PY -m imitation.rl.harvest --ckpts $R/bc_v1_s0/final.pt $R/diff_v1_s0/final.pt $R/dagger_v2/round_3/final.pt $W2 \
    --noise 0.1 0.3 --episodes 1600 --min-per-code 60 --max-episodes 2400 --version harvest_v2 --workers $N --resume \
    >> $R/harvest_v2.log 2>&1
echo "M5b.4 harvest: $(tail -2 $R/harvest_v2.log | tr '\n' ' ')"
$PY -m imitation.rl.iql --init $W2 --versions v1_dagger_diff_r2 v1_failures harvest_v2 --out $R/iql_v3 \
    --adv-norm --beta 1.0 --adv-clip 20 --max-per-episode 12 --strata uniform > $R/iql_v3.log 2>&1
$PY -m imitation.rl.probe --run $R/iql_v3 --versions harvest_v2 > $R/iql_v3.probe.log 2>&1
$PY -m imitation.evaluate --ckpt $R/iql_v3/final.pt --n 200 --workers $N > $R/iql_v3.eval.log 2>&1
echo "M5b.4 IQL: $(grep success $R/iql_v3.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"

# M5b.5 -- deterministic re-evaluation + repeat check
for run in bc_v1_s0 bc_v1_s1 abl_proprio_s0 abl_proprio_corners_s0; do
  $PY -m imitation.evaluate --ckpt $R/$run/final.pt --n 200 --workers $N --out $R/$run/eval_det.json > $R/$run.eval_det.log 2>&1
done
$PY -m imitation.evaluate --ckpt $R/bc_v1_s0/final.pt --n 200 --workers $N --out $R/bc_v1_s0/eval_det_repeat.json > $R/bc_v1_s0.eval_det_repeat.log 2>&1
echo "M5b.5 determinism: $($PY -m imitation.viz.pick same --a $R/bc_v1_s0/eval_det.json --b $R/bc_v1_s0/eval_det_repeat.json)"

# detector, figures, demos
[ -f outputs/imitation/datasets/v1_failures_img128/manifest.json ] || \
    $PY -m imitation.vision.replay --version v1_failures --out v1_failures_img128 --workers $N > $R/replay_fail128.log 2>&1
$PY -m imitation.vision.success train --versions v1_img128 v1_failures_img128 --out $R/success_v2 > $R/success_v2.log 2>&1
$PY -m imitation.vision.success agree --ckpt $R/success_v2/detector.pt \
    --versions $(for d in outputs/imitation/datasets/v1_img128_distill_v2_r*; do [ -f $d/manifest.json ] && basename $d; done) \
    > $R/success_v2.agree.log 2>&1
echo "detector: $(tail -1 $R/success_v2.log) $(grep '\"rate\"' $R/success_v2.agree.log)"
$PY -m imitation.viz.report_figures --images v1_img128 --student harvest_v2 --out docs/reports/media > $R/figures.log 2>&1
$PY -m imitation.demo --ckpt $W2 --seeds 100000 100001 100002 100003 --out docs/reports/media/half_fold_privileged.mp4 > $R/demo_priv.log 2>&1
$PY -m imitation.demo --ckpt $V2 --seeds 100000 100001 100002 100003 --out docs/reports/media/half_fold_sensor.mp4 > $R/demo_sensor.log 2>&1
echo "demos: $(tail -1 $R/demo_priv.log) | $(tail -1 $R/demo_sensor.log)"
echo PHASE5B_DONE

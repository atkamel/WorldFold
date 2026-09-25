#!/bin/sh
# Track C: after A and B -- detector on 128 frames, figures, demos.
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
until grep -q TRACK_A_DONE $R/phase5b_a.out 2>/dev/null && grep -q TRACK_B_DONE $R/phase5b_b.out 2>/dev/null; do sleep 60; done
W2=outputs/imitation/runs/dagger_diff/round_1/final.pt
V2=$($PY -m imitation.viz.pick best --history $R/distill_v2/history.json)
[ -f outputs/imitation/datasets/v1_failures_img128/manifest.json ] || \
    $PY -m imitation.vision.replay --version v1_failures --out v1_failures_img128 --workers 14 > $R/replay_fail128.log 2>&1
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

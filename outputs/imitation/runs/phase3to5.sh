#!/bin/sh
# After dagger_v2: M2.4 eval, M5.1 harvest, M5.3 IQL + eval, M4.2 distillation, M4.3 detector, demo.
set -e
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
BEST=$($PY -c "import json;h=json.load(open('$R/dagger_v2/history.json'));print([x for x in h if x['kept']][-1]['checkpoint'])")
LAST=$($PY -c "import json;h=json.load(open('$R/dagger_v2/history.json'));print(h[-1]['dataset'])")
echo "best dagger: $BEST  last version: $LAST"

$PY -m imitation.evaluate --ckpt $R/diff_v1_s0/final.pt --n 200 --workers 14 > $R/diff_v1_s0.eval.log 2>&1
echo "M2.4 diffusion evaluated: $(grep success $R/diff_v1_s0.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"

$PY -m imitation.rl.harvest --ckpt $BEST --episodes 1000 --version harvest_v1 --workers 14 > $R/harvest_v1.log 2>&1
echo "M5.1 harvest: $(tail -1 $R/harvest_v1.log)"

$PY -m imitation.rl.iql --init $BEST --versions $LAST v1_failures harvest_v1 --out $R/iql_v1 > $R/iql_v1.log 2>&1
$PY -m imitation.evaluate --ckpt $R/iql_v1/final.pt --n 200 --workers 14 > $R/iql_v1.eval.log 2>&1
echo "M5.3 IQL evaluated: $(grep success $R/iql_v1.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"

$PY -m imitation.evaluate --ckpt $R/vision_bc_v1/final.pt --n 200 --workers 14 > $R/vision_bc_v1.eval.log 2>&1
echo "vision BC evaluated: $(grep success $R/vision_bc_v1.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"

$PY -m imitation.vision.distill --teacher $BEST --dataset v1_img --out $R/distill_v1 --rounds 3 --workers 14 > $R/distill_v1.out 2>&1
echo "M4.2 distill: $(grep -E 'round [0-9]:|best student' $R/distill_v1.out | tr '\n' '|')"

[ -f $R/success_v1/detector.pt ] || $PY -m imitation.vision.success train --versions v1_img v1_failures_img --out $R/success_v1 > $R/success_v1.log 2>&1
$PY -m imitation.vision.success agree --ckpt $R/success_v1/detector.pt --versions $(for d in outputs/imitation/datasets/v1_img_distill_v1_r*; do [ -f $d/manifest.json ] && basename $d; done) > $R/success_v1.agree.log 2>&1 || true
echo "M4.3 detector: $(tail -1 $R/success_v1.log) $(grep -E '\"rate\"' $R/success_v1.agree.log)"
echo PHASE3TO5_DONE

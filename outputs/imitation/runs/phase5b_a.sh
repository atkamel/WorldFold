#!/bin/sh
# Track A (vision): GPU training, then distillation rollouts on 10 CPU workers.
set -e
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
W2=outputs/imitation/runs/dagger_diff/round_1/final.pt
$PY -m imitation.train --policy vision --dataset v1_img128 --run $R/vision_t0_128 --batch 256 --teacher $W2 > $R/vision_t0_128.log 2>&1
$PY -m imitation.dagger --init $R/vision_t0_128/final.pt --dataset v1_img128 --out $R/distill_v2 --teacher $W2 \
    --relabel --rounds 3 --episodes 128 --recovery-fraction 0.6 --shift-fraction 0.3 --dagger-weight 2 \
    --score-sets id_easy id_hard recovery --workers 10 > $R/distill_v2.out 2>&1
echo "M5b.3 best vision: $($PY -m imitation.viz.pick best --history $R/distill_v2/history.json) | $(grep -E 'round [0-9]:' $R/distill_v2.out | tr '\n' '|')"
echo TRACK_A_DONE

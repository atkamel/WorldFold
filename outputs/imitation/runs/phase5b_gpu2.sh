#!/bin/sh
# Final pre-VLA queue, lane C (GPU), started inside lane A's M5c.3 DAgger window so the GPU
# has independent work while lane A simulates. Its evaluation waits for the shared CPU slot.
#   vision student seed 1 (same recipe as vision_t0_128: 128², teacher dagger_diff/round_1)
set -e
export PYTHONUNBUFFERED=1
export IMITATION_CPU_SLOT=outputs/imitation/scratch/cpu.slot
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
W2=$R/dagger_diff/round_1/final.pt
N=10
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "train vision_t0_128_s1 (GPU)"
$PY -m imitation.train --policy vision --dataset v1_img128 --run $R/vision_t0_128_s1 --batch 256 --teacher $W2 \
    --seed 1 > $R/vision_t0_128_s1.log 2>&1
step "trained vision_t0_128_s1: $(tail -n 1 $R/vision_t0_128_s1.log | cut -c1-80)"
$PY -m imitation.evaluate --ckpt $R/vision_t0_128_s1/final.pt --n 200 --workers $N --replan-every 2 \
    > $R/vision_t0_128_s1.eval_r2.log 2>&1
step "vision_t0_128_s1 replan 2: $(grep success $R/vision_t0_128_s1.eval_r2.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"
step PHASE5B_GPU2_DONE

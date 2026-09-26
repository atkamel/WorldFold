#!/bin/sh
# Final pre-VLA queue, lane B (GPU). Trains while lane A (phase5b_final.sh) simulates;
# its evaluations wait for the shared CPU slot, so the CPU never runs two sim pools.
#   diffusion BC seed variance on v1 (seeds 1, 2; seed 0 = diff_v1_s0), evaluated at n=200
set -e
export PYTHONUNBUFFERED=1
export IMITATION_CPU_SLOT=outputs/imitation/scratch/cpu.slot
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
N=10
REST_OUT=${REST_OUT:?set to phase5b_rest.sh output file}
step() { echo "[$(date +%H:%M:%S)] $*"; }

for s in 1 2; do
  step "train diff_v1_s$s (GPU)"
  $PY -m imitation.train --policy diffusion --dataset v1 --run $R/diff_v1_s$s --seed $s > $R/diff_v1_s$s.log 2>&1
  step "trained diff_v1_s$s: $(tail -1 $R/diff_v1_s$s.log | cut -c1-80)"
done
# phase5b_rest.sh predates the CPU slot: don't simulate until it has finished
until grep -q PHASE5B_REST_DONE "$REST_OUT" 2>/dev/null; do sleep 60; done
for s in 1 2; do
  $PY -m imitation.evaluate --ckpt $R/diff_v1_s$s/final.pt --n 200 --workers $N > $R/diff_v1_s$s.eval.log 2>&1
  step "diff_v1_s$s eval: $(grep success $R/diff_v1_s$s.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"
done
step PHASE5B_GPU_DONE

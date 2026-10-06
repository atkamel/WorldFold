#!/bin/sh
# Phase 2 runs: BC baseline (2 seeds) + obs-subset ablation, each evaluated at n=200 per set.
set -e
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
for run in "bc_v1_s0 --seed 0" "bc_v1_s1 --seed 1" "abl_proprio_s0 --seed 0 --obs-subset proprio" "abl_proprio_corners_s0 --seed 0 --obs-subset proprio_corners"; do
  set -- $run; name=$1; shift
  $PY -m imitation.train --dataset v1 --run $R/$name "$@" > $R/$name.log 2>&1
  echo "trained $name"
done
for name in bc_v1_s0 bc_v1_s1 abl_proprio_s0 abl_proprio_corners_s0; do
  $PY -m imitation.evaluate --ckpt $R/$name/final.pt --n 200 --workers 14 > $R/$name.eval.log 2>&1
  echo "evaluated $name: $(grep success $R/$name.eval.log | tr -s ' ' | cut -c1-60 | tr '\n' '|')"
done
echo PHASE2_DONE

#!/bin/sh
# Track B (offline RL + hygiene): CPU harvest -> GPU IQL -> CPU eval -> CPU re-evals, 6 workers.
set -e
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
W2=outputs/imitation/runs/dagger_diff/round_1/final.pt
$PY -m imitation.rl.harvest --ckpts $R/bc_v1_s0/final.pt $R/diff_v1_s0/final.pt $R/dagger_v2/round_3/final.pt $W2 \
    --noise 0.1 0.3 --episodes 1600 --min-per-code 60 --max-episodes 2400 --version harvest_v2 --workers 6 \
    > $R/harvest_v2.log 2>&1
echo "M5b.4 harvest: $(tail -2 $R/harvest_v2.log | tr '\n' ' ')"
$PY -m imitation.rl.iql --init $W2 --versions v1_dagger_diff_r2 v1_failures harvest_v2 --out $R/iql_v3 \
    --adv-norm --beta 1.0 --adv-clip 20 --max-per-episode 12 --strata uniform > $R/iql_v3.log 2>&1
$PY -m imitation.rl.probe --run $R/iql_v3 --versions harvest_v2 > $R/iql_v3.probe.log 2>&1
$PY -m imitation.evaluate --ckpt $R/iql_v3/final.pt --n 200 --workers 6 > $R/iql_v3.eval.log 2>&1
echo "M5b.4 IQL: $(grep success $R/iql_v3.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"
for run in bc_v1_s0 bc_v1_s1 abl_proprio_s0 abl_proprio_corners_s0; do
  $PY -m imitation.evaluate --ckpt $R/$run/final.pt --n 200 --workers 6 --out $R/$run/eval_det.json > $R/$run.eval_det.log 2>&1
done
$PY -m imitation.evaluate --ckpt $R/bc_v1_s0/final.pt --n 200 --workers 6 --out $R/bc_v1_s0/eval_det_repeat.json > $R/bc_v1_s0.eval_det_repeat.log 2>&1
echo "M5b.5 determinism: $($PY -m imitation.viz.pick same --a $R/bc_v1_s0/eval_det.json --b $R/bc_v1_s0/eval_det_repeat.json)"
echo TRACK_B_DONE

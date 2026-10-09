#!/bin/sh
# Final pre-VLA queue, lane A (CPU). Runs next to phase5b_gpu.sh (lane B, GPU training);
# both set IMITATION_CPU_SLOT, so their rollouts never overlap (<= 10 sim workers total).
#   M5c.3  one DAgger round from the privileged best at the adopted replan 4, with a GPU
#          utilisation sampler (every 5 s) running for the whole queue
#   M5b.4  iql_v4 at n=200, replan 8 and 4 (fair against the privileged best at each)
#   demos  re-rendered at the adopted operating points (privileged replan 4, vision replan 2)
set -e
export PYTHONUNBUFFERED=1
export IMITATION_CPU_SLOT=outputs/imitation/scratch/cpu.slot
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
W2=$R/dagger_diff/round_1/final.pt
V2=$R/vision_t0_128/final.pt
N=10
step() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p outputs/imitation/scratch
# phase5b_rest.sh predates the CPU slot: don't simulate until it has finished
REST_OUT=${REST_OUT:?set to phase5b_rest.sh output file}
until grep -q PHASE5B_REST_DONE "$REST_OUT" 2>/dev/null; do sleep 60; done
GPU=$R/m5c3_gpu.csv
echo "epoch,util_pct,mem_mib" > $GPU
( while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')" >> $GPU
    sleep 5
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

step "M5c.3 window start: DAgger r4 (1 round, 64 episodes, replan 4) from $W2"
echo "dagger_start,$(date +%s)" > $R/m5c3_window.csv
# round 0 = the replan-4 evaluation M5b.6 already ran on the same seeds (deterministic eval),
# so the run resumes at round 1 instead of spending ~45 CPU-minutes re-measuring it
mkdir -p $R/dagger_r4
[ -f $R/dagger_r4/history.json ] || $PY -c "
import json; e = json.load(open('$W2'.replace('final.pt', 'eval_r4.json')))['results']
json.dump([{'round': 0, 'checkpoint': '$W2', 'dataset': 'v1_dagger_diff_r1', 'eval': e, 'kept': True}],
          open('$R/dagger_r4/history.json', 'w'), indent=1)"
$PY -m imitation.dagger --init $W2 --dataset v1_dagger_diff_r1 --out $R/dagger_r4 --rounds 1 --episodes 64 \
    --replan-every 4 --score-sets id_easy id_hard recovery --workers $N --resume > $R/dagger_r4.out 2>&1
echo "dagger_end,$(date +%s)" >> $R/m5c3_window.csv
step "M5c.3 window end: $(grep -E 'round [0-9]:|id_easy|id_hard|recovery' $R/dagger_r4.out | tail -4 | tr -s ' ' | cut -c1-60 | tr '\n' '|')"

for rp in 8 4; do
  $PY -m imitation.evaluate --ckpt $R/iql_v4/final.pt --n 200 --workers $N --replan-every $rp > $R/iql_v4.eval_r$rp.log 2>&1
  step "M5b.4 iql_v4 replan $rp: $(grep success $R/iql_v4.eval_r$rp.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"
done

$PY -m imitation.demo --ckpt $W2 --replan-every 4 --seeds 100000 100001 100002 100003 \
    --out docs/reports/media/half_fold_privileged.mp4 > $R/demo_priv.log 2>&1
$PY -m imitation.demo --ckpt $V2 --replan-every 2 --seeds 100000 100001 100002 100003 \
    --out docs/reports/media/half_fold_sensor.mp4 > $R/demo_sensor.log 2>&1
step "demos: $(tail -1 $R/demo_priv.log) | $(tail -1 $R/demo_sensor.log)"
step PHASE5B_FINAL_DONE

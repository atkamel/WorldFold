#!/bin/sh
# Resume the final pre-VLA queue after the 2026-09-25 15:16 pause (lanes A/B/C stopped).
# One sequential lane, <= 10 sim workers. Run it as a visible task:
#     sh outputs/imitation/runs/phase5b_resume.sh
# Already done before the pause: diff_v1_s1 eval, diff_v1_s2 + vision_t0_128_s1 trained,
# DAgger r4 round-1 rollouts frozen (v1_dagger_diff_r1_dagger_r4_r1, reused by --resume).
set -e
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
W2=$R/dagger_diff/round_1/final.pt
V2=$R/vision_t0_128/final.pt
N=10
step() { echo "[$(date +%H:%M:%S)] $*"; }

# M5c.3 part 2: part 1 (m5c3_gpu.csv, window start in m5c3_window.csv) covered the rollouts
GPU=$R/m5c3_gpu_part2.csv
echo "epoch,util_pct,mem_mib" > $GPU
( while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')" >> $GPU
    sleep 5
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

step "M5c.3 part 2 start: DAgger r4 round 1 (reuses frozen rollouts): train + eval at replan 4"
echo "dagger_start,$(date +%s)" > $R/m5c3_window_part2.csv
$PY -m imitation.dagger --init $W2 --dataset v1_dagger_diff_r1 --out $R/dagger_r4 --rounds 1 --episodes 64 \
    --replan-every 4 --score-sets id_easy id_hard recovery --workers $N --resume >> $R/dagger_r4.out 2>&1
echo "dagger_end,$(date +%s)" >> $R/m5c3_window_part2.csv
step "DAgger r4: $(grep -E 'round 1:|success' $R/dagger_r4.out | tail -n 4 | tr -s ' ' | cut -c1-60 | tr '\n' '|')"
kill $SAMPLER 2>/dev/null || true
step "M5c.3 part 2: $($PY -m imitation.viz.gpu_busy --samples $GPU --window $R/m5c3_window_part2.csv | tr -d '\n ')"

$PY -m imitation.evaluate --ckpt $R/diff_v1_s2/final.pt --n 200 --workers $N > $R/diff_v1_s2.eval.log 2>&1
step "diff_v1_s2: $(grep success $R/diff_v1_s2.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"
$PY -m imitation.evaluate --ckpt $R/vision_t0_128_s1/final.pt --n 200 --workers $N --replan-every 2 \
    > $R/vision_t0_128_s1.eval_r2.log 2>&1
step "vision_t0_128_s1 replan 2: $(grep success $R/vision_t0_128_s1.eval_r2.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"

for rp in 8 4; do
  $PY -m imitation.evaluate --ckpt $R/iql_v4/final.pt --n 200 --workers $N --replan-every $rp > $R/iql_v4.eval_r$rp.log 2>&1
  step "iql_v4 replan $rp: $(grep success $R/iql_v4.eval_r$rp.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"
done

$PY -m imitation.demo --ckpt $W2 --replan-every 4 --seeds 100000 100001 100002 100003 \
    --out docs/reports/media/half_fold_privileged.mp4 > $R/demo_priv.log 2>&1
$PY -m imitation.demo --ckpt $V2 --replan-every 2 --seeds 100000 100001 100002 100003 \
    --out docs/reports/media/half_fold_sensor.mp4 > $R/demo_sensor.log 2>&1
step "demos: $(tail -n 1 $R/demo_priv.log) | $(tail -n 1 $R/demo_sensor.log)"

$PY -m imitation.viz.profile --episodes 20 --workers $N --state $W2 --vision $V2 --out outputs/imitation/profile.json \
    > $R/profile.log 2>&1
step "M5c.1 profile (label look-ahead timed): $(head -n 4 $R/profile.log | cut -c1-200 | tr '\n' '|')"
step PHASE5B_RESUME_DONE

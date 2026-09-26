#!/bin/sh
# Rest of Phase 5b + queue-2 items, sequential, <= 10 sim workers.
set -e
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=outputs/imitation/runs
W2=$R/dagger_diff/round_1/final.pt
V2=$R/vision_t0_128/final.pt
N=10
step() { echo "[$(date +%H:%M)] $*"; }

step "M5b.4 IQL (diffusion AWR) on harvest_v2"
$PY -m imitation.rl.iql --init $W2 --versions v1_dagger_diff_r2 v1_failures harvest_v2 --out $R/iql_v3 \
    --adv-norm --beta 1.0 --adv-clip 20 --max-per-episode 12 --strata uniform > $R/iql_v3.log 2>&1
$PY -m imitation.rl.probe --run $R/iql_v3 --versions harvest_v2 > $R/iql_v3.probe.log 2>&1
step "M5b.4 probe: $(grep -A3 summary $R/iql_v3.probe.log | tr -s ' \n' ' ')"
$PY -m imitation.evaluate --ckpt $R/iql_v3/final.pt --n 200 --workers $N > $R/iql_v3.eval.log 2>&1
step "M5b.4 IQL eval: $(grep success $R/iql_v3.eval.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"

step "M5b.6 replan sweep (privileged + vision, replan 4 and 2)"
for ck in $W2 $V2; do for rp in 4 2; do
  $PY -m imitation.evaluate --ckpt $ck --n 200 --workers $N --replan-every $rp > $ck.replan$rp.log 2>&1
  step "  $ck replan $rp: $(grep success $ck.replan$rp.log | tr -s ' ' | cut -c1-40 | tr '\n' '|')"
done; done

step "M5b.5 deterministic re-evaluation + repeat check"
for run in bc_v1_s0 bc_v1_s1 abl_proprio_s0 abl_proprio_corners_s0; do
  $PY -m imitation.evaluate --ckpt $R/$run/final.pt --n 200 --workers $N --out $R/$run/eval_det.json > $R/$run.eval_det.log 2>&1
  step "  $run: $(grep success $R/$run.eval_det.log | tr -s ' ' | cut -c1-30 | tr '\n' '|')"
done
$PY -m imitation.evaluate --ckpt $R/bc_v1_s0/final.pt --n 200 --workers $N --out $R/bc_v1_s0/eval_det_repeat.json > $R/bc_v1_s0.eval_det_repeat.log 2>&1
step "M5b.5 determinism: $($PY -m imitation.viz.pick same --a $R/bc_v1_s0/eval_det.json --b $R/bc_v1_s0/eval_det_repeat.json)"

step "M5c.1 profile"
$PY -m imitation.viz.profile --episodes 20 --workers $N --state $W2 --vision $V2 --out outputs/imitation/profile.json > $R/profile.log 2>&1
step "M5c.1 done: $(head -4 $R/profile.log | cut -c1-160 | tr '\n' '|')"

step "detector (128), figures, demos"
[ -f outputs/imitation/datasets/v1_failures_img128/manifest.json ] || \
    $PY -m imitation.vision.replay --version v1_failures --out v1_failures_img128 --workers $N > $R/replay_fail128.log 2>&1
$PY -m imitation.vision.success train --versions v1_img128 v1_failures_img128 --out $R/success_v2 > $R/success_v2.log 2>&1
$PY -m imitation.vision.success agree --ckpt $R/success_v2/detector.pt --versions v1_img128_distill_v2_r1 v1_img128_distill_v2_r2 \
    > $R/success_v2.agree.log 2>&1
step "detector: $(tail -1 $R/success_v2.log) $(grep '\"rate\"' $R/success_v2.agree.log)"
$PY -m imitation.viz.report_figures --images v1_img128 --student harvest_v2 --out docs/reports/media > $R/figures.log 2>&1
$PY -m imitation.demo --ckpt $W2 --seeds 100000 100001 100002 100003 --out docs/reports/media/half_fold_privileged.mp4 > $R/demo_priv.log 2>&1
$PY -m imitation.demo --ckpt $V2 --seeds 100000 100001 100002 100003 --out docs/reports/media/half_fold_sensor.mp4 > $R/demo_sensor.log 2>&1
step "demos: $(tail -1 $R/demo_priv.log) | $(tail -1 $R/demo_sensor.log)"
step PHASE5B_REST_DONE

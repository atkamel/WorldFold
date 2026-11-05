#!/usr/bin/env bash
# One teacher run on the GCP L4 VM (as root, after `WITH_TEACHER=1 bash teacher/runpod/setup.sh`): the VLA folds
# N garments while every grasp is logged (oracle/grasp_log.py), under one cloth setting (oracle/isaac_physics.py).
# Speed fixes we already measured, none changing physics or model input: his fast-server patches (ea97031: one
# output transfer, pre-flattened state, padded batch sizes), client-side 224 resize (795a21a), buffer-protocol cloth
# read (84c71f3), persistent JAX compile cache. Retry stays ON (part of his policy).
#   PHYSICS=v2|v2_lowmu|fixed|orig  WORKERS=2  PER_TYPE=5  TYPES="top_long top_short pant_long pant_short"
#   sudo -E bash /workspace/WorldFold/teacher/gcp/run_teacher_grasp.sh
set -uo pipefail
WS=${WS:-/workspace}
WF=$WS/WorldFold/teacher
[ -d /opt/lehome_solution/lehome-challenge/.venv ] && SRC=/opt/lehome_solution || SRC=$WS/lehome_solution
CH=$SRC/lehome-challenge
PHYSICS=${PHYSICS:-v2}
WORKERS=${WORKERS:-2}
PER_TYPE=${PER_TYPE:-5}
TYPES=${TYPES:-"top_long top_short pant_long pant_short"}
MAX_STEPS=${MAX_STEPS:-600}
CKPT=${CKPT:-$WS/checkpoints/lehome_sim}
OUT=$WS/results/teachergrasp/$(date +%Y%m%d-%H%M%S)-$PHYSICS
mkdir -p "$OUT"
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y XDG_RUNTIME_DIR=/tmp
export __GLX_VENDOR_LIBRARY_NAME=nvidia VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
T0=$(date +%s)

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
case "$DRV" in 580.*) ;; *) echo "[tg] FATAL driver $DRV is not 580.x (Isaac 5.1)"; exit 3 ;; esac
test -x "$SRC/.venv/bin/python" || { echo "[tg] FATAL teacher venv missing: WITH_TEACHER=1 bash teacher/runpod/setup.sh"; exit 3; }
bash "$WF/runpod/link_code.sh"

# ---- source patches (idempotent; the image resets them every container) ----
for p in fast_server_patch.py eval_speed_patch.py grasp_log_patch.py grasp_test_patch.py grasp_test_patch2.py arm_port_patch.py attach_patch.py; do
  "$SRC/.venv/bin/python" "$WF/$p" "$SRC" || { echo "[tg] FATAL patch $p"; exit 3; }
done
"$CH/.venv/bin/python" "$WF/oracle/isaac_physics.py" apply "$PHYSICS" "$CH" | tee "$OUT/physics.txt"
# the arm follows the physics choice: fitted arm (arm_port_patch.py) with v2/v2_lowmu/fixed, LeHome's arm with orig
[ "$PHYSICS" = orig ] && export TEACHER_ARM=stock || export TEACHER_ARM=fitted
# jaw-cloth contact: Isaac's own with orig; CONTACT=attach (default) | native with the corrected settings. Isaac's own
# contact at a 50 g shirt closes the jaw fully through the cloth and lets go at 0.4-0.8 N (real 1.3-4.5 N; 2026-10-19)
CONTACT=${CONTACT:-attach}
if [ "$PHYSICS" != orig ] && [ "$CONTACT" = attach ]; then export TEACHER_ATTACH=1; else export TEACHER_ATTACH=0; fi
echo "[tg] arm: $TEACHER_ARM, contact: $([ "$TEACHER_ATTACH" = 1 ] && echo "attach model (ATTACH_MU=${ATTACH_MU:-0.15} ATTACH_N=${ATTACH_N:-9})" || echo "Isaac's own")" | tee -a "$OUT/physics.txt"
ROBOT="$CH/Assets/robots/lerobot/so101_follower_good.usd"
[ -f "$ROBOT.orig" ] || cp "$ROBOT" "$ROBOT.orig"                     # LeHome's robot (round capsule jaws), kept once
if [ "$PHYSICS" != orig ]; then
  cp "$WF/oracle/assets/so101_flatpads.usd" "$ROBOT" && echo "[tg] flat pads"
else
  cp "$ROBOT.orig" "$ROBOT" && echo "[tg] LeHome's capsule jaws"
fi

# ---- garments: the first PER_TYPE seen garments of each type (same list for every physics setting) ----
GL=""
for t in $TYPES; do
  d=$(ls -d "$CH"/Assets/objects/Challenge_Garment/Release/*/ | while read x; do b=$(basename "$x"); \
      [ "$(echo "$b" | tr 'A-Z' 'a-z')" = "$t" ] && echo "$x"; done | head -1)
  [ -n "$d" ] || { echo "[tg] no garment folder for $t"; continue; }
  GL="$GL $(ls "$d" | grep -i _seen_ | sort -t_ -k4 -n | head -"$PER_TYPE" | tr '\n' ' ')"
done
echo "[tg] physics $PHYSICS, $WORKERS workers, garments:$GL" | tee "$OUT/garments.txt"
# garment models: LeHome's worn-shape models with orig; the flattened copies (run_isaac.sh flatten, 2026-10-19:
# the measured-stiff cloth tumbles on the worn shape) with every corrected setting, flattening any garment not done yet
for g in $GL; do
  if [ "$PHYSICS" = orig ]; then
    "$CH/.venv/bin/python" "$WF/oracle/flatten_garment.py" restore "$CH" "$g"
  else
    "$CH/.venv/bin/python" "$WF/oracle/flatten_garment.py" use "$CH" "$g" | grep -q "^FLAT_USE $g" || {
      echo "[tg] flattening $g"; GARMENT=$g KEEP_POD=1 MAX_S=600 bash "$WF/runpod/run_isaac.sh" flatten >"$OUT/flatten_$g.log" 2>&1
      "$CH/.venv/bin/python" "$WF/oracle/flatten_garment.py" use "$CH" "$g" || { echo "[tg] FATAL no flat model for $g"; exit 3; }; }
  fi
done

export TEACHER_FAST_OUT=1 TEACHER_FLAT_STATE=1 TEACHER_BUCKETS=1 TEACHER_FAST_CLOTH_READ=1 TEACHER_RESIZE224=1
export OPENPI_CLIENT_SRC=$SRC/openpi/packages/openpi-client/src
export TEACHER_GRASP_LOG=$OUT/grasps ORACLE_DIR=$WF/oracle
# grasp-on-demand test (grasp_test_patch.py): harvest snapshots into $OUT/snapshots, or restore from TEACHER_RESTORE_DIR
[ -n "${TEACHER_SNAPSHOT_EVERY:-}" ] && export TEACHER_SNAPSHOT_DIR=${TEACHER_SNAPSHOT_DIR:-$OUT/snapshots} && echo "[tg] snapshots every $TEACHER_SNAPSHOT_EVERY steps -> $TEACHER_SNAPSHOT_DIR"
[ -n "${TEACHER_MOVE_AT_STEP:-}" ] && echo "[tg] garment moved at step $TEACHER_MOVE_AT_STEP (shift ${TEACHER_MOVE_SHIFT_CM:-0} cm, rot ${TEACHER_MOVE_ROT_DEG:-0} deg)"
[ -n "${TEACHER_RESTORE_DIR:-}" ] && echo "[tg] restore-driven episodes from $TEACHER_RESTORE_DIR (shift ${TEACHER_RESTORE_SHIFT_CM:-0} cm, rot ${TEACHER_RESTORE_ROT_DEG:-0} deg, max_steps $MAX_STEPS, ${NUM_EPISODES:-1}/garment)"
export JAX_PLATFORMS=cuda JAX_COMPILATION_CACHE_DIR=$WS/cache/jax JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
cd "$SRC"
timeout --kill-after=60 "${MAX_S:-14400}" .venv/bin/python -u scripts/run_eval.py \
  --checkpoint_dir "$CKPT" --config_name pi_modified_bc_rl --num_workers "$WORKERS" --num_episodes "${NUM_EPISODES:-1}" \
  --garment_types $TYPES --garment_list $GL --max_steps "$MAX_STEPS" \
  --no_wandb --no_save_dataset --video_dir "$OUT" 2>&1 | tee "$OUT/run_eval.log" \
  | grep --line-buffered -E "GRASP|ATTACH|GRASPTEST|EVAL SUMMARY|OVERALL|SR=|Traceback|Error|hung|dead|Starting server|ready" \
  | grep --line-buffered -v carb.launcher
echo "[tg] eval exit ${PIPESTATUS[0]} after $(( $(date +%s) - T0 )) s"

"$CH/.venv/bin/python" "$WF/summarize_grasps.py" "$OUT" | tee "$OUT/grasp_summary.txt"
echo "[tg] DONE $OUT"

#!/usr/bin/env bash
# On the GCP L4 VM: run one job in our public Isaac image with the teacher mounts (same docker recipe as WATcloud's
# isaac_teleop.sbatch; the teacher pieces as the Modal image had them).
#   bash docker_run.sh prep                  # docker + NVIDIA runtime present? disk? pull the image (public, ~10 GB)
#   bash docker_run.sh setup                 # one-time: teacher venv + checkpoint (gcp/teacher_setup.sh)
#   bash docker_run.sh grip                  # scripted-grasp calibration (runpod/run_isaac.sh grip)
#   bash docker_run.sh contact               # jaw-cloth contact check, 5 checks vs real targets
#   bash docker_run.sh flatten               # save the shirt's flat shape as its built-in shape (FLAT=1 runs use it)
#   PHYSICS=v2 bash docker_run.sh teacher    # teacher run with the grasp logger (gcp/run_teacher_grasp.sh)
# Code: ~/wf/WorldFold (copied from the laptop). Everything persistent lives under ~/wf on the VM disk.
set -uo pipefail
IMAGE=${IMAGE:-ghcr.io/cimurghe/worldfold-isaac:5.1}
W=${W:-$HOME/wf}
CMD=${1:?usage: docker_run.sh prep|setup|grip|contact|flatten|teacher}

if [ "$CMD" = prep ]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  docker --version || { echo "FATAL: no docker on this VM"; exit 3; }
  docker info 2>/dev/null | grep -qi nvidia || echo "WARNING: NVIDIA container runtime not listed in docker info"
  df -h "$HOME" | tail -1
  t0=$(date +%s); docker pull -q "$IMAGE" && echo "[prep] image pulled in $(( $(date +%s) - t0 )) s"
  exit $?
fi

# no Isaac cache mounts: empty mounts hid files the image ships and Isaac crashed at sim start (2026-10-18)
mkdir -p "$W"/teacher_env/venv "$W"/teacher_env/openpi "$W"/checkpoints "$W"/results "$W"/cache/jax
case "$CMD" in
  setup)   INNER="bash /workspace/WorldFold/teacher/gcp/teacher_setup.sh" ;;
  grip)    INNER="KEEP_POD=1 MAX_S=${MAX_S:-1500} bash /workspace/WorldFold/teacher/runpod/run_isaac.sh grip" ;;
  contact) INNER="KEEP_POD=1 MAX_S=${MAX_S:-2400} bash /workspace/WorldFold/teacher/runpod/run_isaac.sh contact" ;;
  flatten) INNER="KEEP_POD=1 MAX_S=${MAX_S:-900} bash /workspace/WorldFold/teacher/runpod/run_isaac.sh flatten" ;;
  teacher) INNER="bash /workspace/WorldFold/teacher/gcp/run_teacher_grasp.sh" ;;
  dream)   INNER="bash /workspace/WorldFold/teacher/gcp/run_dream.sh" ;;     # DREAM_API_KEY from the host env
  *) echo "unknown $CMD"; exit 2 ;;
esac
# one long-lived container (name wfisaac): whatever Isaac builds on first start (shader / extension caches) stays in its
# writable layer for the next job, instead of being thrown away with --rm
if ! docker ps --format '{{.Names}}' | grep -qx wfisaac; then
  docker ps -a --format '{{.Names}}' | grep -qx wfisaac && docker start wfisaac >/dev/null || \
  docker image inspect wfisaac:patched >/dev/null 2>&1 && IMAGE=wfisaac:patched   # patches + caches committed
  docker run -d --name wfisaac --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e TEACHER_FAST_OUT=1 -e TEACHER_FLAT_STATE=1 -e TEACHER_BUCKETS=1 -e TEACHER_FAST_CLOTH_READ=1 -e TEACHER_RESIZE224=1 \
    -e OPENPI_CLIENT_SRC=/opt/lehome_solution/openpi/packages/openpi-client/src \
    -e ORACLE_DIR=/workspace/WorldFold/teacher/oracle -e TEACHER_GRASP_LOG=/workspace/results/grasps -e TEACHER_ARM=fitted \
    -v "$W/WorldFold":/workspace/WorldFold -v "$W/results":/workspace/results -v "$W/checkpoints":/workspace/checkpoints \
    -v "$W/teacher_env/venv":/opt/lehome_solution/.venv -v "$W/teacher_env/openpi":/opt/lehome_solution/openpi \
    -v "$W/cache/jax":/workspace/cache/jax \
    --entrypoint sleep "$IMAGE" infinity >/dev/null
  docker exec wfisaac bash -c '[ -f /etc/vulkan/icd.d/nvidia_icd.json ] || { mkdir -p /etc/vulkan/icd.d; echo "{\"file_format_version\": \"1.0.0\", \"ICD\": {\"library_path\": \"libGLX_nvidia.so.0\", \"api_version\": \"1.3\"}}" > /etc/vulkan/icd.d/nvidia_icd.json; }'
fi
# Kit's hang detector (isaaclab.python.kit: enabled, 120 s) killed the first headless RTX start on the L4 (2026-10-18).
# His launcher overwrites --kit_args, so switch it off in the experience file (inside this container only)
docker exec wfisaac sed -i 's/^hangDetector.enabled = true/hangDetector.enabled = false/' \
  /opt/lehome_solution/lehome-challenge/third_party/IsaacLab/apps/isaaclab.python.kit
docker exec -e PHYSICS -e WORKERS -e PER_TYPE -e TYPES -e MAX_STEPS -e MAX_S -e GRIP_VARIANTS -e GARMENT -e NOCAM -e CT_REPS -e CT_VARIANTS -e CT_JOBS -e CT_PHYSICS -e FLAT -e FLAT_SETTLE_S -e ATTACH -e ATTACH_MU -e ATTACH_N -e CONTACT -e GARMENTS -e TEACHER_SNAPSHOT_EVERY -e TEACHER_SNAPSHOT_DIR -e TEACHER_RESTORE_DIR -e TEACHER_RESTORE_SHIFT_CM -e TEACHER_RESTORE_ROT_DEG -e TEACHER_RESTORE_SETTLE -e TEACHER_RESTORE_SEED -e NUM_EPISODES -e TEACHER_MOVE_AT_STEP -e TEACHER_MOVE_SHIFT_CM -e TEACHER_MOVE_ROT_DEG \
  -e DREAM_API_KEY -e BATTERY -e FAKE_DREAM -e DREAM_START_POSE \
  wfisaac bash -c "$INNER"

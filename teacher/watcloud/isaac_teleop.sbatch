#!/bin/bash
# One Isaac teleop session on a WATcloud RTX GPU (Slurm job). From the login node:
#   sbatch ~/wf/WorldFold/teacher/watcloud/isaac_teleop.sbatch
# Needs: our code in ~/wf/WorldFold. If the image is private: a GitHub token with read:packages in ~/wf/.ghcr_token.
# Isaac listens on port 7777 of the GPU node (host network). From the laptop (UW VPN on), with the node's IP
# (squeue shows the node; `getent hosts <node>` on the login node gives the IP, thor-slurm1 = 10.0.50.184):
#   ssh -N -L 17777:<node ip>:7777 <you>@wato-login1.ext.watonomous.ca     (the .cluster names are not routed by the VPN)
#   python teacher/mujoco_live/teleop_client.py 127.0.0.1 17777
# Isaac takes ~3-5 min to start: wait for "TELEOP listening" in the job log before connecting (it accepts ONE client).
# The session ends when the client quits, after IDLE_S without input, or at MAX_S; the job then ends.
#SBATCH --job-name=isaac-teleop
#SBATCH --gres=gpu:rtx_3090:1,tmpdisk:81920
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.log
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/cimurghe/worldfold-isaac:5.1}
echo "[wc] node $(hostname) job $SLURM_JOB_ID: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
slurm-start-dockerd.sh > /dev/null
export DOCKER_HOST=unix:///tmp/run/docker.sock
[ -f ~/wf/.ghcr_token ] && docker login ghcr.io -u cimurghe --password-stdin < ~/wf/.ghcr_token > /dev/null
t0=$(date +%s)
docker pull -q "$IMAGE"
echo "[wc] image pulled in $(( $(date +%s) - t0 )) s"
mkdir -p ~/wf/results
# run_isaac.sh as on RunPod; the image's own entrypoint (sshd for RunPod) is skipped, its Vulkan fallback kept
docker run --rm --gpus all --network host -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e KEEP_POD=1 -e MAX_S="${MAX_S:-1500}" -e IDLE_S="${IDLE_S:-600}" -e N_GRABS="${N_GRABS:-20}" -e PRESETS="${PRESETS:-}" -e ORACLE_GRAB="${ORACLE_GRAB:-assist}" -e ORACLE_APP_PUMP_TICKS="${ORACLE_APP_PUMP_TICKS:-1500}" \
  -v ~/wf/WorldFold:/workspace/WorldFold -v ~/wf/results:/workspace/results \
  --entrypoint bash "$IMAGE" -c '
    [ -f /etc/vulkan/icd.d/nvidia_icd.json ] || { mkdir -p /etc/vulkan/icd.d; echo "{\"file_format_version\": \"1.0.0\", \"ICD\": {\"library_path\": \"libGLX_nvidia.so.0\", \"api_version\": \"1.3\"}}" > /etc/vulkan/icd.d/nvidia_icd.json; }
    bash /workspace/WorldFold/teacher/runpod/run_isaac.sh '"${MODE:-teleop}"
echo "[wc] done"

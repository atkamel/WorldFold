#!/bin/bash
# One WorldFold Isaac command on a WATcloud GPU node, in the image isaac/Dockerfile builds (the same Isaac stack as
# the Modal image). The repo is mounted, so outputs land in your checkout and code edits need no rebuild.
#
#   sbatch --gres=shard:<gpu>:<MiB>,tmpdisk:61440 isaac/watcloud.sbatch scripts/f4_retrain.py --no-vision --episodes 200
#   sbatch --gres=shard:<gpu>:<MiB>,tmpdisk:61440 isaac/watcloud.sbatch -m imitation.evaluate --backend isaac_friction ...
#   sbatch --gres=shard:<gpu>:<MiB>,tmpdisk:61440 isaac/watcloud.sbatch isaac/smoke_test.py --mode state
#
# GPU: Isaac Sim 5.1 needs RT cores and lists an RTX 4080 with 16 GB as its minimum; the 2080 Ti nodes (11 GB) used
# for Isaac Sim 4.5 are below it. Pick an RTX node with >= 16 GB in --gres. Driver: on Modal, Isaac Sim 5.1 ran on
# the 580 branch and segfaulted at start-up on 610 (isaac/modal_isaac.py, SUPPORTED_DRIVER), so this job stops on
# any other branch unless ISAAC_ANY_DRIVER=1. --gres replaces the whole request, so it carries tmpdisk too (Docker's
# image and layers live there: ~60 GB).
# Each job builds the image (~10 min, ~20 GB on tmpdisk), or loads $IMAGE_TAR if one was saved by hand.
#SBATCH --job-name=worldfold-isaac
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=watcloud-%j.log
set -euo pipefail

REPO=${REPO:-$HOME/WorldFold}
IMAGE=worldfold-isaac
IMAGE_TAR=${IMAGE_TAR:-$HOME/docker/worldfold-isaac.tar}
CACHE=${CACHE:-$HOME/docker/worldfold-isaac-cache}     # Kit's shader and compute caches, kept across jobs

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), driver $driver"
if [[ $driver != 580.* && ${ISAAC_ANY_DRIVER:-0} != 1 ]]; then
    echo "driver $driver: Isaac Sim 5.1 was only seen working on 580.x; set ISAAC_ANY_DRIVER=1 to try anyway" >&2
    exit 1
fi

slurm-start-dockerd.sh
export DOCKER_HOST=unix:///tmp/run/docker.sock     # the rootless daemon it starts (not /var/run)
if ! docker image inspect $IMAGE > /dev/null 2>&1; then
    if [[ -f $IMAGE_TAR ]]; then
        docker load -i "$IMAGE_TAR"
    else
        docker build -t $IMAGE -f "$REPO/isaac/Dockerfile" "$REPO/isaac"
        # saving streams the ~20 GB image through the job's memory: with --mem=24G it was killed (job 706908)
        if [[ ${SAVE_IMAGE:-0} == 1 ]]; then
            mkdir -p "$(dirname "$IMAGE_TAR")"
            docker save -o "$IMAGE_TAR" $IMAGE || { rm -f "$IMAGE_TAR"; echo "docker save failed"; }
        fi
    fi
fi

mkdir -p "$CACHE"
# LeHome looks for Assets/ at the git root of the working directory, which here is the mounted repo: link it to
# the image's copy (/Assets is gitignored)
# only the GPU SLURM gave this job: with --gpus all, PhysX found "no suitable CUDA GPU" (2026-10-15, jobs 706910 /
# 706912: the job's cgroup only opens its own device)
GPUS=all
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] && GPUS="\"device=$CUDA_VISIBLE_DEVICES\""    # the job's own numbering (device=1 was unknown)
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} -> --gpus $GPUS; host: $(ls /dev/nvidia* | tr '\n' ' ')"
docker run --rm --gpus "$GPUS" $IMAGE bash -c 'ls /dev/nvidia* | tr "\n" " "; echo; "$ISAAC_PY" -c "import torch; print(\"torch cuda\", torch.cuda.is_available(), torch.cuda.device_count())"' || true
# graphics too: with docker's default (compute, utility) Kit has no Vulkan driver, and PhysX then finds "no suitable
# CUDA GPU" even though torch sees it (job 707319)
docker run --rm --gpus "$GPUS" -e NVIDIA_DRIVER_CAPABILITIES=all --network=host \
    -v "$REPO":/workspace/WorldFold -v "$CACHE":/root/.cache -w /workspace/WorldFold $IMAGE \
    bash -c 'ln -sfn /opt/lehome-challenge/Assets Assets && exec "$ISAAC_PY" -u "$@"' -- "$@"

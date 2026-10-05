#!/usr/bin/env bash
# One-time install of the Isaac (LeHome) stack on a RunPod pod - the same recipe as isaac_image in
# teacher/modal_teacher.py (pinned commits, Isaac Sim 5.1, LeHome's IsaacLab fork, assets, our oracle/teleop code).
# Installs under /workspace (a RunPod network volume survives pod restarts; without one, rerun each session).
#
#   bash teacher/runpod/setup.sh            # Isaac stack only (grip test, oracle, teleop)
#   WITH_TEACHER=1 bash teacher/runpod/setup.sh   # + the pi0.5 teacher (openpi) venv
#
# Safe to rerun: every step is skipped if already done.
set -euo pipefail

WS=${WS:-/workspace}
WF_DIR=${WF_DIR:-$WS/WorldFold}                 # this repo, uploaded or cloned (see README.md)
REPO=https://github.com/IliaLarchenko/lehome_solution
OPENPI_SHA=c23745b5ad24e98f66967ea795a07b2588ed6c79
LEHOME_CHALLENGE_SHA=5ea947ed83abf414180f4c503dbb31b9d6aa39f8
SRC=$WS/lehome_solution
CH=$SRC/lehome-challenge
log() { echo "[setup $(date +%H:%M:%S)] $*"; }

# ---- 0. the GPU must be new enough for Isaac Sim 5.1 (driver 580+, RT cores) ----
drv=$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1)
log "GPU: $drv"
major=${drv%%.*}
if [ "${major:-0}" -lt 580 ]; then
  log "WARNING: driver $major < 580 - Isaac Sim 5.1 may not start. Pick a pod with CUDA 13 in RunPod's filter."
fi
[ -d "$WF_DIR/teacher/oracle" ] || { log "missing $WF_DIR/teacher/oracle - upload or clone WorldFold first (README.md)"; exit 1; }

# ---- 1. system packages (same list as the Modal image) ----
if ! dpkg -s libvulkan1 >/dev/null 2>&1; then
  log "apt packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq git git-lfs curl build-essential ffmpeg cmake ninja-build pkg-config python3-dev \
    libgl1-mesa-dev libglfw3 libglfw3-dev libglew-dev xorg-dev libxi-dev libxinerama-dev libxcursor1 libxrandr2 \
    libglu1-mesa libvulkan1 vulkan-tools libxt6 libsm6 libice6 libegl1 libgl1 libglvnd0 >/dev/null
fi
# Isaac calls zenity for GUI dialogs, which blocks headless runs
printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/zenity && chmod +x /usr/local/bin/zenity
# Vulkan needs the NVIDIA ICD file (Modal injects it; plain containers may not have it)
if [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  mkdir -p /etc/vulkan/icd.d
  cat > /etc/vulkan/icd.d/nvidia_icd.json <<'J'
{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3"}}
J
fi

# ---- 2. uv ----
export PATH=$HOME/.local/bin:$PATH UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 PYTHONUNBUFFERED=1
command -v uv >/dev/null || { log "uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
git lfs install --skip-smudge >/dev/null

# ---- 3. lehome_solution + the pinned lehome-challenge submodule ----
[ -d "$SRC/.git" ] || { log "clone lehome_solution"; git clone -q "$REPO" "$SRC"; }
cd "$SRC"
git submodule update --init lehome-challenge
[ "$(git -C "$CH" rev-parse HEAD)" = "$LEHOME_CHALLENGE_SHA" ] && log "CHALLENGE_SHA_OK" || { log "lehome-challenge commit mismatch"; exit 1; }

# ---- 4. Isaac Sim 5.1 + LeHome (the big layer) ----
if [ ! -f "$CH/.isaac_ok" ]; then
  log "uv sync (Isaac Sim 5.1, torch 2.7, lerobot 0.4.3) - the long step"
  cd "$CH" && uv sync
  echo yes > "$CH/.venv/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED"
  [ -d "$CH/third_party/IsaacLab" ] || git clone -q --depth 1 https://github.com/lehome-official/IsaacLab.git "$CH/third_party/IsaacLab"
  # TERM: isaaclab.sh calls tput, which aborts over a non-interactive ssh -> its deps (omegaconf, h5py, ...) never install
  TERM=xterm bash -c "source .venv/bin/activate && ./third_party/IsaacLab/isaaclab.sh -i none" || log "isaaclab.sh failed, continuing (as on Modal)"
  uv pip install 'setuptools<70' wheel --python .venv/bin/python
  uv pip install flatdict==4.0.1 --no-build-isolation --python .venv/bin/python
  uv pip install -e third_party/IsaacLab/source/isaaclab --python .venv/bin/python
  uv pip install -e source/lehome --python .venv/bin/python
  uv pip install warp-lang==1.12.1 --python .venv/bin/python   # last warp that Isaac Sim 5.1 works with
  uv pip install h5py --python .venv/bin/python   # isaaclab_tasks needs it; isaaclab.sh (which fails) would have installed it
  .venv/bin/python -c "import isaacsim; print('ISAACSIM_IMPORT_OK')"
  touch "$CH/.isaac_ok"
fi

# ---- 5. LeHome assets (garments) ----
if [ ! -d "$CH/Assets/objects/Challenge_Garment/Release" ]; then
  log "assets"
  cd "$CH" && .venv/bin/python -m pip --version >/dev/null 2>&1 || true
  uvx --from 'huggingface_hub[cli]' hf download lehome/asset_challenge --repo-type dataset --local-dir "$CH/Assets"
fi
ls "$CH/Assets/objects/Challenge_Garment/Release" | head -3

# ---- 6. optional: the pi0.5 teacher venv (only needed to serve the VLA) ----
if [ "${WITH_TEACHER:-0}" = 1 ] && [ ! -f "$SRC/.teacher_ok" ]; then
  log "teacher venv (openpi)"
  cd "$SRC"
  git submodule update --init openpi
  [ "$(git -C openpi rev-parse HEAD)" = "$OPENPI_SHA" ] || { log "openpi commit mismatch"; exit 1; }
  python3 - <<'PY'
import re, pathlib
p = pathlib.Path("pyproject.toml"); s = p.read_text()
old = 'override-dependencies = ["lerobot>=0.4.3", "rerun-sdk>=0.22,<0.24"]'
new = 'override-dependencies = ["lerobot>=0.4.3", "rerun-sdk>=0.22,<0.24", "ml-dtypes==0.4.1", "tensorstore==0.1.74"]'
if old in s:
    s = s.replace(old, new)
s = re.sub(r'^\s*"(pyrealsense2|torchcodec)[^"]*",?\s*\n', "", s, flags=re.M)
s = re.sub(r'^\s*(pyrealsense2|torchcodec)\s*=\s*\{[^\n]*\}\s*\n', "", s, flags=re.M)
p.write_text(s)
PY
  uv venv --python 3.11 .venv && uv lock && uv sync
  uv pip install nvidia-cudnn-cu12==9.5.1.17 nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-runtime-cu12==12.6.77 nvidia-cuda-nvcc-cu12==12.9.41
  python3 "$WF_DIR/teacher/fast_server_patch.py" "$SRC"
  touch "$SRC/.teacher_ok"
fi

# ---- 7. our code where the scripts expect it (same paths as the Modal image) ----
bash "$WF_DIR/teacher/runpod/link_code.sh"
log "SETUP DONE"

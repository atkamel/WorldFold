#!/usr/bin/env bash
# Builds the Linux Isaac Sim stack WorldFold runs on: LeHome's locked lehome-challenge env (Isaac Sim 5.1.0,
# Python 3.11, torch 2.7.0), LeHome's IsaacLab fork, the garment assets and the imitation pipeline's extra packages.
# One script for both images, so they cannot drift: the Modal image (isaac/modal_isaac.py) and the WATcloud Docker
# image (isaac/Dockerfile). Runs as root on Debian bookworm with Python 3.11; installs into $CH.
set -euo pipefail

CH=${CH:-/opt/lehome-challenge}
PY=$CH/.venv/bin/python
# Upstream lehome-challenge (Roy's teacher uses Ilia's fork, which adds only his eval loop: same pyproject and
# uv.lock). LeHome's IsaacLab fork forces PhysX GPU dynamics in DirectRLEnv, which the particle cloth needs on
# the CPU device; Roy cloned it unpinned, its HEAD is pinned here.
LEHOME_CHALLENGE=https://github.com/lehome-official/lehome-challenge.git
LEHOME_CHALLENGE_SHA=a805ad2f7ab52a4583066fc4ee5180459a7f9d15
ISAACLAB=https://github.com/lehome-official/IsaacLab.git
ISAACLAB_SHA=69f6fa548c3a3520e3cb26ed24bb8abe60baeef3

checkout() { git init -q "$3" && git -C "$3" fetch -q --depth 1 "$1" "$2" && git -C "$3" checkout -q FETCH_HEAD; }

apt-get update
apt-get install -y --no-install-recommends git curl ca-certificates build-essential cmake ninja-build pkg-config \
    python3-dev libgl1-mesa-dev libglfw3 libglfw3-dev libglew-dev xorg-dev libxi-dev libxinerama-dev libxcursor1 \
    libxrandr2 libglu1-mesa libvulkan1 vulkan-tools libxt6 libsm6 libice6 libegl1 libgl1 libglvnd0
rm -rf /var/lib/apt/lists/*
pip install --no-cache-dir "huggingface_hub[cli]>=0.30"
# Isaac calls zenity for GUI dialogs, which blocks headless runs
printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/zenity && chmod +x /usr/local/bin/zenity
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH UV_LINK_MODE=copy

checkout $LEHOME_CHALLENGE $LEHOME_CHALLENGE_SHA "$CH"
cd "$CH"
uv sync     # locked: isaacsim[all,extscache]==5.1.0, torch 2.7.0, lerobot 0.4.3
echo yes > "$CH/.venv/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED"
checkout $ISAACLAB $ISAACLAB_SHA "$CH/third_party/IsaacLab"
bash -c 'source .venv/bin/activate && ./third_party/IsaacLab/isaaclab.sh -i none' || echo ISAACLAB_SH_FAILED_CONTINUING
uv pip install 'setuptools<70' wheel --python "$PY"
uv pip install flatdict==4.0.1 --no-build-isolation --python "$PY"
uv pip install -e third_party/IsaacLab/source/isaaclab --python "$PY"
uv pip install -e source/lehome --python "$PY"
# IsaacLab leaves warp-lang unpinned; 1.12.1 is the last release Isaac Sim 5.1 works with
uv pip install warp-lang==1.12.1 --python "$PY"
# the imitation pipeline and lab_scene's config loading, at the versions of isaac/requirements-isaaclab-windows.lock
uv pip install omegaconf==2.3.1 tensorboard==2.21.0 h5py==3.14.0 --python "$PY"
"$PY" -c "import isaacsim, gymnasium, omegaconf; print('ISAACSIM_IMPORT_OK')"
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets

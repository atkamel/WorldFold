# The Isaac (LeHome) "engine" image for RunPod: the isaac_image recipe in teacher/modal_teacher.py, line for line and
# in the same order (pinned commits, Isaac Sim 5.1, LeHome's IsaacLab fork, assets). Our code (teacher/) is NOT in
# here: it changes often and is copied onto the pod at start (deploy.md), so editing code never means rebuilding.
#   docker build -t worldfold-isaac:5.1 teacher/runpod/image
# Not included: the pi0.5 teacher venv (openpi) - only needed to serve the VLA, not for grip tests / teleop.

# Three stages: build (the Modal recipe) -> pack (split into <=1 GB chunks) -> final (base + the chunks as layers).
# Why: GitHub's registry rejects layers over 10 GB and uploads over 10 minutes; the install is one 19 GB folder.
# The final image also leaves out uv's 19 GB download cache.

# BUILD_IMAGE: pack from an already-built image instead of the build stage (skips the 16 min Isaac download):
#   docker build --build-arg BUILD_IMAGE=worldfold-isaac:5.1 -t ghcr.io/cimurghe/worldfold-isaac:5.1 .
ARG BUILD_IMAGE=build

# Modal's base: debian_slim(python_version="3.11")
FROM python:3.11-slim-bookworm AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
    GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    TERM=xterm
# TERM: isaaclab.sh calls tput, which aborts without one (RunPod session 1: omegaconf/h5py never installed)

# ---- Modal `image` layers that the Isaac stack uses ----
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs curl build-essential ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy<2" "websockets>=14.1" psutil "huggingface_hub[cli]>=0.30" nvidia-ml-py
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && git lfs install --skip-smudge
ARG REPO=https://github.com/IliaLarchenko/lehome_solution
ARG SRC=/opt/lehome_solution
RUN git clone ${REPO} ${SRC}

# ---- isaac_image ----
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build pkg-config python3-dev libgl1-mesa-dev \
      libglfw3 libglfw3-dev libglew-dev xorg-dev libxi-dev libxinerama-dev libxcursor1 libxrandr2 libglu1-mesa \
      libvulkan1 vulkan-tools libxt6 libsm6 libice6 \
      libegl1 libgl1 libglvnd0 \
    && rm -rf /var/lib/apt/lists/*
# Isaac calls zenity for GUI dialogs, which blocks headless runs
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/zenity && chmod +x /usr/local/bin/zenity

ARG LEHOME_CHALLENGE_SHA=5ea947ed83abf414180f4c503dbb31b9d6aa39f8
ENV CH=/opt/lehome_solution/lehome-challenge
RUN cd ${SRC} && git submodule update --init lehome-challenge \
    && test "$(git -C ${CH} rev-parse HEAD)" = ${LEHOME_CHALLENGE_SHA} && echo CHALLENGE_SHA_OK

# locked: isaacsim[all,extscache]==5.1.0, torch 2.7.0, lerobot 0.4.3 (the big layer)
RUN cd ${CH} && uv sync

# same order as Modal: isaaclab.sh first, then the pins on top of whatever it installed
RUN echo yes > ${CH}/.venv/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED \
    && git clone --depth 1 https://github.com/lehome-official/IsaacLab.git ${CH}/third_party/IsaacLab \
    && (cd ${CH} && bash -c 'source .venv/bin/activate && ./third_party/IsaacLab/isaaclab.sh -i none' \
        || echo ISAACLAB_SH_FAILED_CONTINUING)
RUN cd ${CH} && uv pip install 'setuptools<70' wheel --python .venv/bin/python \
    && uv pip install flatdict==4.0.1 --no-build-isolation --python .venv/bin/python \
    && uv pip install -e third_party/IsaacLab/source/isaaclab --python .venv/bin/python \
    && uv pip install -e source/lehome --python .venv/bin/python \
    && uv pip install warp-lang==1.12.1 --python .venv/bin/python
# (warp 1.12.1: IsaacLab leaves it unpinned; the last release Isaac Sim 5.1 works with)

# build-time checks (no GPU needed): everything the oracle/teleop imports at the top level
RUN cd ${CH} && .venv/bin/python -c "import isaacsim; print('ISAACSIM_IMPORT_OK')" \
    && .venv/bin/python -c "import omegaconf, h5py, gymnasium, toml, warp, torch, numpy; \
print('DEPS_OK torch', torch.__version__, 'warp', warp.__version__, 'numpy', numpy.__version__)"

# LeHome assets (garments, scenes, robots)
RUN cd ${CH} && hf download lehome/asset_challenge --repo-type dataset --local-dir Assets \
    && ls Assets/objects/Challenge_Garment/Release | head -5

# ================= pack: move the install into chunk dirs (this stage is never uploaded) =================
FROM ${BUILD_IMAGE} AS pack
COPY pack.py /pack.py
RUN python3 /pack.py /opt/lehome_solution /chunks 1000 40

# ================= final: what RunPod pulls =================
FROM python:3.11-slim-bookworm AS final
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
    GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    TERM=xterm
# same first layers as the build stage (cache hits)
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs curl build-essential ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy<2" "websockets>=14.1" psutil "huggingface_hub[cli]>=0.30" nvidia-ml-py
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && git lfs install --skip-smudge
# the build stage's system packages + openssh for RunPod logins
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build pkg-config python3-dev libgl1-mesa-dev \
      libglfw3 libglfw3-dev libglew-dev xorg-dev libxi-dev libxinerama-dev libxcursor1 libxrandr2 libglu1-mesa \
      libvulkan1 vulkan-tools libxt6 libsm6 libice6 \
      libegl1 libgl1 libglvnd0 openssh-server \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd \
    && printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/zenity && chmod +x /usr/local/bin/zenity

# /opt/lehome_solution, one layer per <=1 GB chunk (empty chunks make empty layers)
COPY --from=pack /chunks/skel/ /
COPY --from=pack /chunks/00/ /
COPY --from=pack /chunks/01/ /
COPY --from=pack /chunks/02/ /
COPY --from=pack /chunks/03/ /
COPY --from=pack /chunks/04/ /
COPY --from=pack /chunks/05/ /
COPY --from=pack /chunks/06/ /
COPY --from=pack /chunks/07/ /
COPY --from=pack /chunks/08/ /
COPY --from=pack /chunks/09/ /
COPY --from=pack /chunks/10/ /
COPY --from=pack /chunks/11/ /
COPY --from=pack /chunks/12/ /
COPY --from=pack /chunks/13/ /
COPY --from=pack /chunks/14/ /
COPY --from=pack /chunks/15/ /
COPY --from=pack /chunks/16/ /
COPY --from=pack /chunks/17/ /
COPY --from=pack /chunks/18/ /
COPY --from=pack /chunks/19/ /
COPY --from=pack /chunks/20/ /
COPY --from=pack /chunks/21/ /
COPY --from=pack /chunks/22/ /
COPY --from=pack /chunks/23/ /
COPY --from=pack /chunks/24/ /
COPY --from=pack /chunks/25/ /
COPY --from=pack /chunks/26/ /
COPY --from=pack /chunks/27/ /
COPY --from=pack /chunks/28/ /
COPY --from=pack /chunks/29/ /
COPY --from=pack /chunks/30/ /
COPY --from=pack /chunks/31/ /
COPY --from=pack /chunks/32/ /
COPY --from=pack /chunks/33/ /
COPY --from=pack /chunks/34/ /
COPY --from=pack /chunks/35/ /
COPY --from=pack /chunks/36/ /
COPY --from=pack /chunks/37/ /
COPY --from=pack /chunks/38/ /
COPY --from=pack /chunks/39/ /

ENV CH=/opt/lehome_solution/lehome-challenge \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
    __GLX_VENDOR_LIBRARY_NAME=nvidia VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json XDG_RUNTIME_DIR=/tmp \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
# NVIDIA_DRIVER_CAPABILITIES=all: without "graphics" the container gets no Vulkan driver (Modal sets this for us)

# ---- RunPod plumbing (Modal does these for us): SSH login, Vulkan ICD fallback, stay alive ----
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]

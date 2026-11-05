#!/usr/bin/env bash
# One-time, inside our Isaac image (ghcr.io/cimurghe/worldfold-isaac:5.1 = Isaac + LeHome + assets, as on WATcloud):
# add the pi0.5 teacher exactly as the Modal image did (teacher/modal_teacher.py `image`: openpi pinned, his uv
# overrides, exact nvidia wheels, his eval importable) and stage the checkpoint as Modal's stage_checkpoint did.
# The venv, the openpi checkout and the checkpoint sit on host mounts (gcp/docker_run.sh), so later runs skip this.
set -euo pipefail
SRC=/opt/lehome_solution
OPENPI_SHA=c23745b5ad24e98f66967ea795a07b2588ed6c79
CKPT=${CKPT:-/workspace/checkpoints/lehome_sim}
export PATH=$HOME/.local/bin:$PATH GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
cd "$SRC"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

if [ ! -f "$SRC/openpi/pyproject.toml" ]; then
  echo "[setup] openpi @ $OPENPI_SHA"
  git clone -q https://github.com/Physical-Intelligence/openpi.git /tmp/openpi
  git -C /tmp/openpi checkout -q "$OPENPI_SHA"
  cp -a /tmp/openpi/. "$SRC/openpi/"
fi
test "$(git -C "$SRC/openpi" rev-parse HEAD)" = "$OPENPI_SHA" && echo "[setup] OPENPI_SHA_OK"

python3 - <<'PY'
import re, pathlib
p = pathlib.Path("pyproject.toml"); s = p.read_text()
old = 'override-dependencies = ["lerobot>=0.4.3", "rerun-sdk>=0.22,<0.24"]'
new = 'override-dependencies = ["lerobot>=0.4.3", "rerun-sdk>=0.22,<0.24", "ml-dtypes==0.4.1", "tensorstore==0.1.74"]'
if new not in s:
    assert old in s, "override line not found -- pyproject changed upstream, stop and read"
    s = s.replace(old, new)
s = re.sub(r'^\s*"(pyrealsense2|torchcodec)[^"]*",?\s*\n', "", s, flags=re.M)
s = re.sub(r'^\s*(pyrealsense2|torchcodec)\s*=\s*\{[^\n]*\}\s*\n', "", s, flags=re.M)
p.write_text(s)
print("[setup] pyproject patched")
PY

[ -x "$SRC/.venv/bin/python" ] || uv venv --python 3.11 .venv
# his repo has no lock file: resolve as of the Modal build (2026-09-24), which worked; pyarrow 26 (Oct 9) needs
# numpy 2 while openpi pins 1.26
export UV_EXCLUDE_NEWER=2026-09-24T00:00:00Z
uv lock && uv sync
# jax 0.5.3 fresh-install cuDNN init failure (jax#27874): pin exactly openpi's lock
uv pip install nvidia-cudnn-cu12==9.5.1.17 nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-runtime-cu12==12.6.77 \
  nvidia-cuda-nvcc-cu12==12.9.41
uv pip install toml && uv pip install --no-deps -e lehome-challenge/source/lehome/
.venv/bin/python -c "import jax, torch, openpi, lehome_solution.eval; \
from lehome_solution.training.config import get_config; \
print('[setup] IMPORT_OK jax', jax.__version__, 'torch', torch.__version__, get_config('pi_modified_bc_rl').name)"

if [ ! -f "$CKPT/params/manifest.ocdbt" ]; then
  mkdir -p "$CKPT"
  hf download IliaLarchenko/lehome_sim --local-dir "$CKPT"
fi
for f in params/_METADATA params/manifest.ocdbt assets/inference_config.json assets/sim_rl/norm_stats.json \
         assets/sim_rl/fast_tokenizer/tokenizer.json; do
  test -f "$CKPT/$f" || { echo "[setup] FATAL checkpoint incomplete: $f"; exit 4; }
done
echo "[setup] CHECKPOINT_OK $(du -sh "$CKPT" | cut -f1)"
echo "[setup] TEACHER_SETUP_DONE"

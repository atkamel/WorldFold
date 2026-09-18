#!/bin/bash
# WATcloud job for cloud/watcloud/watcloud.sh: unpack the repo on tmpdisk, build a
# venv with uv (everything under /tmp so nothing persists on the cluster), run the
# staged command until 5 minutes before the wall clock, tar the --out paths back.
# Resources (gres, cpus, mem, time, partition) come from the driver's sbatch flags.
set -euo pipefail
: "${STAGE:?}"
. "$STAGE/job.env"     # CMD, REQ, OUTS, GPU

echo "== node =="; hostname; date
[ "$GPU" = 1 ] && nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv || true
lscpu | grep -E 'Model name|^CPU\(s\)' || true; echo "cpuset: $(taskset -cp $$ 2>/dev/null | cut -d: -f2)"

W=/tmp/worldfold
mkdir -p "$W"
tar xf "$STAGE/repo.tar" -C "$W"                                    # -> $W/repo
[ -f "$STAGE/inputs.tar" ] && tar xf "$STAGE/inputs.tar" -C "$W/repo"
find "$W/repo" -name '._*' -delete
R="$W/repo"

echo "== venv (uv, on tmpdisk) =="
export UV_INSTALL_DIR="$W/uv" UV_CACHE_DIR="$W/uv-cache" UV_PYTHON_INSTALL_DIR="$W/py" XDG_DATA_HOME="$W/xdg"
curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null   # never edits ~/.profile
UV="$W/uv/uv"
"$UV" venv --python 3.12 "$W/venv" >/dev/null
"$UV" pip install --python "$W/venv/bin/python" -r "$R/$REQ" 2>&1 | tail -n 3
export PATH="$W/venv/bin:$PATH" MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 WARP_CACHE_PATH="$W/warp-cache"
python -c "import torch, mujoco, warp; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mujoco', mujoco.__version__, '| warp', warp.__version__)"

end=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
budget=$(( end - $(date +%s) - 300 ))
echo "== run ($budget s budget): $CMD =="
cd "$R"
bash -c "timeout -s INT -k 120 $budget $CMD" || echo "command exited rc=$? (124 = deadline)"

if [ -n "$OUTS" ]; then
    echo "== pack: $OUTS =="
    for p in $OUTS; do mkdir -p "$p"; done
    tar cf "$STAGE/out.tar" $OUTS
fi
touch "$STAGE/DONE"
echo "== done =="; date

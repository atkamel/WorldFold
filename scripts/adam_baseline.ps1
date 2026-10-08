# Phase F comparison row: Adam's Isaac friction grasp as he built it (origin/main PR #15, isaac/half_fold_demo.py's
# pinch, ported unchanged as IsaacArmExpert in I2.1), on the same LeHome task sets as our friction expert (F3).
#   lehome profile: CPU device, LeHome's arm + gripper drives, closed jaw -0.1, adhesion 0, no dynamics DR
#   expert: pinch 1.0 cm above the table, no placement overshoot, no pre-close alignment, no F3b re-grasp fixes
# One Isaac process (CPU, not vectorised): run it only when at most one other Isaac process is running.
# Resumable per set.   powershell -File scripts/adam_baseline.ps1 [-N 50]
param([int]$N = 50, [string]$Out = "outputs/imitation/isaac/adam_asbuilt")
$ErrorActionPreference = "Stop"
. ./isaac/env_windows.ps1
New-Item -ItemType Directory -Force $Out | Out-Null
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
$env:WORLDFOLD_EXPERT_PARAMS = "PINCH_HEIGHT=0.010,ALIGN_TOL=0.0,OVERSHOOT_SCALE=0.0,RESYNC_IK_HOME=0,DESCEND_RETRY=0,CART_ARRIVE=0"
foreach ($set in @("id_easy", "id_hard", "recovery")) {
    $json = "$Out/eval_expert_$set.json"
    if (Test-Path $json) { "skip $set (exists)"; continue }
    & .venv-isaac/Scripts/python.exe -u -m imitation.evaluate --backend isaac --ckpt expert --sets $set --n $N `
        --workers 1 --out $json 2>&1 | Out-File -Encoding utf8 "$Out/eval_expert_$set.log"
    "$set exit $LASTEXITCODE"
}

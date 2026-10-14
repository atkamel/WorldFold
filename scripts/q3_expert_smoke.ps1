# Phase Q3, step 1: the quarter-fold expert (bimanual fold-over second fold) on isaac_friction, tune seeds only.
# n = 16 on tune_id_easy with per-episode rows (stage reached, failure code by stage), then 3 demo videos of the same
# seeds for a visual check of the fold-over. One sim job at a time (heat cap); run after the queued Q0/Q1 evals.
#   powershell -ExecutionPolicy Bypass -File scripts\q3_expert_smoke.ps1 [-N 16] [-Tag smoke]
param([int]$N = 16, [string]$Tag = "smoke", [int]$EnvsPerProc = 8)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"
$out = "outputs\imitation\isaac_friction\q3"
New-Item -ItemType Directory -Force $out | Out-Null
"STEP expert_$Tag $(Get-Date -Format s)"
$ErrorActionPreference = "Continue"
& $py -u -m imitation.evaluate --backend isaac_friction --task quarter_fold --ckpt expert --sets tune_id_easy `
    --n $N --workers 2 --out "$out\expert_$Tag.json" --resume *>&1 | Out-File -Append -Encoding utf8 "$out\expert_$Tag.log"
if ($LASTEXITCODE -ne 0) { "FAIL expert_$Tag exit $LASTEXITCODE"; exit 1 }
"DONE expert_$Tag $(Get-Date -Format s)"
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
& $py -u -m imitation.demo --backend isaac_friction --task quarter_fold --ckpt expert --seeds 660000 660001 660002 `
    --out "docs\reports\media\quarter_fold_expert_$Tag.mp4" *>&1 | Out-File -Append -Encoding utf8 "$out\demo_$Tag.log"
"Q3 SMOKE COMPLETE exit $LASTEXITCODE"

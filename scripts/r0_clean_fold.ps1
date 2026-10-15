# Phase Q rethink, R0 (2026-10-15): the fold-over second fold from a perfect half fold (isaac/diag_clean_fold.py).
# Tune seeds only, one Isaac process at a time (heat cap), thermal_guard alongside.
#   -Step smoke : geometry of the clean start on 2 seeds (recovery_replay stops at once: the start is stage 1),
#                 summarised by isaac/fold1_summary.py, plus a 1-seed video to look at the start.
#   -Step eval  : the fold-over expert (attempt F settings) on tune_id_easy, n = 16, 1 x 8 envs, then 3 seed videos.
#   powershell -ExecutionPolicy Bypass -File scripts\r0_clean_fold.ps1 -Step smoke
# -Params "A=1,B=2" sets WORLDFOLD_EXPERT_PARAMS (IsaacArmExpert knobs, R2 attempts); -Tag names the outputs.
param([ValidateSet("smoke", "eval")][string]$Step = "smoke", [int]$N = 16, [int]$EnvsPerProc = 8, [string]$Params = "",
      [string]$Tag = "clean", [string]$Second = "fold_over")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$env:WORLDFOLD_QF_SECOND = $Second      # fold_over (R0) or left_fold (R2, option 1)
$env:WORLDFOLD_QF_START = "clean"
$env:WORLDFOLD_EXPERT_PARAMS = $Params
$out = "outputs\imitation\isaac_friction\r0"
New-Item -ItemType Directory -Force $out | Out-Null
$ErrorActionPreference = "Continue"
if ($Step -eq "smoke") {
    "STEP r0_smoke $(Get-Date -Format s)"
    & $py -u isaac/recovery_replay.py --backend isaac_friction --task quarter_fold --no-knock --stop-at-stage 1 `
        --seeds 660000:660002 --out "$out\clean_start.jsonl" *>&1 | Out-File -Append -Encoding utf8 "$out\clean_start.log"
    if ($LASTEXITCODE -ne 0) { "FAIL r0_smoke exit $LASTEXITCODE"; exit 1 }
    & .venv\Scripts\python.exe isaac/fold1_summary.py "$out\clean_start.jsonl"
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
    & $py -u -m imitation.demo --backend isaac_friction --task quarter_fold --ckpt expert --seeds 660000 `
        --out "$out\clean_start_660000.mp4" *>&1 | Out-File -Append -Encoding utf8 "$out\demo_smoke.log"
    "R0 SMOKE COMPLETE exit $LASTEXITCODE"
} else {
    "STEP r0_eval $(Get-Date -Format s)"
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"
    & $py -u -m imitation.evaluate --backend isaac_friction --task quarter_fold --ckpt expert --sets tune_id_easy `
        --n $N --workers 1 --out "$out\expert_$Tag.json" --resume *>&1 | Out-File -Append -Encoding utf8 "$out\expert_$Tag.log"
    if ($LASTEXITCODE -ne 0) { "FAIL r0_eval exit $LASTEXITCODE"; exit 1 }
    "DONE r0_eval $(Get-Date -Format s)"
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
    & $py -u -m imitation.demo --backend isaac_friction --task quarter_fold --ckpt expert --seeds 660000 660001 660002 `
        --out "$out\expert_$Tag.mp4" *>&1 | Out-File -Append -Encoding utf8 "$out\demo_$Tag.log"
    "R0 EVAL COMPLETE exit $LASTEXITCODE"
}

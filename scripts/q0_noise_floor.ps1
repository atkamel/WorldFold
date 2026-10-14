# Phase Q0: Isaac run-to-run noise floor. The F4 BC checkpoint is evaluated twice on the same held-out selection
# seeds (sel_id_hard + sel_knock_arm, n = 100, replan 4); `python -m imitation.stats --flips` then gives the share of
# seeds whose outcome flips between identical runs. Run A doubles as the Q1 loss A/B baseline row.
# Resumable (evaluate --resume). Heat cap: one eval at a time (2 Isaac processes x 8 envs); thermal_guard alongside.
#   powershell -ExecutionPolicy Bypass -File scripts\q0_noise_floor.ps1
param([int]$N = 100, [int]$EnvsPerProc = 8)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"
$out = "outputs\imitation\isaac_friction\q0"
New-Item -ItemType Directory -Force $out | Out-Null
$ckpt = "outputs\imitation\runs\isaac_v1_friction_diff_s0\final.pt"
foreach ($run in "a", "b") {
    "STEP noise_$run $(Get-Date -Format s)"
    $ErrorActionPreference = "Continue"
    & $py -u -m imitation.evaluate --backend isaac_friction --ckpt $ckpt --sets sel_id_hard sel_knock_arm --n $N `
        --replan-every 4 --workers 2 --out "$out\noise_$run.json" --resume *>&1 |
        Out-File -Append -Encoding utf8 "$out\noise_$run.log"
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { "FAIL noise_$run exit $LASTEXITCODE"; exit 1 }
    "DONE noise_$run $(Get-Date -Format s)"
}
& $py -m imitation.stats "$out\noise_a.json" "$out\noise_b.json" --flips | Tee-Object "$out\flips.txt"
"Q0 NOISE COMPLETE"

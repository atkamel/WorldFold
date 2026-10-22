# Portfolio renders (2026-10-21): the R1-preset quarter-fold expert (scripted, parallel second fold, real fold 1) on one
# seed, seen through one or more look-at cameras (imitation.demo --hero-cam), with no text on the frames. One Isaac
# process, one episode; keep <= 2 Isaac processes running (heat cap) and thermal_guard alongside. Changes no defaults.
#   -Cams  : one "px py pz tx ty tz fovy" per camera, ";"-separated (world frame, metres / degrees)
#   powershell -ExecutionPolicy Bypass -File scripts\hero_render.ps1 -Seed 660000 -Out docs\reports\media\hero\raw.mp4 `
#       -FramesDir outputs\imitation\hero\frames -Cams "0.40 0.25 0.80 0.0 -0.17 0.45 40"
param([int]$Seed = 660000, [string]$Out = "outputs\imitation\hero\hero_raw.mp4", [string]$FramesDir = "",
      [string]$Cams = "0.40 0.25 0.80 0.0 -0.17 0.45 40", [string]$Size = "1600x1000", [int]$StopAfter = 0)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$env:WORLDFOLD_QF_SECOND = "parallel"                     # as qd_demo.ps1: the R1 preset is the quarter fold's default
Remove-Item Env:WORLDFOLD_QF_START -ErrorAction SilentlyContinue
$env:WORLDFOLD_EXPERT_PARAMS = ""
$env:WORLDFOLD_GRASP_KNOBS = ""
$env:WORLDFOLD_QF_MAX_STEPS = ""
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
$camArgs = @(); foreach ($c in ($Cams -split ";")) { $camArgs += @("--hero-cam") + ($c.Trim() -split "\s+") }
$extra = @(); if ($FramesDir) { $extra += @("--frames-dir", $FramesDir) }; if ($StopAfter -gt 0) { $extra += @("--stop-after", "$StopAfter") }
New-Item -ItemType Directory -Force (Split-Path -Parent $Out) | Out-Null
$ErrorActionPreference = "Continue"
"STEP hero seed $Seed $(Get-Date -Format s) cams='$Cams'"
& $env:ISAAC_PY -u -m imitation.demo --backend isaac_friction --task quarter_fold --ckpt expert --seeds $Seed `
    @camArgs --hero-size $Size --no-overlay @extra --out $Out
"DONE hero exit $LASTEXITCODE $(Get-Date -Format s)"

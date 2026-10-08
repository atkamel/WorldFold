# Phase F3b -> F4 chain: the F3 expert gate (id_easy, id_hard, the perturbation suite, legacy recovery; 2 x 8) with a
# given expert config, then `verify F3`; F4 (scripts/f4_retrain.ps1) starts with the same config only on PASS, so the
# retrain never starts on an expert that missed the gate and no hour is lost between them.
#   powershell -ExecutionPolicy Bypass -File scripts\f3_then_f4.ps1 -Expert "RESYNC_IK_HOME=1,DESCEND_RETRY=2"
# Resumable: the gate skips sets already evaluated; F4 skips finished stages.
param([string]$Expert = "", [string]$Out = "outputs/imitation/isaac_friction/f3")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$log = "outputs/imitation/isaac_friction/f3_then_f4.log"
function Note($m) { $m; Add-Content -Encoding utf8 $log "$(Get-Date -Format s) $m" }
Note "GATE start expert='$Expert' out=$Out"
$ex = @()
if ($Expert) { $ex = @("-Expert", $Expert) }          # an empty -Expert "" arrives as a missing argument
& powershell -ExecutionPolicy Bypass -File scripts\f3_gate.ps1 -Lane eval -Out $Out -N 100 @ex
& .venv/Scripts/python.exe -m imitation.verify F3 *>&1 | Tee-Object -Variable verdict | Out-Null
$verdict | ForEach-Object { Note "  $_" }
if (-not ($verdict -match "^PASS F3")) { Note "GATE FAIL: F4 not started"; exit 1 }
Note "GATE PASS: F4 starting"
& powershell -ExecutionPolicy Bypass -File scripts\f4_retrain.ps1 @ex
Note "F4 exit $LASTEXITCODE"

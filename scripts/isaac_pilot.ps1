# Phase I, I3.2: the imitation pipeline on Isaac Sim at pilot scale, through the real CLIs.
# Resumable: every stage is skipped when its output exists (collect resumes per episode). Logs to
# outputs/imitation/isaac/pilot/pilot.log with STEP / DONE / FAIL markers for a monitor.
#   powershell -ExecutionPolicy Bypass -File scripts\isaac_pilot.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$out = "outputs\imitation\isaac\pilot"
$runs = "outputs\imitation\runs"
$ds = "outputs\imitation\datasets"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out "pilot.log"

function Note($msg) { $msg; Add-Content -Path $log -Value $msg -Encoding utf8 }   # UTF-8, so a monitor can grep it

function Stage($name, $done, [scriptblock]$body) {
    if ($done -and (Test-Path $done)) { Note "SKIP $name (have $done)"; return }
    Note "STEP $name $(Get-Date -Format s)"
    $t0 = Get-Date
    # Windows PowerShell turns a native program's redirected stderr into error records; under "Stop" the first
    # Kit warning would end the script, so the exit code alone decides a stage
    $ErrorActionPreference = "Continue"
    & $body *>&1 | Out-File -Append -Encoding utf8 (Join-Path $out "$name.log")
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { Note "FAIL $name exit $LASTEXITCODE"; exit 1 }
    Note "DONE $name $([int]((Get-Date) - $t0).TotalSeconds)s"
}

$cams = "main=128,left_wrist_cam=64,right_wrist_cam=64"
Stage "collect" "$ds\isaac_pilot\manifest.json" {
    & $py -u -m imitation.data.collect --backend isaac --episodes 80 --workers 2 --recovery-fraction 0.3 `
        --render --cameras $cams --version isaac_pilot --root $ds --resume }
Stage "train_mlp" "$runs\isaac_pilot_mlp\final.pt" {
    & $py -u -m imitation.train --policy chunk_mlp --dataset isaac_pilot --root $ds --run "$runs\isaac_pilot_mlp" --steps 5000 --seed 0 }
Stage "train_diff" "$runs\isaac_pilot_diff\final.pt" {
    & $py -u -m imitation.train --policy diffusion --dataset isaac_pilot --root $ds --run "$runs\isaac_pilot_diff" --steps 5000 --seed 0 }
Stage "train_vision" "$runs\isaac_pilot_vision\final.pt" {
    & $py -u -m imitation.train --policy vision --dataset isaac_pilot --root $ds --run "$runs\isaac_pilot_vision" --steps 3000 --seed 0 }
foreach ($e in @(@("mlp", "id_easy", 20), @("mlp", "recovery", 10), @("diff", "id_easy", 20), @("diff", "id_hard", 10),
                 @("diff", "recovery", 10))) {
    $k, $set, $n = $e
    Stage "eval_${k}_$set" "$out\eval_${k}_$set.json" {
        & $py -u -m imitation.evaluate --backend isaac --ckpt "$runs\isaac_pilot_$k\final.pt" --sets $set --n $n `
            --workers 2 --out "$out\eval_${k}_$set.json" }
}
Stage "eval_vision_id_easy" "$out\eval_vision_id_easy.json" {
    & $py -u -m imitation.evaluate --backend isaac --ckpt "$runs\isaac_pilot_vision\final.pt" --sets id_easy --n 10 `
        --replan-every 2 --workers 2 --out "$out\eval_vision_id_easy.json" }
Stage "dagger" "$runs\isaac_pilot_dagger\history.json" {
    & $py -u -m imitation.dagger --backend isaac --init "$runs\isaac_pilot_diff\final.pt" --dataset isaac_pilot --root $ds `
        --out "$runs\isaac_pilot_dagger" --rounds 1 --episodes 16 --train-steps 5000 --eval-n 20 --eval-sets id_easy `
        --score-sets id_easy --workers 2 --labels takeover --takeover-p 0.3 }
Stage "detector_train" "$runs\isaac_pilot_detector\detector.pt" {
    & $py -u -m imitation.vision.success train --versions isaac_pilot isaac_pilot_failures --root $ds `
        --out "$runs\isaac_pilot_detector" --steps 2000 }
Stage "detector_agree" "$runs\isaac_pilot_detector\agreement.json" {
    & $py -u -m imitation.vision.success agree --ckpt "$runs\isaac_pilot_detector\detector.pt" `
        --versions isaac_pilot isaac_pilot_failures --root $ds }
Note "PILOT COMPLETE"

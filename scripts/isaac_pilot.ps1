# Phase I, I3.2 (-Backend isaac) and Phase W, W4 (-Backend isaac_weld): the imitation pipeline on Isaac Sim at pilot
# scale, through the real CLIs.
# Resumable: every stage is skipped when its output exists (collect resumes per episode). Logs to
# outputs/imitation/isaac/pilot/pilot.log with STEP / DONE / FAIL markers for a monitor.
#   powershell -ExecutionPolicy Bypass -File scripts\isaac_pilot.ps1 [-Backend isaac_weld]
param([ValidateSet("isaac", "isaac_weld")][string]$Backend = "isaac")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$out = "outputs\imitation\$Backend\pilot"
$tag = "${Backend}_pilot"        # dataset version and run-name prefix: isaac_pilot (I3.2), isaac_weld_pilot (W4)
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
Stage "collect" "$ds\$tag\manifest.json" {
    & $py -u -m imitation.data.collect --backend $Backend --episodes 80 --workers 2 --recovery-fraction 0.3 `
        --render --cameras $cams --version $tag --root $ds --resume }
Stage "train_mlp" "$runs\${tag}_mlp\final.pt" {
    & $py -u -m imitation.train --policy chunk_mlp --dataset $tag --root $ds --run "$runs\${tag}_mlp" --steps 5000 --seed 0 }
Stage "train_diff" "$runs\${tag}_diff\final.pt" {
    & $py -u -m imitation.train --policy diffusion --dataset $tag --root $ds --run "$runs\${tag}_diff" --steps 5000 --seed 0 }
Stage "train_vision" "$runs\${tag}_vision\final.pt" {
    & $py -u -m imitation.train --policy vision --dataset $tag --root $ds --run "$runs\${tag}_vision" --steps 3000 --seed 0 }
foreach ($e in @(@("mlp", "id_easy", 20), @("mlp", "recovery", 10), @("diff", "id_easy", 20), @("diff", "id_hard", 10),
                 @("diff", "recovery", 10))) {
    $k, $set, $n = $e
    Stage "eval_${k}_$set" "$out\eval_${k}_$set.json" {
        & $py -u -m imitation.evaluate --backend $Backend --ckpt "$runs\${tag}_$k\final.pt" --sets $set --n $n `
            --workers 2 --out "$out\eval_${k}_$set.json" }
}
Stage "eval_vision_id_easy" "$out\eval_vision_id_easy.json" {
    & $py -u -m imitation.evaluate --backend $Backend --ckpt "$runs\${tag}_vision\final.pt" --sets id_easy --n 10 `
        --replan-every 2 --workers 2 --out "$out\eval_vision_id_easy.json" }
Stage "dagger" "$runs\${tag}_dagger\round_1\final.pt" {
    & $py -u -m imitation.dagger --backend $Backend --init "$runs\${tag}_diff\final.pt" --dataset $tag --root $ds `
        --out "$runs\${tag}_dagger" --rounds 1 --episodes 16 --train-steps 5000 --eval-n 20 --eval-sets id_easy `
        --score-sets id_easy --workers 2 --labels takeover --takeover-p 0.3 --resume }
Stage "detector_train" "$runs\${tag}_detector\detector.pt" {
    & $py -u -m imitation.vision.success train --versions $tag "${tag}_failures" --root $ds `
        --out "$runs\${tag}_detector" --steps 2000 }
Stage "detector_agree" "$runs\${tag}_detector\agreement.json" {
    & $py -u -m imitation.vision.success agree --ckpt "$runs\${tag}_detector\detector.pt" `
        --versions $tag "${tag}_failures" --root $ds }
Note "PILOT COMPLETE"

# Phase Q1: loss v2 A/B on the half fold (isaac_v1_friction, diffusion seed 0, 20k steps, the F4 BC recipe).
#   arms:  F4 BC (pre-Q1 loss; scored in q0\noise_a.json)  |  q1_grip (gripper BCE head)  |
#          q1_all (gripper head + horizon tail 0.5 + phase x3)
#   then:  checkpoint selection -- best_val.pt of each Q1 arm, against its final.pt
# Every eval: sel_id_hard + sel_knock_arm, n = 100, replan 4, paired on shared seeds (imitation.stats).
# Trainings run on the GPU right away; evals wait for the Q0 noise-floor runs (one sim job at a time, heat cap).
# Resumable: each stage is skipped when its output exists.
#   powershell -ExecutionPolicy Bypass -File scripts\q1_ab.ps1
param([int]$N = 100, [int]$EnvsPerProc = 8, [int]$Steps = 20000)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"
$out = "outputs\imitation\isaac_friction\q1"
$runs = "outputs\imitation\runs"
New-Item -ItemType Directory -Force $out | Out-Null
$arms = [ordered]@{
    "q1_grip" = "gripper_head=true"
    "q1_all"  = "gripper_head=true,horizon_tail=0.5,phase_weight=3.0" }

function Run($name, $done, [scriptblock]$body) {
    if (Test-Path $done) { "SKIP $name"; return }
    "STEP $name $(Get-Date -Format s)"
    $ErrorActionPreference = "Continue"
    & $body *>&1 | Out-File -Append -Encoding utf8 "$out\$name.log"
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { "FAIL $name exit $LASTEXITCODE"; exit 1 }
    "DONE $name $(Get-Date -Format s)"
}

foreach ($arm in $arms.Keys) {
    $cfg = $arms[$arm]
    Run "train_$arm" "$runs\isaac_v1_friction_$arm\final.pt" {
        & $py -u -m imitation.train --policy diffusion --dataset isaac_v1_friction --root outputs\imitation\datasets `
            --run "$runs\isaac_v1_friction_$arm" --steps $Steps --seed 0 --save-every 2000 --loss-cfg $cfg }
}

while (-not (Test-Path "outputs\imitation\isaac_friction\q0\flips.txt")) {
    "WAIT q0 noise floor $(Get-Date -Format s)"; Start-Sleep -Seconds 120 }

function Eval($name, $ckpt) {
    Run "eval_$name" "$out\$name.done" {
        & $py -u -m imitation.evaluate --backend isaac_friction --ckpt $ckpt --sets sel_id_hard sel_knock_arm `
            --n $N --replan-every 4 --workers 2 --out "$out\$name.json" --resume
        if ($LASTEXITCODE -eq 0) { Set-Content "$out\$name.done" (Get-Date -Format s) } }
}
foreach ($arm in $arms.Keys) { Eval "${arm}_final" "$runs\isaac_v1_friction_$arm\final.pt" }
foreach ($arm in $arms.Keys) { Eval "${arm}_bestval" "$runs\isaac_v1_friction_$arm\best_val.pt" }
foreach ($name in "q1_grip_final", "q1_all_final", "q1_grip_bestval", "q1_all_bestval") {
    "== $name vs F4 BC (A = $name)"
    & $py -m imitation.stats "$out\$name.json" "outputs\imitation\isaac_friction\q0\noise_a.json"
}
"Q1 AB COMPLETE"

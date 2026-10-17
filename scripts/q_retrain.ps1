# Phase Q4: the fast-track quarter-fold pipeline on isaac_friction (user, 2026-10-13: 200 eps, 1 seed, 1 DAgger round,
# n = 100 finals; 2026-10-14: the parallel second fold). Stages:
#   collect (200 eps, 50% knocked across both folds, cameras) -> BC diffusion with the pre-Q1 loss and with the Q1 gripper
#   head -> DAgger 1 round (takeover p = 0.6) from the better BC on the sel_ sets -> vision student distilled from the
#   privileged best -> detector -> finals n = 100 on the final_ sets (expert, BC base, BC gripper head, DAgger, vision)
#   -> demo videos.
# Resumable: every stage is skipped when its output exists. One sim job at a time with GPU trainings never overlapping
# a sim (2026-10-13 PhysX crash); <= 2 Isaac processes; thermal_guard alongside. Log: outputs\imitation\isaac_friction\q4\q4.log
#   powershell -ExecutionPolicy Bypass -File scripts\q_retrain.ps1 [-Only <stage>] [-Episodes 200] [-FinalN 100]
param([string]$Only = "", [int]$Episodes = 200, [int]$FinalN = 100, [int]$SelN = 100, [int]$BCSteps = 20000,
      [int]$EnvsPerProc = 8, [int]$CameraEnvsPerProc = 6, [int]$DaggerEpisodes = 128, [switch]$NoVision)
# -NoVision (QD-2, 2026-10-16): state-only demos at the full envs per process. The camera collect ran at ~4.7 s per step
# (2 x 6 envs, 3 cameras, GPU 97%): 27 min to the first episodes, ~7 h for 200. The vision stages then have no images.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$B = "isaac_friction"
$T = "quarter_fold"
$tag = "isaac_qf_v1"
$out = "outputs\imitation\$B\q4"
$runs = "outputs\imitation\runs"
$ds = "outputs\imitation\datasets"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out "q4.log"
$cams = "main=128,left_wrist_cam=64,right_wrist_cam=64"
$sel = @("sel_id_hard", "sel_knock_arm")
$finals = @("final_id_easy", "final_id_hard", "final_knock_arm")
$kinds = @("knock_arm")

function Note($msg) {
    $msg
    foreach ($i in 1..10) {
        try { [IO.File]::AppendAllText((Join-Path $repo $log), "$msg`r`n"); return } catch { Start-Sleep -Milliseconds 300 }
    }
}
function Stage($name, $done, [scriptblock]$body) {
    if ($Only -and $Only -ne $name) { return }
    if ($done -and (Test-Path $done)) { Note "SKIP $name (have $done)"; return }
    Note "STEP $name $(Get-Date -Format s)"
    $t0 = Get-Date
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"
    $ErrorActionPreference = "Continue"
    & $body *>&1 | Out-File -Append -Encoding utf8 (Join-Path $out "$name.log")
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { Note "FAIL $name exit $LASTEXITCODE"; exit 1 }
    Note "DONE $name $([int]((Get-Date) - $t0).TotalSeconds)s"
}
function Best($run) {
    & $py -c "import json,sys; h=json.load(open(sys.argv[1])); print([r for r in h if r['kept']][-1]['checkpoint'])" `
        "$run\history.json"
}
# pooled sel_ success of an eval JSON (picks the better BC as the DAgger start)
function Pooled($json) {
    & $py -c "import json,sys; r=json.load(open(sys.argv[1]))['results']; print(sum(v['success_rate'] for v in r.values()))" $json
}
function Eval($name, $ckpt, $sets, $n, $replan, $envs) {
    Stage $name "$out\$name.done" {
        $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$envs"
        & $py -u -m imitation.evaluate --backend $B --task $T --ckpt $ckpt --sets $sets --n $n --replan-every $replan `
            --workers 2 --out "$out\$name.json" --resume
        if ($LASTEXITCODE -eq 0) { Set-Content "$out\$name.done" (Get-Date -Format s) } }
}

# 1. demos with images (the vision student needs them); the camera rig caps envs per process
Stage "collect" "$ds\$tag\manifest.json" {
    if ($NoVision) {
        & $py -u -m imitation.data.collect --backend $B --task $T --episodes $Episodes --workers 2 --recovery-fraction 0.5 `
            --perturb-kinds $kinds --version $tag --root $ds --resume
    } else {
        $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$CameraEnvsPerProc"
        & $py -u -m imitation.data.collect --backend $B --task $T --episodes $Episodes --workers 2 --recovery-fraction 0.5 `
            --perturb-kinds $kinds --render --cameras $cams --version $tag --root $ds --resume
    } }

# 2. BC: the pre-Q1 loss and the Q1 gripper head (kept, results.md), same seed and recipe (GPU only)
Stage "train_bc_base" "$runs\${tag}_bc_base\final.pt" {
    & $py -u -m imitation.train --policy diffusion --dataset $tag --root $ds --run "$runs\${tag}_bc_base" `
        --steps $BCSteps --seed 0 --save-every 2000 }
Stage "train_bc_grip" "$runs\${tag}_bc_grip\final.pt" {
    & $py -u -m imitation.train --policy diffusion --dataset $tag --root $ds --run "$runs\${tag}_bc_grip" `
        --steps $BCSteps --seed 0 --save-every 2000 --loss-cfg "gripper_head=true" }
Eval "sel_bc_base" "$runs\${tag}_bc_base\final.pt" $sel $SelN 4 $EnvsPerProc
Eval "sel_bc_grip" "$runs\${tag}_bc_grip\final.pt" $sel $SelN 4 $EnvsPerProc

# 3. DAgger, 1 round from the better BC on the sel_ sets (the keep rule is paired, Q0)
$bcBest = "$runs\${tag}_bc_grip\final.pt"
if ((Test-Path "$out\sel_bc_base.json") -and (Test-Path "$out\sel_bc_grip.json")) {
    if ([double](Pooled "$out\sel_bc_base.json") -gt [double](Pooled "$out\sel_bc_grip.json")) { $bcBest = "$runs\${tag}_bc_base\final.pt" }
}
$priv = "$runs\${tag}_dagger"
Stage "dagger" "$priv\done.txt" {
    & $py -u -m imitation.dagger --backend $B --task $T --init $bcBest --dataset $tag --root $ds --out $priv --rounds 1 `
        --episodes $DaggerEpisodes --train-steps 15000 --eval-n $SelN --eval-sets $sel --score-sets $sel `
        --workers 2 --labels takeover --takeover-p 0.6 --resume --recovery-fraction 0.5 --perturb-kinds $kinds --replan-every 4
    if ($LASTEXITCODE -eq 0) { Set-Content "$priv\done.txt" (Get-Date -Format s) } }
$bp = if (Test-Path "$priv\history.json") { Best $priv } else { $bcBest }

# 4. vision student: teacher-relabelled BC from the privileged best, with the gripper head (GPU only)
Stage "train_vision" "$runs\${tag}_vision\final.pt" {
    & $py -u -m imitation.train --policy vision --dataset $tag --root $ds --run "$runs\${tag}_vision" `
        --steps 30000 --batch 256 --seed 0 --teacher $bp --loss-cfg "gripper_head=true" }
$bv = "$runs\${tag}_vision\final.pt"

# 5. detector (labels free in sim)
Stage "detector_train" "$runs\${tag}_detector\detector.pt" {
    & $py -u -m imitation.vision.success train --versions $tag "${tag}_failures" --root $ds `
        --out "$runs\${tag}_detector" --steps 2000 }
Stage "detector_agree" "$runs\${tag}_detector\agreement.json" {
    & $py -u -m imitation.vision.success agree --ckpt "$runs\${tag}_detector\detector.pt" `
        --versions $tag "${tag}_failures" --root $ds }

# 6. finals: n = 100 on the final_ blocks, every row on the same seeds (paired comparisons, Q0)
Eval "final_expert" "expert" $finals $FinalN 4 $EnvsPerProc
Eval "final_bc_base" "$runs\${tag}_bc_base\final.pt" $finals $FinalN 4 $EnvsPerProc
Eval "final_bc_grip" "$runs\${tag}_bc_grip\final.pt" $finals $FinalN 4 $EnvsPerProc
Eval "final_dagger" $bp $finals $FinalN 4 $EnvsPerProc
Eval "final_vision" $bv $finals $FinalN 2 $CameraEnvsPerProc

# 7. demo videos: 3 episodes of each final set, privileged and sensor-only
foreach ($pair in @(@("privileged", $bp, 4), @("sensor", $bv, 2))) {
    $who, $ck, $rp = $pair
    foreach ($set in @("id_easy", "id_hard", "knock_arm")) {
        $mp4 = "docs\reports\media\quarter_fold_${who}_$set.mp4"
        Stage "demo_${who}_$set" $mp4 {
            $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
            & $py -u -m imitation.demo --backend $B --task $T --ckpt $ck --replan-every $rp --set $set --n 3 --out $mp4 }
    }
}
if (-not $Only) { Note "Q4 COMPLETE" }

# Phase W, W5: the full imitation pipeline on the Isaac weld baseline (isaac_weld), with the MuJoCo recipe:
# diffusion BC (2 seeds) -> diffusion DAgger (takeover labels) -> vision student distilled from the best
# privileged policy -> detector -> final n = 200 evals -> demo videos.
# Resumable: every stage is skipped when its output exists (collect and DAgger resume inside). Logs to
# outputs/imitation/isaac_weld/w5/w5.log with STEP / DONE / FAIL markers for a monitor.
#   powershell -ExecutionPolicy Bypass -File scripts\isaac_retrain.ps1 [-Phase privileged|vision|final|all]
#   powershell -ExecutionPolicy Bypass -File scripts\isaac_retrain.ps1 -Only <stage>     # one stage (used for overlap)
# Heat budget (user, 2026-10-06): at most 2 Isaac processes x 4 envs, no overlapping sims, and keep the laptop under
# 94 C -- run scripts/thermal_guard.py alongside; rollouts and training pause while its flag is set (imitation/thermal.py).
param([ValidateSet("privileged", "vision", "final", "all")][string]$Phase = "all", [string]$Only = "",
      [int]$EnvsPerProc = 4)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"
$B = "isaac_weld"
$tag = "isaac_v1_weld"
$out = "outputs\imitation\$B\w5"
$runs = "outputs\imitation\runs"
$ds = "outputs\imitation\datasets"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out "w5.log"
$cams = "main=128,left_wrist_cam=64,right_wrist_cam=64"
$sets = @("id_easy", "id_hard", "recovery")

function Note($msg) {     # a reader holding the log open must not end the run: retry, then give up on this line
    $msg
    foreach ($i in 1..10) {
        try { [IO.File]::AppendAllText((Join-Path $repo $log), "$msg`r`n"); return } catch { Start-Sleep -Milliseconds 300 }
    }
}

$script:sideStages = @()
function Stage($name, $done, [scriptblock]$body) {
    if ($Only -and $Only -ne $name) { return }
    if ($script:sideStages -contains $name) { return }      # running in a Side process
    if ($done -and (Test-Path $done)) { Note "SKIP $name (have $done)"; return }
    Note "STEP $name $(Get-Date -Format s)"
    $t0 = Get-Date
    $ErrorActionPreference = "Continue"     # Kit warnings on stderr must not end the script; the exit code decides
    & $body *>&1 | Out-File -Append -Encoding utf8 (Join-Path $out "$name.log")
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { Note "FAIL $name exit $LASTEXITCODE"; exit 1 }
    Note "DONE $name $([int]((Get-Date) - $t0).TotalSeconds)s"
}

# Run one stage of this script in a second process (overlap); returns the process to wait on.
function Side($name) {
    if ($Only) { return $null }
    $null = Note "SIDE $name"      # Note echoes; keep it out of the return value
    $script:sideStages += $name
    Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList @("-ExecutionPolicy", "Bypass", "-File",
        $PSCommandPath, "-Only", $name, "-EnvsPerProc", "$EnvsPerProc")
}
function Join($p, $name, $done) {
    if (-not $p) { return }
    $p.WaitForExit()
    Note "JOIN $name"
    if (-not (Test-Path $done)) { Note "FAIL $name (side process ended without $done)"; exit 1 }
}

# Best checkpoint of a DAgger run: the last kept round in history.json.
function Best($run) {
    & $env:ISAAC_PY -c "import json,sys; h=json.load(open(sys.argv[1])); print([r for r in h if r['kept']][-1]['checkpoint'])" `
        "$run\history.json"
}

$priv = "$runs\${tag}_dagger"
$vis = "$runs\${tag}_distill"

if ($Phase -in "privileged", "all" -or $Only) {
    Stage "collect" "$ds\$tag\manifest.json" {
        & $py -u -m imitation.data.collect --backend $B --episodes 400 --workers 2 --recovery-fraction 0.3 `
            --render --cameras $cams --version $tag --root $ds --resume }
    # both seeds train at once on the GPU (no Isaac process running)
    $s1 = Side "train_diff_s1"
    Stage "train_diff_s0" "$runs\${tag}_diff_s0\final.pt" {
        & $py -u -m imitation.train --policy diffusion --dataset $tag --root $ds --run "$runs\${tag}_diff_s0" `
            --steps 30000 --seed 0 }
    Stage "train_diff_s1" "$runs\${tag}_diff_s1\final.pt" {
        & $py -u -m imitation.train --policy diffusion --dataset $tag --root $ds --run "$runs\${tag}_diff_s1" `
            --steps 30000 --seed 1 }
    Join $s1 "train_diff_s1" "$runs\${tag}_diff_s1\final.pt"
    # DAgger from seed 0 (its round 0 is the seed-0 BC eval). Seed 1 is the seed-variance check: id_easy / id_hard
    # finished in eval_diff_s1.log before the heat cap; only its recovery set runs here, alone.
    Stage "dagger" "$priv\done.txt" {
        # one round (fast track, user 2026-10-06): MuJoCo's gain came almost all from round 1 (M5b.1)
        & $py -u -m imitation.dagger --backend $B --init "$runs\${tag}_diff_s0\final.pt" --dataset $tag --root $ds `
            --out $priv --rounds 1 --episodes 128 --train-steps 15000 --eval-n 100 --eval-sets $sets `
            --score-sets $sets --min-gain-se 1 --workers 2 --labels takeover --takeover-p 0.3 --resume
        if ($LASTEXITCODE -eq 0) { Set-Content "$priv\done.txt" (Get-Date -Format s) } }
    Stage "eval_diff_s1_recovery" "$out\eval_diff_s1_recovery_r8.json" {
        & $py -u -m imitation.evaluate --backend $B --ckpt "$runs\${tag}_diff_s1\final.pt" --sets recovery --n 100 `
            --workers 2 --out "$out\eval_diff_s1_recovery_r8.json" }
    if (-not $Only) { Note "PRIVILEGED COMPLETE best=$(Best $priv)" }
}

# Demo videos: 3 episodes of one eval set (its poses and knock), one Isaac env on the CPU pipeline.
function Demos($policy, $ckpt, $replan) {
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
    foreach ($set in $sets) {
        $mp4 = "docs\reports\media\half_fold_isaac_${policy}_$set.mp4"
        if (Test-Path $mp4) { continue }
        & $py -u -m imitation.demo --backend $B --ckpt $ckpt --replan-every $replan --set $set --n 3 --out $mp4
        if ($LASTEXITCODE -ne 0) { return }
    }
}

# Operating points from M5b.6: privileged replan 4, vision replan 2.
$bp = if (Test-Path "$priv\history.json") { Best $priv } else { "" }
$bv = "$runs\${tag}_vision_t0\final.pt"     # fast track: the teacher-relabelled vision BC is the student (M5b.3: round 0 won)

if ($Phase -in "vision", "all" -or $Only) {
    # the vision student trains on the GPU while the privileged demos run (1 CPU-pipeline sim): 2 jobs at once
    $dp = Side "demos_privileged"
    Stage "demos_privileged" "docs\reports\media\half_fold_isaac_privileged_recovery.mp4" { Demos "privileged" $bp 4 }
    Stage "train_vision_t0" "$runs\${tag}_vision_t0\final.pt" {
        & $py -u -m imitation.train --policy vision --dataset $tag --root $ds --run "$runs\${tag}_vision_t0" `
            --steps 30000 --batch 256 --seed 0 --teacher $bp }
    Join $dp "demos_privileged" "docs\reports\media\half_fold_isaac_privileged_recovery.mp4"
    Stage "final_privileged" "$out\final_privileged_r4.json" {
        & $py -u -m imitation.evaluate --backend $B --ckpt $bp --sets $sets --n 200 --replan-every 4 --workers 2 `
            --out "$out\final_privileged_r4.json" }
    Stage "detector_train" "$runs\${tag}_detector\detector.pt" {
        & $py -u -m imitation.vision.success train --versions $tag "${tag}_failures" --root $ds `
            --out "$runs\${tag}_detector" --steps 2000 }
    Stage "detector_agree" "$runs\${tag}_detector\agreement.json" {
        & $py -u -m imitation.vision.success agree --ckpt "$runs\${tag}_detector\detector.pt" `
            --versions $tag "${tag}_failures" --root $ds }
    if (-not $Only) { Note "VISION COMPLETE" }
}

if ($Phase -in "final", "all" -or $Only) {
    Stage "final_vision" "$out\final_vision_r2.json" {
        & $py -u -m imitation.evaluate --backend $B --ckpt $bv --sets $sets --n 200 --replan-every 2 --workers 2 `
            --out "$out\final_vision_r2.json" }
    Stage "demos_sensor" "docs\reports\media\half_fold_isaac_sensor_recovery.mp4" { Demos "sensor" $bv 2 }
    if (-not $Only) { Note "W5 COMPLETE" }
}

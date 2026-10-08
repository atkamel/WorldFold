# Phase F, F4: the full imitation pipeline on the physical (friction) grasp, isaac_friction, from W5's script:
# diffusion BC (2 seeds) -> diffusion DAgger (takeover labels) -> vision student distilled from the best
# privileged policy -> detector -> final n = 200 evals -> demo videos.
# Resumable: every stage is skipped when its output exists (collect and DAgger resume inside). Logs to
# outputs/imitation/isaac_friction/f4/f4.log with STEP / DONE / FAIL markers for a monitor.
# Differences from W5: 50% knocked demos (recovery is the gap), DAgger 2 rounds at takeover p = 0.6, LeHome's task
# sets (imitation.seeds.LEHOME_TASK_BACKENDS), demo videos named half_fold_friction_*. Fast path (user, 2026-10-07):
# 2 x 8 envs, one BC seed (W5 measured the seed variance), DAgger selection on id_hard + recovery only (id_easy is
# saturated; the finals measure all three); -WaitF3 starts collection the moment the F3 expert eval finishes.
#   powershell -ExecutionPolicy Bypass -File scripts\f4_retrain.ps1 [-Phase privileged|vision|final|all]
#   powershell -ExecutionPolicy Bypass -File scripts\f4_retrain.ps1 -Only <stage>     # one stage (used for overlap)
# Heat budget (user, 2026-10-06): at most 2 Isaac processes x 4 envs, no overlapping sims, and keep the laptop under
# 94 C -- run scripts/thermal_guard.py alongside; rollouts and training pause while its flag is set (imitation/thermal.py).
param([ValidateSet("privileged", "vision", "final", "all")][string]$Phase = "all", [string]$Only = "",
      [int]$EnvsPerProc = 8, [int]$CameraEnvsPerProc = 6, [switch]$WaitF3, [string]$Expert = "",
      [switch]$NoVision, [int]$Rounds = 2)
# -NoVision (user, 2026-10-08: the 9 h budget): state-only demos at the full envs per process, no vision student,
# detector, vision finals or sensor demos; the privileged demos run after the finals (never 3 Isaac processes)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"
$B = "isaac_friction"
$tag = "isaac_v1_friction"
$out = "outputs\imitation\$B\f4"
$runs = "outputs\imitation\runs"
$ds = "outputs\imitation\datasets"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out "f4.log"
$cams = "main=128,left_wrist_cam=64,right_wrist_cam=64"
# F3b (2026-10-08): the perturbation suite replaces the legacy recovery set; finals at n = 200 on the clean, shifted
# and re-grasp sets, n = 100 on the noise sets and the legacy knock (reported only)
$sets = @("id_easy", "id_hard", "knock_arm")         # forced drop skipped (user, 2026-10-08)
$sets100 = @("joint_noise", "overshoot")       # the legacy all-dims knock (recovery) is no longer run
$kinds = @("knock_arm")                     # perturbed demos and DAgger rollouts (noise kinds need DART labels)

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
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"     # each stage starts at the default; camera stages lower it
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

if ($WaitF3) {
    $f3 = "outputs\imitation\$B\f3\eval_expert_recovery.json"
    Note "WAIT for $f3"
    while (-not (Test-Path $f3)) { Start-Sleep 30 }
    Start-Sleep 30       # its Isaac processes exit after the JSON is written
}
$sel = @("id_hard", "knock_arm")
$env:WORLDFOLD_EXPERT_PARAMS = $Expert       # the expert config F3b kept (IsaacArmExpert attributes), for every stage

if ($Phase -in "privileged", "all" -or $Only) {
    Stage "collect" "$ds\$tag\manifest.json" {
        if ($NoVision) {
            & $py -u -m imitation.data.collect --backend $B --episodes 400 --workers 2 --recovery-fraction 0.5 `
                --perturb-kinds $kinds --version $tag --root $ds --resume
        } else {
            $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$CameraEnvsPerProc"     # rig cameras: 2 x 8 state-only uses 14 GB
            & $py -u -m imitation.data.collect --backend $B --episodes 400 --workers 2 --recovery-fraction 0.5 `
                --perturb-kinds $kinds --render --cameras $cams --version $tag --root $ds --resume
        } }
    Stage "train_diff_s0" "$runs\${tag}_diff_s0\final.pt" {
        & $py -u -m imitation.train --policy diffusion --dataset $tag --root $ds --run "$runs\${tag}_diff_s0" `
            --steps 30000 --seed 0 }
    # DAgger from seed 0 (its round 0 is the seed-0 BC eval on the selection sets)
    Stage "dagger" "$priv\done.txt" {
        # two rounds at takeover p = 0.6 (W5's one round at 0.3 gave +5 pp recovery on 403 labels)
        & $py -u -m imitation.dagger --backend $B --init "$runs\${tag}_diff_s0\final.pt" --dataset $tag --root $ds `
            --out $priv --rounds $Rounds --episodes 128 --train-steps 15000 --eval-n 100 --eval-sets $sel `
            --score-sets $sel --min-gain-se 1 --workers 2 --labels takeover --takeover-p 0.6 --resume `
            --recovery-fraction 0.5 --perturb-kinds $kinds
        if ($LASTEXITCODE -eq 0) { Set-Content "$priv\done.txt" (Get-Date -Format s) } }
    if (-not $Only) { Note "PRIVILEGED COMPLETE best=$(Best $priv)" }
}

# Demo videos: 3 episodes of one eval set (its poses and knock), one Isaac env.
function Demos($policy, $ckpt, $replan) {
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
    foreach ($set in $sets) {
        $mp4 = "docs\reports\media\half_fold_friction_${policy}_$set.mp4"
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
    $dp = if ($NoVision) { $null } else { Side "demos_privileged" }
    if ($NoVision) { $script:sideStages += "demos_privileged" }       # runs after the finals instead
    Stage "demos_privileged" "docs\reports\media\half_fold_friction_privileged_knock_arm.mp4" { Demos "privileged" $bp 4 }
    if (-not $NoVision) {
    Stage "train_vision_t0" "$runs\${tag}_vision_t0\final.pt" {
        & $py -u -m imitation.train --policy vision --dataset $tag --root $ds --run "$runs\${tag}_vision_t0" `
            --steps 30000 --batch 256 --seed 0 --teacher $bp }
    Join $dp "demos_privileged" "docs\reports\media\half_fold_friction_privileged_knock_arm.mp4"
    }
    Stage "final_privileged" "$out\final_privileged_r4.done" {
        & $py -u -m imitation.evaluate --backend $B --ckpt $bp --sets $sets --n 200 --replan-every 4 --workers 2 `
            --out "$out\final_privileged_r4.json" --resume
        if ($LASTEXITCODE -ne 0) { return }
        & $py -u -m imitation.evaluate --backend $B --ckpt $bp --sets $sets100 --n 100 --replan-every 4 --workers 2 `
            --out "$out\final_privileged_r4_n100.json" --resume
        if ($LASTEXITCODE -eq 0) { Set-Content "$out\final_privileged_r4.done" (Get-Date -Format s) } }
    if ($NoVision) {
        $script:sideStages = @($script:sideStages | Where-Object { $_ -ne "demos_privileged" })
        Stage "demos_privileged" "docs\reports\media\half_fold_friction_privileged_knock_arm.mp4" {
            Demos "privileged" $bp 4 }
        if (-not $Only) { Note "F4 COMPLETE (no vision)" }
        exit 0
    }
    Stage "detector_train" "$runs\${tag}_detector\detector.pt" {
        & $py -u -m imitation.vision.success train --versions $tag "${tag}_failures" --root $ds `
            --out "$runs\${tag}_detector" --steps 2000 }
    Stage "detector_agree" "$runs\${tag}_detector\agreement.json" {
        & $py -u -m imitation.vision.success agree --ckpt "$runs\${tag}_detector\detector.pt" `
            --versions $tag "${tag}_failures" --root $ds }
    if (-not $Only) { Note "VISION COMPLETE" }
}

if ($Phase -in "final", "all" -or $Only) {
    # over 2 h in all (camera rendering at replan 2): each set is saved as it finishes and --resume skips it,
    # so a job-limit cut costs at most one set; the .done marker says all three are in
    Stage "final_vision" "$out\final_vision_r2.done" {
        $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$CameraEnvsPerProc"
        & $py -u -m imitation.evaluate --backend $B --ckpt $bv --sets $sets --n 200 --replan-every 2 --workers 2 `
            --out "$out\final_vision_r2.json" --resume
        if ($LASTEXITCODE -ne 0) { return }
        & $py -u -m imitation.evaluate --backend $B --ckpt $bv --sets $sets100 --n 100 --replan-every 2 --workers 2 `
            --out "$out\final_vision_r2_n100.json" --resume
        if ($LASTEXITCODE -eq 0) { Set-Content "$out\final_vision_r2.done" (Get-Date -Format s) } }
    Stage "demos_sensor" "docs\reports\media\half_fold_friction_sensor_knock_arm.mp4" { Demos "sensor" $bv 2 }
    if (-not $Only) { Note "F4 COMPLETE" }
}

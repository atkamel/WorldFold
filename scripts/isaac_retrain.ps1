# Phase W, W5: the full imitation pipeline on the Isaac weld baseline (isaac_weld), with the MuJoCo recipe:
# diffusion BC (2 seeds) -> diffusion DAgger (takeover labels) -> vision student distilled from the best
# privileged policy -> detector -> final n = 200 evals -> demo videos.
# Resumable: every stage is skipped when its output exists (collect and DAgger resume inside). Logs to
# outputs/imitation/isaac_weld/w5/w5.log with STEP / DONE / FAIL markers for a monitor.
#   powershell -ExecutionPolicy Bypass -File scripts\isaac_retrain.ps1 [-Phase privileged|vision|final|all]
#   powershell -ExecutionPolicy Bypass -File scripts\isaac_retrain.ps1 -Only <stage>     # one stage (used for overlap)
# At most 3 Isaac processes (CPU heat budget); camera stages use 2: 3 x B = 8 with the rig cameras overflows 16 GB VRAM.
param([ValidateSet("privileged", "vision", "final", "all")][string]$Phase = "all", [string]$Only = "",
      [int]$EnvsPerProc = 8)
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
    # DAgger from seed 0 (its round 0 is the seed-0 BC eval); seed 1 is evaluated alongside as the seed check
    $e1 = Side "eval_diff_s1"
    Stage "eval_diff_s1" "$out\eval_diff_s1_r8.json" {
        & $py -u -m imitation.evaluate --backend $B --ckpt "$runs\${tag}_diff_s1\final.pt" --sets $sets --n 100 `
            --workers 1 --out "$out\eval_diff_s1_r8.json" }
    Stage "dagger" "$priv\done.txt" {
        & $py -u -m imitation.dagger --backend $B --init "$runs\${tag}_diff_s0\final.pt" --dataset $tag --root $ds `
            --out $priv --rounds 3 --episodes 128 --train-steps 15000 --eval-n 100 --eval-sets $sets `
            --score-sets $sets --min-gain-se 1 --workers 2 --labels takeover --takeover-p 0.3 --resume
        if ($LASTEXITCODE -eq 0) { Set-Content "$priv\done.txt" (Get-Date -Format s) } }
    Join $e1 "eval_diff_s1" "$out\eval_diff_s1_r8.json"
    if (-not $Only) { Note "PRIVILEGED COMPLETE best=$(Best $priv)" }
}

if ($Phase -in "vision", "all" -or $Only) {
    $teacher = if (Test-Path "$priv\history.json") { Best $priv } else { "" }
    # round 0 of the M5b.3 protocol: vision BC on every collected step relabelled by the privileged teacher
    Stage "train_vision_t0" "$runs\${tag}_vision_t0\final.pt" {
        & $py -u -m imitation.train --policy vision --dataset $tag --root $ds --run "$runs\${tag}_vision_t0" `
            --steps 30000 --batch 256 --seed 0 --teacher $teacher }
    Stage "distill" "$vis\done.txt" {
        & $py -u -m imitation.dagger --backend $B --init "$runs\${tag}_vision_t0\final.pt" --dataset $tag --root $ds `
            --out $vis --teacher $teacher --relabel --rounds 2 --episodes 128 --recovery-fraction 0.6 `
            --shift-fraction 0.3 --replan-every 2 --dagger-weight 2 --train-steps 15000 --eval-n 100 `
            --eval-sets $sets --score-sets $sets --min-gain-se 1 --workers 2 --resume
        if ($LASTEXITCODE -eq 0) { Set-Content "$vis\done.txt" (Get-Date -Format s) } }
    Stage "detector_train" "$runs\${tag}_detector\detector.pt" {
        & $py -u -m imitation.vision.success train --versions $tag "${tag}_failures" --root $ds `
            --out "$runs\${tag}_detector" --steps 2000 }
    Stage "detector_agree" "$runs\${tag}_detector\agreement.json" {
        & $py -u -m imitation.vision.success agree --ckpt "$runs\${tag}_detector\detector.pt" `
            --versions $tag "${tag}_failures" --root $ds }
    if (-not $Only) { Note "VISION COMPLETE best=$(Best $vis)" }
}

if ($Phase -in "final", "all" -or $Only) {
    $bp = if (Test-Path "$priv\history.json") { Best $priv } else { "" }
    $bv = if (Test-Path "$vis\history.json") { Best $vis } else { "" }
    # operating points from M5b.6: privileged replan 4, vision replan 2
    $fv = Side "final_vision"
    Stage "final_privileged" "$out\final_privileged_r4.json" {
        & $py -u -m imitation.evaluate --backend $B --ckpt $bp --sets $sets --n 200 --replan-every 4 --workers 2 `
            --out "$out\final_privileged_r4.json" }
    Stage "final_vision" "$out\final_vision_r2.json" {
        & $py -u -m imitation.evaluate --backend $B --ckpt $bv --sets $sets --n 200 --replan-every 2 --workers 1 `
            --out "$out\final_vision_r2.json" }
    Join $fv "final_vision" "$out\final_vision_r2.json"
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"     # demos render one episode at a time
    Stage "demo_privileged" "docs\reports\media\half_fold_isaac_privileged.mp4" {
        & $py -u -m imitation.demo --backend $B --ckpt $bp --replan-every 4 --seeds 100000 100001 100002 100003 `
            --out "docs\reports\media\half_fold_isaac_privileged.mp4" }
    Stage "demo_sensor" "docs\reports\media\half_fold_isaac_sensor.mp4" {
        & $py -u -m imitation.demo --backend $B --ckpt $bv --replan-every 2 --seeds 100000 100001 100002 100003 `
            --out "docs\reports\media\half_fold_isaac_sensor.mp4" }
    if (-not $Only) { Note "W5 COMPLETE" }
}

# QD (2026-10-16): the parallel quarter-fold demo. Expert runs on tune seeds, no knobs unless -Params is given, a real
# fold 1 (never WORLDFOLD_QF_START=clean). One Isaac process per call; keep <= 2 running (heat cap), thermal_guard alongside.
#   -Step eval  : expert on -Sets (default tune_id_easy; "tune_id_hard tune_knock_arm"), n = -N, 1 x -EnvsPerProc
#                 -> outputs\imitation\isaac_friction\qd\expert_<Tag>.json (--resume: sets already there are kept)
#   -Step demo  : video of -Seeds (space-separated)                 -> -Out (default docs\reports\media\quarter_fold_parallel_expert.mp4)
#   -Step trace : recovery_replay per-step traces of -Seeds (a:b range) -> qd\trace_<Tag>.jsonl + qd\traces_<Tag>\
#   powershell -ExecutionPolicy Bypass -File scripts\qd_demo.ps1 -Step eval -Tag base
param([ValidateSet("eval", "demo", "trace")][string]$Step = "eval", [int]$N = 16, [int]$EnvsPerProc = 8,
      [string]$Params = "", [string]$Tag = "base", [string]$Seeds = "660002 660004 660009", [string]$Out = "",
      [string]$Set = "", [string]$Ckpt = "expert", [int]$Replan = 8, [int]$MaxSteps = 0, [int]$StopAtStage = -1, [string]$Grasp = "",
      [string]$Sets = "tune_id_easy")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$env:WORLDFOLD_QF_SECOND = "parallel"
Remove-Item Env:WORLDFOLD_QF_START -ErrorAction SilentlyContinue
$env:WORLDFOLD_EXPERT_PARAMS = $Params
$env:WORLDFOLD_GRASP_KNOBS = $Grasp      # diagnostic only (isaac_env), e.g. "particle_friction=1.0"
$env:WORLDFOLD_QF_MAX_STEPS = $(if ($MaxSteps -gt 0) { "$MaxSteps" } else { "" })
$dir = "outputs\imitation\isaac_friction\qd"      # not $out: PowerShell names are case-insensitive, -Out would be clobbered
New-Item -ItemType Directory -Force $dir | Out-Null
$ErrorActionPreference = "Continue"
"STEP qd_$Step $Tag $(Get-Date -Format s) params='$Params'"
if ($Step -eq "eval") {
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"
    & $py -u -m imitation.evaluate --backend isaac_friction --task quarter_fold --ckpt $Ckpt --sets @($Sets -split " ") `
        --n $N --workers 1 --replan-every $Replan --out "$dir\expert_$Tag.json" --resume *>&1 | Out-File -Append -Encoding utf8 "$dir\expert_$Tag.log"
} elseif ($Step -eq "demo") {
    if (-not $Out) { $Out = "docs\reports\media\quarter_fold_parallel_expert.mp4" }
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
    $seedArgs = @(); if ($Seeds) { $seedArgs = @("--seeds") + ($Seeds -split " ") }
    $setArgs = @(); if ($Set) { $setArgs = @("--set", $Set, "--n", "$N") }
    & $py -u -m imitation.demo --backend isaac_friction --task quarter_fold --ckpt $Ckpt --replan-every $Replan `
        @seedArgs @setArgs --out $Out *>&1 | Out-File -Append -Encoding utf8 "$dir\demo_$Tag.log"
} else {
    $stopArgs = @(); if ($StopAtStage -ge 0) { $stopArgs = @("--stop-at-stage", "$StopAtStage") }
    & $py -u isaac/recovery_replay.py --backend isaac_friction --task quarter_fold --no-knock --seeds $Seeds @stopArgs `
        --trace-dir "$dir\traces_$Tag" --out "$dir\trace_$Tag.jsonl" *>&1 | Out-File -Append -Encoding utf8 "$dir\trace_$Tag.log"
}
"DONE qd_$Step $Tag exit $LASTEXITCODE $(Get-Date -Format s)"

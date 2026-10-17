# QD-2 at scale (2026-10-16): the gentle release at n = 100, paired, plus a half-fold A/B of the same release.
# Queues behind any running collect / training / q_retrain chain (heat cap: <= 2 Isaac processes, no GPU training
# alongside), then runs each phase as two Isaac processes of 1 worker x 8 envs, with thermal_guard alongside:
#   phase 1  quarter fold, sel_id_hard + sel_knock_arm, n = 100: gentle preset (default) || WORLDFOLD_QF_GENTLE=0
#   phase 2  half fold, same sets, n = 100: baseline || the gentle knobs (QF_GENTLE_GRASP / QF_GENTLE_EXPERT's release part)
#   phase 3  half fold settled placement miss (isaac/place_miss.py), 20 clean tune seeds: baseline || gentle
#   q_r1     (2026-10-17, by hand) quarter fold, same sets and seeds, the R1-adopted preset (fold-1 friction + lay-down
#            on top of gentle, WORLDFOLD_QF_FOLD1 unset = default on) -> expert_r1.json, paired against expert_gentle.json;
#            -Workers 2 runs it on both slots (2 x 8 envs) when the machine is otherwise clear
# Everything is resumable: re-run the same command. Outputs: outputs\imitation\isaac_friction\qd2\
#   powershell -ExecutionPolicy Bypass -File scripts\qd2_scale.ps1            # orchestrate (waits for a clear machine)
#   powershell -ExecutionPolicy Bypass -File scripts\qd2_scale.ps1 -Job q_gentle   # one job (used by the orchestrator)
param([string]$Job = "", [int]$N = 100, [int]$EnvsPerProc = 8, [int]$Workers = 1, [string]$MissSeeds = "610000:610020", [switch]$NoWait)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$dir = "outputs\imitation\isaac_friction\qd2"
New-Item -ItemType Directory -Force $dir | Out-Null
$env:WORLDFOLD_QF_SECOND = "parallel"
Remove-Item Env:WORLDFOLD_QF_START -ErrorAction SilentlyContinue
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "$EnvsPerProc"
$env:WORLDFOLD_QF_FOLD1 = "0"     # R1 (2026-10-17) adds fold-1 knobs to the quarter-fold preset; this A/B stays on the QD-2 preset
$GentleGrasp = "release_open=0.6,release_open_steps=80,jaw_open_rate=0.04"
$GentleExpert = "OPEN_DWELL=18,SETTLE_DWELL=6,RETREAT_SPEED=0.3"
$sets = @("sel_id_hard", "sel_knock_arm")

function Eval($task, $out) {
    & $py -u -m imitation.evaluate --backend isaac_friction --task $task --ckpt expert --sets @sets --n $N `
        --workers $Workers --replan-every 8 --out "$dir\$out.json" --resume
}

if ($Job) {
    $ErrorActionPreference = "Continue"
    "START $Job $(Get-Date -Format s)"
    switch ($Job) {
        "q_gentle" { $env:WORLDFOLD_QF_GENTLE = "1"      # the default preset, also set explicitly (belt and braces)
                     $env:WORLDFOLD_GRASP_KNOBS = $GentleGrasp
                     $env:WORLDFOLD_EXPERT_PARAMS = "$GentleExpert,STAGE1_ARC_HEIGHT=0.06,STAGE1_CARRY_SPEED=0.5"
                     Eval "quarter_fold" "expert_gentle" }
        "q_old"    { $env:WORLDFOLD_QF_GENTLE = "0"; Eval "quarter_fold" "expert_old" }
        "q_r1"     { Remove-Item Env:WORLDFOLD_QF_FOLD1 -ErrorAction SilentlyContinue     # the adopted preset, as shipped
                     Eval "quarter_fold" "expert_r1" }
        "h_base"   { Eval "half_fold" "half_base" }
        "h_gentle" { $env:WORLDFOLD_GRASP_KNOBS = $GentleGrasp; $env:WORLDFOLD_EXPERT_PARAMS = $GentleExpert
                     Eval "half_fold" "half_gentle" }
        "m_base"   { & $py -u isaac/place_miss.py --seeds $MissSeeds --out "$dir\half_miss_base.jsonl" }
        "m_gentle" { $env:WORLDFOLD_GRASP_KNOBS = $GentleGrasp; $env:WORLDFOLD_EXPERT_PARAMS = $GentleExpert
                     & $py -u isaac/place_miss.py --seeds $MissSeeds --out "$dir\half_miss_gentle.jsonl" }
        default    { throw "unknown job $Job" }
    }
    "DONE $Job exit $LASTEXITCODE $(Get-Date -Format s)"
    exit $LASTEXITCODE
}

# ---- orchestrator ----
function Busy() {
    # other Isaac / training work this machine's heat cap must not be stacked on: known launchers, anything run from
    # the Isaac venv or the isaac/ folder, and any large spawned worker (Isaac workers are 7-9 GB; a launcher's
    # Stop-Process leaves them alive). A HOLD file in the output folder (another session's slot) also blocks.
    $pat = "imitation\.data\.collect|imitation\.train|train_bc|imitation\.dagger|q_retrain\.ps1|imitation\.evaluate|recovery_replay|place_miss|imitation\.demo|qd_demo|r0_clean_fold|\.venv-isaac|isaac[\\/][a-z_0-9]+\.py"
    $mine = $PID
    if (Test-Path "$dir\HOLD") { "HOLD file $dir\HOLD (delete it to release)" }
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $mine -and $_.Name -notmatch "^(bash|sh|cmd|conhost)\.exe$" -and $_.CommandLine -and $_.CommandLine -notmatch "qd2_scale\.ps1|thermal_guard\.py" -and
        (($_.CommandLine -match $pat) -or ($_.Name -eq "python.exe" -and $_.WorkingSetSize -gt 1GB -and $_.CommandLine -match "spawn_main|multiprocessing"))
    } | ForEach-Object { "$($_.ProcessId) $($_.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length)))" }
}
function WaitClear() {
    if ($NoWait) { return }
    $said = $false
    while ($true) {
        $b = @(Busy)
        if ($b.Count -eq 0) { if ($said) { "CLEAR $(Get-Date -Format s)" }; return }
        if (-not $said) { "WAIT $(Get-Date -Format s) behind: $($b -join ' | ')"; $said = $true }
        Start-Sleep -Seconds 120
    }
}
function Guard() {
    $g = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -match "thermal_guard\.py" }
    if (-not $g) {
        Start-Process -WindowStyle Hidden -FilePath ".venv\Scripts\python.exe" -ArgumentList "scripts/thermal_guard.py --hot 90 --cool 84" `
            -RedirectStandardOutput "$dir\thermal_guard.log" -RedirectStandardError "$dir\thermal_guard.err"
        "GUARD started $(Get-Date -Format s)"
    }
}
function Phase($name, $jobs) {
    WaitClear
    Guard
    "PHASE $name $(Get-Date -Format s): $($jobs -join ', ')"
    $procs = @()
    foreach ($j in $jobs) {
        $procs += Start-Process -PassThru -WindowStyle Hidden -FilePath "powershell.exe" `
            -ArgumentList "-ExecutionPolicy Bypass -File scripts\qd2_scale.ps1 -Job $j -N $N -EnvsPerProc $EnvsPerProc -MissSeeds $MissSeeds" `
            -RedirectStandardOutput "$dir\$j.log" -RedirectStandardError "$dir\$j.err"
        Start-Sleep -Seconds 20       # stagger Kit start-up
    }
    $procs | Wait-Process
    "PHASE $name done $(Get-Date -Format s) exits: $(($procs | ForEach-Object { $_.ExitCode }) -join ',')"
}

"QD2 orchestrator start $(Get-Date -Format s) (N = $N, 1 x $EnvsPerProc per job, 2 jobs per phase)"
Phase "quarter" @("q_gentle", "q_old")
Phase "half" @("h_base", "h_gentle")
Phase "miss" @("m_base", "m_gentle")
"QD2 COMPLETE $(Get-Date -Format s)"

# Phase F, F3: the friction expert's gate on isaac_friction (LeHome's task sets), n = 100 per set, plus the Isaac
# check_resync (100 seeds x 3 modes, one env, resumable). Lane "eval" runs the three sets vectorised (2 processes x 8
# envs, the heat cap); lane "resync" is one more Isaac process, so run it only while "eval" is not running, or alone.
#   powershell -File scripts/f3_gate.ps1 -Lane eval   [-N 100] [-Out outputs/imitation/isaac_friction/f3] [-Expert "REGRASP=2"]
#   powershell -File scripts/f3_gate.ps1 -Lane resync
# Resumable: a set whose report exists is skipped; check_resync resumes from <out>/check_resync.json.rows.jsonl.
# -Expert sets IsaacArmExpert class attributes through WORLDFOLD_EXPERT_PARAMS (k=v,...), read by the teacher.
param([ValidateSet("eval", "resync")][string]$Lane = "eval", [int]$N = 100,
      [string]$Out = "outputs/imitation/isaac_friction/f3", [string]$Expert = "")
$ErrorActionPreference = "Stop"
. ./isaac/env_windows.ps1
New-Item -ItemType Directory -Force $Out | Out-Null
$env:WORLDFOLD_EXPERT_PARAMS = $Expert
if ($Lane -eq "eval") {
    $env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "8"      # user, 2026-10-07: 2 x 8 allowed (thermal guard on)
    foreach ($set in @("id_easy", "id_hard", "recovery")) {
        $json = "$Out/eval_expert_$set.json"
        if (Test-Path $json) { "skip $set (exists)"; continue }
        & .venv-isaac/Scripts/python.exe -u -m imitation.evaluate --backend isaac_friction --ckpt expert --sets $set `
            --n $N --workers 2 --out $json 2>&1 | Out-File -Encoding utf8 "$Out/eval_expert_$set.log"
        "$set exit $LASTEXITCODE"
    }
} else {
    $env:WORLDFOLD_N_ISAAC = "1"
    & .venv-isaac/Scripts/python.exe -u -m imitation.check_resync --backend isaac_friction --episodes $N `
        --out "$Out/check_resync.json" 2>&1 | Out-File -Encoding utf8 -Append "$Out/check_resync.log"
    "check_resync exit $LASTEXITCODE"
}

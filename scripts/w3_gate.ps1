# Phase W, W3: MuJoCo's expert on the isaac_weld backend, n = 100 per MuJoCo eval set (resumable: a set whose report
# exists is skipped). Run from the repo root: powershell -File scripts/w3_gate.ps1
param([int]$N = 100, [string[]]$Sets = @("id_easy", "id_hard", "recovery"), [string]$Out = "outputs/imitation/isaac_weld/w3")
$ErrorActionPreference = "Stop"
. ./isaac/env_windows.ps1
$out = $Out
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($set in $Sets) {
    $json = "$out/eval_expert_$set.json"
    if (Test-Path $json) { "skip $set (exists)"; continue }
    & .venv-isaac/Scripts/python.exe -u -m imitation.evaluate --backend isaac_weld --ckpt expert --sets $set --n $N --out $json 2>&1 |
        Out-File -Encoding utf8 "$out/eval_expert_$set.log"
    "$set exit $LASTEXITCODE"
}

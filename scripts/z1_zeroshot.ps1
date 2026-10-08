# Phase W, Z1: MuJoCo-trained checkpoints evaluated zero-shot on the isaac_weld backend (no training), n = 20 per
# set, 1 Isaac worker. Resumable: a report that exists is skipped. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/z1_zeroshot.ps1
param([int]$N = 20)
$ErrorActionPreference = "Stop"
. ./isaac/env_windows.ps1
$out = "outputs/imitation/isaac_weld/z1"
New-Item -ItemType Directory -Force $out | Out-Null
$runs = "outputs/imitation/runs"
# name, checkpoint, replan (each at its MuJoCo operating point)
$ckpts = @(@("diff_dagger_r4", "$runs/dagger_diff/round_1/final.pt", 4),
           @("diff_bc_s2_r8", "$runs/diff_v1_s2/final.pt", 8),
           @("vision_r2", "$runs/vision_t0_128/final.pt", 2))
foreach ($c in $ckpts) {
    $name, $ckpt, $replan = $c
    foreach ($set in "id_easy", "id_hard", "recovery") {
        $json = "$out/${name}_$set.json"
        if (Test-Path $json) { "skip $name $set"; continue }
        $ErrorActionPreference = "Continue"
        & .venv-isaac/Scripts/python.exe -u -m imitation.evaluate --backend isaac_weld --ckpt $ckpt --sets $set --n $N `
            --replan-every $replan --workers 1 --out $json *>&1 | Out-File -Encoding utf8 "$out/${name}_$set.log"
        $ErrorActionPreference = "Stop"
        "$name $set exit $LASTEXITCODE"
    }
}
"Z1 COMPLETE"

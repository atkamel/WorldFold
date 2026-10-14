# Phase Q sequencing (2026-10-14). A GPU training next to the 2 GPU-physics Isaac eval processes crashed PhysX
# ("Failed to get DOF velocities", CUDA stream errors) 53 min into the first noise-floor set. So sims wait until both
# Q1 trainings are done. Then: the quarter-fold expert smoke (Q3 step 1, ~30 min; the critical path), the Q0 noise
# floor, and q1_ab.ps1 (already waiting on q0\flips.txt) runs the Q1 evals. One sim job at a time.
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
while (-not (Test-Path "outputs\imitation\runs\isaac_v1_friction_q1_all\final.pt")) {
    "WAIT q1 trainings $(Get-Date -Format s)"; Start-Sleep -Seconds 120 }
& powershell -ExecutionPolicy Bypass -File scripts\q3_expert_smoke.ps1
& powershell -ExecutionPolicy Bypass -File scripts\q0_noise_floor.ps1

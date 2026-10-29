# F6 report videos: the friction-grasp policy (BC, replan 4) and the expert, 1 env, sequential
$repo = "C:\Users\ethan\Documents\Projects\watai\WorldFold"
Set-Location $repo
. (Join-Path $repo "isaac\env_windows.ps1")
$py = $env:ISAAC_PY
$env:WORLDFOLD_ISAAC_ENVS_PER_PROC = "1"
$ck = "outputs\imitation\runs\isaac_v1_friction_diff_s0\final.pt"
$M = "docs\reports\media"
function Run($name, [string[]]$a) {
    "START $name $(Get-Date -Format T)"
    & $py -u -m imitation.demo --backend isaac_friction @a --out "$M\$name.mp4" *> "outputs\imitation\isaac_friction\f4\demo_$name.log"
    "END $name exit $LASTEXITCODE $(Get-Date -Format T)"
}
Run "half_fold_friction_privileged_id_easy" @("--ckpt", $ck, "--replan-every", "4", "--set", "id_easy", "--n", "3")
Run "half_fold_friction_expert_knock_arm" @("--ckpt", "expert", "--set", "knock_arm", "--n", "3")
Run "half_fold_friction_privileged_knock_arm" @("--ckpt", $ck, "--replan-every", "4", "--set", "knock_arm", "--n", "3")
Run "half_fold_friction_privileged_id_hard" @("--ckpt", $ck, "--replan-every", "4", "--set", "id_hard", "--n", "3")
"ALL DONE"

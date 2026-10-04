# Friction-grasp bench (track G, IG.1-IG.2) in one Isaac process. Resumable: re-run the same command after a stop.
#   powershell -File scripts\grasp_bench.ps1 -Seeds 600000:600020 -Out outputs\isaac\grasp\baseline
#       [-Knobs particle_friction=1.0,gripper_drive.stiffness=50] [-Expert PINCH_INSET=0.01] [-Main <checkout with Assets\ and .venv-isaac\>]
# The code that runs is this checkout's (PYTHONPATH first); cwd is -Main, because LeHome resolves Assets\ from cwd.
param(
    [Parameter(Mandatory = $true)][string]$Seeds,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Knobs = "",
    [string]$Expert = "",
    [string]$Overshoot = "",
    [string]$Main = "C:\Users\ethan\Documents\Projects\watai\WorldFold"
)
$ErrorActionPreference = "Stop"
. (Join-Path $Main "isaac\env_windows.ps1")       # sets its own $repo (= $Main), so ours is set after it
$code = Split-Path -Parent $PSScriptRoot
$env:WORLDFOLD_N_ISAAC = "1"
$env:PYTHONPATH = $code
$py = Join-Path $Main ".venv-isaac\Scripts\python.exe"
$outAbs = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $code $Out }
if (-not $outAbs.StartsWith($code)) { throw "-Out must be inside $code" }
New-Item -ItemType Directory -Force $outAbs | Out-Null
$log = Join-Path $outAbs "bench.log"
$script = Join-Path $code "isaac\grasp_bench.py"
$cmd = "`"$py`" -u `"$script`" --seeds $Seeds --out `"$outAbs`""
if ($Knobs) { $cmd += " --knobs $Knobs" }        # k=v,... (no spaces, no quotes)
if ($Expert) { $cmd += " --expert $Expert" }
if ($Overshoot) { $cmd += " --overshoot=$Overshoot" }   # lx,ly,rx,ry; "=" so a leading minus is not an option
Add-Content -Encoding utf8 $log "== $(Get-Date -Format s) $cmd"
Push-Location $Main
try {
    cmd /c "$cmd >> `"$log`" 2>&1"
    Add-Content -Encoding utf8 $log "== exit $LASTEXITCODE $(Get-Date -Format s)"
} finally {
    Pop-Location
}

# Reproducible, hash-verified install of the local Isaac Sim 5.1 stack into .venv-isaac (Windows).
# Reviewed in isaac/INSTALL_REVIEW.md. Run from the repo root after the source checkouts exist in
# .venv-isaac\src (lehome-challenge @a805ad2f, IsaacLab fork @69f6fa54) and the venv + uv bootstrap:
#
#   py -3.11 -m venv .venv-isaac
#   .venv-isaac\Scripts\python.exe -m pip install --require-hashes --only-binary :all: `
#       "uv==0.12.22" --hash=sha256:1c45b1e544901b9eef14806ad964cd1aca49338748e3d8030e41c4aed8b17b00
#   powershell -File isaac\setup_windows.ps1
#
# Never run `uv sync` in the lehome checkout afterwards: it would put PyPI's CPU-only torch back.

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "env_windows.ps1")
$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repo ".venv-isaac"
$uv = Join-Path $venv "Scripts\uv.exe"
$py = $env:ISAAC_PY
$indexes = @("--index-url", "https://pypi.org/simple", "--extra-index-url", "https://pypi.nvidia.com")
$build = @("--build-constraints", (Join-Path $PSScriptRoot "build-constraints.txt"))

function Step($name, [scriptblock]$body) {
    Write-Host "== $name" ; & $body
    if ($LASTEXITCODE -ne 0) { throw "step failed: $name" }
}

# 1. LeHome's dependencies re-resolved for Windows (isaac/requirements-lehome.in, seeded with its uv.lock), every entry sha256-pinned;
#    isaacsim from NVIDIA's official index. The locks are complete resolved sets, so they install with --no-deps
#    (re-resolving would trip over LeHome's deliberate packaging==23.0 override)
Step "lehome lock" { & $uv pip install --python $py --require-hashes --no-deps @indexes @build `
    -r (Join-Path $PSScriptRoot "requirements-lehome-windows.lock") }
# 2. the IsaacLab fork's own requirements + imitation deps, compiled against LeHome's versions, hashed
Step "isaaclab + imitation lock" { & $uv pip install --python $py --require-hashes --no-deps @indexes @build `
    -r (Join-Path $PSScriptRoot "requirements-isaaclab-windows.lock") }
# 3. CUDA torch: PyPI's Windows torch is CPU-only; swap in the cu128 build (hash-pinned, official index)
Step "torch cu128" { & $uv pip install --python $py --require-hashes --no-deps --reinstall `
    --index-url https://download.pytorch.org/whl/cu128 -r (Join-Path $PSScriptRoot "requirements-torch-cu128-windows.lock") }
# 4. the two reviewed packages, editable, no dependency resolution
Step "editables" { & $uv pip install --python $py --no-deps `
    -e (Join-Path $venv "src\IsaacLab\source\isaaclab") -e (Join-Path $venv "src\lehome-challenge\source\lehome") }
# 5. the two robot asset files, at a pinned dataset revision, checked against the reviewed sha256
$env:HF_HUB_OFFLINE = "0"
$assets = Join-Path $venv "assets"
Step "assets" { & (Join-Path $venv "Scripts\hf.exe") download lehome/asset_challenge --repo-type dataset `
    --revision bea65fd960ad5a1bb3bd3fa77164b28001c08ef9 --local-dir $assets `
    robots/lerobot/so101_follower_good.usd robots/so101_new_calib.urdf }
$env:HF_HUB_OFFLINE = "1"
$usd = Join-Path $assets "robots\lerobot\so101_follower_good.usd"
$sha = (Get-FileHash $usd -Algorithm SHA256).Hash.ToLower()
if ($sha -ne "af8d2f281cff5c3bd3478fbce9341782979ce042872ca2a9a0638934a5bfa46f") { throw "asset hash mismatch: $sha" }
# lehome's ASSETS_ROOT is <git root of the cwd>/Assets; from this repo that is ./Assets, a gitignored junction
$link = Join-Path $repo "Assets"
if (-not (Test-Path $link)) { New-Item -ItemType Junction -Path $link -Target $assets | Out-Null }
Write-Host "== done"

# Session-only environment for the local Isaac Sim stack (dot-source it: `. isaac\env_windows.ps1`).
# Nothing here is persisted: no setx, no registry, no PATH edits. Every cache lands inside .venv-isaac.
# See isaac/INSTALL_REVIEW.md for why each setting is here.

$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repo ".venv-isaac"
$cache = Join-Path $venv "cache"

$env:ISAAC_PY = Join-Path $venv "Scripts\python.exe"
$env:UV_CACHE_DIR = Join-Path $cache "uv"
$env:PIP_CACHE_DIR = Join-Path $cache "pip"
$env:HF_HOME = Join-Path $cache "hf"
$env:TORCH_HOME = Join-Path $cache "torch"
$env:WARP_CACHE_PATH = Join-Path $cache "warp"
$env:HF_HUB_OFFLINE = "1"                # runtime is offline; setup_windows.ps1 lifts this for its one download
$env:LEHOME_DISABLE_KEYBOARD = "1"       # lehome's pynput keyboard device is never used
$env:OMNI_KIT_ACCEPT_EULA = "YES"        # accepted by the user on 2026-10-02 (docs/status.md pass log)
# Kit settings read by isaac.isaac_env.start_app: portable root (Kit's data, logs and shader cache stay in
# .venv-isaac\kit), anonymous telemetry off, and no online extension registry (only the pre-cached extscache).
$kit = Join-Path $venv "kit"
$env:WORLDFOLD_KIT_ARGS = "--portable-root $kit --/telemetry/enableAnonymousData=false " +
    "--/telemetry/enableAnonymousAppName=false --/exts/omni.kit.registry.nucleus/registryEnabled=false"

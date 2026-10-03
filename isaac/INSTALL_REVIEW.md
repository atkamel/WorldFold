# Isaac Sim install review (Windows, `.venv-isaac`)

Phase I, milestone I0.2, 2026-10-02. This is what gets installed for the local Isaac Sim
stack, where it comes from, how it's verified, and what the source review found. Nothing
third-party ran before this review was written.

## Artifacts

| artifact | source | pinned by | integrity | approx. download |
|---|---|---|---|---|
| `uv` (installer) | pypi.org (Astral) | exact version | pip hash | 20 MB |
| isaacsim 5.1.0.0 `[all,extscache]`, 25 wheels | **pypi.nvidia.com** (NVIDIA's official index) | LeHome `uv.lock` | sha256 from the lock, enforced | 4.0 GB |
| ~160 other Python packages | pypi.org | LeHome `uv.lock` | sha256 from the lock, enforced | ~0.6 GB |
| torch 2.7.0+cu128, torchvision 0.22.0+cu128 | download.pytorch.org/whl/cu128 (official) | exact version | sha256 from the index, enforced | ~3.3 GB |
| lehome-challenge source | github.com/lehome-official | commit `a805ad2f7ab52a4583066fc4ee5180459a7f9d15` | git object ids | fetched (in `.venv-isaac/src`) |
| LeHome IsaacLab fork source | github.com/lehome-official/IsaacLab | commit `69f6fa548c3a3520e3cb26ed24bb8abe60baeef3` | git object ids | fetched (in `.venv-isaac/src`) |
| `robots/lerobot/so101_follower_good.usd` | huggingface.co/datasets/lehome/asset_challenge | revision `bea65fd960ad5a1bb3bd3fa77164b28001c08ef9` | LFS sha256 `af8d2f28…a5bfa46f` | 23 MB |
| `robots/so101_new_calib.urdf` | same | same revision | git blob | 16 KB |

Totals: about 8 GB downloaded and about 15–20 GB installed, all inside `.venv-isaac/`.

## Findings and verdicts

**LeHome IsaacLab fork:** SAFE. It is upstream isaac-sim/IsaacLab `88d94ea8` plus 2 commits
across 3 files. The rest of the diff is reformatting. The fork's changes:
- `envs/direct_rl_env.py:118-119` forces PhysX GPU dynamics (`overwrite_gpu_setting(1)`).
  Benign; the particle cloth needs it.
- `app/app_launcher.py` always loads `apps/isaaclab.python.kit`, the full rendering
  experience, even when headless. Benign: a repo-local file. The cost is startup time and
  VRAM.
- `sensors/camera/tiled_camera.py` has its `--enable_cameras` check commented out. Benign.

`source/isaaclab/setup.py` only reads `extension.toml` and calls `setup()`: no cmdclass hooks,
downloads or shell-outs. The startup modules have no subprocess, socket, requests, urllib,
ctypes or base64 calls. `utils/string.py:154` has an `eval` that resolves config callables;
that is upstream code.

**lehome-challenge:** SAFE.
- `source/lehome/setup.py` is metadata only; `install_requires` is `psutil`.
- No pickle, ctypes, socket, exec or obfuscated strings anywhere in `source/` or `scripts/`.
- The only network code is `scripts/eval_policy/docker_policy.py`, the challenge's policy
  server client. It is not in our import path.
- The import-time side effect is `utils/logger.py:184`, which creates a `logs/` directory in
  the clone (inside `.venv-isaac`).
- Device classes (pynput keyboard, serial leader arm) only run when instantiated, and we never
  instantiate them. We set `LEHOME_DISABLE_KEYBOARD=1` anyway.

**Lock (`uv.lock`):** it has every registry entry sha256-pinned, with no git or URL
dependencies.
- It has win_amd64 cp311 wheels for isaacsim 5.1.0.0, torch 2.7.0 and open3d 0.19.
- The isaacsim wheel URLs point at the mirror `pypi.nvidia.cn`. Spot-checked hashes
  (`isaacsim-kernel`, `isaacsim-core`) are identical to NVIDIA's official
  `pypi.nvidia.com` index. We install from the official index with the lock's hashes
  enforced, so any byte difference fails the install.
- The lock's torch is the PyPI wheel, which is CPU-only on Windows. It gets replaced by the
  hash-pinned cu128 build. `uv sync` is never re-run afterwards, because it would restore
  the CPU wheel.
- The lock's `pinocchio` 0.4.3 is an unrelated pure-Python PyPI package. Our `isaac/pinch.py`
  doesn't use it.

**Assets:** the whole HF dataset has only USD/USDZ/GLB, JPG/PNG, JSON, TXT and the URDF. It
has no executables or pickles. We download only the 2 files the env loads, at a pinned
revision.

**Telemetry:** the experience file loads `omni.kit.telemetry` with anonymous usage data on
(`isaaclab.python.kit:210-212`). We turn it off per process:
`--/telemetry/enableAnonymousData=false --/telemetry/enableAnonymousAppName=false`.

**Online extension registry:** enabled by default (`isaaclab.python.kit:265`). We disable it
per process (`--/exts/omni.kit.registry.nucleus/registryEnabled=false`), so only the
pre-cached `extscache` extensions load.

## Scope

- Everything lives in `.venv-isaac/` (gitignored):
  - the venv
  - `src/` checkouts
  - `assets/`
  - `cache/`: uv, pip, HF, torch and warp caches, set per session by `isaac/env_windows.ps1`
  - `kit/`: Kit's portable root
- LeHome's `ASSETS_ROOT` is `<git root of the cwd>/Assets`, with no override. From WorldFold
  it resolves to `WorldFold/Assets`, so that path is a gitignored junction to
  `.venv-isaac/assets`.
- No admin rights. No registry, PATH, firewall, service or driver changes. Env vars are
  set only inside the session script.
- Runtime is offline: `HF_HUB_OFFLINE=1` and the extension registry is disabled.
- A before/after listing of `%USERPROFILE%`, `%LOCALAPPDATA%`, `%APPDATA%` and `%TEMP%`
  documents anything written outside `.venv-isaac` (I0.3).
- Uninstall: delete `.venv-isaac/` and the `Assets` junction, plus any leftovers that I0.3
  documents.

## Changes found while installing (I0.3, 2026-10-03)

- **LeHome's `uv.lock` was never resolved for Windows.** `isaacsim-core` 5.1.0.0 on Windows pins
  `pywin32==306`, `networkx==3.3`, `filelock==3.13.1` and `fsspec==2024.6.1`.
  - `isaac/requirements-lehome-windows.lock` is therefore re-resolved from LeHome's
    `pyproject.toml` dependencies (`isaac/requirements-lehome.in`, with its `numpy==1.26.0` and
    `packaging==23.0` overrides), seeded with LeHome's own lock.
  - Exactly those 4 versions changed, and 15 Windows-only packages were added (`ipython`,
    `ipywidgets` and their dependencies, pulled in by open3d on Windows).
  - Everything is still sha256-pinned and installed with `--require-hashes --no-deps`.
- **`isaaclab_tasks` and `isaaclab_assets`** (same fork checkout) are installed editable too.
  LeHome's `lehome.tasks` imports `isaaclab_tasks.utils`, and the kit file loads `isaaclab_tasks`
  as an extension.
  - Neither was touched by the fork; its only changes are the 3 files in `source/isaaclab`.
  - Their `setup.py` files have no install hooks. `isaaclab_tasks` adds `numba` and `protobuf`.
  - Their deps, plus `omegaconf` (used by lehome) and `h5py`, are in the hashed
    `isaac/requirements-isaaclab-windows.lock`.
- **`h5py==3.14.0`, not 3.16.** Kit loads `isaacsim.sensors.rtx`'s `hdf5.dll` (HDF5 1.14.6) at
  startup, and Windows reuses an already-loaded DLL by name. h5py 3.16 bundles HDF5 2.0.0 and
  dies on import with `0xc0000139` (entry point not found). 3.14.0 bundles 1.14.6.
- **D3D12 instead of Vulkan** (`--/app/vulkan=false` in `isaac/env_windows.ps1`). With driver
  616.56, Kit's Vulkan path access-violates in `omni.hydra.rtx.plugin.dll` at renderer start-up,
  even for a plain `SimulationApp` with no stage. D3D12 starts clean.
  - This was isolated step by step: with and without our Kit args, with and without
    IsaacLab, and with and without stage creation. Logs are in `outputs/isaac/debug/`.

## Measured footprint outside `.venv-isaac` (I0.3 scope check, 2026-10-03)

This compares `outputs/isaac/scope_before.txt` and `scope_after.txt`, top-level listings of
`%USERPROFILE%`, `%LOCALAPPDATA%`, `%APPDATA%` and `%TEMP%`. Kit honours `--portable-root` for
most of its data (`.venv-isaac\kit\{cache,data,logs}`), but not all of it.

| location | what | size | written by |
|---|---|---|---|
| `%USERPROFILE%\.nvidia-omniverse\{logs,pycache}` | Kit logs | 1.3 MB | every Kit start |
| `%LOCALAPPDATA%\ov\{cache,data}` | Kit texture/shader cache | ~560 MB | Kit start-up; partly the isolation runs without our Kit args |
| `%APPDATA%\uv\credentials` | empty uv credentials stub | 0 | uv |
| `%TEMP%\isaaclab\logs`, `%TEMP%\mat-debug-*.log`, `%TEMP%\x*.0`, `swx*` | IsaacLab/Kit temp and MDL logs | small | Kit |

These are all per-user caches and logs. There are no system, registry, PATH or driver
changes, and nothing in Program Files.

**Uninstall:**
1. Delete `.venv-isaac\` and the `Assets` junction (the junction only, not its target).
2. Delete the four locations above.

## Open risks

- Driver 616.56 is newer than Isaac Sim 5.1's validated Windows driver (580.88). TiledCamera
  hangs have been reported on this GPU family. The fallback is `isaacsim.sensors.camera.Camera`;
  a driver change would be the user's call.
- The NVIDIA Isaac Sim EULA must be accepted by the user before first launch.

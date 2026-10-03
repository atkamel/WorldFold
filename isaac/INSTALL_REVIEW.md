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

## Open risks

- Driver 616.56 is newer than Isaac Sim 5.1's validated Windows driver (580.88). TiledCamera
  hangs have been reported on this GPU family. The fallback is `isaacsim.sensors.camera.Camera`;
  a driver change would be the user's call.
- The NVIDIA Isaac Sim EULA must be accepted by the user before first launch.

"""Runs isaac/smoke_test.py (IsaacClothFoldEnv + half fold) on a Modal L40S, instead of WATcloud.

    pip install modal && modal setup                   # once
    modal run isaac/modal_isaac.py                     # state then hybrid, ~20-30 min on L40S
    modal run isaac/modal_isaac.py --modes hybrid
    modal volume get worldfold-isaac smoke/<stamp> .   # logs + hybrid RGB/depth frames

Isaac Sim 4.5.0 from NVIDIA's pip index, the same version as the WATcloud container the env was
written against. Isaac needs an RTX GPU (L40S / RTX PRO 6000); A100/H100/H200 cannot run it.
Each mode runs in its own process because Isaac Sim's World is one per process.
Long runs: add --detach, then check `modal app list` and stop the app when it is done.
"""

import pathlib

import modal

_REPO = pathlib.Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/WorldFold"
VOL_PATH = "/vol"

app = modal.App("worldfold-isaac")
vol = modal.Volume.from_name("worldfold-isaac", create_if_missing=True)

# Vulkan/GL user-space libs and env as in teacher/modal_teacher.py's isaac_image, which runs Isaac
# headless on Modal L40S; the NVIDIA driver itself is injected by Modal's GPU runtime.
image = (
    modal.Image.debian_slim(python_version="3.10")      # Isaac Sim 4.5 wheels are cp310 only
    .apt_install("libvulkan1", "vulkan-tools", "libegl1", "libgl1", "libglu1-mesa", "libglib2.0-0",
                 "libxt6", "libsm6", "libice6", "libxrandr2", "libxinerama1", "libxcursor1", "libxi6",
                 "libxext6", "libx11-6")
    # Isaac calls zenity for GUI dialogs, which blocks headless runs
    .run_commands("printf '#!/bin/sh\\nexit 0\\n' > /usr/local/bin/zenity && chmod +x /usr/local/bin/zenity")
    .pip_install("isaacsim[all,extscache]==4.5.0", extra_index_url="https://pypi.nvidia.com")
    .pip_install("gymnasium", "pillow")
    .env({"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y",
          "__GLX_VENDOR_LIBRARY_NAME": "nvidia", "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
          "XDG_RUNTIME_DIR": "/tmp"})
    .run_commands(
        "python -c \"import importlib.util, pathlib; "
        "kit = pathlib.Path(importlib.util.find_spec('isaacsim').origin).parent / 'kit'; "
        "(kit / 'EULA_ACCEPTED').write_text('yes'); print('EULA', kit)\"")
    # local code last, so editing it does not rebuild the Isaac layers
    .add_local_dir(str(_REPO / "isaac"), f"{REMOTE_REPO}/isaac", ignore=["assets", "__pycache__"])
    .add_local_dir(str(_REPO / "mujuco"), f"{REMOTE_REPO}/mujuco", ignore=["simulations", "__pycache__"])
    .add_local_dir(str(_REPO / "cloth_fold_rl"), f"{REMOTE_REPO}/cloth_fold_rl", ignore=["__pycache__"])
)


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, timeout=3600, volumes={VOL_PATH: vol})
def smoke(modes: str = "state,hybrid"):
    import shutil
    import subprocess
    import time

    # run from a writable copy: the first run fetches the SO101 MJCF into isaac/assets
    repo = "/tmp/WorldFold"
    shutil.copytree(REMOTE_REPO, repo, dirs_exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path(VOL_PATH, "smoke", stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip()
    print("GPU", gpu, flush=True)

    results = {}
    for mode in modes.split(","):
        cmd = ["python", "-u", f"{repo}/isaac/smoke_test.py", "--mode", mode]
        if mode == "hybrid":
            cmd += ["--save-frame", str(out_dir / "hybrid")]
        t0 = time.time()
        with open(out_dir / f"{mode}.log", "w") as log:
            proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log.write(line)
                print(f"[{mode}] {line.rstrip()}", flush=True)
            proc.wait()
        results[mode] = {"exit": proc.returncode, "wall_s": round(time.time() - t0, 1)}
        print("RESULT", mode, results[mode], flush=True)
        vol.commit()
    return {"stamp": stamp, "gpu": gpu, "results": results}


@app.local_entrypoint()
def main(modes: str = "state,hybrid"):
    out = smoke.remote(modes)
    print(out)
    print(f"logs and frames: modal volume get worldfold-isaac smoke/{out['stamp']} .")
    failed = [m for m, r in out["results"].items() if r["exit"] != 0]
    if failed:
        raise SystemExit(f"smoke test failed: {failed}")

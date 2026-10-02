"""Runs isaac/smoke_test.py (IsaacClothFoldEnv on LeHome's stack: contract, friction grasp, wrappers) on a Modal L40S.

    pip install modal && modal setup                   # once
    modal run isaac/modal_isaac.py                     # state and hybrid, one container each, in parallel
    modal run isaac/modal_isaac.py --modes state
    modal volume get worldfold-isaac smoke/<stamp> .   # logs + hybrid RGB/depth frames
    modal run isaac/modal_isaac.py::scripted --episodes 10    # scripted half-fold baseline, multi-seed

Same Isaac stack as isaac_image in teacher/modal_teacher.py (branch ROY-vla-teacher): LeHome's locked
lehome-challenge env (Isaac Sim 5.1.0, Python 3.11, torch 2.7.0) plus LeHome's IsaacLab fork and the
garment assets, without the pi0.5 teacher layers. WorldFold's env runs in that venv, so it shares one
Isaac install with LeHome's garment env. Isaac needs an RTX GPU (L40S / RTX PRO 6000); A100/H100/H200
cannot run it. The image build (~10 min) is CPU only and cached after the first run.
Long runs: add --detach, then check `modal app list` and stop the app when it is done.
"""

import pathlib

import modal

_REPO = pathlib.Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/WorldFold"
VOL_PATH = "/vol"

# Upstream lehome-challenge (Roy's teacher uses Ilia's fork, which adds only his eval loop: same pyproject and
# uv.lock). LeHome's IsaacLab fork forces PhysX GPU dynamics in DirectRLEnv, which the particle cloth needs on
# the CPU device; Roy cloned it unpinned, its HEAD is pinned here.
LEHOME_CHALLENGE = "https://github.com/lehome-official/lehome-challenge.git"
LEHOME_CHALLENGE_SHA = "a805ad2f7ab52a4583066fc4ee5180459a7f9d15"
ISAACLAB = "https://github.com/lehome-official/IsaacLab.git"
ISAACLAB_SHA = "69f6fa548c3a3520e3cb26ed24bb8abe60baeef3"
CH = "/opt/lehome-challenge"
PY = f"{CH}/.venv/bin/python"


def _checkout(url, sha, path):
    return f"git init -q {path} && git -C {path} fetch -q --depth 1 {url} {sha} && git -C {path} checkout -q FETCH_HEAD"


app = modal.App("worldfold-isaac")
vol = modal.Volume.from_name("worldfold-isaac", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential", "cmake", "ninja-build", "pkg-config", "python3-dev",
                 "libgl1-mesa-dev", "libglfw3", "libglfw3-dev", "libglew-dev", "xorg-dev", "libxi-dev",
                 "libxinerama-dev", "libxcursor1", "libxrandr2", "libglu1-mesa", "libvulkan1", "vulkan-tools",
                 "libxt6", "libsm6", "libice6", "libegl1", "libgl1", "libglvnd0")
    .pip_install("huggingface_hub[cli]>=0.30")
    # Isaac calls zenity for GUI dialogs, which blocks headless runs
    .run_commands("printf '#!/bin/sh\\nexit 0\\n' > /usr/local/bin/zenity && chmod +x /usr/local/bin/zenity",
                  "curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin", "UV_LINK_MODE": "copy"})
    .run_commands(_checkout(LEHOME_CHALLENGE, LEHOME_CHALLENGE_SHA, CH),
                  f"cd {CH} && uv sync")     # locked: isaacsim[all,extscache]==5.1.0, torch 2.7.0, lerobot 0.4.3
    .run_commands(
        f"echo yes > {CH}/.venv/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED",
        _checkout(ISAACLAB, ISAACLAB_SHA, f"{CH}/third_party/IsaacLab"),
        f"cd {CH} && bash -c 'source .venv/bin/activate && ./third_party/IsaacLab/isaaclab.sh -i none' || echo ISAACLAB_SH_FAILED_CONTINUING",
        f"cd {CH} && uv pip install 'setuptools<70' wheel --python {PY}",
        f"cd {CH} && uv pip install flatdict==4.0.1 --no-build-isolation --python {PY}",
        f"cd {CH} && uv pip install -e third_party/IsaacLab/source/isaaclab --python {PY}",
        f"cd {CH} && uv pip install -e source/lehome --python {PY}",
        # IsaacLab leaves warp-lang unpinned; 1.12.1 is the last release Isaac Sim 5.1 works with
        f"cd {CH} && uv pip install warp-lang==1.12.1 --python {PY}",
        f"{PY} -c \"import isaacsim, gymnasium; print('ISAACSIM_IMPORT_OK')\"",
    )
    .run_commands(f"cd {CH} && hf download lehome/asset_challenge --repo-type dataset --local-dir Assets")
    .env({"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y",
          "__GLX_VENDOR_LIBRARY_NAME": "nvidia", "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
          "XDG_RUNTIME_DIR": "/tmp"})
    # local code last, so editing it does not rebuild the Isaac layers
    .add_local_dir(str(_REPO / "isaac"), f"{REMOTE_REPO}/isaac", ignore=["__pycache__"])
    .add_local_dir(str(_REPO / "mujuco"), f"{REMOTE_REPO}/mujuco", ignore=["simulations", "__pycache__"])
    .add_local_dir(str(_REPO / "cloth_fold_rl"), f"{REMOTE_REPO}/cloth_fold_rl", ignore=["__pycache__"])
)


def _run(tag, args, out_dir):
    """Runs `PY args` from a copy of the repo, streaming its output and saving it to out_dir/<tag>.log on the volume."""
    import shutil
    import subprocess
    import time

    repo = "/tmp/WorldFold"
    shutil.copytree(REMOTE_REPO, repo, dirs_exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip()
    print(f"[{tag}] GPU {gpu}", flush=True)
    t0 = time.time()
    with open(out_dir / f"{tag}.log", "w") as log:
        proc = subprocess.Popen([PY, "-u"] + args, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        for line in proc.stdout:
            log.write(line)
            print(f"[{tag}] {line.rstrip()}", flush=True)
        proc.wait()
    vol.commit()
    return {"tag": tag, "exit": proc.returncode, "wall_s": round(time.time() - t0, 1)}


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, timeout=3600, volumes={VOL_PATH: vol})
def smoke(mode: str, stamp: str):
    out_dir = pathlib.Path(VOL_PATH, "smoke", stamp)
    args = ["isaac/smoke_test.py", "--mode", mode]
    if mode == "hybrid":
        args += ["--save-frame", str(out_dir / "hybrid")]
    return _run(mode, args, out_dir)


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, timeout=3600, volumes={VOL_PATH: vol})
def scripted(episodes: int = 10, seed_base: int = 0):
    """Multi-seed eval of the open-loop scripted half fold (cloth_fold_rl/scripted_half_fold.py) on Isaac."""
    import time
    out_dir = pathlib.Path(VOL_PATH, "scripted", time.strftime("%Y%m%d-%H%M%S"))
    args = ["-m", "cloth_fold_rl.scripted_half_fold", "--isaac", "--episodes", str(episodes),
            "--seed-base", str(seed_base)]
    return _run("scripted", args, out_dir)


@app.local_entrypoint()
def main(modes: str = "state,hybrid"):
    import time
    stamp = time.strftime("%Y%m%d-%H%M%S")
    names = modes.split(",")
    results = list(smoke.map(names, [stamp] * len(names)))
    for r in results:
        print("RESULT", r)
    print(f"logs and frames: modal volume get worldfold-isaac smoke/{stamp} .")
    failed = [r["tag"] for r in results if r["exit"] != 0]
    if failed:
        raise SystemExit(f"smoke test failed: {failed}")

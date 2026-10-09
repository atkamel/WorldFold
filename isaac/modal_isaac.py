"""Runs isaac/smoke_test.py (IsaacClothFoldEnv on LeHome's stack: contract, friction half fold, wrappers) on a Modal
A10G.

    pip install modal && modal setup                   # once
    modal run isaac/modal_isaac.py::main               # state and hybrid, one container each, in parallel
    modal run isaac/modal_isaac.py::main --modes state
    modal volume get worldfold-isaac smoke/<stamp> .   # logs + hybrid RGB/depth frames
    modal run isaac/modal_isaac.py::half_fold --episodes 3    # scripted friction half fold, with videos
    # the imitation pipeline (#17): one command per job, outputs on the volume under pipeline/
    modal run isaac/modal_isaac.py::pipeline --cmd "-m imitation.evaluate --backend isaac_friction ..."
    modal run --detach isaac/modal_isaac.py::pipeline --background --timeout-min 720 \
        --cmd "scripts/f4_retrain.py --no-vision --final-n 100 --noise-n 50 --episodes 200 --bc-steps 20000"

The image is built by isaac/install_lehome.sh, the same script as the WATcloud image (isaac/Dockerfile):
LeHome's locked lehome-challenge env (Isaac Sim 5.1.0, Python 3.11, torch 2.7.0) plus LeHome's IsaacLab
fork, the garment assets and the imitation pipeline's extra packages. Isaac needs an RTX GPU with RT cores
(A10G / L40S / RTX PRO 6000; A100/H100/H200 cannot run it) on the 580 driver branch (SUPPORTED_DRIVER).
The image build (~10 min) is CPU only and cached after the first run.
Long runs: add --detach, then check `modal app list` and stop the app when it is done.
"""

import pathlib

import modal

_REPO = pathlib.Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/WorldFold"
VOL_PATH = "/vol"
CH = "/opt/lehome-challenge"
PY = f"{CH}/.venv/bin/python"
# Isaac Sim 5.1 starts on NVIDIA's 580 driver branch. Ruby's 2026-10-04 probe (feat/isaac-half-fold, 5 containers per
# type): A10G and RTX PRO 6000 all on 580.95.05; L40S 3/5 and L4 5/5 on 610.57.04, where every Isaac start
# segfaults in librtx.scenedb (state mode too). _run refuses a 610 host up front instead of crashing mid-job.
SUPPORTED_DRIVER = "580."
PIPELINE_GPU = "A10G"

app = modal.App("worldfold-isaac")
vol = modal.Volume.from_name("worldfold-isaac", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # the same build as the WATcloud image (isaac/Dockerfile)
    .add_local_file(str(_REPO / "isaac" / "install_lehome.sh"), "/opt/install_lehome.sh", copy=True)
    .run_commands("bash /opt/install_lehome.sh")
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin", "UV_LINK_MODE": "copy",
          "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y",
          "__GLX_VENDOR_LIBRARY_NAME": "nvidia", "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
          "XDG_RUNTIME_DIR": "/tmp"})
    # local code last, so editing it does not rebuild the Isaac layers
    .add_local_dir(str(_REPO / "isaac"), f"{REMOTE_REPO}/isaac", ignore=["__pycache__"])
    .add_local_dir(str(_REPO / "mujuco"), f"{REMOTE_REPO}/mujuco", ignore=["simulations", "__pycache__"])
    .add_local_dir(str(_REPO / "cloth_fold_rl"), f"{REMOTE_REPO}/cloth_fold_rl", ignore=["__pycache__"])
    .add_local_dir(str(_REPO / "imitation"), f"{REMOTE_REPO}/imitation", ignore=["__pycache__"])
    .add_local_dir(str(_REPO / "scripts"), f"{REMOTE_REPO}/scripts", ignore=["__pycache__"])
)


def _run(tag, args, out_dir, isaac=True, limit_s=None):
    """Runs `PY args` from a copy of the repo, streaming its output and saving it to out_dir/<tag>.log on the volume.
    isaac=False skips the driver check, for commands that never start Isaac (training). limit_s kills the command
    after that many seconds."""
    import shutil
    import subprocess
    import threading
    import time

    repo = "/tmp/WorldFold"
    shutil.copytree(REMOTE_REPO, repo, dirs_exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip()
    print(f"[{tag}] GPU {gpu}", flush=True)
    driver = gpu.split(",")[-1].strip()
    if isaac and not driver.startswith(SUPPORTED_DRIVER):
        print(f"[{tag}] driver {driver}: Isaac Sim 5.1 crashes on it (needs {SUPPORTED_DRIVER}x), not starting", flush=True)
        return {"tag": tag, "exit": "unsupported_driver", "driver": driver}
    t0 = time.time()
    with open(out_dir / f"{tag}.log", "w") as log:
        proc = subprocess.Popen([PY, "-u"] + args, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        if limit_s:
            threading.Timer(limit_s, proc.kill).start()
        for line in proc.stdout:
            log.write(line)
            print(f"[{tag}] {line.rstrip()}", flush=True)
        proc.wait()
    vol.commit()
    return {"tag": tag, "exit": proc.returncode, "wall_s": round(time.time() - t0, 1)}


@app.function(image=image, gpu=PIPELINE_GPU, cpu=8, memory=32768, timeout=1200, volumes={VOL_PATH: vol})
def smoke(mode: str, stamp: str):
    out_dir = pathlib.Path(VOL_PATH, "smoke", stamp)
    args = ["isaac/smoke_test.py", "--mode", mode]
    if mode == "hybrid":
        args += ["--save-frame", str(out_dir / "hybrid")]
    return _run(mode, args, out_dir)


@app.function(image=image, gpu=PIPELINE_GPU, cpu=8, memory=32768, timeout=1200, volumes={VOL_PATH: vol})
def half_fold(episodes: int = 3):
    """The scripted two-arm friction half fold (isaac/half_fold_demo.py), with a video per episode."""
    import time
    out_dir = pathlib.Path(VOL_PATH, "half_fold", time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return _run("half_fold", ["isaac/half_fold_demo.py", "--episodes", str(episodes), "--video-dir", str(out_dir)],
                out_dir)


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


@app.function(image=image, gpu=PIPELINE_GPU, cpu=8, memory=65536, timeout=24 * 3600, volumes={VOL_PATH: vol})
def pipeline_job(args: list, tag: str, env: dict, timeout_min: int):
    """One imitation pipeline command, `python <args>`: a driver such as scripts/f4_retrain.py, or -m imitation.<module>.
    outputs/ and docs/reports/media/ live on the volume under pipeline/, so datasets, checkpoints and videos persist
    across runs and a resumed run picks them up. The volume is committed every 5 min, so a killed job loses at most
    that much. (After Ruby's imitation_job on feat/isaac-half-fold.)"""
    import os
    import threading
    os.environ.update(env)
    repo = pathlib.Path("/tmp/WorldFold")      # _run's working copy of the repo
    for rel in ("outputs", "docs/reports/media"):
        target = pathlib.Path(VOL_PATH, "pipeline", rel)
        target.mkdir(parents=True, exist_ok=True)
        link = repo / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(target)
    stop = threading.Event()

    def commit():
        while not stop.wait(300):
            vol.commit()
    threading.Thread(target=commit, daemon=True).start()
    try:
        # training never starts Isaac, so it runs on any driver
        return _run(tag, args, pathlib.Path(VOL_PATH, "pipeline", "logs"), isaac=args[:2] != ["-m", "imitation.train"],
                    limit_s=timeout_min * 60)
    finally:
        stop.set()


@app.local_entrypoint()
def pipeline(cmd: str, tag: str = "", timeout_min: int = 360, env_json: str = "{}", background: bool = False):
    """Runs one pipeline command on Modal (an A10G, 8 CPUs, 64 GB). timeout_min bounds what it can cost: the command
    is killed then. (The job's resources are fixed in its decorator: Function.with_options, which could set them per
    call, is missing from the Modal client this was written against, 1.3.5.) background: spawn and return at once
    (with `modal run --detach`); the job then runs on its own and its log lands on the volume. env_json: extra
    environment, e.g. '{"WORLDFOLD_ISAAC_ENVS_PER_PROC": "4"}'."""
    import json
    import shlex
    import time
    args = shlex.split(cmd)
    tag = tag or time.strftime("%Y%m%d-%H%M%S-") + (args[1].split(".")[-1] if args[0] == "-m" else pathlib.Path(args[0]).stem)
    log = f"modal volume get worldfold-isaac pipeline/logs/{tag}.log ."
    if background:
        call = pipeline_job.spawn(args, tag, json.loads(env_json), timeout_min)
        print(f"spawned {call.object_id}; log: {log}")
        return
    print("RESULT", pipeline_job.remote(args, tag, json.loads(env_json), timeout_min))
    print(f"log: {log}; outputs: modal volume get worldfold-isaac pipeline/outputs .")

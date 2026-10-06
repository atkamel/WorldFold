"""Runs isaac/smoke_test.py (IsaacClothFoldEnv on LeHome's stack: contract, friction half fold, wrappers) on a Modal
L40S.

    pip install modal && modal setup                   # once
    modal run isaac/modal_isaac.py                     # state and hybrid, one container each, in parallel
    modal run isaac/modal_isaac.py --modes state
    modal volume get worldfold-isaac smoke/<stamp> .   # logs + hybrid RGB/depth frames
    modal run isaac/modal_isaac.py::half_fold --episodes 3    # scripted friction half fold, with videos
    modal run isaac/modal_isaac.py::check_expert --shards 10  # IsaacHalfFoldExpert via the imitation pipeline
    modal run --detach isaac/modal_isaac.py::pipeline --cmd "imitation.data.collect --episodes 300 --workers 8" --cpu 32
                                                              # any imitation module, IMITATION_SIM=isaac

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
SUPPORTED_DRIVER = "580."     # NVIDIA driver branch Isaac Sim 5.1 starts on; see _run
# 2026-10-04 driver probe (5 containers each): A10G and RTX-PRO-6000 all on 580.95.05, L40S 3/5 and L4 5/5 on 610.57.04,
# where Isaac Sim 5.1 crashes. The imitation jobs take the GPU type as an option.
IMITATION_GPU = "A10G"
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
    .add_local_dir(str(_REPO / "imitation"), f"{REMOTE_REPO}/imitation", ignore=["__pycache__"])
)


def _run(tag, args, out_dir, isaac=True):
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
    driver = gpu.split(",")[-1].strip()
    if isaac and not driver.startswith(SUPPORTED_DRIVER):
        # seen 2026-10-03: on 610.57.04 hosts every Isaac Sim 5.1 start segfaults in librtx.scenedb (state mode too);
        # the same image runs on 580.95.05 hosts
        print(f"[{tag}] driver {driver}: Isaac Sim 5.1 crashes on it (needs {SUPPORTED_DRIVER}x), not starting", flush=True)
        return {"tag": tag, "exit": "unsupported_driver", "driver": driver}
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


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, timeout=1200, volumes={VOL_PATH: vol})
def smoke(mode: str, stamp: str):
    out_dir = pathlib.Path(VOL_PATH, "smoke", stamp)
    args = ["isaac/smoke_test.py", "--mode", mode]
    if mode == "hybrid":
        args += ["--save-frame", str(out_dir / "hybrid")]
    return _run(mode, args, out_dir)


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, timeout=1200, volumes={VOL_PATH: vol})
def half_fold(episodes: int = 3):
    """The scripted two-arm friction half fold (isaac/half_fold_demo.py), with a video per episode."""
    import time
    out_dir = pathlib.Path(VOL_PATH, "half_fold", time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return _run("half_fold", ["isaac/half_fold_demo.py", "--episodes", str(episodes), "--video-dir", str(out_dir)],
                out_dir)


def _imitation_run(tag, module_args, log_dir, isaac=True):
    """`python -m imitation.<module_args>` on Isaac (IMITATION_SIM=isaac) with outputs/imitation on the volume, so
    datasets and checkpoints persist across runs and resume after a kill. Commits the volume every few minutes."""
    import os
    import threading
    os.environ["IMITATION_SIM"] = "isaac"
    root = pathlib.Path(VOL_PATH, "imitation")
    root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def commit():
        while not stop.wait(300):
            vol.commit()
    threading.Thread(target=commit, daemon=True).start()
    # in _run's working copy of the repo (a link in REMOTE_REPO would be copied through, volume and all)
    link = pathlib.Path("/tmp/WorldFold/outputs/imitation")
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(root)
    try:
        return _run(tag, ["-m"] + module_args, log_dir, isaac)
    finally:
        stop.set()


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, timeout=1800, volumes={VOL_PATH: vol})
def expert_check(shard: int, stamp: str, episodes: int = 2, workers: int = 2, check_seed: int = -1, params: str = "",
                 variant: int = -1, seed_shard: int = -1):
    """IsaacHalfFoldExpert through the imitation pipeline: `episodes` clean expert demos on train seeds
    shard*episodes.., `workers` Isaac processes in this container. With check_seed >= 0, also isaac/expert_check.py
    on that seed (label agreement, phase timeline, video)."""
    import json
    import os
    if params:      # constants to override: "env.<NAME>" in isaac/isaac_env.py, the rest in isaac/half_fold_expert.py
        p = json.loads(params)
        os.environ["ISAAC_ENV_PARAMS"] = json.dumps({k[4:]: v for k, v in p.items() if k.startswith("env.")})
        os.environ["ISAAC_EXPERT_PARAMS"] = json.dumps({k: v for k, v in p.items()
                                                       if not k.startswith("env.") and k != "max_steps"})
        if "max_steps" in p:
            os.environ["IMITATION_MAX_STEPS"] = str(p["max_steps"])
    name = f"v{variant}_s{shard}" if variant >= 0 else f"s{shard}"
    seed_shard = shard if seed_shard < 0 else seed_shard
    log_dir = pathlib.Path(VOL_PATH, "imitation", "checks", stamp)
    results = [_imitation_run(f"collect_{name}", [
        "imitation.data.collect", "--episodes", str(episodes), "--seed-base", str(seed_shard * episodes),
        "--workers", str(workers), "--recovery-fraction", "0", "--version", f"check_{stamp}_{name}",
        "--root", str(pathlib.Path(VOL_PATH, "imitation", "datasets"))], log_dir)]
    if check_seed >= 0:
        import os
        os.environ["IMITATION_SIM"] = "isaac"
        video_dir = log_dir / name
        video_dir.mkdir(parents=True, exist_ok=True)
        results.append(_run(f"check_{name}_seed{check_seed}", ["isaac/expert_check.py", "--seeds", str(check_seed),
                                                                "--video-dir", str(video_dir)], log_dir))
    return results


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


@app.local_entrypoint()
def check_expert(shards: int = 10, episodes: int = 2, workers: int = 2, multi_shards: int = -1,
                 gpu: str = IMITATION_GPU):
    """multi_shards: how many shards run `workers` Isaac processes in their container (the rest run one), to compare
    throughput and check that concurrent Isaac apps coexist on one GPU. Default: all."""
    import time
    stamp = time.strftime("%Y%m%d-%H%M%S")
    multi = shards if multi_shards < 0 else multi_shards
    # the expert_check (label agreement + video) runs on the last shard, a single-worker one when there is one
    check = expert_check.with_options(gpu=gpu)
    calls = [check.spawn(k, stamp, episodes, workers if k < multi else 1, 0 if k == shards - 1 else -1)
             for k in range(shards)]
    for c in calls:
        for r in c.get():
            print("RESULT", r)
    print(f"logs and video: modal volume get worldfold-isaac imitation/checks/{stamp} .")


@app.function(image=image, gpu="L40S", cpu=8, memory=65536, timeout=6 * 3600, volumes={VOL_PATH: vol})
def imitation_job(cmd: list, tag: str, env: dict = None):
    """One imitation pipeline command (collect / train / evaluate / dagger / demo) in one container: its rollout
    workers are Isaac processes sharing this GPU, training uses it too. Outputs land in the volume's imitation/.
    imitation.train never starts Isaac, so it runs on any driver."""
    import os
    os.environ.update(env or {})
    return _imitation_run(tag, cmd, pathlib.Path(VOL_PATH, "imitation", "logs"), isaac=cmd[0] != "imitation.train")


@app.local_entrypoint()
def pipeline(cmd: str, cpu: int = 8, tag: str = "", gpu: str = IMITATION_GPU, memory_gb: int = 64,
             timeout_min: int = 360, max_steps: int = 0, background: bool = False, env_json: str = "{}"):
    """background: spawn the job and return at once (use with `modal run --detach`); it then runs on Modal on its own,
    independent of this machine, and its log and outputs land on the volume."""
    import shlex
    import time
    tag = tag or time.strftime("%Y%m%d-%H%M%S-") + cmd.split()[0].split(".")[-1]
    # cpu, memory_gb and timeout_min bound the run's cost, with the GPU's rate
    job = imitation_job.with_options(cpu=cpu, gpu=gpu, memory=memory_gb * 1024, timeout=timeout_min * 60)
    import json
    env = {"IMITATION_MAX_STEPS": str(max_steps)} if max_steps else {}
    env.update(json.loads(env_json))      # e.g. {"ISAAC_ENV_PARAMS": "{\"GRASP_MODE\": \"anchor\"}"}
    if background:
        call = job.spawn(shlex.split(cmd), tag, env)
        print(f"spawned {call.object_id}; log: modal volume get worldfold-isaac imitation/logs/{tag}.log .")
        return
    print("RESULT", job.remote(shlex.split(cmd), tag, env))
    print(f"log: modal volume get worldfold-isaac imitation/logs/{tag}.log .")


@app.local_entrypoint()
def sweep_expert(variants: str, shards_per: int = 3, episodes: int = 2, workers: int = 2, gpu: str = IMITATION_GPU,
                 video: bool = False, cpu: int = 8, memory_gb: int = 32, timeout_min: int = 30):
    """Each variant (a JSON list of ISAAC_EXPERT_PARAMS dicts) on the same train seeds 0..shards_per*episodes-1."""
    import json
    import time
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # cpu, memory_gb and timeout_min bound the sweep's cost, with the GPU's rate
    check = expert_check.with_options(gpu=gpu, cpu=cpu, memory=memory_gb * 1024, timeout=timeout_min * 60)
    # with video, each variant's first shard also runs isaac/expert_check.py on seed 0 (video, phase timeline)
    calls = [(v, check.spawn(v * shards_per + k, stamp, episodes, workers, 0 if video and k == 0 else -1,
                             json.dumps(p), v, k))
             for v, p in enumerate(json.loads(variants)) for k in range(shards_per)]
    for v, c in calls:
        for r in c.get():
            print("RESULT", v, r)
    print(f"logs: modal volume get worldfold-isaac imitation/checks/{stamp} .")


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={VOL_PATH: vol})
def merge_job(version: str, shards: list):
    """imitation.data.merge on the volume (no GPU: it only writes a manifest)."""
    return _imitation_run(f"merge_{version}", ["imitation.data.merge", "--version", version, "--from", *shards,
                                               "--root", str(pathlib.Path(VOL_PATH, "imitation", "datasets"))],
                          pathlib.Path(VOL_PATH, "imitation", "logs"), isaac=False)


@app.local_entrypoint()
def collect_shards(version: str, shards: int = 20, episodes: int = 35, workers: int = 4, cpu: int = 16,
                   recovery_fraction: float = 0.3, gpu: str = IMITATION_GPU, retries: int = 3, memory_gb: int = 64,
                   timeout_min: int = 360, stop_after_min: float = 0, max_steps: int = 0, env_json: str = "{}"):
    """Expert demos (imitation.data.collect) in `shards` containers, `workers` Isaac processes each, on train seeds
    shard*episodes..; shards that land on an unsupported driver are relaunched. Then merges the shards into
    <version> (successes) and <version>_failures."""
    root = str(pathlib.Path(VOL_PATH, "imitation", "datasets"))
    # the container size and timeout bound what a run can cost: shards x timeout x (GPU + cpu + memory rates)
    job = imitation_job.with_options(cpu=cpu, gpu=gpu, memory=memory_gb * 1024, timeout=timeout_min * 60)
    import json
    env = {"IMITATION_MAX_STEPS": str(max_steps)} if max_steps else {}
    env.update(json.loads(env_json))      # e.g. {"ISAAC_ENV_PARAMS": "{\"GRASP_MODE\": \"anchor\"}"}

    def launch(k):
        cmd = ["imitation.data.collect", "--episodes", str(episodes), "--seed-base", str(k * episodes),
               "--workers", str(workers), "--recovery-fraction", str(recovery_fraction),
               "--version", f"{version}_s{k:02d}", "--root", root]
        if stop_after_min:
            cmd += ["--stop-after-min", str(stop_after_min)]
        return job.spawn(cmd, f"collect_{version}_s{k:02d}", env)

    import time
    pending = {k: (launch(k), 0) for k in range(shards)}
    done, failed = [], []
    while pending:
        time.sleep(10)
        for k, (call, tries) in list(pending.items()):
            try:
                r = call.get(timeout=0)
            except (TimeoutError, modal.exception.TimeoutError):      # still running
                continue
            del pending[k]
            print("RESULT", k, r, flush=True)
            if r["exit"] == 0:
                done.append(k)
            elif r["exit"] == "unsupported_driver" and tries < retries:
                pending[k] = (launch(k), tries + 1)
            else:
                failed.append(k)
    names = [f"{version}_s{k:02d}" for k in sorted(done)]
    if names:
        print("RESULT merge", merge_job.remote(version, names))
        print("RESULT merge", merge_job.remote(f"{version}_failures", [n + "_failures" for n in names]))
    print(f"shards merged: {len(names)}/{shards}; failed: {failed}")


@app.local_entrypoint()
def merge(version: str, shards: str):
    """Merge frozen shard versions (comma-separated) into <version> on the volume."""
    print("RESULT", merge_job.remote(version, shards.split(",")))

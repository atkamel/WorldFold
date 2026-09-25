"""Phase 0: stand up the lehome_sim (pi0.5) policy server on Modal and benchmark it.

    PYTHONUTF8=1 python -m modal run teacher/modal_teacher.py::stage_checkpoint   # CPU, one-time
    PYTHONUTF8=1 python -m modal run teacher/modal_teacher.py::smoke              # L40S benchmark

Recipe follows the runbook: openpi pinned at c23745b, NEVER run his setup.sh, openpi's
uv overrides added by hand, exact nvidia wheels pinned (jax 0.5.3 cuDNN init bug),
JAX_PLATFORMS=cuda so a CPU fallback is a hard error, persistent JAX compile cache on
the volume. Everything the job produces lands on the volume (atomic write + commit).
"""

from __future__ import annotations

import os
import pathlib

import modal

_REPO = pathlib.Path(__file__).resolve().parent.parent      # WorldFold repo root
_TEACHER = _REPO / "teacher"
# Optional second sim (Roy's stats-worldfold mujuco/ dir, arms-on-flanks rig). Set ROY_SIM_DIR to use sim="roy".
_ROY_SIM = pathlib.Path(os.environ["ROY_SIM_DIR"]) if os.environ.get("ROY_SIM_DIR") else None
_SO101_VENDOR = _TEACHER / "vendor" / "so101_nexus"


def _ensure_so101_assets():
    """so101-nexus needs py>=3.12 but the teacher image is 3.11; WorldFold only needs its SO101 MJCF.
    Copy the assets out of a locally installed so101_nexus (pip install -r cloth_fold_rl/requirements.txt)."""
    if (_SO101_VENDOR / "assets" / "SO101" / "so101_new_calib.xml").exists():
        return True
    try:
        import shutil, so101_nexus  # noqa: E401
        src = pathlib.Path(so101_nexus.__file__).parent / "assets" / "SO101"
        (_SO101_VENDOR / "assets").mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, _SO101_VENDOR / "assets" / "SO101", dirs_exist_ok=True)
        (_SO101_VENDOR / "__init__.py").write_text("# vendored SO101 MJCF assets only (see modal_teacher.py)\n")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[modal_teacher] SO101 assets not vendored ({e!r}); MuJoCo rollout/physics functions will not work. "
              "Server/benchmark functions are unaffected.")
        return False

APP = "lehome-teacher"
REPO = "https://github.com/IliaLarchenko/lehome_solution"
OPENPI_SHA = "c23745b5ad24e98f66967ea795a07b2588ed6c79"
SRC = "/opt/lehome_solution"
VOL_PATH = "/vol"
CKPT = f"{VOL_PATH}/checkpoints/lehome_sim"
HF_REPO = "IliaLarchenko/lehome_sim"

app = modal.App(APP)
vol = modal.Volume.from_name("lehome-teacher", create_if_missing=True)

# ---- image ------------------------------------------------------------------
# Python 3.11 is required by his pyproject (>=3.11,<3.12). The nvidia CUDA/cuDNN
# user-space libs come from pip wheels (openpi README: no system CUDA needed);
# the driver is injected by Modal's GPU runtime.
PATCH_PYPROJECT = r'''
import re, pathlib
p = pathlib.Path("pyproject.toml"); s = p.read_text()
old = 'override-dependencies = ["lerobot>=0.4.3", "rerun-sdk>=0.22,<0.24"]'
new = ('override-dependencies = ["lerobot>=0.4.3", "rerun-sdk>=0.22,<0.24", '
       '"ml-dtypes==0.4.1", "tensorstore==0.1.74"]')
assert old in s, "override line not found -- pyproject changed upstream, stop and read"
s = s.replace(old, new)
# Hardware/video deps unused on the serve path (runbook breakage #3).
s = re.sub(r'^\s*"(pyrealsense2|torchcodec)[^"]*",?\s*\n', "", s, flags=re.M)
s = re.sub(r'^\s*(pyrealsense2|torchcodec)\s*=\s*\{[^\n]*\}\s*\n', "", s, flags=re.M)
p.write_text(s)
print("patched pyproject:\n" + "\n".join(l for l in s.splitlines() if "override" in l or "torch" in l))
'''

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "build-essential", "ffmpeg")
    .pip_install("numpy<2", "websockets>=14.1", "psutil", "huggingface_hub[cli]>=0.30", "nvidia-ml-py")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "git lfs install --skip-smudge",
        f"git clone {REPO} {SRC}",
        f"cd {SRC} && git submodule update --init openpi",
        f"cd {SRC} && test \"$(git -C openpi rev-parse HEAD)\" = {OPENPI_SHA} && echo OPENPI_SHA_OK",
        f"cd {SRC} && python3 - <<'PY'\n{PATCH_PYPROJECT}\nPY",
    )
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin", "GIT_LFS_SKIP_SMUDGE": "1",
          "UV_LINK_MODE": "copy", "PYTHONUNBUFFERED": "1"})
    .run_commands(
        f"cd {SRC} && uv venv --python 3.11 .venv && uv lock && uv sync",
        # jax 0.5.3 fresh-install cuDNN init failure (jax#27874): pin exactly openpi's lock.
        f"cd {SRC} && uv pip install nvidia-cudnn-cu12==9.5.1.17 nvidia-cublas-cu12==12.6.4.1 "
        "nvidia-cuda-runtime-cu12==12.6.77 nvidia-cuda-nvcc-cu12==12.9.41",
        f"cd {SRC} && .venv/bin/python -c \"import jax, torch, openpi, lehome_solution; "
        "from lehome_solution.training.config import get_config; "
        "print('IMPORT_OK jax', jax.__version__, 'torch', torch.__version__, get_config('pi_modified_bc_rl').name)\"",
    )
    .run_commands(f"cd {SRC} && uv pip install yappi==1.6.10 py-spy")   # thread-aware profilers
    # ---- WorldFold MuJoCo sim in the container's own python (co-located with the server) ----
    .apt_install("libegl1", "libgl1", "libglvnd0", "libosmesa6", "libglew2.2")
    # so101-nexus needs py>=3.12 (container is 3.11 for his stack); WorldFold only uses its SO101 MJCF -> vendored
    .pip_install("mujoco==3.10.0", "gymnasium==1.3.0", "imageio", "imageio-ffmpeg")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
)

# local code mounts (last layers; changing these files does not rebuild the image)
image = (image
         .add_local_dir(str(_REPO / "mujuco"), remote_path="/opt/sims/main/mujuco")
         .add_local_file(str(_TEACHER / "vendor" / "mjviser.py"), remote_path="/opt/vendor/mjviser.py")
         .add_local_file(str(_TEACHER / "wf_adapter.py"), remote_path="/opt/teacher/wf_adapter.py")
         .add_local_file(str(_TEACHER / "yappi_serve.py"), remote_path="/opt/teacher/yappi_serve.py"))
if modal.is_local() and _ensure_so101_assets():
    image = image.add_local_dir(str(_SO101_VENDOR), remote_path="/opt/vendor/so101_nexus")
if _ROY_SIM is not None:
    image = image.add_local_dir(str(_ROY_SIM), remote_path="/opt/sims/roy/mujuco")

# ---- volume staging (CPU only, never on the GPU clock) -----------------------
@app.function(image=image, volumes={VOL_PATH: vol}, timeout=3600, cpu=4, memory=8192)
def stage_checkpoint():
    import os, subprocess, pathlib
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    pathlib.Path(CKPT).mkdir(parents=True, exist_ok=True)
    subprocess.run(["hf", "download", HF_REPO, "--local-dir", CKPT], check=True)
    need = ["params/_METADATA", "params/manifest.ocdbt", "assets/inference_config.json",
            "assets/sim_rl/norm_stats.json", "assets/sim_rl/fast_tokenizer/tokenizer.json"]
    missing = [f for f in need if not pathlib.Path(CKPT, f).exists()]
    assert not missing, f"checkpoint incomplete: {missing}"
    total = sum(p.stat().st_size for p in pathlib.Path(CKPT).rglob("*") if p.is_file())
    vol.commit()
    print(f"CHECKPOINT_OK {total/1e9:.2f} GB at {CKPT}")
    return total


# ---- shared: start scripts/serve.py and wait for it ---------------------------
def _start_server(log_path, profile_out=None):
    import os, subprocess, sys, threading, time
    env = dict(os.environ, JAX_PLATFORMS="cuda", JAX_COMPILATION_CACHE_DIR=f"{VOL_PATH}/jax_cache",
               JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION="0.8", PYTHONUNBUFFERED="1")
    cmd = [f"{SRC}/.venv/bin/python"] + (["-m", "cProfile", "-o", str(profile_out)] if profile_out else []) + [
           "scripts/serve.py", "--port", "8000",
           "policy:checkpoint", "--policy.config", "pi_modified_bc_rl", "--policy.dir", CKPT]
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ready = threading.Event()

    def pump():
        for line in proc.stdout:
            logf.write(line); logf.flush()
            sys.stdout.write("[serve] " + line); sys.stdout.flush()
            if "listening on" in line:
                ready.set()
    threading.Thread(target=pump, daemon=True).start()
    t0 = time.time()
    while not ready.is_set():
        if proc.poll() is not None:
            raise SystemExit(f"server died during load, exit {proc.returncode}; see {log_path}")
        time.sleep(2)
    print(f"SERVER_READY load_s={time.time() - t0:.1f}", flush=True)
    return proc, logf


def _stop_server(proc, logf):
    import subprocess
    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
    logf.close()


# ---- teacher-driven WorldFold rollouts (sim co-located with the server) -------
@app.function(image=image, gpu="L40S", volumes={VOL_PATH: vol}, timeout=3 * 3600, cpu=8, memory=49152)
def rollout(episodes: int = 1, max_steps: int = 600, seed0: int = 0, video_every: int = 2, garment_id: int = -1,
            cloth_spacing: float = 0.0, cloth_shift_y: float = 0.0, tag: str = "", run_name: str = "",
            sim: str = "main"):
    """run_name set -> resumable cache mode: fixed dir /vol/cache/<run_name>, one file per seed,
    seeds whose completion marker (ep_XXXX.json with sha256) exists are skipped."""
    import json, pathlib, sys, time
    assert sim in ("roy", "main"), sim
    sys.path.insert(0, "/opt/vendor"); sys.path.insert(0, f"/opt/sims/{sim}"); sys.path.insert(0, "/opt/teacher")
    import os, subprocess
    probe = ("import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'); "
             "r = mujoco.Renderer(m, 64, 64); r.render(); print('ok')")
    egl_ok = subprocess.run(["python", "-c", probe], env=dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl"),
                            capture_output=True, text=True, timeout=120).returncode == 0
    gl = "egl" if egl_ok else "osmesa"
    os.environ["MUJOCO_GL"] = gl; os.environ["PYOPENGL_PLATFORM"] = gl
    print(f"MUJOCO_GL={gl} sim={sim}", flush=True)
    import wf_adapter as A

    import hashlib
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{sim}" + (f"-{tag}" if tag else "") + f"-s{seed0}"
    out_dir = (pathlib.Path(VOL_PATH, "cache", run_name) if run_name
               else pathlib.Path(VOL_PATH, "rollouts", stamp))
    out_dir.mkdir(parents=True, exist_ok=True)

    def done(seed):
        m = out_dir / f"ep_{seed:04d}.json"; z = out_dir / f"ep_{seed:04d}.npz"
        if not (m.exists() and z.exists()):
            return False
        try:
            return json.loads(m.read_text())["sha256"] == hashlib.sha256(z.read_bytes()).hexdigest()
        except Exception:  # noqa: BLE001  torn/invalid marker -> redo the episode
            return False

    todo = [seed0 + e for e in range(episodes) if not (run_name and done(seed0 + e))]
    if not todo:
        print(f"SKIP all {episodes} seeds from {seed0} already cached in {out_dir}", flush=True)
        return []
    logs = pathlib.Path(VOL_PATH, "logs"); logs.mkdir(exist_ok=True)
    proc, logf = _start_server(logs / f"serve-rollout-{stamp}.log")
    cfg_all = json.load(open(f"{CKPT}/assets/inference_config.json"))["per_garment_type"]
    summaries = []
    try:
        env = A.make_env(max_episode_steps=max_steps, cloth_spacing=cloth_spacing or None); cams = A.Cameras(env)
        t = time.perf_counter(); cams.render(); print(f"RENDER_OK 3 cams in {time.perf_counter() - t:.2f}s (MUJOCO_GL={gl})", flush=True)
        teacher = A.WsTeacher()
        for seed in todo:
            t0 = time.perf_counter()
            summary, arrays, frames = A.run_episode(env, cams, teacher, cfg_all, seed=seed, max_steps=max_steps,
                                                    garment_id=None if garment_id < 0 else garment_id,
                                                    video_every=video_every, log=lambda s: print(s, flush=True),
                                                    cloth_shift_y=cloth_shift_y)
            summary["wall_s"] = time.perf_counter() - t0
            npz, mp4 = A.save_episode(str(out_dir), f"ep_{seed:04d}", summary, arrays, frames)
            # completion marker written LAST (atomic), carries the npz checksum -> resume trusts disk truth
            marker = dict(summary, sha256=hashlib.sha256(pathlib.Path(npz).read_bytes()).hexdigest(),
                          npz=pathlib.Path(npz).name, finished_at=time.time())
            tmp = out_dir / f"ep_{seed:04d}.json.tmp"; tmp.write_text(json.dumps(marker))
            os.replace(tmp, out_dir / f"ep_{seed:04d}.json")
            vol.commit()
            summaries.append(summary)
            print("EPISODE " + json.dumps(summary), flush=True)
        if not run_name:
            (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2)); vol.commit()
    finally:
        _stop_server(proc, logf)
        print(f"DONE rollouts={len(summaries)} dir={out_dir}", flush=True)
    return summaries


@app.local_entrypoint()
def pilot(n: int = 3, seed0: int = 0, max_steps: int = 450, video_every: int = 10,
          cloth_spacing: float = 0.045, cloth_shift_y: float = 0.0, tag: str = "pilot", sim: str = "main"):
    """N episodes in N parallel containers (each with its own server). Prints one summary line per episode."""
    import json
    args = [(1, max_steps, seed0 + i, video_every, -1, cloth_spacing, cloth_shift_y, tag, "", sim) for i in range(n)]
    for res in rollout.starmap(args):
        for s in res:
            print("PILOT " + json.dumps(s), flush=True)


@app.local_entrypoint()
def cache(run_name: str = "cache1", n_seeds: int = 60, seed0: int = 1000, per_container: int = 5,
          max_steps: int = 450, video_every: int = 0, cloth_spacing: float = 0.045, sim: str = "main"):
    """Resumable parallel caching: n_seeds episodes split over ceil(n/per_container) L40S containers.
    Re-running the same command skips every seed that already has a verified completion marker."""
    import json
    chunks = [(min(per_container, n_seeds - i), max_steps, seed0 + i, video_every, -1, cloth_spacing, 0.0, "", run_name, sim)
              for i in range(0, n_seeds, per_container)]
    print(f"CACHE run={run_name} seeds {seed0}..{seed0 + n_seeds - 1} in {len(chunks)} containers", flush=True)
    n = 0
    for res in rollout.starmap(chunks, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"CONTAINER_FAILED {type(res).__name__}: {res}", flush=True); continue
        for s in res:
            n += 1
            print(f"CACHED {n} " + json.dumps({k: s[k] for k in ('seed', 'fold_score_max', 'steps', 'wall_s')}), flush=True)
    print(f"CACHE_DONE new_episodes={n}", flush=True)


# ---- GPU smoke + benchmark ---------------------------------------------------
@app.function(image=image, gpu="L40S", volumes={VOL_PATH: vol}, timeout=3600, cpu=8, memory=49152)
def smoke(batch_sizes: str = "1,4", candidates: str = "1,2,3", calls: int = 12):
    import asyncio, base64, json, os, pathlib, subprocess, sys, threading, time
    import numpy as np, psutil, websockets

    t0 = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    logs = pathlib.Path(VOL_PATH, "logs"); logs.mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path(VOL_PATH, "results"); results_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"serve-{stamp}.log"
    cache = pathlib.Path(VOL_PATH, "jax_cache"); cache.mkdir(exist_ok=True)
    cache_before = sorted(p.name for p in cache.rglob("*") if p.is_file())

    env = dict(os.environ,
               JAX_PLATFORMS="cuda",
               JAX_COMPILATION_CACHE_DIR=str(cache),
               JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",   # so nvidia-smi shows real use
               XLA_PYTHON_CLIENT_MEM_FRACTION="0.8",
               PYTHONUNBUFFERED="1")
    cmd = [f"{SRC}/.venv/bin/python", "scripts/serve.py", "--port", "8000",
           "policy:checkpoint", "--policy.config", "pi_modified_bc_rl", "--policy.dir", CKPT]
    print("SERVER_CMD", " ".join(cmd), flush=True)
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ready = threading.Event()

    def pump():
        for line in proc.stdout:
            logf.write(line); logf.flush()
            sys.stdout.write("[serve] " + line); sys.stdout.flush()
            if "listening on" in line:
                ready.set()
    threading.Thread(target=pump, daemon=True).start()

    def gpu_mem():
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                           "--format=csv,noheader,nounits"], text=True)
            u, t = out.strip().split(",")
            return {"used_mb": int(u), "total_mb": int(t)}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def enc(a):
        a = np.ascontiguousarray(a)
        return {"base64": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": str(a.dtype)}

    GARMENT = "top_long"
    cfg_all = json.load(open(f"{CKPT}/assets/inference_config.json"))["per_garment_type"][GARMENT]
    rng = np.random.default_rng(0)
    imgs = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for k in ("top_rgb", "left_rgb", "right_rgb")}
    state = np.array([-1.1363, 0, 0, 1.0, -0.4, 0.0, 1.1363, 0, 0, 1.0, 0.3, 0.0], np.float32)

    def make_msg(n_cand, sid):
        cfg = dict(cfg_all); cfg["num_rollout_candidates"] = n_cand
        return json.dumps({"type": "infer_chunk", "session_id": sid, "garment_type_id": 0,
                           "observation.images.top_rgb": enc(imgs["top_rgb"]),
                           "observation.images.left_rgb": enc(imgs["left_rgb"]),
                           "observation.images.right_rgb": enc(imgs["right_rgb"]),
                           "observation.state": state.tolist(), "inference_config": cfg})

    async def one_call(msg, timeout):
        async with websockets.connect("ws://localhost:8000", max_size=100 * 1024 * 1024,
                                      open_timeout=60, ping_interval=None) as ws:
            t = time.perf_counter()
            await ws.send(msg)
            r = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            dt = time.perf_counter() - t
        if "error" in r:
            raise RuntimeError(r["error"])
        return dt, r

    async def concurrent(n, n_cand, timeout):
        # n independent observations in flight at once -> exercises his InferenceBatcher
        msgs = [make_msg(n_cand, f"s{i}") for i in range(n)]
        t = time.perf_counter()
        outs = await asyncio.gather(*(one_call(m, timeout) for m in msgs))
        wall = time.perf_counter() - t
        return wall, [o[0] for o in outs], outs[0][1]

    # -- wait for server --
    while not ready.is_set():
        if proc.poll() is not None:
            raise SystemExit(f"server died during load, exit {proc.returncode}; see {log_path}")
        time.sleep(2)
    load_s = time.time() - t0
    print(f"SERVER_READY load_s={load_s:.1f} gpu={gpu_mem()}", flush=True)
    sproc = psutil.Process(proc.pid)
    sproc.cpu_percent(None)

    async def ping():
        async with websockets.connect("ws://localhost:8000", max_size=100 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"type": "ping"})); return json.loads(await ws.recv())
    print("PING", asyncio.run(ping()), flush=True)

    res = {"stamp": stamp, "gpu": "L40S", "garment": GARMENT, "cfg": cfg_all, "load_s": load_s,
           "gpu_mem_after_load": gpu_mem(), "runs": []}

    def save():
        tmp = results_dir / f"phase0-{stamp}.json.tmp"
        tmp.write_text(json.dumps(res, indent=2)); os.replace(tmp, results_dir / f"phase0-{stamp}.json")
        vol.commit()

    try:
        for n_cand in [int(x) for x in candidates.split(",")]:
            for bs in [int(x) for x in batch_sizes.split(",")]:
                # first call at a new (batch, candidates) shape = JIT compile
                t = time.perf_counter()
                wall, lats, r = asyncio.run(concurrent(bs, n_cand, timeout=1800))
                compile_s = time.perf_counter() - t
                acts = np.asarray(r["actions"], np.float32)
                print(f"COMPILED bs={bs} cand={n_cand} first_call_s={compile_s:.1f} actions{acts.shape} "
                      f"exec={r.get('execute_in_n_steps')} keep={r.get('actions_to_keep')} "
                      f"garment_pred={r.get('garment_type_pred')} gpu={gpu_mem()}", flush=True)
                walls, per = [], []
                sproc.cpu_percent(None)
                for _ in range(calls):
                    w, l, _r = asyncio.run(concurrent(bs, n_cand, timeout=300))
                    walls.append(w); per.extend(l)
                cpu = sproc.cpu_percent(None)
                walls = np.array(walls); per = np.array(per)
                run = {"batch": bs, "candidates": n_cand, "first_call_s": compile_s,
                       "wall_median_s": float(np.median(walls)), "wall_p95_s": float(np.percentile(walls, 95)),
                       "req_median_s": float(np.median(per)), "req_p95_s": float(np.percentile(per, 95)),
                       "obs_per_s": float(bs / np.median(walls)), "server_cpu_pct": cpu,
                       "gpu_mem": gpu_mem(), "actions_shape": list(acts.shape),
                       "actions_finite": bool(np.isfinite(acts).all()),
                       "actions_min": float(acts.min()), "actions_max": float(acts.max())}
                res["runs"].append(run); save()
                print("RUN", json.dumps(run), flush=True)
    finally:
        cache_after = sorted(p.name for p in cache.rglob("*") if p.is_file())
        res["jax_cache_entries_before"] = len(cache_before)
        res["jax_cache_entries_after"] = len(cache_after)
        res["total_s"] = time.time() - t0
        save()
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        print(f"DONE total_s={res['total_s']:.0f} cache_entries {len(cache_before)}->{len(cache_after)} "
              f"results={results_dir}/phase0-{stamp}.json", flush=True)
    return res


# =============================================================================================
# Profiling: physics (CPU-only container) and teacher server (H200), run in parallel.
#   PYTHONUTF8=1 python -m modal run teacher/modal_teacher.py::profile_all
# =============================================================================================
@app.function(image=image, volumes={VOL_PATH: vol}, timeout=1800, cpu=4, memory=8192)
def physics_profile(sim: str = "main", episode: str = "", steps: int = 150, cloth_spacing: float = 0.0):
    """Replay a recorded teacher episode (real arm/cloth contact) and break the physics cost down."""
    import glob, json, os, pathlib, sys, time
    import numpy as np
    sys.path.insert(0, "/opt/vendor"); sys.path.insert(0, f"/opt/sims/{sim}"); sys.path.insert(0, "/opt/teacher")
    import mujoco
    import mujuco.sim_main as SM
    import wf_adapter as A

    z = np.load(glob.glob(f"{VOL_PATH}/rollouts/{episode}/ep_*.npz")[0])
    summ = json.loads(str(z["summary"]))
    acts, tg = z["env_actions"][:steps], z["targets"][:steps]
    base_dt = SM.ARM_TIMESTEP
    out = {"sim": sim, "episode": episode, "steps": int(len(acts)), "variants": []}
    stages = (("collision", mujoco.mj_collision),
              ("fwdPosition", mujoco.mj_fwdPosition),
              ("fwdVelocity", mujoco.mj_fwdVelocity),
              ("fwdActuation", mujoco.mj_fwdActuation),
              ("fwdAcceleration", mujoco.mj_fwdAcceleration),
              ("fwdConstraint_solver", mujoco.mj_fwdConstraint))

    def replay(dt, iters=None):
        SM.ARM_TIMESTEP = dt
        env = A.make_env(max_episode_steps=len(acts) + 5, cloth_spacing=cloth_spacing or None)
        m, d = env.model, env.data
        if iters:
            m.opt.iterations = iters
        env.reset(seed=summ["seed"], options={"task": 0})
        A.set_teacher_init_pose(env)
        step_wall, stage, niter, ncon, nefc = 0.0, {}, [], [], []
        exploded = False
        for t, a in enumerate(acts):
            env._grip_target["left_"], env._grip_target["right_"] = float(tg[t, 5]), float(tg[t, 11])
            if t % 25 == 10:
                for name, fn in stages:
                    t0 = time.perf_counter()
                    for _ in range(10):
                        fn(m, d)
                    stage.setdefault(name, []).append((time.perf_counter() - t0) / 10 * 1e3)
                t0 = time.perf_counter()
                for _ in range(10):
                    mujoco.mj_forward(m, d)
                stage.setdefault("mj_forward_total", []).append((time.perf_counter() - t0) / 10 * 1e3)
            t0 = time.perf_counter()
            env.step(a)
            step_wall += time.perf_counter() - t0
            niter.append(int(d.solver_niter[0])); ncon.append(int(d.ncon)); nefc.append(int(d.nefc))
            if not np.all(np.isfinite(d.qpos)) or env._failed():
                exploded = True
                break
        t0 = time.perf_counter()
        for _ in range(50):
            mujoco.mj_step(m, d)
        mj_step_ms = (time.perf_counter() - t0) / 50 * 1e3
        n_done = len(niter)
        env_ms = step_wall / max(n_done, 1) * 1e3
        res = {"dt_ms": dt * 1e3, "iterations_cap": int(m.opt.iterations), "substeps_per_control": int(env.n_substeps),
               "control_steps": n_done, "exploded": exploded,
               "env_step_ms": env_ms, "mj_step_ms": mj_step_ms,
               "python_overhead_ms_per_control": env_ms - mj_step_ms * env.n_substeps,
               "stage_ms": {k: float(np.mean(v)) for k, v in stage.items()},
               "solver_iter_mean": float(np.mean(niter)), "solver_iter_max": int(np.max(niter)),
               "ncon_mean": float(np.mean(ncon)), "nefc_mean": float(np.mean(nefc)),
               "nv": int(m.nv), "nbody": int(m.nbody), "ngeom": int(m.ngeom), "neq": int(m.neq),
               "fold_score_end": float(env._fold_score()),
               "corners_end": d.xpos[env._corner_ids].tolist(),
               "episode_450_est_s": env_ms * 450 / 1e3}
        print("PHYS " + json.dumps({k: v for k, v in res.items() if k != "corners_end"}), flush=True)
        return res

    for dt, iters in ((0.0005, None), (0.001, None), (0.002, None), (0.0005, 30)):
        try:
            out["variants"].append(replay(dt, iters))
        except Exception as e:  # noqa: BLE001
            out["variants"].append({"dt_ms": dt * 1e3, "iterations_cap": iters, "error": repr(e)})
            print(f"PHYS_ERROR dt={dt} iters={iters} {e!r}", flush=True)
    SM.ARM_TIMESTEP = base_dt
    if "corners_end" in out["variants"][0]:
        ref = np.array(out["variants"][0]["corners_end"])
        for v in out["variants"][1:]:
            if "corners_end" in v:
                v["corner_drift_vs_ref_cm"] = float(np.linalg.norm(np.array(v["corners_end"]) - ref, axis=1).max() * 100)
    pdir = pathlib.Path(VOL_PATH, "profile"); pdir.mkdir(exist_ok=True)
    f = pdir / f"physics-{sim}-{time.strftime('%H%M%S')}.json"
    tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(out, indent=2)); os.replace(tmp, f); vol.commit()
    brief = [{k: v.get(k) for k in ("dt_ms", "iterations_cap", "env_step_ms", "mj_step_ms", "exploded",
                                    "corner_drift_vs_ref_cm", "episode_450_est_s")} for v in out["variants"]]
    print("PHYS_DONE " + sim + " " + json.dumps(brief), flush=True)
    return out


@app.function(image=image, gpu="H200", volumes={VOL_PATH: vol}, timeout=3600, cpu=16, memory=65536)
def server_profile(concurrency: str = "1,4,8,16,32", candidates: int = 3, rounds: int = 8):
    """His websocket server on an H200: latency/throughput vs concurrency, GPU util, and a cProfile of the server."""
    import asyncio, base64, io, json, os, pathlib, pstats, signal, subprocess, threading, time
    import numpy as np, psutil, websockets

    stamp = time.strftime("%H%M%S")
    pdir = pathlib.Path(VOL_PATH, "profile"); pdir.mkdir(exist_ok=True)
    prof_path = pdir / f"server-{stamp}.prof"
    cfg = dict(json.load(open(f"{CKPT}/assets/inference_config.json"))["per_garment_type"]["pant_long"])
    cfg.pop("k_execute", None); cfg.pop("num_steps", None); cfg["num_rollout_candidates"] = candidates

    def enc(a):
        a = np.ascontiguousarray(a)
        return {"base64": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": str(a.dtype)}
    rng = np.random.default_rng(0)
    imgs = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for k in ("top_rgb", "left_rgb", "right_rgb")}
    state = [-1.24, -1.69, 1.49, 1.05, -0.08, -0.01, 1.24, -1.69, 1.49, 1.05, -0.08, -0.01]

    def msg(i):
        body = {"type": "infer_chunk", "session_id": f"p{i}", "garment_type_id": 2,
                "observation.state": state, "inference_config": cfg}
        for k, v in imgs.items():
            body[f"observation.images.{k}"] = enc(v)
        return json.dumps(body)

    m0 = msg(0)
    t0 = time.perf_counter()
    for _ in range(20):
        dd = json.loads(m0)
        for k in ("top_rgb", "left_rgb", "right_rgb"):
            v = dd[f"observation.images.{k}"]
            np.frombuffer(base64.b64decode(v["base64"]), dtype=v["dtype"]).reshape(v["shape"])
    decode_ms = (time.perf_counter() - t0) / 20 * 1e3
    t0 = time.perf_counter()
    for _ in range(20):
        msg(0)
    encode_ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"PROTOCOL request_MB={len(m0) / 1e6:.2f} server_decode_ms={decode_ms:.1f} client_encode_ms={encode_ms:.1f}", flush=True)

    gpu_samples = []
    sampling = threading.Event()

    def sampler():
        while True:
            try:
                o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                             "--format=csv,noheader,nounits"], text=True)
                u, mem = o.strip().split(",")
                gpu_samples.append((time.time(), int(u), int(mem), sampling.is_set()))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)
    threading.Thread(target=sampler, daemon=True).start()

    proc, logf = _start_server(pdir / f"server-{stamp}.log", profile_out=prof_path)
    sp = psutil.Process(proc.pid)

    async def one(m, timeout):
        async with websockets.connect("ws://localhost:8000", max_size=100 * 1024 * 1024,
                                      open_timeout=60, ping_interval=None) as ws:
            t = time.perf_counter()
            await ws.send(m)
            r = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if "error" in r:
                raise RuntimeError(r["error"])
            return time.perf_counter() - t

    async def burst(n, timeout):
        ms = [msg(i) for i in range(n)]
        t = time.perf_counter()
        lats = await asyncio.gather(*(one(x, timeout) for x in ms))
        return time.perf_counter() - t, lats

    results = {"gpu": "H200", "candidates": candidates,
               "protocol": {"request_MB": len(m0) / 1e6, "decode_ms": decode_ms, "encode_ms": encode_ms}, "runs": []}
    try:
        for n in [int(x) for x in concurrency.split(",")]:
            t = time.perf_counter()
            asyncio.run(burst(n, 1800))
            compile_s = time.perf_counter() - t
            for _ in range(2):          # the 5 ms batch window may split n -> warm those shapes too
                asyncio.run(burst(n, 1800))
            sp.cpu_percent(None); sampling.set(); t_start = time.time()
            walls, lats = [], []
            for _ in range(rounds):
                w, l = asyncio.run(burst(n, 600))
                walls.append(w); lats += list(l)
            sampling.clear()
            cpu = sp.cpu_percent(None)
            g = [u for (ts, u, mem, on) in gpu_samples if ts >= t_start and on]
            mem = max([mm for (ts, u, mm, on) in gpu_samples if ts >= t_start] or [0])
            run = {"concurrency": n, "first_burst_s": compile_s, "wall_median_s": float(np.median(walls)),
                   "req_median_s": float(np.median(lats)), "req_p95_s": float(np.percentile(lats, 95)),
                   "obs_per_s": float(n / np.median(walls)), "server_cpu_pct": cpu,
                   "gpu_util_mean": float(np.mean(g)) if g else None, "gpu_util_max": int(max(g)) if g else None,
                   "gpu_mem_mb": mem}
            results["runs"].append(run)
            print("SRV " + json.dumps(run), flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
    if prof_path.exists():
        s = io.StringIO(); pstats.Stats(str(prof_path), stream=s).sort_stats("tottime").print_stats(25)
        results["cprofile_tottime_top25"] = s.getvalue()
        s = io.StringIO(); pstats.Stats(str(prof_path), stream=s).sort_stats("cumulative").print_stats(30)
        results["cprofile_cumulative_top30"] = s.getvalue()
        print("CPROFILE_TOTTIME\n" + results["cprofile_tottime_top25"][-6000:], flush=True)
    else:
        print("CPROFILE_MISSING (server did not exit cleanly)", flush=True)
    f = pdir / f"server-{stamp}.json"
    tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(results, indent=2)); os.replace(tmp, f); vol.commit()
    brief = [{k: r[k] for k in ("concurrency", "req_median_s", "obs_per_s", "server_cpu_pct", "gpu_util_mean")}
             for r in results["runs"]]
    print("SRV_DONE " + json.dumps(brief), flush=True)
    return {k: v for k, v in results.items() if not k.startswith("cprofile")}


@app.local_entrypoint()
def profile_all(physics_only: bool = False, server_only: bool = False,
                main_episode: str = "", roy_episode: str = ""):
    """main_episode / roy_episode: folder names under /vol/rollouts in YOUR volume (run ::pilot first)."""
    import json
    calls = []
    if not server_only:
        if main_episode:
            calls.append(physics_profile.spawn(sim="main", episode=main_episode, steps=150, cloth_spacing=0.045))
        if roy_episode:
            calls.append(physics_profile.spawn(sim="roy", episode=roy_episode, steps=150))
    if not physics_only:
        calls.append(server_profile.spawn())
    for c in calls:
        try:
            r = c.get()
            print("RESULT " + json.dumps(r)[:3000], flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"CALL_FAILED {type(e).__name__}: {e}", flush=True)


# =============================================================================================
# Teacher-server bottleneck, round 2 (L40S): (A) yappi inside the server = every thread,
# (B) K server processes sharing one GPU, client compression off.
#   PYTHONUTF8=1 python -m modal run --detach teacher/modal_teacher.py::server_profile2
# =============================================================================================
def _launch_server(port, log_path, mem_fraction="0.8", yappi_out=None):
    import os, subprocess, sys, threading
    env = dict(os.environ, JAX_PLATFORMS="cuda", JAX_COMPILATION_CACHE_DIR=f"{VOL_PATH}/jax_cache",
               JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=mem_fraction, PYTHONUNBUFFERED="1")
    script = ["scripts/serve.py"]
    if yappi_out:
        env["YAPPI_OUT"] = str(yappi_out)
        script = ["/opt/teacher/yappi_serve.py"]
    cmd = [f"{SRC}/.venv/bin/python"] + script + ["--port", str(port), "policy:checkpoint",
           "--policy.config", "pi_modified_bc_rl", "--policy.dir", CKPT]
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ready = threading.Event()

    def pump():
        for line in proc.stdout:
            logf.write(line); logf.flush()
            if "listening on" in line:
                ready.set()
            if "Traceback" in line or "Error" in line:
                sys.stdout.write(f"[serve:{port}] " + line); sys.stdout.flush()
    threading.Thread(target=pump, daemon=True).start()
    return proc, logf, ready


@app.function(image=image, gpu="L40S", volumes={VOL_PATH: vol}, timeout=2400, cpu=16, memory=65536)
def server_profile2(ks: str = "1,2,4", per_server: int = 4, rounds: int = 6, candidates: int = 3):
    import asyncio, base64, json, os, pathlib, signal, subprocess, threading, time
    import numpy as np, psutil, websockets

    stamp = time.strftime("%H%M%S")
    pdir = pathlib.Path(VOL_PATH, "profile"); pdir.mkdir(exist_ok=True)
    cfg = dict(json.load(open(f"{CKPT}/assets/inference_config.json"))["per_garment_type"]["pant_long"])
    cfg.pop("k_execute", None); cfg.pop("num_steps", None); cfg["num_rollout_candidates"] = candidates
    rng = np.random.default_rng(0)
    imgs = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for k in ("top_rgb", "left_rgb", "right_rgb")}
    state = [-1.24, -1.69, 1.49, 1.05, -0.08, -0.01, 1.24, -1.69, 1.49, 1.05, -0.08, -0.01]

    def enc(a):
        a = np.ascontiguousarray(a)
        return {"base64": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": str(a.dtype)}
    body = {"type": "infer_chunk", "garment_type_id": 2, "observation.state": state, "inference_config": cfg}
    for k, v in imgs.items():
        body[f"observation.images.{k}"] = enc(v)
    MSG = json.dumps(body)

    gpu_samples = []
    def sampler():
        while True:
            try:
                o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                             "--format=csv,noheader,nounits"], text=True)
                u, mem = o.strip().split(","); gpu_samples.append((time.time(), int(u), int(mem)))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)
    threading.Thread(target=sampler, daemon=True).start()

    async def one(port, timeout):
        async with websockets.connect(f"ws://localhost:{port}", max_size=100 * 1024 * 1024, open_timeout=120,
                                      ping_interval=None, compression=None) as ws:
            t = time.perf_counter(); await ws.send(MSG)
            r = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if "error" in r:
                raise RuntimeError(r["error"])
            return time.perf_counter() - t

    async def burst(ports, n_each, timeout):
        t = time.perf_counter()
        lats = await asyncio.gather(*(one(p, timeout) for p in ports for _ in range(n_each)))
        return time.perf_counter() - t, lats

    def wait_ready(servers, what):
        t0 = time.time()
        for proc, _, ready in servers:
            while not ready.is_set():
                if proc.poll() is not None:
                    raise SystemExit(f"{what}: server died during load (exit {proc.returncode})")
                time.sleep(1)
        print(f"READY {what} {len(servers)} server(s) in {time.time() - t0:.0f}s", flush=True)

    out = {"stamp": stamp, "gpu": "L40S", "candidates": candidates, "per_server_concurrency": per_server, "phase_b": []}

    # ---------------- phase A: yappi inside one server ----------------
    yout = pdir / f"yappi-{stamp}.txt"
    srv = _launch_server(8000, pdir / f"yappi-serve-{stamp}.log", yappi_out=yout)
    try:
        wait_ready([srv], "A(yappi)")
        for _ in range(3):
            asyncio.run(burst([8000], per_server, 1800))          # compile + warm the batch shapes
        t0 = time.time(); walls = []
        for _ in range(rounds):
            w, _ = asyncio.run(burst([8000], per_server, 600)); walls.append(w)
        g = [u for ts, u, m in gpu_samples if ts >= t0]
        out["phase_a"] = {"obs_per_s": per_server / float(np.median(walls)), "gpu_util_mean": float(np.mean(g)) if g else None}
        print("PHASE_A " + json.dumps(out["phase_a"]), flush=True)
    finally:
        srv[0].send_signal(signal.SIGUSR1)
        try:
            srv[0].wait(90)
        except subprocess.TimeoutExpired:
            srv[0].kill()
        srv[1].close()
    if yout.exists():
        txt = yout.read_text(); out["yappi_file"] = str(yout)
        print("YAPPI\n" + txt[:9000], flush=True)
    else:
        print("YAPPI_MISSING", flush=True)
    vol.commit()

    # ---------------- phase B: K servers on one GPU ----------------
    for k in [int(x) for x in ks.split(",")]:
        ports = [8100 + i for i in range(k)]
        frac = f"{min(0.8, 0.9 / k):.2f}"
        servers = [_launch_server(p, pdir / f"multi-{stamp}-k{k}-{p}.log", mem_fraction=frac) for p in ports]
        try:
            wait_ready(servers, f"B(k={k})")
            for _ in range(3):
                asyncio.run(burst(ports, per_server, 1800))
            t0 = time.time(); walls, lats = [], []
            cpus = [psutil.Process(s[0].pid) for s in servers]
            for c in cpus:
                c.cpu_percent(None)
            for _ in range(rounds):
                w, l = asyncio.run(burst(ports, per_server, 600)); walls.append(w); lats += list(l)
            g = [u for ts, u, m in gpu_samples if ts >= t0]
            mem = max([m for ts, u, m in gpu_samples if ts >= t0] or [0])
            run = {"servers": k, "total_inflight": k * per_server, "obs_per_s": k * per_server / float(np.median(walls)),
                   "req_median_s": float(np.median(lats)), "req_p95_s": float(np.percentile(lats, 95)),
                   "gpu_util_mean": float(np.mean(g)) if g else None, "gpu_mem_mb": mem,
                   "server_cpu_pct_each": [c.cpu_percent(None) for c in cpus]}
            out["phase_b"].append(run); print("PHASE_B " + json.dumps(run), flush=True)
        finally:
            for proc, logf, _ in servers:
                proc.terminate()
            for proc, logf, _ in servers:
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                logf.close()
        f = pdir / f"server2-{stamp}.json"; tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2)); os.replace(tmp, f); vol.commit()
    print("DONE2 " + json.dumps({"phase_a": out.get("phase_a"), "phase_b": out["phase_b"]}), flush=True)
    return out

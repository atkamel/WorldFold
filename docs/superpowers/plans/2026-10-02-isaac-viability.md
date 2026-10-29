# Phase I — Run the imitation pipeline on Isaac Sim (viability pass, self-verified)

> **For agentic workers:** execute milestone by milestone with superpowers:executing-plans
> (inline, with the checkpoints below). Steps use `- [ ]` checkboxes. After approval this file
> is copied to `docs/superpowers/plans/2026-10-02-isaac-viability.md` and Phase I is added to
> `docs/roadmap.md`.

**Goal:** pull `origin/main`'s new Isaac Sim env. Then make the existing imitation pipeline
run end-to-end on it, locally and at pilot scale:

> expert → dataset → chunk-MLP / diffusion BC → eval → DAgger → vision BC → success detector → demo

Installs are reviewed and contained before any third-party code runs. Every milestone closes
on a verify command plus committed evidence.

**Architecture:** one simulator switch (`backend="mujoco"|"isaac"`) at the env factory, the
rollout worker and the CLIs. Datasets, trainers, policies, DAgger loop and eval stay shared.
The Isaac side runs in its own Python 3.11 venv, `.venv-isaac/`, at the repo root and
gitignored, like the existing `.venv-warp/`. The MuJoCo venv
`.venv` is untouched and stays the regression baseline.

**Tech stack:** Isaac Sim 5.1.0 (pip), LeHome lehome-challenge `a805ad2` + LeHome IsaacLab fork
`69f6fa5`, PhysX particle cloth, torch 2.7.0+cu128; existing `imitation/`.

**Spec:** this file, plus `docs/superpowers/specs/2026-09-24-isaac-sim-port-design.md` and
`isaac/README.md` (the env as built on main).

---

## Context

`origin/main` (ce02b44, PR #15 "feat/isaac-lehome") adds `isaac/`: WorldFold's cloth-fold
env rebuilt on LeHome's Isaac Sim 5.1 stack.
- PhysX particle cloth, LeHome SO101 USD arms
- **friction grasp, no weld**
- shared constants in `mujuco/cloth_params.py`
- accessor-based fold wrappers

`feature/imitation` (PR #12) finished the whole pre-VLA pipeline on MuJoCo. This pass makes
that pipeline **viable on Isaac**: every stage runs through the real CLIs and produces valid
artifacts. Full-scale retraining and grasp work come later.

**Your decisions**
- **Local Windows only** (RTX 5080 Laptop 16 GB, 64 GB RAM, 16 threads, 873 GB free, long
  paths already on). If LeHome's stack won't run on Windows, stop at gate I0.3 and report.
- **Viability, not performance.** The pipeline stages behind the keepers run at pilot scale:
  - expert, data
  - chunk-MLP + diffusion BC
  - DAgger
  - vision BC (128² main + wrists)
  - detector
  - demo

  Pilot numbers are recorded and labelled as pilot: wide intervals, not benchmarks.
- **Grasping deferred.** The friction grasp is used as built. Its success rate is measured and
  recorded, not tuned. Grasp reliability, the expert ceiling gate and the n = 200 retrain
  become future roadmap milestones, with their bars written down (see "Future milestones").
- **Safe installs.** Every download is reviewed, pinned and hash-checked. It's contained in
  one folder, with no admin rights or system changes.

**How this got cheaper**

| | previous draft | this plan |
|---|---|---|
| Isaac sim time | ~50–60 h, plus an open-ended grasp loop | **~2–3 h** pilot, plus ~10 min per e2e test run |
| Isaac processes | up to 4 | 2 (less heat) |
| expert data | 400 episodes | 80 episodes |
| evals | n = 200 × 3 sets per model, 2 seeds, replan sweep | n = 10–20, 1 seed, no sweep |
| DAgger labels | look-ahead in the simulator | kinematic look-ahead, with no extra sim steps |
| randomization | visual DR | none this pass (deferred) |
| downloads | full LeHome asset dataset | only the robot files the env loads |

**Facts established while planning** (read-only exploration and two research agents)
- **`main`'s `isaac/` is the newest Isaac work.** `feat/isaac-sim` is the old Isaac Sim 4.5 +
  weld version. `ROY-vla-teacher` runs LeHome garments with pi0.5 and is reference only.
- **Merge:**
  - conflicts in `cloth_fold_rl/quarter_fold_env.py` (3 hunks) and `.gitignore`
  - `quarter_fold_expert.py` auto-merges
  - `mujuco/sim_main.py` only moved constants to `cloth_params.py`; `CAMERA_FOVY_DEG = 45`
    is MuJoCo's default
- **Silent-breakers** a naive merge would leave in place:
  - main's `QuarterFoldEnv.__init__(stages=STAGES)` sets `self.stages` on the instance, which
    shadows our `HalfFoldEnv.stages = STAGES[:1]` and turns the half fold into a 2-stage
    quarter fold
  - main's `isaac/tests` stub lacks `n_tasks`, `_get_obs` and `_domain_params`
  - our `OBS_TASK_START = 50 + 69` is a hard-coded size
- **MuJoCo coupling to undo:**
  - `import imitation.teachers` pulls in MuJoCo, and every rollout worker imports it
  - `sim_state.py` is `mj_getState`-only
  - `vision/replay.py` needs bit-exact replay
  - `benchmark_expert` and `check_resync` build several envs per process
  - Isaac's `_failed` can end on NaN observations
- **Isaac env as built:**
  - one env per process
  - `main` camera only
  - ~5 steps/s per env
  - jitter ±1 cm
  - the fold takes ~230 steps, so the cap is 400
  - the scripted fold succeeds 1/3 times
  - `grasp_active` is a 6 cm proximity proxy
  - no domain randomization
- **Local environment:** Python 3.11.9 is installed and every repo `.py` parses under 3.11.
  `uv` isn't installed. `.venv` is Python 3.13.7.

## Global constraints (every task inherits these)

- **MuJoCo stays put:**
  - `.venv` pins in `imitation/requirements.txt` unchanged
  - expert benchmark **48/50, failing seeds 24 and 35, rows identical** to
    `outputs/imitation/expert_benchmark_m1_5.json`
- **The Isaac venv** is `.venv-isaac/` at the repo root (gitignored; `$ISAAC_PY` =
  `.venv-isaac\Scripts\python.exe`):
  - Python 3.11.9, isaacsim 5.1.0, torch 2.7.0+cu128
  - lehome-challenge `a805ad2f7ab5`, IsaacLab fork `69f6fa548c3a`
  - pins and hashes in `isaac/requirements-windows.lock`, never merged with another pin file
- **Isaac processes:**
  - ≤ `N_ISAAC` (default 2), one sim pool at a time
  - long jobs run as visible background tasks with a Monitor, and are resumable
  - orphan check at session start
- **Sizes and data:**
  - obs/action sizes come from `imitation/spec.py` (`OBS_DIM` 139, `ACTION_DIM` 12); no
    literal dims
  - datasets are write-once
  - Isaac versions are named `isaac_*`, and their manifests carry `backend`, the stack pins
    and the Isaac knob hash before freezing
- **Pilot numbers:** Wilson 95% intervals, labelled "pilot". Never compared with MuJoCo rows
  except in a labelled side-by-side.
- **Ask first:** I ask in chat before:
  - the first download (artifact list with sizes)
  - accepting NVIDIA's EULA
  - any `git push`
  - deleting anything outside the repo

## Install safety and scope (I0.2 reviews, I0.3 installs)

| artifact | source | pinned by | integrity | review |
|---|---|---|---|---|
| isaacsim 5.1.0 `[all,extscache]` | pypi.nvidia.com (NVIDIA's official index) | version, lock | sha256 in lock | vendor package; no runtime registry downloads |
| torch 2.7.0+cu128, torchvision | download.pytorch.org (official) | version, lock | sha256 in lock | vendor package |
| every other Python dep | pypi.org | LeHome's `uv.lock` (or our hashed lock) | sha256 in lock | lock reviewed; delta vs LeHome listed |
| `uv` | pypi.org (Astral) | exact version | pip hash | widely used installer |
| lehome-challenge | github.com/lehome-official | commit `a805ad2f…` | git object ids | **source review** (below) |
| IsaacLab fork | github.com/lehome-official/IsaacLab | commit `69f6fa54…` | git object ids | **diff vs upstream IsaacLab** |
| LeHome robot assets | huggingface.co/datasets/lehome/asset_challenge | pinned revision, robot files only | HF sha256 | file-type audit |

**Review, before any of their code runs.**
1. Clone both repos at their SHAs. This is source only, nothing executes. Confirm with
   `git rev-parse HEAD`.
2. Diff the IsaacLab fork against the upstream IsaacLab release it forked from, and read every
   changed file. Main's README says the fork only forces GPU dynamics in `DirectRLEnv`.
3. Read the code that will run:
   - `setup.py` / `pyproject.toml` of `source/isaaclab` and `source/lehome`, which execute at
     install
   - every lehome module the env imports: `assets/object/Garment`,
     `assets/robots/lerobot`, `utils/constant`, and the particle cfg
4. Grep both trees for `subprocess`, `os.system`, `exec(`, `eval(`, `pickle`, `ctypes`,
   `socket`, `requests`/`urllib`, `base64` and obfuscated strings. Each hit gets a written
   verdict.
5. **Never run the repos' shell scripts** (`isaaclab.sh`/`.bat` etc.). Install only the
   reviewed packages, with explicit `uv pip install -e`.
6. Install Python packages hash-verified:
   - `uv sync --locked` against LeHome's lock, or
   - if Windows needs another resolution, our own lock compiled with `--generate-hashes`
     from the three official indexes only, with its delta vs LeHome's lock reviewed.
7. Assets: download only the robot files the env loads, found by grepping lehome's asset
   paths (`so101_*` USD + URDF), at a pinned revision. Reject anything executable or pickled:
   `.exe .dll .bat .ps1 .sh .py .pkl .pt .pth`.
8. Run `pip-audit` (PyPA, in a throwaway `uvx` env) on the final lock. Findings are reported;
   versions stay pinned by Isaac 5.1.
9. Write `isaac/INSTALL_REVIEW.md`:
   - sources, SHAs and lock
   - every review finding with its verdict
   - touched paths, network hosts, telemetry settings
   - uninstall steps

**Scope:**
- Everything lives inside the one gitignored folder `.venv-isaac/` at the repo root:
  - the venv itself
  - the two pinned checkouts in `.venv-isaac\src\`; that's where pip puts editable VCS
    installs anyway
  - the robot assets, inside the lehome checkout where `ASSETS_ROOT` expects them
  - every cache:
    - `isaac/env_windows.ps1` points `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`
      and `WARP_CACHE_PATH` at `.venv-isaac\cache\`
    - Kit's portable root goes to `.venv-isaac\kit\`, so its shader cache, logs and data
      don't land in `%LOCALAPPDATA%`
    - all of this is set **for the session only**, never globally

  Path length is fine there: `LongPathsEnabled = 1` on this machine, and Python 3.11 is
  long-path aware. If the extension cache still hits a path error, the folder moves to a
  shorter path; that's the only reason to leave the repo.
- pytest's `testpaths` are explicit, so it never walks into `.venv-isaac`. ripgrep and git
  ignore it too.
- No admin rights, no registry, firewall, PATH or service changes. The GPU driver is
  untouched.
- **Offline at runtime:**
  - `HF_HUB_OFFLINE=1`
  - Kit's online extension registry is disabled, so only the hash-verified `extscache`
    extensions load
  - the Kit log is checked for downloads
- **Telemetry and data collection are off** (privacy-preserving defaults; I don't opt in).
- I take a before/after listing of the top level of `%USERPROFILE%`, `%LOCALAPPDATA%`,
  `%APPDATA%` and `%TEMP%`. Every new location gets documented, and anything unexpected gets
  reported to you.
- Uninstall means deleting `.venv-isaac/` plus any leftovers the scope check documents. I ask
  before deleting.

## Architecture

```
                 .venv (py3.13, MuJoCo)                   .venv-isaac (py3.11, Isaac 5.1; gitignored)
make_env(backend="mujoco") → HalfFoldEnv(ClothFoldEnv)    make_env(backend="isaac") → HalfFoldEnv(base_env=IsaacClothFoldEnv)
CameraRig (mujoco.Renderer)                               IsaacCameraRig (TiledCameras: main 128², wrists 64²)
QuarterFoldExpert(FoldExpert: MjData IK, weld)            QuarterFoldExpert(IsaacArmExpert: PinchIK, friction pinch as built)
labels: mj_getState look-ahead                            labels: KinematicProxy look-ahead (no sim steps)
                 └──── shared: data store, train, policies, rollout/EnvPool, dagger, evaluate, verify ────┘
```

- **Factory.** `imitation.tasks.half_fold.make_env(backend, **kw)` replaces the 7 direct
  `HalfFoldEnv(...)` constructions.
- **Workers.** Isaac `EnvPool` workers run one Kit app each. They clear `sys.argv` before
  `start_app()`, and on close they reply, then `os._exit(0)`.
- **One venv per backend.** The whole Isaac pipeline runs in the Isaac venv (sim, training,
  eval). That avoids cross-interpreter pickles and torch 2.13 → 2.7 checkpoint loading.

---

## Self-verification protocol (every milestone)

1. **Preflight:**
   - re-read `docs/status.md` and confirm this milestone is next
   - orphan-process check (python, sh, Kit)
2. **Build** test-first for code: failing test → code → passing test.
3. **Run the milestone's Verify commands** and read the real output. Nothing is inferred.
4. **`python -m imitation.verify <ID>` → `PASS <ID>`** (once I0.4 lands; I0.1 is re-checked
   then). It reads committed artifacts only and writes
   `outputs/imitation/isaac/verify/<ID>.json`.
5. **Regression:**
   - `.venv\Scripts\python.exe -m pytest -m "not slow"` (+ `-m slow` if sim, env or wrapper
     changed)
   - `$ISAAC_PY -m pytest -m isaac` if Isaac-side code changed
6. **Docs:**
   1. `results.md` first (n, seeds, dataset version, Wilson; "pilot" label)
   2. `status.md` (Updated, Phase, Where we are, Next action, tracker row, test counts,
      pass-log line + commit)
   3. `roadmap.md` (✅ only when the exit artifact exists)
   4. `imitation.md` (spec changes)
7. **Commit** code, docs and verify evidence together.
8. **On failure:** superpowers:systematic-debugging, fix, re-verify. A ❌ close with evidence
   is allowed only where marked.
9. **Code review:** I0.1 (merge) and I1.1 (backend) get a fresh-eyes review
   (superpowers:requesting-code-review) before closing.

### Verification at a glance

| ID | milestone | verify (+ `python -m imitation.verify <ID>`) | pass when |
|---|---|---|---|
| I0.1 | pull / merge main | pytest fast + slow (`.venv`); `benchmark_expert` | green; 48/50, seeds 24/35 fail, rows identical |
| I0.2 | install review | `isaac/INSTALL_REVIEW.md` complete; lock hashed | every artifact sourced, pinned, reviewed; your OK on the list + EULA |
| I0.3 | **scoped install + smoke (gate)** | import line; `smoke_test.py` state + hybrid; scope listing | cu128 + CUDA True; `SMOKE OK` ×2; nothing outside documented paths; no runtime downloads |
| I0.4 | verifier | `pytest tests/imitation/test_verify.py` | green; `--list` shows every ID |
| I1.1 | backend switch | pytest in both venvs | MuJoCo unchanged; Isaac obs 139-D finite; imports without mujoco; pool closes < 60 s |
| I1.2 | Isaac seed sets | `test_seeds.py`; `isaac/reach_check.py` | ranges disjoint; id_hard ring reachable 200/200 |
| I1.3 | camera rig | `test_isaac_rig.py`; frames looked at | 3 cameras, right shapes; wrist pose within 1 mm / 0.5° |
| I2.1 | Isaac expert (grasp as built) | teacher tests; `evaluate --ckpt expert` n = 20; `check_resync` n = 10 | runs clean; rates recorded (no threshold) |
| I2.2 | DAgger labels | `test_isaac_labels.py` on 5 expert episodes | ≥ 80% of chunks within 0.05 (pilot sanity bar) |
| I3.1 | e2e micro chain | `pytest -m "isaac and slow" tests/imitation/test_isaac_chain.py` | collect → train → eval → dagger → vision → detector green |
| I3.2 | pilot run | the real CLIs at pilot scale | every artifact valid (dataset hash, loss ↓, evals, DAgger round, vision, detector, videos) |
| I3.3 | close-out | `verify --all` in both venvs | all PASS; future milestones in roadmap |

---

## Milestones

### I0.1 — Pull: merge `origin/main` into `feature/imitation` (MuJoCo must not move)

This happens on the current branch, so PR #12 becomes conflict-free with `main`. Phase I then
branches off as `feature/isaac-imitation`. Commits stay local.

**Files:**
- merge: `.gitignore`, `cloth_fold_rl/quarter_fold_env.py`
- modify: `isaac/tests/test_half_fold.py` (stub), `pytest.ini`, `imitation/tasks/half_fold.py`
- create: `tests/imitation/test_merge_invariants.py`

- [ ] **Preflight:**
  - `git status`: the 3 untracked demo mp4s stay untracked
  - orphan check
  - HEAD is `feature/imitation` @ 1cca9d7
- [ ] **Fetch:** `git fetch --prune origin`, then `git fetch origin main:main` (a fast-forward,
  28 behind).
- [ ] **Merge:** `git merge --no-ff origin/main`.
- [ ] **Resolve `.gitignore`:** take the union.
- [ ] **Resolve `quarter_fold_env.py`:**
  - from main: `base_env=`, the lazy `ClothFoldEnv`, `cloth_params` imports,
    `HALF_FOLD_MAX_STEPS`, `HalfFoldEnv`
  - from ours: the 139-D `_observe`, `_set_goals`, `cloth_offset_xy`, info goals
  - then fix the silent-breakers:

```python
class QuarterFoldEnv(gym.Wrapper):
    stages = STAGES                       # subclasses may run a prefix (imitation.tasks.HalfFoldEnv)

    def __init__(self, max_episode_steps=MAX_STEPS, seed=None, cloth_jitter=CLOTH_JITTER,
                 base_env=None, stages=None):
        ...
        if stages is not None:            # explicit argument wins; else the class attribute
            self.stages = tuple(stages)
        spaces = env.observation_space
        task_start = spaces["proprio"].shape[0] + spaces["cloth_state"].shape[0]   # no literal 50 + 69
        self._onehot = slice(task_start, task_start + env.n_tasks)
```

  The remaining `data.xpos` / `site_xpos` reads become `cloth_positions()` /
  `gripper_position()`. `imitation/tasks/half_fold.py` imports `HALF_FOLD_MAX_STEPS` from
  `cloth_fold_rl.quarter_fold_env`.
- [ ] **Stub:** give `isaac/tests/test_half_fold.py`'s `ScriptedBase` sizes 50/69/22,
  `n_tasks = 4`, `_get_obs` and `_domain_params`.
- [ ] **pytest.ini:** `testpaths += isaac/tests cloth_fold_rl/tests`, and register the marker
  `isaac: needs Isaac Sim (Isaac venv)`.
- [ ] **`test_merge_invariants.py`:**
  - fast: imitation `HalfFoldEnv` has 1 stage, `QuarterFoldEnv` has 2, and the obs length is
    `OBS_DIM`
  - slow: on MuJoCo, `env._start == cloth_positions()`, and main's `cloth_fold_rl.HalfFoldEnv`
    is 1-stage and 139-D

**Verify**
- `.venv\Scripts\python.exe -m pytest -m "not slow"` and `-m slow`: all pass. That includes
  main's `test_folds_on_nominal_seed`.
- `.venv\Scripts\python.exe -m imitation.benchmark_expert --out outputs/imitation/expert_benchmark_i0_1.json`:
  48/50, seeds 24 and 35 fail, rows identical to `expert_benchmark_m1_5.json` (checked with a
  one-line JSON diff).

**Exit:**
- merge commit with the docs:
  - status pass log
  - results.md post-merge row
  - roadmap Phase I section, including the future milestones
  - this plan in `docs/superpowers/plans/`
- then `git switch -c feature/isaac-imitation`

### I0.2 — Install review (nothing third-party executes)

**Files:** create `isaac/INSTALL_REVIEW.md` and `isaac/requirements-windows.lock` (hashed).

- [ ] **Clone** into `.venv-isaac\src\` (source only; `.venv-isaac/` is added to
  `.gitignore` first): lehome-challenge @ `a805ad2f7ab52a4583066fc4ee5180459a7f9d15`;
  IsaacLab fork @ `69f6fa548c3a3520e3cb26ed24bb8abe60baeef3`.
- [ ] **Review the code:**
  - fork vs upstream diff
  - setup files and imported lehome modules
  - risky-pattern grep, with a verdict per hit (steps 1–5 of Install safety)
- [ ] **Lock:**
  - check whether LeHome's `uv.lock` resolves for `win_amd64` / cp311 with uv's
    resolve-only commands (nothing installs)
  - otherwise compile our hashed lock from the three official indexes
  - list the delta vs LeHome's lock
- [ ] **Assets:** list the robot files the env needs, by grepping lehome's asset paths, with
  sizes, at a pinned HF revision.
- [ ] **Ask you:**
  - post the artifact list with sources, sizes and hashes
  - ask for your OK, and whether you accept NVIDIA's Isaac Sim EULA

**Verify:** the review doc covers every row of the Install-safety table, with a verdict per
finding. The lock has a sha256 for every artifact. Your OK and the EULA answer are in chat.

### I0.3 — Scoped install + smoke test  **(GATE: stop and report if it fails)**

**Files:**
- create: `isaac/setup_windows.ps1` (reproducible, hash-checked install),
  `isaac/env_windows.ps1` (session-only env)
- modify: minimal Windows fixes in `isaac/`, e.g. `lab_scene._spawn_cloth`'s
  `"/" + os.path.relpath(...)` asset path; `.gitignore`; `CLAUDE.md` (Isaac venv,
  `-m isaac` step)

- [ ] **Listing before:** take the before-listing of the four user directories.
- [ ] **Install:** `py -3.11 -m venv .venv-isaac`, then a hash-verified install
  from the lock. The reviewed editable packages come next. Then the robot assets only, at the
  pinned revision.
- [ ] **Session env:** EULA env var (only after your yes), telemetry off, registry off,
  offline flags, caches redirected (`env_windows.ps1`).
- [ ] **Windows fixes:** only what the smoke test trips on, cross-platform. The Modal path
  stays unchanged.
- [ ] **Throughput:** quick runs with N = 1 and N = 2 processes for state steps/s and peak
  VRAM. Plus a determinism repeat: the same seed and actions in two fresh processes, and the
  max particle deviation. These go to `outputs/isaac/runtime.json`. That sets `N_ISAAC` (2
  unless VRAM says 1).

**Verify**
- `$ISAAC_PY -c "import isaacsim, isaaclab, lehome, torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`:
  `2.7.0+cu128 True NVIDIA GeForce RTX 5080 Laptop GPU`.
- `isaac/smoke_test.py --mode state` and `--mode hybrid --save-frame …`:
  - both exit 0 with `SMOKE OK`
  - logs go to `outputs/isaac/smoke/<stamp>/`
  - I open the RGB frame and confirm the cloth, table and both arms are visible
- `$ISAAC_PY -m pytest isaac/tests -q`: pass.
- **Scope check:**
  - the after-listing shows new entries only inside `.venv-isaac/`, plus the locations
    documented in the review
  - the Kit log shows no extension downloads and telemetry off
  - `pip-audit` results recorded

**Stop rule:** if the stack can't import, or the smoke test fails on Windows after this
milestone's fixes, I record the evidence in results.md/status.md, commit, and stop and report.

### I0.4 — Verifier (`python -m imitation.verify`)

**Files:** create `imitation/verify.py`, `tests/imitation/test_verify.py`.

- It is sim-free (stdlib, numpy, json, git), so it runs in either venv.
- Checks that need Isaac report `SKIP` from `.venv`.
- Helpers:
  - `wilson(k, n)`
  - `eval_counts(path)`, which reads `imitation.evaluate`'s
    `{"results": {set: {"n", "success_rate"}}}`
  - `manifest_ok(version)`, `loss_dropped(history, ratio)`
  - `results_md_has(token)`, `git_tracked(path)`

```python
@dataclass
class Result:
    milestone: str
    passed: bool
    evidence: list[str]

CHECKS: dict[str, Callable[[], Result]] = {}
def check(mid: str): ...          # decorator: registers a milestone check
```

- [ ] **Tests first:**
  - pass and fail on synthetic artifacts
  - a missing artifact fails and names its path
  - `wilson(97, 100)` gives (91.5, 99.0), matching results.md M1.7
- [ ] **Implement.** Then `python -m imitation.verify I0.1 I0.4`: PASS.

### I1.1 — Backend switch (the pipeline runs on either sim)

**Files:**
- `imitation/tasks/half_fold.py`: `make_env(backend=...)`, Isaac cap 400, Isaac jitter
- `imitation/rollout.py`: worker backend, `sys.argv` clear, Isaac `os._exit` on close, join
  timeout then terminate
- lazy MuJoCo imports in `imitation/teachers/__init__.py`, `imitation/sim_state.py`,
  `imitation/vision/render.py`, `cloth_fold_rl/quarter_fold_expert.py`
- `--backend {mujoco,isaac}` on `collect`, `evaluate`, `dagger`, `demo`, `benchmark_expert`
  and `check_resync`:
  - the last two use `EnvPool` on Isaac
  - `check_resync` gains `--out`
  - `collect` records `backend`, stack pins and knob hash in its config and `meta`
- non-finite guard: an Isaac `unstable` ending keeps the last finite obs and is stored as a
  failure
- new `imitation/isaac_runtime.py`: `N_ISAAC`, `ISAAC_MAX_STEPS = 400`
- tests: `tests/imitation/test_backend.py` (fast, no sim); `tests/imitation/test_isaac_backend.py`
  (`pytest.mark.isaac`, skipped without `isaacsim`)

- [ ] **Failing tests first:**
  - the default backend is MuJoCo
  - Isaac obs is `(OBS_DIM,)` and finite after reset + 20 random steps
  - `import imitation.rollout, .evaluate, .dagger, .train, .teachers` works with
    `find_spec("mujoco") is None`
  - `EnvPool(2, {"backend": "isaac"})` resets, steps and closes in < 60 s
  - `cloth_offset_xy` stays within ±1 cm
  - a NaN final obs is stored as `unstable`
- [ ] **Implement** until both suites are green.

**Verify**
- `.venv` fast + slow: green, MuJoCo unchanged.
- `$ISAAC_PY -m pytest -m isaac -q`: green.
- `$ISAAC_PY -m pytest tests/imitation -m "not slow and not isaac" -q`: the sim-free pipeline
  tests also pass under Isaac's numpy and torch 2.7.
- `python -m imitation.verify I1.1`.

### I1.2 — Isaac seed sets

**Files:** modify `imitation/seeds.py` (`eval_set(name, n, backend="mujoco")`). Create
`isaac/reach_check.py`.

- **`id_easy`:** the default reset (±1 cm + the env's ≤10° drop tilt).
- **`id_hard`:** one axis is pushed to sign·U(1.0, R_max) cm. R_max is the largest offset
  ≤ 2 cm where `PinchIK` reaches every pinch and place pose within 4 mm on 200/200 seeds. No
  physics is needed.
- **`recovery`:** id_easy starts with `k ~ U[8,16)` random steps at `t ~ U[35,140)`, which is
  MuJoCo's 15–60 scaled to the ~230-step Isaac fold. Training recovery demos scale the same
  way.

**Verify:** `test_seeds.py` (disjoint; same seed gives the same pose);
`$ISAAC_PY isaac/reach_check.py --n 200` → `outputs/isaac/reach.json`;
`python -m imitation.verify I1.2`.

### I1.3 — Isaac camera rig (main + wrists; no visual DR this pass)

**Files:**
- `isaac/isaac_env.py` and `isaac/lab_scene.py`: optional `cameras={name: size}`. The
  defaults keep main's behaviour, and `check_contract` and the smoke test still pass.
- create `imitation/vision/isaac_render.py` and `tests/imitation/test_isaac_rig.py`.

- [ ] **Cameras:**
  - `main` at 128²
  - `left_wrist_cam` and `right_wrist_cam` at 64², parented to each gripper link with the
    so101-nexus MJCF `wrist_cam` offset: pos (0, 0.04, −0.04), euler (−0.5, 0, 6.28),
    fovy 75, OpenGL convention
- [ ] **`IsaacCameraRig`** has the `CameraRig` interface: `render() -> {name: uint8[3,H,W]}`,
  `reset(seed)`, `.cameras`, `last_render_s`.

**Verify**
- `test_isaac_rig.py`:
  - shapes and dtypes
  - each wrist camera's world pose equals the gripper-link pose ∘ MJCF offset, within 1 mm /
    0.5°
- I open the sample frames in `outputs/isaac/rig/` and confirm each wrist view shows its jaws.
- `python -m imitation.verify I1.3`.

### I2.1 — Isaac expert (friction grasp as built), wired as the teacher

**Files:**
- create `isaac/fold_expert.py`
- modify `cloth_fold_rl/quarter_fold_expert.py`: `expert_cls=` injection, default the MuJoCo
  `FoldExpert` imported lazily
- modify `imitation/teachers/scripted.py`: expert by backend; `PHASE_FIELDS` per expert
  class; Isaac phases added to `_GROUPS`
- create `tests/imitation/test_isaac_expert.py`

**`IsaacArmExpert`** ports `isaac/half_fold_demo.py`'s plan unchanged into a per-arm phase
machine: pinch → close dwell → arc of IK waypoints → place → open dwell → retreat.
- It uses `PinchIK` and joint-arrival tolerances.
- It reads the sim only through accessors.
- It provides `PHASES`, `RELEASE_PHASES`, `PHASE_FIELDS`, `reset()`, `act()` and
  `infer_phase(placed)`, so `QuarterFoldExpert`'s release gate, retry and `resync` work.
- No grasp tuning.

**Verify**
- `test_isaac_expert.py`: shadowing its own episode and labelling reproduces the executed
  actions, through `step_state`. This is the Isaac version of `test_teacher.py:70-117`.
- `$ISAAC_PY -m imitation.evaluate --backend isaac --ckpt expert --sets id_easy --n 20 --workers 2 --out outputs/imitation/isaac/pilot/eval_expert.json`
- `$ISAAC_PY -m imitation.check_resync --backend isaac --episodes 10 --workers 2 --out outputs/imitation/isaac/pilot/check_resync.json`
- Rates go to results.md as an "Isaac pilot" with Wilson intervals. **There is no threshold.**
  Exit means it runs clean and the numbers are recorded.

### I2.2 — DAgger labels on Isaac: `KinematicProxy` (cheapest path)

**Files:** create `imitation/teachers/kinematic.py`; modify `imitation/sim_state.py`
(backend dispatch: Isaac uses the proxy); create `tests/imitation/test_isaac_labels.py`
(isaac + slow).

`KinematicProxy(env)` freezes the accessor values at the label point. The expert then runs K
steps against the proxy:
- joints advance through a per-joint lag fitted from the I2.1 episodes
- `gripper_pose` comes from `PinchIK.site_pose`
- the cloth stays frozen, except a held corner rides with the gripper

The live env is never stepped or restored, so there are no extra sim steps. True Isaac
snapshot/restore is left to the scale-up milestone.

**Verify:** on 5 expert episodes, label chunks vs the executed next K actions: ≥ 80% of chunks
with max |Δ| ≤ 0.05 (pilot sanity bar) → `outputs/isaac/labels/agreement.json`. The
scale-up bar is ≥ 95% within 0.02.

### I3.1 — End-to-end micro chain test (the Isaac version of M1.6; reusable regression guard)

**Files:** create `tests/imitation/test_isaac_chain.py`, marked `isaac` and `slow`, about 10
minutes. It runs through the CLIs' Python entry points:
1. collect 3 episodes with images
2. chunk-MLP and diffusion, 200 steps each
3. eval n = 2
4. 1 DAgger round on 2 episodes
5. vision BC, 100 steps
6. detector, 100 steps

It asserts that each stage's output exists and is valid:
- dataset hash verifies
- finite actions inside ±1
- eval JSON schema
- labels have shape `[L, K, 12]`

**Verify:** `$ISAAC_PY -m pytest -m "isaac and slow" tests/imitation/test_isaac_chain.py`:
green.

### I3.2 — Pilot run through the real CLIs (~2–3 h sim, `N_ISAAC = 2`)

Each step runs as a visible background task with a Monitor, and is resumable.

1. **Collect:**
   `$ISAAC_PY -m imitation.data.collect --backend isaac --episodes 80 --recovery-fraction 0.3 --render --version isaac_pilot --workers 2 --resume`.
   Successes go to `isaac_pilot`, failures to `isaac_pilot_failures`, with images (main 128,
   wrists 64) captured at collection, because Isaac can't replay.
2. **Train** (GPU, 1 seed each): `--policy chunk_mlp`, `--policy diffusion`, `--policy vision`
   on `isaac_pilot`, with short step budgets (5k / 5k / 3k).
3. **Eval** with `--backend isaac`:
   - chunk-MLP: id_easy n = 20, recovery n = 10
   - diffusion: id_easy n = 20, id_hard n = 10, recovery n = 10
   - vision: id_easy n = 10 at replan 2
4. **DAgger:** `dagger --backend isaac --init <diffusion pilot ckpt> --dataset isaac_pilot --rounds 1 --episodes 16 --eval-n 20 --eval-sets id_easy --workers 2`.
5. **Detector:** `vision.success train --versions isaac_pilot isaac_pilot_failures`, then
   `agree` on held-out episodes.
6. **Demos:** `demo --backend isaac` on seed 100000 for the expert, diffusion and vision, at
   the main camera 512², written to `docs/reports/media/isaac_pilot_*.mp4`. I watch them via
   extracted frames.

**Verify:** `python -m imitation.verify I3.2`:
- the `isaac_pilot` manifest re-hashes, with `backend == "isaac"` and images on every episode
- each run's `history.json` shows loss dropping by ≥ 50% from its first logged value
- every eval JSON has its n and valid failure codes
- the DAgger round wrote a labelled version and a checkpoint
- detector agreement is above the majority-class rate
- the videos exist
- results.md has every pilot row

### I3.3 — Close-out

- `docs/imitation.md` gets a backend section:
  - Isaac obs semantics: grasp-flag proxy, gripper target, finite-difference velocities,
    −13.5 cm y shift
  - the eval sets
  - the 400-step cap
  - no DR
- `docs/pipeline.md` gets the `--backend isaac` commands.
- status.md "Next action" becomes the first future milestone, IG.1.

**Verify:** `python -m imitation.verify --all` passes in both venvs. `.venv` fast + slow and
`$ISAAC_PY -m isaac` are green. The MuJoCo expert benchmark still has identical rows.

---

## Future milestones (written into the roadmap now, not run in this pass)

| | milestone | exit |
|---|---|---|
| IG.1 | Grasp bench: per-arm acquired / held-through-carry / placed / released, anchor drift, failure attribution | baseline on tuning seeds 600 000+ |
| IG.2 | Grasp reliability loop: one knob per iteration (pinch geometry, squeeze, contact/friction ≤ 2.0, motion). Adhesion stays 0; no attachments | per arm, n = 100 on two blocks plus a fresh block: acquired / held / released ≥ 98, placed ≥ 95 |
| IG.3 | Expert ceiling gate | id_easy ≥ 95, id_hard ≥ 90, recovery ≥ 90, check_resync ≥ 90 (n = 100) |
| IS.1–IS.6 | Full retrain at scale: `isaac_v1` (400 eps), BC ×2 seeds, replan sweep, DAgger (labels at the strict 95% / 0.02 bar, with true snapshot/restore if the proxy misses), vision ×2 seeds, detector ≥ 90% | n = 200 × 3 sets, Wilson, gains in SE |
| I-DR | Visual DR in the Isaac rig and dynamics DR (mass, friction, damping) | robustness eval on held-out DR draws |

## Critical files

- **New:**
  - `imitation/verify.py`, `imitation/isaac_runtime.py`, `imitation/vision/isaac_render.py`,
    `imitation/teachers/kinematic.py`
  - `isaac/fold_expert.py`, `isaac/reach_check.py`
  - `isaac/setup_windows.ps1`, `isaac/env_windows.ps1`, `isaac/requirements-windows.lock`,
    `isaac/INSTALL_REVIEW.md`
  - tests: `test_merge_invariants`, `test_verify`, `test_backend`, `test_isaac_backend`,
    `test_isaac_rig`, `test_isaac_expert`, `test_isaac_labels`, `test_isaac_chain`
- **Modified:**
  - `cloth_fold_rl/quarter_fold_env.py` (merge), `cloth_fold_rl/quarter_fold_expert.py`
  - `imitation/tasks/half_fold.py`, `rollout.py`, `seeds.py`, `sim_state.py`
  - `imitation/teachers/{__init__,scripted}.py`, `data/collect.py`
  - `imitation/evaluate.py`, `dagger.py`, `demo.py`, `benchmark_expert.py`, `check_resync.py`
  - `isaac/isaac_env.py`, `isaac/lab_scene.py` (opt-in cameras, Windows path fix)
  - `pytest.ini`, `CLAUDE.md`, `.gitignore`
  - docs: roadmap, status, results, imitation, pipeline
- **Reused as-is:**
  - `IsaacClothFoldEnv` accessors, `isaac.pinch.PinchIK`
  - `isaac/half_fold_demo.py` (the fold plan), `isaac/smoke_test.py`,
    `cloth_params.check_contract`
  - `EnvPool`, `evaluate.summarize`, the write-once data store, `PolicyTeacher`,
    `dagger.py`, `train.py`

## Risks and stop rules

1. **LeHome on Windows** (lock platform, fork, lerobot deps): stop rule at I0.3. A research
   agent is still checking this. Its findings go into I0.2 before anything downloads, and a
   hard Linux-only blocker gets reported to you first.
2. **Review finds something suspicious:** I stop and show you the finding before installing.
3. **Weak expert (~1/3):** fewer pilot demos, around 25. That's fine for viability; the
   numbers are labelled pilot, and IG fixes the expert.
4. **Kinematic labels may be loose in contact-heavy phases:** a pilot sanity bar only. The
   strict bar comes at scale-up.
5. **Isaac nondeterminism:** measured in I0.3. Images are captured at collection, and eval
   variance is noted.
6. **Teammates' `isaac/` code:** changes are opt-in arguments with the old defaults. The Modal
   path and `smoke_test.py` keep working.

---

## Amendments after approval (2026-10-02)

From the research agent's report (sources in the session; summarised here):

- **I0.2/I0.3 install recipe:**
  - LeHome's `uv.lock` resolves for Windows (cp311 win_amd64 wheels for all `isaacsim-*`
    5.1.0.0, open3d, av); Linux-only packages are marker-gated.
  - The lock's torch is the CPU wheel on Windows, so after the hash-verified sync, replace it
    with `torch==2.7.0+cu128` / `torchvision==0.22.0+cu128` from download.pytorch.org, with
    hashes pinned.
  - Never re-run `uv sync` afterwards: it restores CPU torch and drops the editables.
- **Assets:** `lehome/utils/constant.py` sets `ASSETS_ROOT` from the *git root of the
  working directory*. Run from WorldFold, it resolves to `WorldFold/Assets`. The robot assets
  stay in `.venv-isaac`, reached through a gitignored `Assets` junction at the repo root (or
  an env override, if constant.py offers one).
- **Driver risk:**
  - 616.56 is outside Isaac Sim 5.1's validated Windows driver (580.88).
  - TiledCamera hangs have been reported on GB203 laptops under 59x drivers.
  - Fallback order: `isaacsim.sensors.camera.Camera` instead of TiledCamera, then an R580
    driver (your decision; I don't change drivers).
- **IsaacLab fork** = upstream + 2 commits:
  - forced GPU dynamics
  - the full rendering experience file loaded even headless
  - TiledCamera's `--enable_cameras` check removed

  These three commits are what the I0.2 diff review covers.
- **Worker lifecycle** (I1.1):
  - Kit aborts after 120 s without a tick, so idle workers poll the pipe with a timeout and
    tick Kit.
  - `os._exit` on close and on `EOFError`.
  - Start workers one at a time on a warm shader cache.
  - Per-command deadlines restart a stuck worker.
- **Determinism:** PhysX does not guarantee determinism for cloth. Gates stay statistical, and
  images are captured at collection.
- **I2.2 labels (approved by the user 2026-10-02):** executed-expert *takeover* labels
  replace `KinematicProxy`. At a replan point, with probability ~0.3, the expert resyncs and
  runs K steps for real, and that executed chunk is the label for the student-visited state.
  - The label is exact, with no restore and no extra sim.
  - Pilot acceptance: on student rollouts, takeovers complete without resync/IK errors in
    ≥ 98% of cases.
- **I1.3 cameras:** borrow LeHome's camera render settings (FXAA, no DLSS at 64²) and check
  the first frame after reset is fresh.
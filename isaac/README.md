# isaac

WorldFold's cloth-fold env on LeHome's Isaac Sim stack (lehome-challenge `a805ad2`, Isaac Sim 5.1, PhysX particle
cloth, LeHome's SO-101 arms), and the scripted experts, RL and tools that run on it. `IsaacClothFoldEnv` is a drop-in
for `mujoco_sim.sim_main.ClothFoldEnv` in `joint_delta` mode (same 14-dim action, observation dict, reward and
termination), so the `cloth_fold_rl` wrappers run on it unchanged. Stay on Isaac Sim 5.1, the last release with
particle cloth.

| folder | what it holds |
|---|---|
| `sim/` | `isaac_env.py` (the env, `PROFILES`, `IsaacClothFoldBatch` for several copies in one scene, snapshot and restore), `lab_scene.py` (the IsaacLab scene), `pinch.py` (IK on LeHome's URDF) |
| `experts/` | `fold_expert.py` (`IsaacArmExpert`, the open-loop pinch or slide grasp), `servo_expert.py` (closed-loop corner steering), `place_servo.py` (the RL base with a residual, a release vote and the gentle release), `half_fold_demo.py` (shared constants and a scripted demo) |
| `rl/` | `precision_env.py` (residual-RL window env over B copies, the validity gate), `precision_rl.py` (train, demo, sweep), `corrector_env.py` and `corrector_rl.py` (push-or-stop corrector on settled folds), `rlpd.py` (the SAC learner) |
| `tools/` | `quarter_batch.py` (scripted quarter fold over B copies), `grasp_bench.py` and `grasp_metrics.py` (grasp tuning), `rescore_gate.py`, `smoke_test.py` (install check) |
| `tests/` | sim-free tests, part of `pytest -m "not slow"` |

## Profiles

- **`lehome`** is the env as ported from LeHome, on the CPU device with LeHome's drop and drives.
- **`friction`** is the working profile. It runs on the GPU pipeline (scene copies, dynamics randomization, snapshots)
  with MuJoCo's stiff arm drives, LeHome's gripper drive, a closed jaw at +0.05 rad, adhesion 0.1 and pad friction
  2.0. The cloth sits 13.5 cm toward the arms (`CLOTH_CENTER`) so the pinch can reach both ends of the half fold.
- **`quarter2`** is `friction` with the left arm north of the cloth and the right arm south-east, the two-arm layout
  for the quarter fold's second fold.

Nothing attaches the cloth. The jaws hold it by friction and adhesion, through a high-friction lining on each finger.

## Running on WATcloud

`watcloud.sbatch` builds `isaac/Dockerfile` in the job (about 10 minutes on trpro, 30 on thor) and runs the command
in the container with the repo mounted. It passes the `WORLDFOLD_*` tuning variables through (`WORLDFOLD_EXPERT_PARAMS`
and `WORLDFOLD_PLACE_PARAMS` override expert class attributes for a run).

```bash
sbatch --time=2:00:00 --cpus-per-task=6 --mem=32G --gres=gpu:rtx_3090:1,tmpdisk:200000 \
    isaac/watcloud.sbatch isaac/tools/quarter_batch.py --copies 8 --seeds 0:8 --profile quarter2 --fold2 two \
    --drag none --video --out outputs/isaac/quarter/q2
```

Sync code with `tar czf - <files> | ssh watcloud 'cd ~/WorldFold && tar xzf -'` (rsync fails mid-protocol there).
Isaac scripts end with `os._exit` because Kit can hang on shutdown.

Natively on Windows, `setup_windows.ps1` builds `.venv-isaac` from the hashed locks here and `env_windows.ps1` sets
up the shell. `INSTALL_REVIEW.md` is the security review of that install.

## Using the env

```python
from isaac.sim.isaac_env import IsaacClothFoldBatch, IsaacClothFoldEnv, make_half_fold_env

env = IsaacClothFoldEnv(profile="friction")         # one env, one scene
batch = IsaacClothFoldBatch(8, profile="friction")  # 8 copies in one scene, batch.envs[i]
half = make_half_fold_env()                          # cloth_fold_rl.quarter_fold_env.HalfFoldEnv on Isaac
```

IsaacLab's simulation context is a singleton, so build one scene per process. The GPU sim is not deterministic
(two restores of one snapshot differ by 0.1 to 1.8 cm), so every comparison is statistical over paired seeds.

## Scoring

`cloth_fold_rl/fold_alignment.py` scores how well the layers line up (mean and edge spread, bar 1.0 / 1.5 cm for the
half fold, 2.0 / 3.0 cm for the quarter fold). A score only counts when `rl/precision_env.gate` passes it, meaning
the cloth is flat on the table, still, and both jaws are open and clear of it.

## World model (full-state predictor)

`world_model/` collects data for the half fold's world model and scores a trained model live. The state is the half
fold's 393 values (121 cloth vertices, then 15 per arm) and the action its 12. Training and the benchmark are
sim-agnostic and live in `cloth_angles/` (`train_state_predictor`, `benchmark_state_predictor`). `docs/world_model.md`
has the protocol.

- `world_model/collect.py` records episodes of two kinds into one store format. `play` has both arms pinch random
  reachable vertices on their own side of the cloth and lift, drag, shake, drop or force-release them. The scripted
  half fold (`world_model/half_fold.py`, the place-servo base `rl/precision_env` runs) records as `expert`,
  `expert_noisy` and `expert_release`. Frames from the imitation camera rig (RGB, plus depth in mm from the main and
  wrist cameras) go to `frames_<n>.npz` beside each episode unless `--no-frames`.
- `world_model/run.py` predicts each next state before the step and scores it after. It drives the scripted half fold
  by default, `--driver play` plays, `--actions` replays a collected episode. The predictor never drives.

Smoke run, two train and one test episode per kind (about 4,000 control steps). Submit each job once the last ends.
Training and the benchmark need no GPU and also run in the `.venv` here.

```bash
J="sbatch --gres=gpu:rtx_3090:1,tmpdisk:200000 isaac/watcloud.sbatch"
$J -m isaac.world_model.collect --per-kind 3 --test-per-kind 1 --play-steps 200 \
  --output outputs/cloth_angles/isaac_wm_smoke
$J -m cloth_angles.train_state_predictor --data outputs/cloth_angles/isaac_wm_smoke --steps 500 \
  --output outputs/cloth_angles/isaac_wm_smoke_model/state_l1_seed0.pt
$J -m cloth_angles.benchmark_state_predictor --data outputs/cloth_angles/isaac_wm_smoke --steps 500 \
  --output outputs/cloth_angles/isaac_wm_smoke_benchmark
$J -m isaac.world_model.run --checkpoint outputs/cloth_angles/isaac_wm_smoke_model/state_l1_seed0.pt
```

Full collection in two jobs, each under the 12 h limit, then the benchmark on their union.

```bash
$J -m isaac.world_model.collect --output outputs/cloth_angles/isaac_wm_a
$J -m isaac.world_model.collect --kinds play --seed-base 1100000 --output outputs/cloth_angles/isaac_wm_b
$J -m cloth_angles.benchmark_state_predictor --data outputs/cloth_angles/isaac_wm_a \
  outputs/cloth_angles/isaac_wm_b --output outputs/cloth_angles/isaac_wm_benchmark
```

Each collected line (`EPISODE ...`) carries the episode's steps per second and its disturbance statistics. No Isaac
dataset or checkpoint exists yet.

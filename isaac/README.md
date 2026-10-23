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

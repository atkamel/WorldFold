# isaac: WorldFold in Isaac Sim

`IsaacClothFoldEnv` is a drop-in for `mujuco.sim_main.ClothFoldEnv` in
`joint_delta` mode: same action, observations, reward, and termination, so
`cloth_fold_rl.fold_env.SingleCornerFoldEnv(base_env=IsaacClothFoldEnv())`
runs the SB3 pipeline unchanged. Scene constants are shared through
`mujuco/cloth_params.py`. Design: `docs/superpowers/specs/2026-09-24-isaac-sim-port-design.md`.

Milestone 1 covers the environment only. Not ported: `ee_delta` mode, the
physical grabber, mass/friction/damping randomisation, the viewer.

## FOR ZECH: getting into WATcloud

Docs: https://cloud.watonomous.ca/docs/compute-cluster/ssh

- Username: `zechwang1204`
- Login node: `wato-login1.ext.watonomous.ca` (or `wato-login2`)
- SSH key: `~/.ssh/id_ed25519_watcloud` (the public key is registered in the
  WATcloud onboarding form; `~/.ssh/config` applies it to `*.watonomous.ca`,
  but its `User` line still says `YOUR_WATCLOUD_USERNAME`, so fix that or pass
  the user explicitly)

```powershell
ssh zechwang1204@wato-login1.ext.watonomous.ca
```

Do not pass the public key text with `-i`; `-i` wants the private key path.

Once in, GPUs are only on compute nodes via SLURM. What is free and what is
queued:

```bash
sinfo -o "%P %D %t %G"            # node state and GPUs (drain = unavailable)
squeue --start -u $(whoami)       # your jobs and estimated start times
```

RTX 3090 / 4090 nodes are usually taken; the 2080 Ti node (`delta-slurm1`)
is the reliable one for Isaac Sim 4.5. Hold a GPU while waiting with a batch
job and attach later:

```bash
sbatch --job-name isaac3090 --gres shard:rtx_3090:16384,tmpdisk:40960 --cpus-per-task 8 --mem 32G --time 4:00:00 --wrap "sleep infinity"
srun --jobid <JOBID> --pty bash   # once RUNNING
scancel <JOBID>                   # when done, so the GPU is not held idle
```

NGC: `docker login nvcr.io` with username `$oauthtoken` and an API key from
https://ngc.nvidia.com (profile > Setup > Generate API Key). The key is shown
once; generate a new one if lost. Never commit it.

The Docker image lives on the node's `/tmp` and is re-pulled every job.

## Running on WATcloud

Isaac Sim needs an RTX GPU. The 2080 Ti nodes run Isaac Sim 4.5.0.

```bash
# login node
srun --gres shard:rtx_2080_ti:10240,tmpdisk:40960 --cpus-per-task 4 --mem 16G --time 4:00:00 --pty bash

# compute node
slurm-start-dockerd.sh
docker login nvcr.io            # user: $oauthtoken, password: NGC API key
docker pull nvcr.io/nvidia/isaac-sim:4.5.0
docker run --name isaac-sim --entrypoint bash -it --gpus all --rm --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v ~/WorldFold:/workspace/WorldFold \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  nvcr.io/nvidia/isaac-sim:4.5.0

# inside the container
./python.sh -m pip install gymnasium
./python.sh /workspace/WorldFold/isaac/smoke_test.py --mode state
./python.sh /workspace/WorldFold/isaac/smoke_test.py --mode hybrid
```

The first run downloads the `so101-nexus` wheel and unpacks the SO101 MJCF
and meshes into `isaac/assets/` (gitignored). The container cannot install
`mujoco` or `so101-nexus` (they need Python 3.12), which is why nothing in
`isaac/` imports `sim_main`.

## Using the env

```python
from isaac.isaac_env import IsaacClothFoldEnv
from cloth_fold_rl.fold_env import SingleCornerFoldEnv

env = SingleCornerFoldEnv(base_env=IsaacClothFoldEnv(observation_mode="state"))
```

Only one env per process: Isaac Sim's `World` is a singleton. Use
`observation_mode="hybrid"` for RGB and depth from the shared camera pose.

## Knobs

Isaac-only tuning constants sit at the top of `isaac_env.py`: physics dt,
cloth stiffness and damping, particle offsets, joint drive gains, the cloth
speed limit used for the failure check. Everything geometric (table, cloth
size, arm bases, camera, grasp radius) lives in `mujuco/cloth_params.py`.

## Grasp

Same cheat as MuJoCo's weld: when a gripper closes within `GRASP_RADIUS` of
its corner particle, that particle's mass is set to zero and its position is
written to the gripper frame every substep. Opening restores the mass.

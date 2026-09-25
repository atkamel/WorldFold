# isaac: WorldFold in Isaac Sim

`IsaacClothFoldEnv` is a drop-in for `mujuco.sim_main.ClothFoldEnv` in
`joint_delta` mode: same action, observations, reward, and termination, so
`cloth_fold_rl.fold_env.SingleCornerFoldEnv(base_env=IsaacClothFoldEnv())`
runs the SB3 pipeline unchanged. Scene constants are shared through
`mujuco/cloth_params.py`. Design: `docs/superpowers/specs/2026-09-24-isaac-sim-port-design.md`.

Milestone 1 covers the environment only. Not ported: `ee_delta` mode, the
physical grabber, mass/friction/damping randomisation, the viewer.

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

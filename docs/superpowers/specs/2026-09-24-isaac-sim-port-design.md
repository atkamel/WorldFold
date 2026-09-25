# Isaac Sim port of ClothFoldEnv: milestone 1 design

Date: 2026-09-24

## Goal

Run the single-corner fold task in NVIDIA Isaac Sim instead of MuJoCo, for
more realistic cloth, RTX rendering of RGB and depth, and a path to sim-to-real
on the physical SO101 arms.

Milestone 1 is the environment only: a Gymnasium env that is a drop-in
replacement for `ClothFoldEnv` in `joint_delta` mode, so
`cloth_fold_rl.fold_env.SingleCornerFoldEnv` and the SB3 pipeline run against
it unchanged. Training, demo collection, and the world model are later
milestones. Existing MuJoCo checkpoints are not expected to transfer.

## Non-goals (milestone 1)

- `ee_delta` action mode and the differential IK.
- Friction-based gripping. The weld cheat stays.
- Mass, friction, and damping domain randomisation. Only the cloth XY offset
  (`cloth_pose` option) is supported, because the fold wrapper's jitter needs it.
- Quarter fold, physical grabber, `cloth_angles` data collection, the mjviser
  viewer.
- Vectorised or GPU-parallel envs (Isaac Lab).
- Interactive GUI on the cluster.

## Environment and versions

- Isaac Sim 4.5.0 container (`nvcr.io/nvidia/isaac-sim:4.5.0`), the version
  that runs on the RTX 2080 Ti nodes available on WATcloud. Newer versions on
  3090/4090 nodes are a later concern.
- Inside the container: `pip install mujoco so101-nexus gymnasium numpy` into
  Isaac's Python (`./python.sh -m pip install ...`). `mujoco` and `so101-nexus`
  are needed only so `mujuco/sim_main.py` can be imported for its constants and
  the SO101 MJCF asset path.
- Runs headless. Images come from the Isaac camera sensor, not a window.

## Package layout

```
isaac/
  __init__.py
  README.md            how to run on WATcloud (SLURM, container, pip installs)
  convert_so101.py     one-shot: so101-nexus MJCF -> isaac/assets/so101.usd
  assets/so101.usd     committed output of convert_so101.py
  isaac_env.py         IsaacClothFoldEnv
  smoke_test.py        runnable check: contract + scripted grasp and lift
```

One change outside the package: `cloth_fold_rl/fold_env.py` stops reading
MuJoCo internals (`env.data.xpos`, `env._corner_ids`, `env._site_id`) and
calls two new accessor methods instead. `ClothFoldEnv` gains those two methods.

## Interface contract

`IsaacClothFoldEnv(control_dt=0.05, max_episode_steps=200,
observation_mode="state", image_size=(84, 84), n_cloth_samples=9, n_tasks=4,
grasp_corners=None, grasp_radius=GRASP_RADIUS)`

- `action_space`: `Box(-1, 1, (14,))`. Slots 0..4 left joint deltas, 6 left
  gripper, 7..11 right joint deltas, 13 right gripper. Slots 5 and 12 (rotation
  deltas in `ee_delta` mode) are ignored, as in `joint_delta` mode today.
- `observation_space`: same `Dict` as `ClothFoldEnv` for the same
  `observation_mode`:
  - `proprio` (per arm: 5 joint pos, 5 joint vel, gripper command, EE pos 3,
    EE quat 4 wxyz, EE lin vel 3, EE ang vel 3, grasp flag) = 25 per arm, 50 total.
  - `task` (one-hot n_tasks, stage, goal corners 12, corner progress 4,
    time left) = 22 for n_tasks=4.
  - `cloth_state` (state/hybrid): corners 12, corner vel 12, K samples,
    CoM 3, height min/max/mean 3, corner-to-goal 12.
  - `image` (pixels/hybrid): `uint8 (H, W, 3)`.
  - `depth` (pixels/hybrid): `float32 (H, W, 1)`, metres, 0 = invalid, after
    the same noise, quantisation, and grazing-angle model as MuJoCo.
- `reset(seed, options)`: honours `cloth_pose` (XY offset), `task`, and
  `goal_pose`. Ignores `randomization`. Returns the same `info` keys.
- `step(action)`: same reward (`-mean corner dist` plus success bonus), same
  `terminated`/`truncated` rules, same `info` keys. `left_ik_success` and
  `right_ik_success` are always `True` (no IK).
- Public helpers used by wrappers: `grasp_active(prefix)`, `set_gripper`,
  `apply_joint_delta`, `_failed()`, `_step_count`, `max_episode_steps`,
  `prefixes`, `n_tasks`, `weld_mask`, and the two new accessors below.

New accessors on both envs:

```python
def corner_positions(self) -> np.ndarray   # (4, 3) world, order cloth_0, cloth_10, cloth_110, cloth_120
def gripper_position(self, prefix) -> np.ndarray   # (3,) world, gripper frame origin
```

`SingleCornerFoldEnv._corners` and `_gripper_pos` call these.

## Shared constants

`isaac_env.py` imports from `mujuco/sim_main.py`: `TABLE_TOP_Z`,
`CLOTH_COUNT`, `CLOTH_SPACING`, `CLOTH_MASS`, `CLOTH_HALF`, `ARM_XML_PATH`,
`ARM_BASE_LEFT`, `ARM_BASE_RIGHT`, `ARM_BASE_QUAT`, `ARM_JOINTS`,
`GRIPPER_OPEN`, `GRIPPER_CLOSED`, `JOINT_DELTA_SCALE`, `GRASP_CORNERS`,
`GRASP_RADIUS`, `HOLD_STEPS`, `WORKSPACE_XY`, `SUCCESS_FOLD_SCORE`,
`CORNER_PLACED_DIST`, `TASK_NAMES`, `DEPTH_*`, `CAMERA_POS`, `CAMERA_TARGET`,
and the functions `camera_axes` and the depth noise model (extracted from
`ClothFoldEnv._sensor_depth` into a module-level function so both envs call
it). Geometry cannot drift between the two sims.

Isaac-only knobs live at the top of `isaac_env.py`, each a plain constant:
physics dt (1/240), cloth stretch/bend/shear stiffness, cloth damping,
particle contact offset, joint drive stiffness and damping, gripper drive
gains, pinned-particle velocity limit for the failure check.

## Scene

Built in Python on `isaacsim.core.api.World` at env construction.

- Ground plane and a fixed table box, top at `TABLE_TOP_Z`, 0.6 m square.
- Cloth: PhysX particle cloth from a plane mesh of `CLOTH_COUNT` x
  `CLOTH_COUNT` vertices spaced `CLOTH_SPACING`, centred at the origin, resting
  on the table. Vertices ordered row-major with index `ix * CLOTH_COUNT + iy`
  so `GRASP_CORNERS`, corner indices, and sample indices match MuJoCo. Total
  mass `CLOTH_MASS`. Self-collision off (matches MuJoCo).
- Arms: `isaac/assets/so101.usd`, produced by `convert_so101.py` from
  `ARM_XML_PATH` with the Isaac MJCF importer. Two references at
  `ARM_BASE_LEFT` and `ARM_BASE_RIGHT`, both rotated by `ARM_BASE_QUAT`. Joints
  in `ARM_JOINTS` plus `gripper` are position-driven; joint limits come from
  the MJCF. A gripper frame prim per arm reproduces the MJCF `gripperframe`
  site offset (read from the XML by the converter and stored in the USD).
- Camera: `isaacsim.sensors.camera.Camera` at `CAMERA_POS` aimed at
  `CAMERA_TARGET` using `camera_axes`, resolution `image_size`, vertical FOV
  matching the MJCF camera default. Annotators: RGB and
  `distance_to_image_plane`.
- Lighting: one dome light plus one distant light. Not tuned in milestone 1.

## Stepping

`control_dt / physics_dt` substeps per `step` (12 by default). Each substep:
apply pinned-particle targets, then `world.step(render=False)`. After the last
substep, if images are requested, one `world.step(render=True)` equivalent
(render call) and read the annotators. Reset settles the cloth for the same
1.0 s of physics as MuJoCo (`SETTLE_STEPS * ARM_TIMESTEP`), then zeroes the
step counter.

## Grasp

Same semantics as the MuJoCo weld:

- `set_gripper(prefix, command)`: hysteresis identical to MuJoCo. When closed,
  for each corner index in `grasp_corners[prefix]` that is not already pinned
  and is allowed by `weld_mask[prefix]`, if the distance from the gripper frame
  to that particle is below `grasp_radius`, capture the offset in the gripper
  frame and pin the particle. When open, unpin all.
- Pin: set the particle's inverse mass to zero and, each substep, write its
  position to gripper frame plus captured offset with zero velocity. Unpin
  restores the original mass.
- `grasp_active(prefix)` is true when any of that arm's corners is pinned.
- Fallback if per-particle mass is not writable in 4.5: create a
  `PhysxPhysicsAttachment` between the gripper link and the cloth with a small
  attachment radius around the corner, delete it on release. The spike below
  decides which.

## Observations

- Joint pos/vel from the articulation view. Gripper slot is the commanded
  target, as in MuJoCo.
- EE pose and velocity from the gripper frame prim's world pose and the
  gripper link's linear and angular velocity.
- Cloth positions and velocities from the particle cloth prim. Corner and
  sample indices as in MuJoCo.
- Depth: annotator output run through the shared noise model with the same
  per-pixel angle computed from the camera FOV.

## Failure check

`_failed()` returns true if any cloth vertex leaves `WORKSPACE_XY`, or any
particle speed exceeds the velocity-limit knob (replaces MuJoCo's `qacc`
check), or any position is non-finite.

## Verification

All checks run on a WATcloud GPU node inside the container. No local machine
runs Isaac Sim.

1. `mujuco/sim_main.check_contract(IsaacClothFoldEnv(observation_mode=m))`
   for `m` in `state`, `hybrid`: passes, and the printed signature equals the
   MuJoCo env's for the same mode (compared from a MuJoCo run on any machine).
2. `isaac/smoke_test.py`: reset, drive the left arm to its corner with joint
   deltas (scripted, using the articulation's own Jacobian or a fixed joint
   sequence found in the spike), close, assert `grasp_active("left_")`, apply
   lift deltas for 20 steps, assert the corner's z rose by at least 3 cm and
   the two anchor corners moved less than 2 cm. Exits non-zero on failure.
3. `SingleCornerFoldEnv(base_env=IsaacClothFoldEnv(...))` constructs and runs
   50 random steps without error (covered inside `smoke_test.py`).
4. Throughput: `smoke_test.py` prints control steps per second for `state` and
   `hybrid` at 84x84 on the 2080 Ti. Target is at least 5 steps/s in `state`
   mode; below that, cloth resolution or substeps get revisited before
   milestone 2.
5. Existing MuJoCo tests still pass after the `fold_env.py` accessor change:
   `python -m pytest mujuco/tests cloth_angles/tests`.

## Spike (before env code)

Throwaway script, not kept: import the SO101 MJCF into Isaac Sim 4.5, print
joint names, limits, and drive settings, and locate the gripper frame; create
a particle cloth and attempt to zero one particle's inverse mass and move it.
Outcome decides the grasp mechanism and whether the MJCF importer output is
usable or the arm needs the URDF from the SO-ARM100 repo instead.

## Risks

- MJCF importer fidelity: mesh collisions, joint axes, drive gains may differ.
  Mitigation: converter prints a joint table for eyeballing against MuJoCo, and
  drive gains are knobs.
- Gripper frame offset: if the importer drops the site, the converter reads it
  from the XML and adds a fixed prim.
- Cloth behaviour differs from MuJoCo's flex grid; stiffness and damping need
  hand tuning. Knobs exposed, no auto-tuning in milestone 1.
- Throughput on the 2080 Ti with RTX rendering per step may be low. Measured
  in verification step 4.

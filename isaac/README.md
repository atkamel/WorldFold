# isaac: WorldFold in Isaac Sim, on LeHome's stack

`IsaacClothFoldEnv` is a drop-in for `mujuco.sim_main.ClothFoldEnv` in
`joint_delta` mode: same 14-dim action, same observation dict, same reward and
termination, so `cloth_fold_rl`'s wrappers (`SingleCornerFoldEnv`,
`HalfFoldEnv`) run on it unchanged. Underneath it is LeHome's simulation stack
(lehome-challenge `a805ad2`, upstream): IsaacLab on the CPU device with PhysX GPU
dynamics, LeHome's SO101 robot, and LeHome's particle-cloth machinery.

## Running on Modal

```bash
pip install modal && modal setup                      # once
modal run isaac/modal_isaac.py                        # smoke test, state and hybrid in parallel, L40S
modal run isaac/modal_isaac.py::scripted --episodes 10   # open-loop scripted half fold over 10 seeds
modal volume get worldfold-isaac smoke/<stamp> .
```

The image is LeHome's locked `lehome-challenge` env (Isaac Sim 5.1.0, Python
3.11, torch 2.7.0) plus LeHome's IsaacLab fork and assets, all pinned to commits;
the same stack as `isaac_image` in `teacher/modal_teacher.py` on `ROY-vla-teacher`.
The image build (~10 min) happens once and doesn't use a GPU. Isaac Sim 5.1 is
the last release with PhysX particle cloth (6.0 replaced the cloth prims with
error stubs), so stay on 5.1. The earlier Isaac Sim 4.5 version of this env
(MJCF arms, weld grasp, WATcloud container) is in git history on
`feat/isaac-sim` and `feat/isaac-half-fold`.

## Using the env

```python
from isaac.isaac_env import IsaacClothFoldEnv, make_half_fold_env
from cloth_fold_rl.fold_env import SingleCornerFoldEnv

env = SingleCornerFoldEnv(base_env=IsaacClothFoldEnv(observation_mode="state"))
env = make_half_fold_env()          # cloth_fold_rl.quarter_fold_env.HalfFoldEnv on Isaac
```

One env per process (IsaacLab's simulation context is a singleton). About 5
control steps/s in state mode on an L40S: 5 physics steps of a 10,201-particle
cloth per 0.05 s step, about LeHome's own cost per simulated second.

## Scene

From LeHome:

- **Arms**: its SO101 follower USD (convex-decomposed jaws, capsule finger
  pads), its actuators (17.8 N m/rad, 10 N m), link gravity off, self-collision
  on. The USD root sits at a fixed offset from the MJCF base, so
  `usd_root_pose` mounts it there and the arms land exactly where
  `ARM_BASE_*` in `mujuco/cloth_params.py` put the MuJoCo arms (same
  kinematics, under 0.3 mm on six joint poses).
- **Cloth**: WorldFold's square mesh packaged like a LeHome garment (a USD plus
  a garment config) and built, read and reset by LeHome's `GarmentObject`,
  with the particle settings in its `particle_garment_cfg.yaml`, except three
  values (`isaac_env.py`): the whole sheet weighs `CLOTH_MASS` (0.05 kg;
  LeHome's 10 g per particle would make it ~100 kg), gravity is 1x (LeHome's
  undocumented 2x), and adhesion is 0 (LeHome's 0.1 pulls the cloth onto every
  rigid surface, the table included). 101x101 particles at 3 mm; the 121
  MuJoCo grid vertices are a subset, in the same row-major order.
- **Simulation**: IsaacLab `DirectRLEnv` on the CPU device; LeHome's IsaacLab
  fork forces PhysX GPU dynamics, which the particle cloth needs (on the CUDA
  device the grippers pass through it, LeHome issue #36). On the CPU device the
  cloth is read and reset through its USD points, as LeHome does.

From WorldFold (`mujuco/cloth_params.py`): the table, the cloth's size, the arm
base poses and the `main` camera.

**Reset** drops the cloth LeHome-style: `DROP_HEIGHT` (5 cm) above its resting
height with a random roll and pitch of up to `DROP_TILT_DEG` (10 deg), seeded
per episode, then settles for 1 s.

## Grasp

Nothing attaches the cloth: the jaws hold it by friction. Each finger carries
a high-friction lining on its inner face, as on the real gripper:
`PAD_LINING_THICKNESS` (3 mm) of flat pad over LeHome's capsule pads, so the
jaws meet at about -0.015 rad and closing to `GRIPPER_CLOSED` squeezes. Cloth
friction against rigid bodies comes from the cloth's particle material (0.5),
so the lining's own material mostly matters for rigid contacts.

`grasp_active(prefix)` cannot see an attachment, so it reports what the
wrappers ask: the gripper is commanded closed with one of its corners
(`grasp_corners`, filtered by `weld_mask`) within `grasp_radius` of the
gripper frame.

`isaac/pinch.py` solves top-down pinch poses (fingers down, jaw opening across
the corner) with IK on LeHome's URDF. A pinch with the fixed fingertip 1 cm
above the table lifts a corner 10+ cm (smoke test). The arm reaches straight
down to the table only within about 30 cm in front of its base, which covers
the cloth's near corners and not its far ones.

## Sensors

Each item matches MuJoCo's `ClothFoldEnv`:

- Camera `main`: `CAMERA_POS` aimed at `CAMERA_TARGET`, vertical FOV
  `CAMERA_FOVY_DEG`, `image_size` resolution, rendered once per control step.
  Depth is distance to the image plane run through the shared `sensor_depth`
  noise model. Other `camera_names` raise.
- Proprio: joints in MuJoCo order, commanded gripper, gripperframe site pose
  (LeHome's gripper link plus the MJCF site offset, which is the URDF's
  `gripper_frame_link`) with `mju_mat2Quat`'s quaternion sign, and site
  velocity `[angular, linear]` in world axes. Cloth velocities are the mean
  over the last control step.
- Scene look: the MJCF's two light directions and strengths plus the camera
  headlight. `LIGHT_INTENSITY` is not calibrated against MuJoCo.

## Knobs

Isaac-only constants sit at the top of `isaac_env.py`: physics dt, cloth
resolution and position, the three deviations from LeHome's cloth config, drop
height and tilt, lining thickness and friction, lights, and the cloth speed
limit used for the failure check. Everything geometric that MuJoCo shares
(table, cloth size, arm bases, camera, grasp radius) lives in
`mujuco/cloth_params.py`.

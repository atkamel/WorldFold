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
modal run isaac/modal_isaac.py::main                  # smoke test, state and hybrid in parallel, A10G
modal run isaac/modal_isaac.py::half_fold --episodes 3   # scripted friction half fold, a video per episode
modal volume get worldfold-isaac smoke/<stamp> .      # logs
modal volume get worldfold-isaac half_fold/<stamp> .  # half fold videos
```

The image is LeHome's locked `lehome-challenge` env (Isaac Sim 5.1.0, Python
3.11, torch 2.7.0) plus LeHome's IsaacLab fork and assets, all pinned to commits;
the same stack as `isaac_image` in `teacher/modal_teacher.py` on `ROY-vla-teacher`.
The image build (~10 min) happens once and doesn't use a GPU. Isaac Sim 5.1 is
the last release with PhysX particle cloth (6.0 replaced the cloth prims with
error stubs), so stay on 5.1. The earlier Isaac Sim 4.5 version of this env
(MJCF arms, weld grasp, WATcloud container) is in git history on
`feat/isaac-sim` and `feat/isaac-half-fold`.

## Running the imitation pipeline in the cloud

`imitation/` (the expert → BC → DAgger → student pipeline) runs on Linux as is. Only its drivers
(`scripts/*.ps1`) and install (`isaac/setup_windows.ps1`) are Windows-only. `scripts/f4_retrain.py`
is `scripts/f4_retrain.ps1` in Python: same stages, outputs and skip-if-done markers, so a run
started by either driver resumes in the other. `isaac/install_lehome.sh` builds the Isaac stack for
both cloud targets.

**Modal**: one command per job. `outputs/` and `docs/reports/media/` live on the volume under
`pipeline/`, so datasets and checkpoints persist and a rerun resumes.

```bash
modal run isaac/modal_isaac.py::pipeline --cmd "-m imitation.evaluate --backend isaac_friction ..."
modal run --detach isaac/modal_isaac.py::pipeline --background --timeout-min 720 \
    --cmd "scripts/f4_retrain.py --no-vision --final-n 100 --noise-n 50 --episodes 200 --bc-steps 20000"
modal volume get worldfold-isaac pipeline/logs/<tag>.log .
modal volume get worldfold-isaac pipeline/outputs .
```

A job runs on an A10G with 8 CPUs and 64 GB; `--timeout-min` (default 360) bounds what it can cost,
since the command is killed then. Isaac Sim 5.1 segfaults on NVIDIA's 610 driver branch, which some
L40S hosts run (A10G hosts were all on 580 in Ruby's probe), so a job on a 610 host stops at once with
`unsupported_driver`. The smoke test runs on an A10G too.

**WATcloud**: `isaac/watcloud.sbatch` builds `isaac/Dockerfile` once, saves it under `~/docker`,
and runs the command in it with the repo mounted:

```bash
sbatch --gres=shard:<gpu>:<MiB>,tmpdisk:61440 isaac/watcloud.sbatch scripts/f4_retrain.py --no-vision
```

Isaac Sim 5.1 lists an RTX 4080 with 16 GB as its minimum, so the 2080 Ti nodes used for Isaac
Sim 4.5 fall short. The job also stops unless the node's driver is on the 580 branch, the only one
seen working (`ISAAC_ANY_DRIVER=1` overrides).

Neither target has run the pipeline yet. #17's datasets and checkpoints are gitignored, so a
cloud run starts from collection unless they are uploaded to `pipeline/outputs/imitation/`.

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

From WorldFold (`mujuco/cloth_params.py`): the table height, the cloth's size,
the arm base poses and the `main` camera. Isaac-only: the cloth sits 13.5 cm
closer to the arms (`CLOTH_CENTER`), the half fold's reset jitter is ±1 cm
(`CLOTH_JITTER`; MuJoCo uses ±2.5 cm), and the table is 0.70 m. Pointing
straight down, the gripper reaches the table between about 8 and 32 cm from its
arm's base, and the 30 cm fold spans that whole range: far corners 32 cm out,
near corners 7 cm out. More jitter pushes one end out of reach. Tilting the
fingers reaches further (39 cm at 35 deg), but the 5-joint arm can only lean
its fingers along the line out from its base, so a tilted pinch has to sweep
sideways along the cloth's edge, and that sweep didn't hold the cloth.

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
the corner) with IK on LeHome's URDF. `isaac/half_fold_demo.py` is a scripted
two-arm half fold built on it: each arm pinches its far corner, carries it along
an arc over the fold line onto its near corner through IK waypoints about 2 cm
apart, releases and backs off. The env moves each joint at most 0.05 rad per
step from where it is, and the joints lag by different amounts, so the script
waits at each waypoint until the joints arrive; with a fixed step count per
pose the grippers wandered off the arc and swung toward the middle. The fold
takes about 230 steps, so the demo runs 400-step episodes (`HalfFoldEnv`: 250).

The demo scores the whole cloth (`fold_error`): an ideal half fold puts each
far-half grid vertex on the start of its mirror image across the middle row and
leaves the near half where it started. `HalfFoldEnv`'s own test only checks the
two carried corners. Three episodes on Modal: fold score 0.40, 0.44 and 0.61
(mean vertex error 4.9, 4.6 and 3.2 cm), one meeting `HalfFoldEnv`'s success
test. None meets the demo's folded bar (mean under 2 cm, max under 5 cm): the
near edge slides about 4.5 cm during the fold, and the folded edge sags short
of the near edge between the two carried corners.

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

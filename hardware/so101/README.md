# SO-101 hardware, open-loop half fold

What is here, what state it is in, and how to run it on a machine that has never seen this repo.

## Status (2026-11-14)

- **Three ways to fold**, most capable first.
  - `fold_live.py` watches through the iPhone the whole time (RGB and LiDAR), masks the arms out by height, moves
    the place point when the bottom layer slides, and records the run.
  - `fold_camera.py fold` looks once, then folds from where the cloth lies.
  - `replay.py` on the committed trajectory is blind, with the cloth placed by ruler.
- **What the camera does not see.** The folded-over half (dark on dark) and the held corners (under the grippers) are
  not tracked. The grasp check on the jaw readings covers the grasp.
- **What it is not** is the Isaac half-fold expert (`isaac/experts/place_servo.py`), which reads the full cloth state.
- **Verified** by 65 tests on a fake servo bus and synthetic camera frames, and a dry run of the blind trajectory
  against both arms' recorded limits (0 clamped steps).
- **Never run on a real arm or a real phone.** The radian-to-tick map is a hypothesis until `replay.py --check`
  passes on each arm. The Record3D connection follows `hardware/iphone_lidar/usb_viewer.py`, which has
  not seen a phone either. Frame orientation and how well the depth lines up with
  the RGB are unknown until a real frame is recorded.
- **Both followers are configured**, with `main`'s names and folders. `arm2` is the left arm and `arm1` the right,
  seen from behind. Arm 2's motors 2 and 3 have identical limits, so watch its elbow in the check.
- **Known risks.**
  - A 30 cm cloth does not fit the arms' reach at both ends. With the near edge 8 cm in front of the base origins
    the far pinch is tilted about 25 degrees and the release lands about 2.4 cm past the near corner, because the
    arms cannot reach closer than 10 cm with the fingers down. Every look prints both misses.
  - The jaw target is the closed limit, so the squeeze on thin cloth may be weak.
  - Dark cloth can absorb LiDAR. Pixels with no depth count as flat, so the height mask only removes what it can see.

## Rig

- Two followers flat on the table, facing the same way, base origins 30 cm apart ("left" is on the left seen from
  behind the arms).
- Cloth 30 cm square, centred between the arms, near edge 1.5 cm in front of the bases' front edges. Each side edge
  of the cloth then lines up with the centre of an arm.
- Start with both arms roughly upright over their bases.

## Setup on a new machine

```bash
git clone https://github.com/LegendaryAKx3/WorldFold.git && cd WorldFold
git checkout feat/half-fold-deploy
python3 -m venv .venv-hw
.venv-hw/bin/pip install feetech-servo-sdk pyserial
```

That is all `replay.py` needs. The trajectory is committed at `hardware/so101/trajectories/half_fold_30cm.json`.

Then tell the repo about this machine's arms.

1. Find each arm's port with `ls /dev/tty.usbmodem*` (plug one arm in at a time) and put the ports in
   `hardware/so101/arms.json` under `"left"` and `"right"`. Ports differ per machine.
2. Limits for both followers travel with the repo (`arm2/limits.json`, `arm1/limits.json`). To redo one,
   set `MOTOR_ID` in `set_limits.py`, run `.venv-hw/bin/python set_limits.py <arm>` per motor, and sweep the joint to
   both hard stops. For the gripper, record the fully closed end too, because the map takes its lower tick as closed.

## Run order

From `hardware/so101/`. Stop at the first step that fails. Keep a hand on the power. Ctrl-C stops sending and leaves
the torque on, so the arms hold where they are.

```bash
# 0. no motion: live joint angles through the map. Arm limp and stretched straight forward should read near 0
#    on joints 1 to 5; jaw shut should read near -10 and fully open near 120
../../.venv-hw/bin/python replay.py --read arm2
../../.venv-hw/bin/python replay.py --read arm1

# 1. joint signs, one arm at a time: each joint moves +8 deg and back, in the model's positive direction
../../.venv-hw/bin/python replay.py --check arm2
../../.venv-hw/bin/python replay.py --check arm1

# 2. dry run: no motion, should report 0 clamped steps
../../.venv-hw/bin/python replay.py trajectories/half_fold_30cm.json --left arm2 --right arm1 --dry-run

# 3. one arm, no cloth, slow
../../.venv-hw/bin/python replay.py trajectories/half_fold_30cm.json --left arm2 --right arm1 --only left --speed 0.3

# 4. both arms with the cloth (about 34 s at the default half speed), with the run log and the grasp check
../../.venv-hw/bin/python replay.py trajectories/half_fold_30cm.json --left arm2 --right arm1 \
    --log run1.json --grasp-check
```

What to look for

- **Step 1.** With the arm upright, model-positive shoulder_pan turns the arm to its right, seen from
  behind. The others are easiest to judge against the other arm once one is known good. A joint that goes the wrong way
  gets `"sign": {"<motor>": -1}` in `<name>/urdf.json`. The check does not move the gripper.
- **Step 3.** The fingertip touches the table 37.8 cm in front of the base origin (31.3 cm in front of the base's
  front edge), in line with the arm, and the jaw closes fully there. It releases 9 cm in front of the origin. If the
  touch-down is off by more than about 1 cm, fix the zero in `<name>/urdf.json` (`"zero": {"<motor>": ticks}`)
  before using a cloth.
- **Step 3, at the far pinch.** The fingers are tilted, so the model puts the fixed fingertip 2.6 mm below the table
  there. The arm presses on the table at that point, which is expected. Stop if it bows or stalls.
- **Step 4.** Each arm pinches its far corner, carries it over in an arc, opens part way, lifts 5 cm, opens fully
  and stands back up. Before the pinch each jaw shuts once in the air, which is the grasp check's empty reading.

## Folding with the camera

Extra packages on the rig machine, on top of `feetech-servo-sdk`. The arm kinematics are plain numpy
(`kinematics.py`), so neither MuJoCo nor so101-nexus is needed.

```bash
.venv-hw/bin/pip install numpy scipy pillow record3d==1.4.1
```

`record3d` builds from source and needs CMake and a C++ compiler. On the phone, open the Record3D app, turn on USB
Streaming mode in its settings, and do not record (the record button pauses the stream).

Phone placement. Centred between the arms and behind them, lens 60 to 80 cm above the table, tilted 45 to 60 degrees
down, on a rigid mount, with the whole cloth and a margin around it in view. The edge of the image nearest the arms
is `--near` (default `bottom`). Use even light, since a hard shadow reads as cloth.

Do steps 0 to 3 of the run order first. Then, from `hardware/so101/`, with both arms standing over their bases.

```bash
# a. once per camera position: cloth at the ruler position from the Rig section
../../.venv-hw/bin/python fold_camera.py calibrate

# b. no motion: move the cloth a few cm, look, and open look.png
../../.venv-hw/bin/python fold_camera.py look

# c. fold from where the cloth is; flags after the mode's own go to replay.py
../../.venv-hw/bin/python fold_camera.py fold --left arm2 --right arm1 --grasp-check --log run1.json
```

- `look.png` shows the outline it found in green, each arm's pinch point in red and place point in blue. The red
  circles must sit just inside the two far corners. `fold` waits for Enter after writing it.
- The look refuses a dark shape whose sides are more than 15% off the cloth's side, and a pinch the arm model cannot
  reach.
- `--image photo.jpg` runs any mode on a saved frame.
- Calibration is only as good as the ruler placement of the cloth and the joint map. Redo it if the phone moves.

## Folding with the camera watching throughout

Same setup and calibration as the single look. Start with a dry run, which moves nothing and records a few seconds.

```bash
../../.venv-hw/bin/python fold_live.py --left arm2 --right arm1 --dry-run
../../.venv-hw/bin/python fold_live.py --left arm2 --right arm1 --grasp-check
```

- **Look.** Corners from the first frame. With depth, the table plane is fitted from that frame and every pixel more
  than 2 cm above it is left out, so the arms may be in view. Keep the table mostly clear for that first frame.
- **Track.** Every frame, it follows the cloth's near edge, which stays in view through the carry.
- **Correct.** At 60% of the carry, if the near edge has moved 3 mm or more, the place point moves with it (at most
  3 cm), the fold is planned again and the arms continue on the new plan.
- **Record.** Each run goes to `runs/<time>/` with every frame (`frames/*.jpg`, depth in `*.npz`), every control
  step's targets and measured joints (`steps.jsonl`), the look, both plans, the grasp readings, the correction and a
  fold score (`events.jsonl`). About 130 MB per minute at 10 frames per second. `recorder.load(path)` reads a run.
- `--no-depth` uses RGB only, and then the arms must stay off the cloth's outline in the image.

## Grasp check and run log

- Above the far corner each jaw shuts once on nothing. That reading is the empty reference.
- `--grasp-check` reads each jaw 2 cm into the lift. A jaw that is not at least `--grasp-margin` degrees (default 0.5)
  further open than its empty reading has no cloth. Both arms then go back down, open, and pinch again, up to
  `--retries` times (default 2), and carry on regardless after that.
- The printed readings from the first run say whether 0.5 degrees is the right margin for this cloth. A thin cloth
  may read almost the same as empty, in which case the check reports misses that are not real.
- `--log FILE` writes the measured joint angles for every step, to compare with the trajectory afterwards.

## Scoring a fold from photos

Take one photo before the fold and one after, from a camera that does not move, with the arms out of the frame.
JPEG or PNG (an iPhone's HEIC needs converting). This runs on any machine with `numpy`, `scipy` and `pillow`.

```bash
python fold_score.py before.jpg after.jpg --near bottom --out overlay.png
```

`--near` says which side of the image the cloth's near edge is on. It prints, in cm, how far the folded edge is from
the middle of the cloth (`far_edge_mean`, `far_edge_max`), how far the cloth pokes outside its starting square
(`overhang_max`), and the area and coverage of the folded half. Look at the overlay before trusting the numbers. A
wrong `--near`, a shadow or an arm in the frame gives a plausible wrong score.

## Changing the fold

Planning needs only `numpy`.

```bash
python fold_traj.py --out trajectories/my_fold.json --cloth 0.30 --near 0.08 --closed -0.17
```

- `--cloth` cloth side (m). `--near` near edge in front of the base origin (m). The base's front edge is 6.5 cm in
  front of the origin. `--spacing` between the base origins (m).
- `--near 0.07` cuts the far pinch's tilt to about 21 degrees and puts the release 3 cm inside the near edge.
- `--closed` is the jaw target at the pinch (rad). -0.1745 is the model's closed limit.

## Arm configuration

- `arms.json` names every arm and its serial port, in order. The first follower listed is the left one, seen from
  behind the arms. It also holds `layout` (`side_by_side` or `facing`), `spacing_cm` between the bases, and `leaders`,
  which pairs a follower with a limp leader arm for teleop.
- `arms.py` reads it. `arms.port(name)` and `arms.folder(name)` are what the scripts use, and `add_arm_argument` and
  `resolve` give `idle.py` and `set_limits.py` their required `--arm`.
- Each arm has a folder (`arm1/`, `arm2/`, `leader/`) with `limits.json` from `set_limits.py`, `idle_pose.json` from
  `idle.py record`, `twin.json` and `presets.json` from the teleop page, and an optional `urdf.json` that overrides
  the tick map below.
- Ports differ per machine. Find one with `ls /dev/tty.usbmodem*` with only that arm plugged in.

## Teleop and idle

```bash
../../.venv-hw/bin/python ui.py                       # control page for every arm, http://localhost:8101
../../.venv-hw/bin/python ui.py arm2                  # one arm
../../.venv-hw/bin/python idle.py --arm arm2 record   # torque off, pose by hand, Enter saves the idle pose
../../.venv-hw/bin/python idle.py --arm arm2          # glide to the idle pose and hold
../../.venv-hw/bin/python idle.py --arm arm2 release  # torque off, support the arm
```

The page has keyboard and drag control of the gripper position, a mirror mode, leader-arm mode, and a phone page.
The docstring at the top of `ui.py` has the details.

## Flags not shown above

Every script prints its flags with `--help`. The ones the sections above do not use.

| script | flag | what |
|---|---|---|
| `replay.py`, `fold_live.py` | `--allow-clamp` | play a fold the arm's limits clamp by more than 2 degrees |
| `replay.py` | `--no-gripper` | leave the jaws alone |
| `fold_camera.py` | `--device N` | which Record3D phone, if more than one |
| `fold_camera.py` | `--dark N` | calibrate with a fixed grey threshold for the cloth, in place of the automatic one |
| `fold_camera.py` | `--near-edge M`, `--cloth M`, `--spacing M` | the ruler position the calibration assumes, the cloth side and the base spacing, in metres |
| `fold_camera.py` | `--out FILE` | where the look image goes |
| `fold_live.py` | `--yes` | do not wait for Enter after the look |
| `fold_live.py` | `--settle S` | seconds to wait before the last frame, and the length of a dry run |
| `fold_live.py` | `--runs DIR` | where run folders go |
| `fold_score.py` | `--side M`, `--dark N` | the cloth side and a fixed grey threshold |

## What a run folder holds

| file | what |
|---|---|
| `meta.json` | the command's arguments and the camera calibration |
| `frames/000000.jpg`, `.npz` | each RGB frame, and its depth in millimetres (0 where there is no reading) |
| `frames.jsonl` | per frame, its number, time and camera intrinsics |
| `steps.jsonl` | per control step, the row, which plan, the targets and the measured joints in model radians |
| `events.jsonl` | `look`, `look_failed`, `grasp_check`, `place_correction`, `score`, `dry_run`, `stopped` |
| `look.png`, `plan_1.json`, `plan_2.json` | the look image, the first plan, and the corrected plan if there was one |
| `summary.json` | counts of frames written and dropped, steps and events |

`recorder.load(path)` returns all of it, and `recorder.load_frame(path, i)` returns one frame's RGB and depth in metres.

## Tests

`python -m pytest hardware/so101` from the repo root, in the pinned `.venv`. They use a fake servo bus, synthetic
camera frames and a fake `record3d`, so they need no arm and no phone. `test_kinematics.py` also checks the numpy
kinematics against MuJoCo when `mujoco` and `so101-nexus` are installed.

## The tick map

`q_model_rad = sign * (ticks - zero) * 2*pi/4096`.

- Motors 1 to 5 use zero 2047 and sign +1. `set_limits.py` puts the middle of each swept range at tick 2047 and the
  model's arm joints are symmetric about 0, so this holds if each joint was swept stop to stop.
- Gripper. The model's range is -10 to 100 degrees, so its zero is 10 degrees above the lower tick limit, taken to be
  the closed end.
- `<name>/urdf.json` overrides zero and sign per motor.

## Files

| file | what |
|---|---|
| `fold_live.py` | the fold with the iPhone watching throughout, recorded |
| `fold_camera.py` | one look through the iPhone, then the fold from where the cloth lies |
| `rgbd.py` | Record3D stream (RGB, depth), table plane, height above the table |
| `recorder.py` | saves a run, `load` reads it back |
| `kinematics.py` | the arm's forward and inverse kinematics in numpy |
| `fold_traj.py` | plans the half fold, writes a trajectory |
| `replay.py` | plays a trajectory, with `--check`, `--dry-run`, `--only`, `--speed`, `--no-gripper` |
| `trajectories/half_fold_30cm.json` | the fold for the rig above |
| `fold_score.py` | scores a fold from a before and an after photo |
| `arms.py`, `arms.json`, `arm1/`, `arm2/`, `leader/` | ports, limits, per-arm overrides |
| `set_limits.py`, `idle.py` | limit recording, idle pose |
| `ui.py`, `ui.html`, `phone.html` | teleop |
| `test_*.py` | tests, see above |

# SO-101 hardware, open-loop half fold

What is here, what state it is in, and how to run it on a machine that has never seen this repo.

## Bring-up log (2026-11-14, Windows laptop, branch `feat/iphone-lidar`)

Where the first session on the real arms stopped, and what to do next. The sections below it are the reference.

### Set up and checked

- **Arms.** Both followers on one Windows laptop: `arm1` on COM3, `arm2` on COM5 (`arms.json`, `PLATFORM = "windows"`
  in `arms.py`). `arm2` is the left arm. The leader arm has no Windows port and is left out.
- **Limits.** `check_limits.py` read every motor on both arms: min, max and homing offset match `limits.json`.
  Arm 2's joints 2 and 3 hold identical limits (870 / 3225); not swept by hand to confirm.
- **Idle poses.** Recorded for both arms in the folded rest pose, where they stay with torque off.
- **Twin (the 3D render).** `arm1/twin.json` is the calibration checked on the arm; `arm2/twin.json` is a copy of
  it, not calibrated on arm 2 itself.
- **Panel (`ui.py`).** Driven on both real arms. It shows the LiDAR point cloud and records runs into `runs/`. The
  record button was tried with the real phone and a stand-in servo bus, not yet with the real arms.
- **Phone.** Record3D over USB works on Windows once Apple Mobile Device Support is installed
  (`winget install Apple.AppleMobileDeviceSupport`). Once it connected but sent no frames; the cause was not found
  on the laptop. The panel's camera line says when that happens. Fix on the phone, then restart the panel.

### First motion with `replay.py` (arm 2 only, no cloth, `--speed 0.3`)

- `replay.py` glides to the idle pose first and back to it at the end. Dry run of the fold: 0 clamped steps.
- **First fold pose.** The gripper points back over the base, as the model says. Fingertip about 40 to 42 cm above
  the table; the model says 43.7.
- **Zero pose** (`trajectories/zero_pose.json`, every joint at 0). Seen on arm 2: forearm about 15 degrees up toward
  the wrist (model: level), gripper level, upper arm looked upright, forearm about 23 cm above the table.
- The fold itself has not been played on either arm.

### Both arms' zeros set by hand (2026-11-17)

- **Method, no ruler.** Torque off (`idle.py --arm arm2 release`), live angles on (`replay.py --read arm2`), the arm
  held in the model's zero pose: upper arm flush against an upright book, forearm and gripper level and pointing
  forward, jaw shut. What a joint reads there is its error `e`; its new zero is `2047 + e * 11.378`.
- **Readings:** pan -1.1, shoulder lift 3.2, elbow 12.2, wrist 1.3, jaw shut -9.7 (expected -10). Only the elbow was
  corrected: `arm2/urdf.json` is `{"zero": {"3": 2186}}`.
- **Check under power** at the zero pose: forearm level, gripper level, upper arm upright, moving finger on top, jaw
  slightly open (the model's jaw 0 is 10 degrees open). Arm 2's wrist roll is the right way round.
- **Arm 1, the same way**, held with the moving finger on top. Readings: pan -3.0, shoulder lift 1.6, elbow 5.5,
  wrist -0.4, roll 0.7, jaw shut -9.6. Only the elbow was corrected: `arm1/urdf.json` is `{"zero": {"3": 2110}}`.
  Under power the zero pose looks the same as arm 2's. The pan reading was left alone; watch for a sideways miss at
  the touch test.
- The half turn between the two wrist-roll encoders at idle (1030 and 3076) is the two arms resting rolled opposite
  ways (-89 and +90 degrees), not a difference in the map.
- Fold dry run with both new zeros: 0 clamped steps.

### Protection against collisions

Per-joint limits do not stop the arm hitting itself, the table or the other arm; nothing here checks for that.

- `replay.py --dry-run` prints the lowest height the elbow joint centre, the wrist joint centre and the fingertip
  reach over the trajectory (model, table at 0). The fold: fingertip 0.5 cm at the pinch, elbow 13.3, wrist 15.2.
  It does not cover the glide between the idle pose and the first target.
- **Self-collision** (`collision.py`, on the arm model's collision shapes with the base made solid). `fold_traj.py`
  moves a release that would hit the arm's own base back toward the pinch, by the least that keeps every checked
  pair of parts 1 cm apart (`release_short_mm` in the trajectory), and refuses a fold it cannot clear. `replay.py`
  prints the nearest the arm comes to itself and refuses to play under 1 cm; `--no-collision-check` plays anyway.
  The glides to the idle pose, from it to the first target and back are checked after the arms are read and before
  torque goes on; a glide is refused only if it comes nearer than both of its ends. With `--no-gripper` the jaw is
  taken at both ends of its travel. Not checked: parts 1 or 2 joints apart (the folded idle pose rests on them),
  the table and the other arm. It needs `mujoco` and `so101_nexus`, which the minimal `.venv-hw` lacks.
- `replay.py --torque-limit PERCENT` (10 to 100, default 100) caps every joint's torque so a collision stalls
  gently. Each motor's limit is read back, and a motor that did not take it stops the run before torque goes on.
  Full torque is written back when the run ends at the idle pose; after a stop the cap stays on the motors until
  the next replay or a power cycle. Not tuned on the arms: too low and the shoulder sags at full reach.

### First fold on arm 2: it hit its own base (2026-11-17)

Arm 2 alone, no cloth, speed 0.3, torque cap 50%. Stopped by hand at step 244, at the release.

- **The planned path goes through the base.** In the arm model the gripper is inside the shoulder housing from step
  218 to 269 (34 mm at the deepest), from the moment the fingertip comes within 11.9 cm of the base origin. The
  release was planned at 8 cm, and the planner checked reach and tilt, not the arm's own body.
- **The motion was jerky.** `replay.py` sent one target per row, 6 a second at speed 0.3, with nothing in between.
- **Since then** (not yet run on an arm): the committed trajectory releases at 13.5 cm, 5.5 cm short of the near
  edge (the user's choice: same cloth, short fold), 16 mm clear of the base in the model; `replay.py` reaches each
  row in equal moves 50 times a second; the old trajectory is refused.
- `idle.py release` then stopped at arm 2's elbow motor with an error; the cause is not known. After a power cycle
  the motor is fine (user).
- **Arm 2 alone, no cloth, played the new fold as intended** (user, speed 0.3, torque cap 50%).
- **First real frames from the phone** (720 x 960, portrait, behind the arms). Record3D streams over USB with the
  app in the foreground; the record button is not needed. The cloth is light blue on a pale table with a laptop, a
  box and a bag in view: the darkness detector outlined the laptop at every threshold tried, the blue setting
  outlines the cloth. An arm's cable over a corner pulls that corner in by 4 to 5 cm, so clear the corners first.
  No calibration from the live phone has been checked yet.
- **Both arms with the cloth folded it** (user), apart from the grasp check, which printed MISSED three times on
  both arms. The check was blind: the trajectory closed the jaw faster than `replay.py` lets it move, so the empty
  reading was taken mid-close (12.6 deg, shut is -9.7) and the pinch was still closing when the lift began. The
  trajectory is regenerated with the jaw inside the rate cap (379 steps); not rerun on the arms.

### Next

From the repo root, panel stopped, a hand on the power.

1. **Both arms with the cloth again**, `--torque-limit 50 --grasp-check`: the empty reading should print near -9.7
   and a held corner should read wider than that by more than 0.5 deg. The fold lands 5.5 cm short of the near edge.
2. **The camera versions**: `fold_camera.py calibrate`, `fold_live.py --dry-run`, then `fold_live.py --grasp-check`.

Known and not fixed: `set_limits.py` uses Unix-only keyboard handling, so limits cannot be re-recorded on Windows.

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
- **The fold has not been played on a real arm** (bring-up log above). The radian-to-tick map is a hypothesis until
  `replay.py --check` passes on each arm. The elbow zeros were off (12 degrees on arm 2, 5.5 on arm 1) and are corrected
  in each arm's `urdf.json`. The Record3D connection follows `hardware/iphone_lidar/usb_viewer.py`, which has
  not seen a phone either. Frame orientation and how well the depth lines up with
  the RGB are unknown until a real frame is recorded.
- **Both followers are configured**, with `main`'s names and folders. `arm2` is the left arm and `arm1` the right,
  seen from behind. Arm 2's motors 2 and 3 have identical limits, so watch its elbow in the check.
- **Known risks.**
  - A 30 cm cloth needs tilted fingers at its far end. The fingers can point straight down from 10 to 31 cm in
    front of the base origin; outside that the planner leans them by the least that reaches the point (26 degrees at
    the far pinch with the near edge 8 cm out, 29 at the hover above it). Both corners are reached exactly in the
    model. Within about 12 cm of its base an arm cannot lift straight up, so the lift after the release leans outward
    (4 cm over its 5 cm rise at 8 cm). Every look prints the pinch and place misses, which should read 0.
  - The jaw target is the closed limit, so the squeeze on thin cloth may be weak.
  - Dark cloth can absorb LiDAR. Pixels with no depth count as flat, so the height mask only removes what it can see.

## Rig

- Two followers flat on the table, facing the same way, base origins 30 cm apart ("left" is on the left seen from
  behind the arms).
- Cloth 30 cm square, centred between the arms, near edge 1.5 cm in front of the bases' front edges. Each side edge
  of the cloth then lines up with the centre of an arm.
- Start with both arms in their idle poses (`idle.py --arm <name> record` saves one). `replay.py` glides each arm
  to its idle pose first and back to it after the fold. An arm with no saved idle pose starts from where it is
  and stops on the last row, so start that one roughly upright over its base.

## Setup on a new machine

```bash
git clone https://github.com/LegendaryAKx3/WorldFold.git && cd WorldFold
git checkout feat/half-fold-deploy
python3 -m venv .venv-hw
.venv-hw/bin/pip install feetech-servo-sdk pyserial
```

That is all `replay.py` needs. The trajectory is committed at `hardware/so101/trajectories/half_fold_30cm.json`.

Then tell the repo about this machine's arms.

1. Set `PLATFORM` at the top of `hardware/so101/arms.py` to `"mac"` or `"windows"`. Find each arm's port (plug one
   arm in at a time): `ls /dev/tty.usbmodem*` on a Mac, Device Manager > Ports (COM & LPT) on Windows. Put it in
   `hardware/so101/arms.json` under that arm's name and platform. An arm with `null` for a platform is left out there.
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
- **Step 3.** The fingertip touches the table 38.0 cm in front of the base origin (31.5 cm in front of the base's
  front edge), in line with the arm, and the jaw closes fully there. It releases 8 cm in front of the origin. If the
  touch-down is off by more than about 1 cm, fix the zero in `<name>/urdf.json` (`"zero": {"<motor>": ticks}`)
  before using a cloth.
- **Step 3, at the far pinch.** The fingers are tilted, so the model puts the fixed fingertip 2.6 mm below the table
  there. The arm presses on the table at that point, which is expected. Stop if it bows or stalls.
- **Step 4.** Each arm pinches its far corner, carries it over in an arc, opens part way, lifts 5 cm (leaning
  outward this close to the base), opens fully and stands back up. Before the pinch each jaw shuts once in the air,
  which is the grasp check's empty reading.

## Folding with the camera

Extra packages on the rig machine, on top of `feetech-servo-sdk`. The arm kinematics are plain numpy
(`kinematics.py`); the planner's self-collision check (`collision.py`) needs MuJoCo and so101-nexus, at the
versions pinned in `imitation/requirements.txt`. The repo's main `.venv` has all of these.

```bash
.venv-hw/bin/pip install numpy scipy pillow record3d==1.4.1 mujoco so101-nexus
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
- `calibrate --cloth-colour blue` finds a blue cloth by its colour instead of its darkness, for a light cloth or a
  table with darker things on it (a laptop, a bag). The choice is saved in `camera.json`, and `look`, `fold` and
  `fold_live.py` follow it. Keep other blue things out of the area in front of the arms.
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
- `--near 0.07` cuts the far pinch's tilt to about 22 degrees.
- `--closed` is the jaw target at the pinch (rad). -0.1745 is the model's closed limit.

## Arm configuration

- `arms.json` names every arm and its serial port per platform (`mac`, `windows`), in order; `PLATFORM` in `arms.py`
  picks the set. The first follower listed is the left one, seen from behind the arms. It also holds `layout` (`side_by_side` or `facing`), `spacing_cm` between the bases, and `leaders`,
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
| `check_limits.py` | read-only check that an arm's motors hold the limits and offsets in its `limits.json` |
| `ui.py`, `ui.html`, `phone.html` | teleop; the page also shows the iPhone LiDAR point cloud and has a record button |
| `teleop_record.py` | what the record button saves: every arm's joints (encoder steps) plus phone color and depth, 10 Hz, into `runs/` |
| `test_*.py` | tests, see above |

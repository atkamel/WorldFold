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
- **3D view.** The panel draws each arm from `so101.urdf` with the real parts' meshes (the STL files of the
  installed `so101_nexus` package, loaded with `urdf-loader` from the CDN that serves three.js), posed through the
  arm model's tick map (`urdf.json`), so the picture is the arm the collision check sees. Checked in Chrome on a
  stand-in servo bus: the drawn fingertip is where `kinematics.py` puts it. Not yet looked at beside the real arms.
- **Twin.** Only the keyboard and the gripper drag use it: their arm maths runs in the page on rounded link lengths
  from a straight-up reference. `arm1/twin.json` is the calibration checked on the arm; `arm2/twin.json` is a copy
  of it, not calibrated on arm 2 itself.
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
- **`fold_camera.py` died without a word** on most live runs (Windows error 0xc0000374, heap corruption): it let go
  of the Record3D stream object after one frame while the phone was still streaming. Letting go crashed 2 of 2
  trial runs, keeping it survived 3 of 3. `rgbd.Phone` holds every stream object until the process ends, and
  `fold_camera.py` reads its frame through it. Three live `look` runs then ended normally.
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
  to its idle pose first and back to it after the fold. An arm with no saved idle pose glides straight from
  where it is to the fold's first row, 5 cm above the far corner, and stops on the last row above the release, so
  start that one raised and clear of the cloth.

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

# 4. both arms with the cloth, with the run log
../../.venv-hw/bin/python replay.py trajectories/half_fold_30cm.json --left arm2 --right arm1 --log run1.json
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
  outward this close to the base) and opens fully. The fold starts 5 cm above the far corner and ends above the
  release; the glides from and to the idle pose are `replay.py`'s. A trajectory planned with `--grasp-check` also
  shuts each jaw once in the air before the pinch, which is the grasp check's empty reading.

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

Do steps 0 to 3 of the run order first. Then, from `hardware/so101/`, with both arms in their idle poses.

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

## Calibrating the camera from the arms (run on the arms 2026-11-19)

`fold_camera.py calibrate` reads the camera's map off the cloth, which it takes to lie at the ruler position. Press
it after moving the cloth and the arms go back to the ruler position. `arm_calibration.py` makes the map from the
arms instead, one map per arm, so the cloth plays no part in it and the bases may stand anywhere, at any angle.

```bash
# table clear, phone streaming; flags after --arm's go to replay.py as for a fold
../../.venv-hw/bin/python arm_calibration.py touch --arm left --left arm2 --right arm1 --speed 0.3 --torque-limit 50
../../.venv-hw/bin/python arm_calibration.py touch --arm right --left arm2 --right arm1 --speed 0.3 --torque-limit 50
../../.venv-hw/bin/python arm_calibration.py click     # then open http://localhost:8011
../../.venv-hw/bin/python arm_calibration.py fit
../../.venv-hw/bin/python fold_camera.py calibrate --cloth-colour blue    # the cloth anywhere in view
```

- **touch.** The arm glides from its idle pose, touches the table at 9 points with the jaw shut, rests 2 s on each
  (at full speed) and glides back. It is played by `replay.py`, so it gets the same checks, torque cap and step log
  as a fold; add `--dry-run` to see them without motion. A picture is kept of every rest, in
  `runs/arm_calibration/<arm>/`. About 50 s of touches at full speed.
- **`--widest`** is how far to either side of straight ahead the touches span. Nothing checks for the other arm.
  The default, 25 degrees, is for the bases side by side 30 cm apart: every part of the touching arm stays within
  18 cm of its own centre line, and each arm's first touch is on its outer side. The other arm must rest in its
  idle pose, where it takes up 5.5 cm to each side of its own centre line and 10 cm in front of its origin. With
  the bases far apart (arms at the cloth's sides) `--widest 55` fits the map over more of the table; the arm then
  sweeps 31 cm to each side.
- **`--outer`** is how far the touches span on the arm's outer side, `--widest` when not given. `--outer 0` keeps
  every touch between the arm's centre line and its inner edge, for a phone that does not see the outer side
  (2026-11-19: with the phone framed on the cloth, the left arm's fingertip was clear in 4 of 9 pictures of the
  usual fan). The map is then fitted over that strip only.
- **click.** A page on this machine shows each picture. Click the very tip of the closed gripper where it meets
  the table, or say the fingertip is not in the picture. Six clicked touches per arm are the least.
- **fit.** Fits both arms' maps, or neither. It prints how far the touches lie from each map, and how far a touch
  lies from the map the others make without it. That second number is the test: over 15 mm stops the fit and
  names the picture the others agree best without. On a synthetic camera with clicks 1.5 px off, a click 20 mm
  wrong is caught 19 times in 20 and one 10 mm wrong mostly is not, so click with care. The maps go into
  `camera.json` under `"arms"`, with the time and the picture size.
- **`fold_camera.py --inset M`** is how far in from each corner the pinch lands (default 0.01). `calibrate` saves
  it in `camera.json`, so the panel uses it too; on `look` and `fold` it holds for that run. Raise it when the
  grippers come down past the corners (2026-11-19: about 1 cm past on both arms, results.md).
- **`fold_camera.py --overshoot FORWARD OUTWARD`** (m, default 0 0) is how far past the point they are sent to
  the fingertips land; the arms are sent that much short, outward being away from the other arm. `look.png`
  still draws where the fingertips should land. **`--pinch-height M`** (default 0.005) is the fingertip's
  height at the pinch in the arm model; lower it when the fingers stop above the cloth. Both are saved by
  `calibrate` as `--inset` is (2026-11-19: about 1 cm forward, 0.7 cm outward, and 0.5 cm high, results.md).
- **A setting for each arm.** `--pinch-height LEFT RIGHT` and `--overshoot LEFT_FORWARD LEFT_OUTWARD
  RIGHT_FORWARD RIGHT_OUTWARD` give each arm its own; one height or one pair still serves both. The pinch
  height may go down to -0.02: the model's table is not the real one, and not the same for the two arms
  (2026-11-19: the left fingertip met the table at about -0.2 cm, the right was still above it at -0.5).
  The look prints each arm's numbers. To find an arm's table, lower its height 3 mm at a time and read the
  run's `steps.jsonl`: an arm that ends above the height it was sent to has met the table.
- **`fold_camera.py --pinch-tilt DEG`** pulls each pinch straight toward its arm, further into the cloth, until
  the fingers are within DEG of upright there (the arm leans them about 3.5 degrees per cm past 30 cm). Off
  unless given; saved by `calibrate`. On the rig it made the claw steep and the grab worse: the pinch ends up
  several cm inside the far edge, where the closing finger lands on top of the cloth (results.md, 2026-11-19).
- **`fold_camera.py --max-tilt DEG`** (default 30): a fold whose fingers would lean further from upright is
  refused with TOO FLAT, as one out of reach is. The lean grows about 4 degrees per cm past 36 cm from the
  base, and flat fingers close on top of the cloth (results.md, 2026-11-19): move the cloth closer.
- **`replay.py --p-gain GAIN`** (default 16, factory 32) is written to joints 1 to 5 before every run, so the
  camera folds, the touches and the panel's fold get it. At 32 the arms shake while they move.
- **A sideways phone, a faint or lopsided cloth:** `calibrate --near left` (the arms on the picture's left),
  `--dark 245` (a threshold by hand when Otsu drops the hems) and `--cloth 0.28` (the side the 15% check is
  against) are saved the same way. A later `calibrate` without them puts the defaults back.
- **After that** `fold_camera.py look` and `fold` read each arm's pinch and place points through that arm's own
  map. `--spacing` and `--near-edge` go unused. `fold_camera.py calibrate` (and the panel's Calibrate button) only
  learns the cloth's colour and leaves the maps alone, so it is safe with the cloth anywhere. It does NOT renew
  the maps: when the phone or a base moves, redo the touches. Every look prints when the maps were fitted and
  refuses a picture of another size.
- The cloth is searched for in front of both bases' front edges and within 60 cm of a base. The corners are named
  by the picture as before: the near edge is the one at `--near`, and the left arm takes the left corners.
- The map pairs each pixel with the point the arm was sent to, so a joint zero that is a little off is folded in.
  The fingertip rests 5 mm up in the arm model and a cloth corner lies on the table, which the camera's slant
  turns into a few mm along the table.
- A new `touch` replaces an arm's earlier pictures and clicks only once it has pictures of its own. A dry run or a
  refused run leaves them.
- `fold_live.py` does not read this calibration and says so.

## Folding from the control panel

`ui.py` (`http://localhost:8101`) has a "Cloth fold" section: the live camera picture, Calibrate, Look, Fold and
Stop, the last `look.png`, and the script's output as it prints.

- The buttons run `fold_camera.py` as a child process (`fold_job.py`), one job at a time. Its two "Enter to ..."
  pauses become a Continue button.
- The panel keeps the phone. The script gets the current frame as a temporary file, deleted when the job ends; the
  live picture itself is never written to disk.
- For a fold the panel closes both arms' serial ports, so the 3D view and the manual controls are frozen until the
  script ends; then the panel reopens them and holds the arms where they are.
- Stop (and Esc) ends the script outright: the arms hold with torque on, and a torque cap set for the fold stays
  on the motors until the next fold or a power cycle.
- Options: cloth colour (for Calibrate), speed, torque cap, grasp check (off by default: on a thin cloth it reads
  a held corner as a miss and pinches again). Left is `arm2`, right is `arm1` (`fold_job.py`).
- Every fold started here is recorded to `runs/<date_time>/` (see "Run logs"). Calibrate and Look are not.
- A fold that fails to start gives the arms back. Mirror and Leader are refused while a fold has the arms.
- The page only answers requests sent to `localhost` or `127.0.0.1` as JSON, so another web page open in the
  browser cannot drive the arms.
- Checked in a browser against a fake servo bus, a synthetic picture and a stand-in script. Not run on the arms.

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

- The check needs a trajectory planned for it (`fold_traj.py --grasp-check`; `fold_camera.py` and `fold_live.py`
  plan one when they are given `--grasp-check`). In such a plan each jaw shuts once on nothing above the far corner.
  That reading is the empty reference. Without the check the plan leaves that motion out, and `replay.py
  --grasp-check` refuses a trajectory planned without it. The committed trajectories are planned without it.
- `--grasp-check` reads each jaw 2 cm into the lift. A jaw that is not at least `--grasp-margin` degrees (default 0.5)
  further open than its empty reading has no cloth. Both arms then go back down, open, and pinch again, up to
  `--retries` times (default 2), and carry on regardless after that.
- The printed readings from the first run say whether 0.5 degrees is the right margin for this cloth. A thin cloth
  may read almost the same as empty, in which case the check reports misses that are not real.
- Every run that moves an arm is logged; see "Run logs" below.

## Run logs

Each `replay.py` run that moves an arm writes a new folder `runs/<date_time>/` (ignored by git); `--log FILE` puts
the step log somewhere else. A dry run writes nothing. `--check` is not logged.

- `steps.jsonl`: one line per control step, written as it happens, so a stopped run keeps what it did. Each line
  has `t` (`time.time()`), `phase` (`glide to the idle pose`, `glide to the first target`, `fold`,
  `glide back to the idle pose`), `row` (the trajectory row while folding) and, per arm, `target` and `measured`:
  the 6 joint angles sent and read back, in model radians. A glide logs every fifth pose. A grasp check adds a
  line with its jaw readings.
- `replay_meta.json`: the trajectory file, speed, torque cap, flags, and each arm's tick map.
- `plan.json`: the trajectory that was played, with its planned fingertip positions.

A fold started from the control panel is recorded into one such folder in `recorder.py`'s layout, which adds:

- `frames/000000.jpg` and `.npz`: the camera picture and LiDAR depth 5 times a second, with their times in
  `frames.jsonl`. Frame and step times are the same clock, so a picture can be matched to the joints at that moment.
- `look.png`, `camera.json`, `output.txt` (the script's printed lines), `events.jsonl` (each prompt, Continue, Stop,
  and how it ended), `meta.json` (the options chosen), `summary.json`.

`recorder.load(folder)` reads a run; `recorder.load_frame(folder, i)` reads one picture with its depth. About
60 MB per minute. The pictures hold whatever the phone sees.

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
- `--grasp-check` adds the jaw's empty reading above the far corner, about 3 s at full speed.

### Arms at the sides of the cloth (planned, not yet played on the arms)

With both arms behind the near edge, the far corner is a whole cloth away. A 30 cm cloth is the most that
placement reaches, with the fingers leaning 26 degrees and the release cut short to clear the base. With an arm at
the middle of each side edge, facing in, each arm reaches across half the cloth.

```bash
python fold_traj.py --out trajectories/half_fold_30cm_sides.json --cloth 0.30 --sides 0.16
python fold_traj.py --out trajectories/half_fold_60cm_sides.json --cloth 0.60 --sides 0.16
```

- `--sides` is how far each base origin sits outside its side edge (m). At 0.16 the plan is exact and stays 46 mm
  clear of the arm's own body. Nearer than that the carry passes too close to the base.
- The two bases face each other with their origins the cloth's side plus 32 cm apart, which is 62 cm for a 30 cm
  cloth and 92 cm for a 60 cm one. The cloth is centred between them with its middle on the line through the two
  origins. `--left` is the arm on the left seen from the edge the cloth is folded onto.
- 30 cm cloth. Fingers straight down at the pinch and at the release, no shortening.
- 60 cm cloth. Fingers lean 19 degrees at the pinch, no shortening.
- The jaw meets each corner turned by the pan angle (about 50 degrees on the 30 cm cloth, 68 on the 60 cm one).
- Both files are committed. `replay.py` plays them like any other trajectory. The camera fold follows the cloth
  in this placement once the camera is calibrated from the arms (`arm_calibration.py`, with `--widest 55`);
  calibrated from the cloth it still assumes both bases behind the near edge.

## Recording folds and training on them (built and checked on a stand-in rig, not yet on the arms)

Every run that moves the arms can become a training episode: the phone's picture, the joints read and the targets
sent, row by row.

```bash
# 1. record. The panel's fold and its record button already do; from the command line:
../../.venv-hw/bin/python rig_record.py replay trajectories/half_fold_30cm.json --left arm2 --right arm1 --torque-limit 50
../../.venv-hw/bin/python rig_record.py policy --left arm2 --right arm1 --layout behind
# 2. say whether each run folded (rig_record.py asks at the end; for the panel's runs, or to change one):
python rig_dataset.py label runs/20261119_101500 good
python rig_dataset.py list                    # every run: who moved the arms, rows, pictures, label, usable or not
# 3. convert the good ones into a dataset, then train (on a machine with the repo's pinned env and torch)
python rig_dataset.py build --out ../../outputs/rig/datasets --version rig_v0
python -m imitation.train --policy vision --root outputs/rig/datasets --dataset rig_v0 --run outputs/rig/runs/vision_v0
```

- `rig_record.py` passes everything after `replay` or `policy` to that script unchanged, so the checks and the
  motion are the same as without it. It saves a picture and its depth five times a second into a new `runs/`
  folder beside the script's step log, and asks at the end whether the cloth folded.
- `rig_dataset.py` reads four kinds of run folder: a fold played by `replay.py` (the panel's fold, `fold_camera.py`,
  `rig_record.py replay`), one played by `policy_run.py`, `fold_live.py`, and a teleop recording from the panel's
  record button (for that one pass `--left` and `--right` if `arms.json` holds more than two followers).
- A run is left out, with the reason printed, when it has no label, no pictures, one arm only, the gripper left
  out (`--no-gripper`), or a plan faster than `replay.py`'s rate cap (the log then holds the planned rows and not the rows sent).
  Runs labelled bad are left out unless `--failures`.
- The dataset is a version of the write-once store the simulator's students train from, so `imitation.train` reads
  it as it is. Each row holds the picture taken last before it (squashed to 128 x 128, camera name `main`), the
  joints read, their rates, the jaw, and the fingertip from the arm model, and as its action the joint targets
  sent, in units of 2 rad. The observation's simulator-only entries stay zero.
- Rates are per second of wall time, so record at the speed the policy will run at. `build` refuses good folds
  recorded at different speeds.
- Every good fold is lengthened to the longest one plus 16 rows by repeating its last row, the arms at rest where
  the fold ended, and no episode is marked as ended. The trainer fills what follows an ended episode with zeros,
  which for these targets would teach "go to the zero pose".

### Playing a camera policy on the arms (`camera_run.py`, checked on a stand-in rig only)

```bash
../../.venv-hw/bin/python camera_run.py --policy ../../outputs/rig/runs/vision_v0/final.pt \
    --dataset ../../outputs/rig/datasets/rig_v0 --left arm2 --right arm1 --torque-limit 50 --dry-run
```

- It needs `torch` and this repo's `imitation` package on the machine at the arms, besides what `replay.py` needs.
- The dataset version tells it the picture size, how long a row lasted, how many rows the longest good fold took
  and where the good folds began. The arms glide from their idle poses to that beginning, the policy plays that
  many rows, a new picture every 4 rows (`--replan`), and the arms glide back.
- Nothing can be checked ahead of the motion beyond the start and the glides to it: what the policy does depends
  on pictures not taken yet. Every row is checked before it is sent: the arm where it was told (`--track-deg`),
  a picture the phone sent since the last look (`--stale`) and of the size the policy learned on, the row finite,
  inside the limits and clear of the arm's own body, and no part of the arm beyond the shoulder more than 3 mm
  into the table (the arm model's meshes at the pose that will be sent, the open jaw included). A step over
  `replay.py`'s rate cap is cut to the cap. A failed check stops the run with torque on. The other arm and the
  cloth are not checked. The first picture is one the phone sends after the Enter prompt; if none comes, nothing moves.
- Every run is recorded like a `rig_record.py` run, with the pictures the policy was shown, so `rig_dataset.py`
  can read it back. Label it afterwards.
- `policy_run.py` is the other learned fold: the planner's copy, which reads joints and corner positions and no
  pictures, and can be checked whole before the arms move.

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
| `arm_calibration.py` | the camera's map from the arms: touch the table, click the fingertips, fit one map per arm |
| `rig_record.py` | records a command-line fold as the panel records its own, and asks whether it folded |
| `camera_run.py` | plays a camera policy trained on a `rig_dataset.py` version, checked row by row, recorded |
| `rig_dataset.py` | recorded runs into a dataset `imitation.train` reads: `list`, `label`, `build` |
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

## Two-arm gamepad

Run the panel with both followers and the controller plugged into the same computer. From the repository root,
using the panel's Python environment:

```sh
python hardware/so101/ui.py --gamepad
```

The gamepad mode uses NumPy, MuJoCo, and the same `so101-nexus` collision model as `replay.py`. On Windows it uses
XInput controller slot 0. On macOS install the dependencies in [the PowerA setup guide](../controller/README.md)
and close any other process reading the raw USB controller before launching the panel.

The two followers must be configured side by side in `arms.json`, in left-to-right order, with the measured
`spacing_cm` between the bases. The left stick drives `arm2` on this rig and the right stick drives `arm1`.

| control | what it does |
|---|---|
| left stick, right stick | sideways turns that arm's base; forward and back reach the fingertip out and in the way the arm faces |
| click a stick | switches that arm's forward and back to up and down (stick forward is up); sideways still turns the base |
| LB, RB | tilts the left, right wrist joint up while held |
| LT, RT | tilts the left, right wrist joint down while held |
| S, K on the keyboard | while driving: the left, right claw opens a little (30 deg, `CLAW_OPEN`) or shuts (the panel's page must have the keyboard) |
| A, D on the keyboard | while driving: turns the left claw about its wrist, one way and the other, while held (90 deg/s, `ROLL_SPEED`) |
| J, L on the keyboard | the same for the right claw |
| A | starts driving, with the sticks centred and the claw buttons released; A again holds both arms |
| B | stops; A drives again |
| Y | while stopped or holding: both arms glide to their saved idle poses; B stops the glide |
| drag the orange ball in the 3D view | while driving: that arm's fingertips go to where the ball is dropped (Shift-drag: up and down) |

Both arms must have torque enabled in the panel. Let the sticks rest at center while the panel calibrates them,
then press A, with the sticks centred and the bumpers and triggers released. B, the panel's Stop button or a lost
controller holds both arms until A is pressed again, and both sticks go back to over the table. If raw USB has not sent its first input report, move a stick
briefly and then let both sticks rest. From the folded idle pose, raise an arm before reaching out with it.

A missed encoder read skips that one step and driving stays on; only reads missed for 15 steps in a row (0.3 s,
`MAX_FAILS`) switch it off. A command a servo did not take switches it off at once. The Gamepad group of the panel
shows both sticks' readings: the value the loop acts on (-1 to 1, after the centre and the dead zone) and the raw
value under it. Each `--gamepad` session writes
`runs/gamepad_<date_time>.log`: a line whenever the status changes, and every 2 s the loop time, the reads and
writes missed, the most a joint trailed its command, how far each stick was pushed (0 to 1) and the sticks' raw
values at that moment. A stick within 9000 raw units of its centre counts as at rest (`DEAD_ZONE`, 27% of its
travel): a released stick does not come back to the same place, and a stick read as pushed moves its arm. Read it before changing a setting.

Every stretch of driving, from A to whatever ends it, and every glide to idle is also written to its own
`runs/gamepad_<date_time>_<n>.csv`, one row a step (about 50 a second): the status, the buttons, triggers and both
sticks (acted-on and raw), each arm's commanded and measured encoder steps, its joint angles in the model and its
fingertip position. The row after a stretch ends holds why it ended.

The solver works inside each arm's own recorded limits. A step it cannot make exactly, as when the wrist or the
elbow is at its limit, is still taken when the fingertip gets at least 30% of the way in the asked direction
(`FOLLOW_SHARE`), so a lift with a joint pinned still raises the arm, a little off the straight line.

The two arms move at the same time and each on its own: an arm whose step is refused, out of reach, or more than
about 9 deg behind its command holds while the other goes on (`MAX_LAG`; 0.15 s of its top speed where that is
more, `LAG_TIME`), and the panel's status says which and why, with the loop's time per step. With a stick fully
over the fingertip moves 20 cm/s and the base turns 90 deg/s; the wrist tilts 60 deg/s and a claw moves 90 deg/s
(`REACH_SPEED`, `PAN_SPEED`, `WRIST_SPEED`, `JAW_SPEED` in `gamepad.py`). No joint goes over 120 deg/s
(`MAX_JOINT_SPEED`) or the panel's speed slider, which starts at 120 in gamepad mode: pull it down to slow
everything. A step covers the time since the last one, so a slow loop does not slow the arms.

A dragged point goes to the controller loop (`POST /gamepad/target`: the arm and a point in its base frame, three
numbers within a metre), which walks the fingertips to it at the sticks' speed through the same check. A point
dragged past where the arm can go is moved to the nearest place it can: the solver is run toward it from the
arm's pose with the joints inside their recorded limits, and the point is kept 1 cm above the model's table
(`TIP_FLOOR`). A blue ball marks the point, moved or not, until the gripper is there. In the 3D view the left
button is for the arm, the right button turns the view (with Shift it slides it) and the wheel zooms. A stick takes over from it, B drops it, and it is refused while
driving is off. In gamepad mode only the gripper can be dragged; turning one joint by its part is for the panel
without `--gamepad`.

A held roll key is repeated by the page ten times a second (`POST /gamepad/roll`: the arm and a way of -1, 0 or
1). A key the loop stops hearing about for 0.3 s (`ROLL_HOLD`) counts as let go, so a page that loses the keyboard
cannot leave a claw turning. The turn stops at the joint's recorded limits and goes through the same check; the
moving jaw against the wrist is not in that check.

The glide to idle moves every joint of an arm straight to its idle value, the fastest at 30 deg/s
(`HOME_SPEED`), through the same check as driving. The folded idle pose rests closer to itself than the clearances
allow, so on the glide a pair may come as close as it is at idle and no closer. A glide that would touch the
table or the other arm stops and says so: move the arm clear with the sticks and press Y again.

Every step is checked in the two-arm model before it is sent. A joint stays inside its recorded limits. The model
also holds what the limits of one motor cannot see:

| checked | kept clear by |
|---|---|
| parts of one arm three or more joints apart (gripper and base, jaw and upper arm) | 10 mm |
| parts of one arm two joints apart (forearm and shoulder housing, wrist and upper arm, upper arm and base) | 5 mm, or 5 mm closer than at the idle pose where that is less |
| any part of one arm and any part of the other | 10 mm |
| every moving part and the table plane at the bases | 2 mm |

The model draws the folded idle pose up to 12 mm inside itself where the real arm rests clear, so parts two
joints apart are judged against the idle pose: they may come 5 mm closer than they are there (`NEAR_SLACK`). A pair
already inside its clearance may move apart and may not come closer. Not
checked: parts next to each other (the motor limits hold those), the moving jaw against the wrist (the jaw runs at
half torque), the cloth, cables, and anything else on the table. The model's table is not the real one and differs
between the arms.

Gamepad mode reserves the two followers and leaves the leader arm unopened. Start the panel without `--gamepad`
to use keyboard, phone motion, mirror, leader, and cloth-fold controls. Physical validation assumes flat, aligned
bases and a clear table around the arms.

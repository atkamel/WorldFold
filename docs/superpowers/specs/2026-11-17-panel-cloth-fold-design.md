# Cloth fold from the control panel

2026-11-17. Short design, for the user's approval before any code.

## Intent

Run the camera calibration, the look and the camera fold from the existing browser panel (`hardware/so101/ui.py`)
with buttons, and watch the phone's camera picture and LiDAR cloud live while it happens. Today these are three
terminal commands (`fold_camera.py calibrate | look | fold`) and the panel cannot run next to them.

## Goals, non-goals, anti-goals

- **Goals.** Three buttons (Calibrate, Look, Fold) and a Stop. The script's printed output on the page as it
  appears. The `look.png` overlay on the page. A live camera picture beside the LiDAR cloud, before and during a fold.
- **Non-goals.** The blind replay, `fold_live.py`, bring-up steps (`--read`, `--check`, zero poses), and a live 3D
  arm view during the fold.
- **Anti-goals.** A second implementation of the fold or of its safety checks. Anything saved from the phone that
  the user did not ask for.

## Constraints

- One process at a time can hold an arm's serial port, and one process at a time can hold the Record3D stream.
- The fold's safety (collision check, torque cap, glide check, clamp refusal) lives in `replay.py` and
  `fold_traj.py` and uses `urdf.json`; the panel moves arms with other code and `twin.json`.
- Freeing a Record3D stream object mid-stream kills the process, so the panel never gives the phone up.

## Approaches

1. **The panel lends the arms to the existing scripts (picked by the user).** The panel pauses its arm loops,
   closes their serial ports, runs `fold_camera.py` as a child process, shows its output, and takes the arms back
   when it ends. The panel keeps the phone and hands the script one frame as a file (`--image`, which the script
   already supports). Every safety check runs unchanged. Cost: the 3D arm view and the manual controls are frozen
   while a fold runs.
2. **The panel runs the fold through its own arm loop.** Live arm view throughout, but the fold's checks and
   today's calibration would have to be rebuilt and re-proven inside the panel. Rejected: more code and more risk on
   the hardware for a view that matters less than the camera's.

## Design

**New file `hardware/so101/fold_job.py`**: one job at a time.

- `start(mode, options)`: refuses if a job is running. Writes the camera's latest frame to a temporary file. For
  `fold`, lends the two arms first. Starts `python -u fold_camera.py <mode> --image <file> ...` and reads its
  output on a thread.
- State: `idle`, `running`, `waiting` (the script is at one of its two "Enter to ..." prompts), `done`, `failed`,
  `stopped`; the last 200 output lines; the exit code.
- `go_on()`: sends the Enter the prompt waits for. `stop()`: ends the child process; the motors keep their last
  goal with torque on, as after Ctrl-C.
- When the child ends, however it ends: the temporary frame is deleted and the arms are taken back.
- Which arm is left and which is right: a constant in this file (`arm2` left, `arm1` right), as in the commands
  used so far.

**`ui.py`**

- `Arm.lend()`: under the arm's lock, marks the arm lent and closes its port; the control loop does nothing while
  lent. `Arm.take_back()`: reopens the port, reads the joints and the torque state, sets target and command to
  where the arm is, so nothing moves.
- While an arm is lent, the endpoints that move it answer "busy".
- New endpoints: `GET /fold` (state and output), `POST /fold/start`, `POST /fold/continue`, `POST /fold/stop`,
  `GET /camera.jpg` (the latest picture, encoded in memory, nothing written), `GET /look.png`.

**`ui.html`**: a "Cloth fold" section under the camera.

```
Camera            [ live picture ]   [ LiDAR cloud ]
Cloth fold
  Cloth colour (blue v)   Speed (0.3)   Torque cap (50 %)   [ ] Grasp check
  [ Calibrate ]  [ Look ]  [ Fold ]                          [ Stop ]
  look.png, refreshed after each look
  Output: the script's lines, live
  "Check the outline."  [ Continue ]  [ Cancel ]
```

- The picture refreshes about 5 times a second. Buttons are disabled while a job runs, except Stop and Continue.

## Calls made here (the builder's, not the user's)

- Grasp check is off by default: on this thin cloth it reads a held corner as a miss and re-pinches.
- Defaults: speed 0.3, torque cap 50 %, cloth colour taken from `camera.json` when there is one.
- Stop ends the child process outright; it does not try to glide the arms anywhere.
- A recording made during a fold holds the camera frames but frozen joint readings; left as is.

## Tests

`test_fold_job.py`, with a stand-in script in place of `fold_camera.py` and stand-in arms: output is captured in
order; a prompt puts the job in `waiting` and `go_on()` releases it; `stop()` ends it; a second `start` is
refused; the arms are lent before a fold and taken back after it ends, fails or is stopped, and are not lent for
calibrate or look; the temporary frame is gone afterwards. `Arm.lend` / `take_back` against the fake servo SDK.

## Done when

On the real rig, from the page alone: Calibrate, Look and Fold behave as the three terminal commands do, the
camera picture and LiDAR cloud stay live through the fold, and the manual controls work again after it.

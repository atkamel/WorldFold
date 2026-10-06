# Play the Isaac cloth scene with two mice

You fold the LeHome shirt in Isaac Sim with one USB mouse per SO-101 arm. The sessions become demos for the VLA
pipeline (fine-tune the pi0.5 teacher, then distil). State as of 2026-10-06.

**How it works.** Isaac runs on a GPU machine (the *server*: physics + the arm controller). Each tick it streams the 12
joint angles and the ~14.7k cloth points (int16, delta + Blosc2 compressed, ~13-15 kB per frame) to your laptop. The
laptop (*client*) draws the scene itself with Polyscope and sends both mice back. No video is streamed.

```
laptop: teacher/mujoco_live/teleop_client.py  <--TCP-->  GPU node: Isaac + teacher/oracle/oracle_fold.py (teleop)
        (raw mice, draws arms + cloth)                    (controller, native IK, stream server on port 7777)
```

## 0. Laptop setup (once)
Windows (raw two-mouse input uses the Win32 raw input API in `mujoco_live/rawmouse.py`). Mac/Linux: not yet,
see tasks M5/L4; you can still work on the code there.
```bash
git clone -b ROY-vla-teacher https://github.com/atkamel/WorldFold && cd WorldFold
pip install -r teacher/mujoco_live/requirements.txt
python teacher/oracle/native/test_so101_ik.py      # C++ IK: prebuilt .dll/.so are in native/lib/, 6 checks should pass
```
Rebuild the C++ only if you change it: `bash teacher/oracle/native/build.sh` (g++ or clang, WSL works).

## 1. Try it with no GPU (free, 1 minute)
A fake Isaac (no physics: joints follow commands, cloth sticks to a closed gripper) runs the same controller,
server and stream. Use it to work on anything client-side: camera, cursor, HUD, controls.
```bash
cd teacher/oracle && python -m teleop.fake_isaac 7777 --state          # terminal 1
python teacher/mujoco_live/teleop_client.py 127.0.0.1 7777               # terminal 2 (from the repo root)
```
Headless check, no mice needed: `python teacher/mujoco_live/probe_teleop.py 127.0.0.1 7777 12` (prints updates/s and
lag; expect ~30 updates/s, ~12 ms locally). The server accepts **one** client, so the probe and the game can't share it.

The local MuJoCo game (its own cloth, no server): `python teacher/mujoco_live/game.py`.

## 2. Isaac on WATcloud (free with an account, RTX 3090, driver 580)
Access: UW VPN (Cisco Secure Client, `vpn.uwaterloo.ca`, group UW-General), then
`ssh <you>@wato-login1.ext.watonomous.ca`. Slurm tools are in `/opt/slurm/bin` (use `bash -lc` or add to PATH).
```bash
# on the login node, once
mkdir -p ~/wf && git clone -b ROY-vla-teacher https://github.com/atkamel/WorldFold ~/wf/WorldFold
# every session
cd ~/wf/WorldFold && git pull && sbatch teacher/watcloud/isaac_teleop.sbatch
squeue -u $USER                      # which node; then: tail -f isaac-teleop-<jobid>.log
```
First run pulls the ~20 GB image (`ghcr.io/cimurghe/worldfold-isaac:5.1`). Isaac then needs ~3-5 min; wait for
`TELEOP listening`. Then on the laptop:
```bash
ssh -N -L 17777:<node ip>:7777 <you>@wato-login1.ext.watonomous.ca     # thor-slurm1 = 10.0.50.184
python teacher/mujoco_live/teleop_client.py 127.0.0.1 17777
```
Job options (env vars before `sbatch`): `MAX_S` (1500), `IDLE_S` (600), `ORACLE_GRAB=assist|mujoco|tested`,
`MODE=assist` runs the grab test with no client (`N_GRABS`, `PRESETS`).
The session ends when you quit, after `IDLE_S` without input, or at `MAX_S`. Cancel with `scancel <jobid>`.
Logs land in `~/wf/results/isaac/<stamp>-teleop/` (`run.log`, `teleop_demo.jsonl`); the client also keeps a copy
in `teacher/mujoco_live/demos/`.

## 3. Isaac elsewhere
- **RunPod**: `teacher/runpod/README.md` (RTX 4090/A40; `run_isaac.sh teleop`; the pod stops itself).
- **Modal**: `isaac_teleop` in `teacher/modal_teacher.py`. Modal's US-East hosts had driver 610, which crashes Isaac 5.1.
- **Build the image yourself**: `docker build -t worldfold-isaac:5.1 teacher/runpod/image` (~16 min Isaac download;
  see the Dockerfile header). Our code is not baked in: it is mounted at run time, so code edits never need a rebuild.

## Measured speed (2026-10-06)
| Where | Server | On your laptop | Lag |
|---|---|---|---|
| WATcloud 3090, from home over the VPN | 52-57 ticks/s | 15-36 FPS | ~90 ms |
| RunPod Montreal A40 | 28 ticks/s | 28 FPS | ~91 ms |
| WATcloud from campus | same | not measured (guess ~50) | not measured (guess 30-45 ms) |

Physics is ~9 ms/step on an L40S; most of the remaining server time is in LeHome's env step.

## Rules learned the hard way
- **Driver 580 only.** Isaac Sim 5.1 segfaults at startup on driver 610 (`run_isaac.sh` checks and stops).
- **Shell scripts must be LF**, or bash on the pod dies silently. `.gitattributes` enforces it; check with `file x.sh`.
- **Never connect a probe to a live session**: the server takes one client, and the probe ends your session.
- **Don't use Isaac's headless kit** (`isaaclab.python.headless.kit`): it stops the cloth writing back and the shirt
  freezes. Use the full kit with cameras off (`LEHOME_NO_CAMERAS=1` via `teacher/no_cameras_patch.py`).
- Isaac only runs on RTX GPUs (3090/4090/A40/L40S/RTX PRO). Not A100/H100/H200/B200.

## Known problems
1. **Grabs land in the wrong place.** The top view hides height (parallax), and the cursor keeps moving during the
   ~1 s grab, so grabs closed 7-14 cm from the target in the 2026-10-06 session.
2. **No sense of touch.** You can't tell when the fingertips reach the cloth.
3. **Freeze every ~30 s / random crashes in no-camera mode.** The keepalive `app.update()` costs ~0.5 s each time.
   The crash itself is hidden by IsaacLab's signal handler, which recurses, so the real error is unknown.
4. The client draws the table ~2 mm low (the cloth looks like it sinks; physics is fine).
5. The stream stalls when the game window isn't in front (acks only go out from the input loop).
6. The mouse wheel didn't turn the wrist in the Isaac session, though it does locally (cause unknown).
7. Left-arm grabs near its own base stop 2-4 cm above the cloth (reach).

## Open tasks, by size

What we're aiming for (Roy, 2026-10-06): the grab strength is fine; a light pinch drags a whole real shirt. What's
missing is **seeing where the gripper will land** (parallax) and **feeling when the fingertips reach the cloth**.
Reference pictures in `docs/controls/`: the MuJoCo game views we want back (`mujoco_view_*.png`) and the target ray
(`target_ray_sketch.png`, `target_ray_mockup.png`). Line numbers are as of commit `bbb0073`.

Every small and medium task is tested on a laptop with the fake server (section 1); no GPU, no account.

### Small: one file, a few lines, 1-3 hours, no AI needed
| # | Task | Exactly where | Done when |
|---|---|---|---|
| S1 | **Table drawn at the right height** (now 2 mm low, so the cloth looks like it sinks into it) | `mujoco_live/teleop_client_ps.py` ~line 87: the table corners use `z = -0.002`; use `0.0` | the cloth lies on the table, not in it |
| S2 | **Camera like the MuJoCo game**: over the robots' shoulders, ~45-50 deg down | `CAMERA` at the top of `mujoco_live/teleop_client_ps.py` = (look-at y m, distance m, degrees down, fov deg). Try values without editing code: `set SO101_CAM=0.20,0.60,50,60` (cmd) or `$env:SO101_CAM="0.20,0.60,50,60"` (PowerShell) before starting the client. The MuJoCo values are `VIEWS` in `mujoco_live/game.py`, but Isaac's arm bases sit wider (x = +-0.23 m) and 8 cm further back, so expect more distance | it looks like `docs/controls/mujoco_view_over_shoulder.png`, both grippers and the whole shirt visible |
| S3 | **Freeze the aim point while a grab runs** (it kept moving during the ~1 s grab, so grabs closed 7-14 cm from the target) | `oracle/teleop/controller.py`, `_apply_mice` (~line 227, the `for a in ARMS:` loop that adds `moves[a]` to `self.cursor[a]`): skip arms whose `self.phase[a]` is `"down"`, `"closing"`, `"drop_after_close"` or `"opening"` | on the fake server, wiggling the mouse during a grab doesn't move the grab spot |
| S4 | **Drop line**: a straight vertical line from each aim point down to the cloth/table | `mujoco_live/teleop_client_ps.py`: the aim points are drawn in `draw()` (~line 172-175, `self.cursor_pc`). Add a Polyscope curve network (`ps.register_curve_network`) with 2 points per arm: the aim point and the same x, y at the cloth top (highest cloth point within ~1.5 cm, else 0 = table) | matches `docs/controls/target_ray_sketch.png` |

### Medium: 0.5-2 days, touches 2-3 files (AI helps a lot)
| # | Task | Where to start |
|---|---|---|
| M1 | **Stream stalls when the game window isn't in front** (0.2 frames/s): acks only go out with the mouse input, from the drawing loop | `mujoco_live/teleop_client.py` `_reader` (~line 115): after decoding a `state`, send `dict(t="ack", got=self.dec.last)`. The socket is then written by two threads, so wrap `protocol.send` in a lock. The server needs no change: it reads `got` from any message (`server.py` ~line 188), and `merge_inputs` ignores a message with no mice/keys |
| M2 | **Target ring on the cloth that turns green on touch** (the client half of pseudo-touch) | client only: ring at the bottom of the S4 line; green when the gripper tip (from the streamed joints, `RobotVis`) is within ~5 mm of the cloth top below it |
| M3 | **Mouse wheel didn't turn the wrist in Isaac** (works locally) | `controller.py` `_apply_mice` (wheel -> `yaw_cmd`, ~line 218) and `_move_arms` (`roll_for_yaw`, ~line 324). Log the wheel value on the server in one session and see where it's lost |
| M4 | **Left arm stops 2-4 cm high when grabbing near its own base** | `controller.py` `_move_arms` reach fallbacks + `oracle/teleop/isaac_world.py` `ik`; reproduce with the fake server first |
| M5 | **Two mice on Linux** | see "Two mice on Mac / Linux" below |

### Large: several days, needs Isaac on WATcloud (give to someone using AI)
| # | Task | Notes |
|---|---|---|
| L1 | **Hold-to-lower until touch** (pseudo-touch, full): holding the button lowers the gripper until it reaches the cloth, the server sends a "touching" flag, then the pinch | server knows the cloth height under the jaws (`IsaacWorld.surface_under`, C++ `cloth_top`); flag goes in the HUD; needs S3, S4, M2 first |
| L2 | **Crash / freeze every ~30 s in no-camera mode** | step 1: disable IsaacLab's recursive `_abort_signal_handle_callback` to see the real error. Step 2: a cheap keepalive with rendering off (`ORACLE_APP_PUMP_TICKS` sets the period; each `app.update()` costs ~0.5 s now) |
| L3 | **Replay demos in Isaac -> training data** | demos only count after a replay in Isaac reproduces them |
| L4 | **Two mice on Mac** | see below; mostly macOS permissions |

### Two mice on Mac / Linux (M5, L4)
Windows merges all mice into one cursor but its Raw Input API still says which device each move came from;
`mujoco_live/rawmouse.py` uses that. Mac and Linux expose the same information another way:
- **Linux**: each mouse is its own `/dev/input/event*` device; read them with `python-evdev`
  (`REL_X/REL_Y/REL_WHEEL`, `BTN_LEFT/RIGHT/MIDDLE`). Needs read access: add yourself to the `input` group.
  Grab the devices (`dev.grab()`) so the moves don't also drive the desktop cursor.
- **Mac**: IOKit's HID Manager (`IOHIDManagerRegisterInputValueCallback`, reachable from Python via PyObjC or
  ctypes) reports values per device. The terminal needs **Input Monitoring** permission once
  (System Settings -> Privacy & Security).

What to build: a `MiceReader` with the same interface as the Windows one, chosen by `sys.platform`:
- `take()` returns `{device_id: (dx, dy, wheel_notches, left_held, right_held, events)}` and clears the totals;
  `events` lists `"left_down"`, `"left_up"`, `"right_down"`, ... plus `"middle_held"` while the wheel button is down
- `register()` / `listening()` (can just return True), `ok`, and `is_external_mouse(id)` (True for USB/Bluetooth
  mice, False for a built-in touchpad).

The clients also make a few Windows-only calls (`ctypes.windll`: DPI awareness, `ClipCursor`, `FindWindowW` in
`teleop_client.py`, `teleop_client_ps.py`, `game.py`); guard those with `sys.platform == "win32"`.
Done when `python rawmouse.py` prints separate moves/clicks for two mice and the fake-server session plays with both.
Rough size: an afternoon for Linux (M5), about a day for Mac (L4, mostly permissions).

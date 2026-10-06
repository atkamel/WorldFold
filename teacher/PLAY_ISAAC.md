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
Windows (raw two-mouse input uses the Win32 raw input API in `mujoco_live/rawmouse.py`).
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

## Open tasks
| # | Task | Notes |
|---|---|---|
| 1 | **Camera**: MuJoCo-style over-the-shoulder view, ~45 deg, shadows on | `CAMERA` / `SO101_CAM` in `mujoco_live/teleop_client_ps.py`; reference: the old MuJoCo game view |
| 2 | **Aim on the cloth**: the cursor is the spot on the cloth (or table) under the mouse ray; a straight line drops from the gripper to it | client has every cloth point; send that point as the target |
| 3 | **Pseudo-touch**: hold to lower until the fingertips reach the cloth, server sends a "touching" flag, the ring turns green (+ sound), then pinch | server knows the cloth height under the jaws (`IsaacWorld.surface_under`, C++ `cloth_top`) |
| 4 | Freeze the cursor during a grab | `oracle/teleop/controller.py` |
| 5 | **Crash**: disable IsaacLab's recursive `_abort_signal_handle_callback` to see the real error; try a cheap keepalive with rendering off | needs Isaac (WATcloud); `ORACLE_APP_PUMP_TICKS` sets the keepalive period |
| 6 | Table height, stream stall in background, mouse speed, wheel | client-side; test on the fake server |
| 7 | Replay demos in Isaac -> training data | demos only count after replay |

Tasks 1-4, 6 can be done and tested on a laptop with the fake server (section 1).

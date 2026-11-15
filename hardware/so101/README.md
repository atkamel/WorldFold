# SO-101 hardware, open-loop half fold

What is here, what state it is in, and how to run it on a machine that has never seen this repo.

## Status (2026-11-14)

- **What runs** is a scripted, open-loop two-arm half fold with no camera. `fold_traj.py` plans it on the SO-101 arm
  model and `replay.py` plays the joint trajectory on two followers. The cloth goes where a ruler puts it.
- **What it is not** is the Isaac half-fold expert (`isaac/experts/place_servo.py`). That one reads the live cloth
  state and needs a camera pipeline that is not written yet.
- **Verified** by 25 tests on a fake servo bus, and a dry run against both arms' recorded limits (0 clamped steps).
- **Never run on a real arm.** The radian-to-tick map is a hypothesis until `replay.py --check` passes on each arm.
- **Both followers are configured.** `left` is the team's arm 2 and `right` is arm 1 (names from `main`'s `arm1/` and
  `arm2/` folders). Arm 2's limits were copied from `main` and its motors 2 and 3 have identical limits, so watch its
  elbow in the check.
- **Known risks.** The far pinch is at the edge of reach (fingers tilted about 25 degrees). The jaw target is the
  closed limit, so the squeeze on thin cloth may be weak. Nothing detects a missed or slipped corner.

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
2. Limits for both followers travel with the repo (`arms/left/limits.json`, `arms/right/limits.json`). To redo one,
   set `MOTOR_ID` in `set_limits.py`, run `.venv-hw/bin/python set_limits.py <arm>` per motor, and sweep the joint to
   both hard stops. For the gripper, record the fully closed end too, because the map takes its lower tick as closed.

## Run order

From `hardware/so101/`. Stop at the first step that fails. Keep a hand on the power. Ctrl-C stops sending and leaves
the torque on, so the arms hold where they are.

```bash
# 1. joint signs, one arm at a time: each joint moves +8 deg and back, in the model's positive direction
../../.venv-hw/bin/python replay.py --check left
../../.venv-hw/bin/python replay.py --check right

# 2. dry run: no motion, should report 0 clamped steps
../../.venv-hw/bin/python replay.py trajectories/half_fold_30cm.json --left left --right right --dry-run

# 3. one arm, no cloth, slow
../../.venv-hw/bin/python replay.py trajectories/half_fold_30cm.json --left left --right right --only left --speed 0.3

# 4. both arms with the cloth (about 31 s at the default half speed)
../../.venv-hw/bin/python replay.py trajectories/half_fold_30cm.json --left left --right right
```

What to look for

- **Step 1.** With the arm upright, model-positive shoulder_pan turns the arm to its right, seen from
  behind. The others are easiest to judge against the other arm once one is known good. A joint that goes the wrong way
  gets `"sign": {"<motor>": -1}` in `arms/<name>/urdf.json`. The check does not move the gripper.
- **Step 3.** The fingertip touches the table 37.8 cm in front of the base origin (31.3 cm in front of the base's
  front edge), in line with the arm, and the jaw closes fully there. It releases 9 cm in front of the origin. If the
  touch-down is off by more than about 1 cm, fix the zero in `arms/<name>/urdf.json` (`"zero": {"<motor>": ticks}`)
  before using a cloth.
- **Step 4.** Each arm pinches its far corner, carries it over in an arc, opens part way, lifts 5 cm, opens fully
  and stands back up.

## Changing the fold

Planning needs the arm model, so it runs in an environment with `mujoco` and `so101-nexus` (the repo's pinned
`.venv`, or `pip install mujoco==3.10.0 numpy && pip install --no-deps so101-nexus==0.4.8`).

```bash
python fold_traj.py --out trajectories/my_fold.json --cloth 0.30 --near 0.08 --closed -0.17
```

- `--cloth` cloth side (m). `--near` near edge in front of the base origin (m). The base's front edge is 6.5 cm in
  front of the origin. `--spacing` between the base origins (m).
- `--near 0.07` cuts the far pinch's tilt to about 21 degrees and puts the release 3 cm inside the near edge.
- `--closed` is the jaw target at the pinch (rad). -0.1745 is the model's closed limit.

## The tick map

`q_model_rad = sign * (ticks - zero) * 2*pi/4096`.

- Motors 1 to 5 use zero 2047 and sign +1. `set_limits.py` puts the middle of each swept range at tick 2047 and the
  model's arm joints are symmetric about 0, so this holds if each joint was swept stop to stop.
- Gripper. The model's range is -10 to 100 degrees, so its zero is 10 degrees above the lower tick limit, taken to be
  the closed end.
- `arms/<name>/urdf.json` overrides zero and sign per motor.

## Files

| file | what |
|---|---|
| `fold_traj.py` | plans the half fold, writes a trajectory |
| `replay.py` | plays a trajectory, with `--check`, `--dry-run`, `--only`, `--speed`, `--no-gripper` |
| `trajectories/half_fold_30cm.json` | the fold for the rig above |
| `arms.json`, `arms/<name>/` | ports, limits, per-arm overrides |
| `set_limits.py`, `idle.py` | limit recording, idle pose |
| `ui.py`, `ui.html`, `phone.html` | teleop |
| `test_replay.py`, `test_fold_traj.py` | tests on a fake bus, run with `python -m pytest hardware/so101` from the repo root |

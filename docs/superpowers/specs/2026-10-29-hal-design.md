# HAL: the real SO-101 arms (2026-10-29)

Branch `feat/hal`. The first hardware code in the repo: a thin layer over the physical LeRobot SO-101 leader
and follower so that (1) the arms can be exercised and checked, (2) any policy in the lerobot observation and
action convention can drive the follower and record episodes, and (3) this repo's sim checkpoints can be sent
to the follower through the same loop. The direct purpose is the plan agreed on 2026-10-29: a VLA out of the
box (MolmoAct2 through the so101-nexus adapter, branch `molmoact-import-policy`) collects real demos, an
imitation policy is trained on them, RL fine-tunes later if it earns its place.

## Intent

- **Goal:** everything needed to put a policy on the follower safely and record what happened, without
  hand-writing a servo protocol. Done when the test moves run on the real arm, the mirror loop follows the
  leader, and a rollout with a dummy policy records a dataset that lerobot can load back.
- **Non-goals (this pass):** building the 139-D sim observation on the real arm; converting recorded episodes
  into the repo's write-once episode store; running the VLA itself (it plugs in as a policy, nothing more);
  camera calibration or placement.
- **Anti-goals:** a second robot abstraction next to lerobot's; a config system; anything that moves a servo
  without a speed cap and limit clamp.

## Constraints

- Driver is `lerobot[feetech]` **0.6.1**, pinned in the root `requirements.txt`. It resolves on the pinned
  `.venv` (Python 3.14.5, checked with a dry-run install 2026-10-29). The HAL never talks to the serial port
  itself.
- Windows first: ports are `COMx`. Nothing in the HAL may assume `/dev/tty*`.
- Tests run without hardware under `pytest -m "not slow"`.
- Joint order everywhere is lerobot's: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
  gripper`, motor IDs 1 to 6. The sim's five arm joints use the same names (`cloth_fold_rl/expert.py:123`),
  and the sim gripper is the 6th slot.
- Units at the HAL boundary are lerobot's with `use_degrees=True`: joints in degrees, gripper 0 to 100.
  Everything the sim speaks (radians, deltas, the -1..1 gripper command) is converted in one place,
  `HAL/sim_bridge.py`.
- Plain code (repo style): one step per line, no comprehensions, no lambdas.

## Layout

```
HAL/
  __init__.py
  config.py       ports, ids, calibration, cameras, rates, limits, sim mapping  (one dataclass)
  arm.py          Follower / Leader wrappers over lerobot, with the safety clamp
  test_moves.py   CLI: ports, read, rest, sequence, mirror
  rollout.py      fixed-rate policy loop + LeRobotDataset recording
  sim_bridge.py   sim <-> real unit and frame conversion, SimPolicy adapter
tests/hal/
  conftest.py     FakeBus / FakeFollower
  test_arm.py, test_sim_bridge.py, test_rollout.py
```

## 1. `config.py`

One frozen dataclass `HalConfig` with defaults, loaded from a JSON file at `HAL/hal_config.json` if present
(`HalConfig.load(path=None)`); otherwise the defaults. The JSON file is gitignored, since it holds this
machine's ports. Fields:

| field | default | meaning |
|---|---|---|
| `follower_port`, `leader_port` | `"COM3"`, `"COM4"` | serial ports |
| `follower_id`, `leader_id` | `"worldfold_follower"`, `"worldfold_leader"` | lerobot calibration ids: the files land in lerobot's calibration dir under these names |
| `motor_ids` | `(1, 2, 3, 4, 5, 6)` | bus ids in lerobot joint order; passed through to the bus config |
| `cameras` | `{}` | name -> `(index, width, height, fps)`; built into `OpenCVCameraConfig` at connect. Empty = no cameras |
| `control_hz` | `30` | loop rate for mirror and rollout |
| `max_step_deg` | `5.0` | largest joint move per command, degrees; passed to lerobot as `max_relative_target` and enforced again in `arm.py` |
| `joint_limits_deg` | per-joint `(lo, hi)` | clamp for every goal; hard-coded conservative defaults in `config.py` (well inside the SO-101 mechanical range, gripper 0..100), overridable in the JSON. Not read from the calibration file: a clamp must not depend on a file that may be missing |
| `rest_pose_deg` | six floats | the "go home" target used by `rest` and at every disconnect. The default is a guess; bring-up step 4 measures the real one with torque off before `rest` is ever run |
| `resync_deg` | `10.0` | `SimPolicy`: measured-vs-commanded gap that resets the command to the measurement |
| `sim_sign` | `(1, 1, 1, 1, 1)` | per-joint sign real -> sim for the five arm joints |
| `sim_offset_rad` | `(0, 0, 0, 0, 0)` | per-joint offset: `q_sim = sign * deg2rad(q_real) + offset` |
| `gripper_open_deg`, `gripper_closed_deg` | `100.0`, `0.0` | what the sim's open / closed commands map to |

`sim_sign` and `sim_offset_rad` are the calibration knobs that make a recorded real trajectory replayable in sim
and a sim action sendable to the arm. They are measured once by hand (rest pose in both frames) and written into
the JSON. Default identity means "not yet calibrated", and `sim_bridge` prints a warning once when it is used
with the identity mapping.

## 2. `arm.py`

Two small classes, each wrapping one lerobot object and adding only what lerobot does not do for us.

`Follower(config)`:
- `connect()`: builds `SO101FollowerConfig(port, id, cameras, max_relative_target=max_step_deg,
  use_degrees=True)`, constructs `SO101Follower`, calls `connect()`. If the calibration file for `follower_id`
  does not exist, lerobot prompts for calibration interactively; the HAL does not wrap that, the README says
  to run `lerobot-calibrate` first.
- `read() -> dict`: `robot.get_observation()` unchanged (joint keys `*.pos` in degrees, camera keys as arrays).
- `goto(joints_deg: dict)`: clamps every value to `joint_limits_deg`, then clamps the move to
  `max_step_deg` from the last read position, then `robot.send_action`. Returns what was actually sent. The
  double clamp is deliberate: lerobot's `max_relative_target` is a config value that can be left unset by a
  caller; ours cannot.
- `rest()`: walks to `rest_pose_deg` in `max_step_deg` steps at `control_hz`.
- `disconnect()`: `rest()` then `robot.disconnect()`. Also installed as the `__exit__` of the context manager
  and as an `atexit` hook, so a crash mid-sequence still parks the arm. Idempotent.
- `__init__` takes an optional `robot=` for tests, so a `FakeFollower` replaces lerobot's class.

`Leader(config)`: `connect()`, `read() -> dict` (`get_action()`, joint keys in degrees), `disconnect()`. Same
`robot=` injection.

## 3. `test_moves.py`

`python -m HAL.test_moves <cmd>`; argparse subcommands, each a plain function so tests can call them with a
fake:

| cmd | what |
|---|---|
| `ports` | list serial ports (`serial.tools.list_ports`) |
| `read [--leader] [--torque-off]` | connect, print joint positions once, disconnect. `--torque-off` disables torque first so the arm can be placed by hand; `disconnect` then skips the walk to rest |
| `rest` | connect, go to rest, disconnect |
| `sequence` | connect, run `SEQUENCE` (a hard-coded list of six-joint targets with a dwell each, defined at the top of the file), print the read-back position after each, disconnect. Pass criterion printed per waypoint: every joint within `--tol` degrees (default 3) of its target |
| `mirror [--seconds N]` | leader -> follower at `control_hz` through `Follower.goto`, so the clamp applies; stop on Ctrl-C or after N seconds |

`SEQUENCE` is small and conservative: rest, pan left 20°, pan right 20°, lift 15°, gripper close, gripper
open, rest. It is the "is it alive" check and nothing in it approaches a joint limit.

## 4. `rollout.py`

A `Policy` is any object with `act(observation: dict) -> dict` where both dicts use lerobot keys. That is the
whole interface; the VLA adapter and `sim_bridge.SimPolicy` both satisfy it.

`run_episode(follower, policy, config, max_steps, dataset=None, task="")`:
- fixed-rate loop at `control_hz` (perf_counter, sleep the remainder, log when a tick overran)
- each step: `obs = follower.read()`, `action = policy.act(obs)`, `sent = follower.goto(action)`, and if
  `dataset` is given `dataset.add_frame({**obs, "action": sent_as_array, "task": task})`
- ends on `max_steps`, Ctrl-C, or `policy.done` if the attribute exists and is true
- returns the number of steps and whether it was interrupted

The recorded `action` is always the six `*.pos` values actually sent after clamping, in lerobot joint order,
for every policy kind. `hold` therefore emits the same six keys as everything else. What the arm was told to
do is the training label; the raw policy output is not stored.

`record(config, policy, repo_id, episodes, max_steps, task)`: creates a `LeRobotDataset` with
`features` built from the follower's `observation_features` and `action_features` (lerobot's own helpers),
then loops until `episodes` episodes are kept. After each completed episode it asks `keep? [Y/n]`: `n`
clears the episode buffer and does not count; `y` calls `save_episode()`. Then a reset prompt (print and wait
for Enter). Ctrl-C discards the episode in progress and ends the session; what was kept so far is finalized.
`finalize()` at the end either way. Local only, no hub push. The dataset root defaults to
`outputs/hal/<repo_id>`, which is already gitignored under `outputs/`.

CLI: `python -m HAL.rollout --policy {hold,leader,sim} [--ckpt PATH] --episodes N --steps M --task "..."
--repo-id NAME`. `hold` is a policy that returns the current position (the dummy for checking the recording
path); `leader` wraps the leader arm as a policy so teleop demos use the identical recording code; `sim` loads
a checkpoint through `sim_bridge.SimPolicy`.

## 5. `sim_bridge.py`

Everything that knows both frames lives here.

- `real_to_sim(obs: dict, config) -> np.ndarray[5]`: joint degrees -> `sign * rad + offset`.
- `sim_to_real(q_sim: np.ndarray[5], gripper_cmd: float, config) -> dict`: inverse, plus gripper: `> 0.3`
  -> `gripper_open_deg`, `< -0.3` -> `gripper_closed_deg`, otherwise hold the last value. That is the sim's
  hysteresis rule (`docs/imitation.md` §3) applied at the boundary. The "last value" starts as the measured
  gripper position on the first call, so a hold on step one keeps the gripper where it is.
- `SimPolicy(ckpt_path, config, arm="left")`: loads a repo checkpoint with
  `imitation.policies.common.load_policy` (the normalizer travels inside the checkpoint, so nothing is
  re-normalized here). State policies only (`chunk_mlp`, `diffusion` with `images=None`); a checkpoint
  with `needs_images` raises with a clear message, the vision student is a later pass.
  - **Observation built per step**, a 139-D vector laid out as `docs/imitation.md` §2.3: dims 0-4 the five
    joints through `real_to_sim`; dims 5-9 joint velocities by finite difference of the two latest reads
    (zero on the first step); dim 10 the gripper command, 1.0 if the measured gripper is above the midpoint
    of open/closed, else 0.0; every other dim zero. The `arm` argument puts these into the left block (0-10)
    or the right block (25-35). A history of `policy.obs_horizon` vectors is kept, the first step repeating
    the current one.
  - **Chunks:** when no chunk is pending, `predict` on the history gives `[1, K, 12]`; the `arm` slice
    `[0:6]` or `[6:12]` is taken. Replan every `replan_every` actions (default 8, as in sim).
  - **Integration state is the last commanded target,** not the measured position:
    `q_next = q_cmd + delta * JOINT_DELTA_SCALE`, so servo lag does not shrink the motion. If the measured
    position drifts more than `resync_deg` (config, default 10) from the command, the command is reset to the
    measurement and a line is logged. The first command is the measured position.
  - **Rate:** the sim's control step is 50 ms (`control_dt = 0.05`, imitation.md §1), so `run_episode` with a
    `SimPolicy` runs at 20 Hz regardless of `control_hz`. One sim step is at most 0.05 rad = 2.9°, under the
    5° cap, so the clamp never alters a legal sim action.
  - **Known limit, stated in the docstring:** the input is the sensor-only proprio subset with the cloth,
    end-effector and second-arm dims zero. Enough to send a checkpoint's motions to the arm and watch them,
    not enough to expect a fold. Building the real observation is the next pass.

## 6. Tests (no hardware)

`tests/hal/conftest.py`: `FakeFollower` records every `send_action` dict, returns a settable position from
`get_observation`, has `observation_features` / `action_features` like lerobot's. `FakeLeader` returns a
scripted sequence of positions.

- `test_arm.py`: `goto` clamps to joint limits; `goto` clamps the step to `max_step_deg`; `disconnect` sends the
  rest pose and is idempotent; `rest` reaches the rest pose within `max_step_deg` per step.
- `test_sim_bridge.py`: `real_to_sim(sim_to_real(q)) == q` for random q with a non-identity sign/offset;
  gripper hysteresis (open, close, hold); `SimPolicy` with a stub chunk policy integrates deltas and replans
  after `replan_every` calls.
- `test_rollout.py`: `run_episode` with `hold` sends `max_steps` actions and records that many frames into a
  fake dataset; `policy.done` stops early; a `sequence` run against `FakeFollower` reports every waypoint
  within tolerance.

Marks: all fast. Nothing imports lerobot at module level in tests; `arm.py` imports lerobot lazily inside
`connect()` so the test suite and `imitation/spec.py`-style imports stay independent of the driver.

## 7. Bring-up order (goes in `HAL/README.md`)

1. `pip install -r requirements.txt` (adds lerobot).
2. Plug in the follower; `python -m HAL.test_moves ports`; write `HAL/hal_config.json` with the ports.
3. `lerobot-calibrate --robot.type=so101_follower --robot.port=COMx --robot.id=worldfold_follower`, and the
   leader equivalent. Motor IDs must already be 1 to 6 (lerobot's `lerobot-setup-motors` if not).
4. Place the follower by hand in the pose you want as rest, run `read --torque-off`, copy the six numbers
   into `rest_pose_deg` in the JSON. Only then `rest`, `sequence`, `mirror`. The first `rest` is run with a
   hand near the arm.
5. `python -m HAL.rollout --policy hold --episodes 1 --steps 60 --repo-id smoke`, then load it back with
   `LeRobotDataset("smoke", root=outputs/hal/smoke)` and check the frame count.
6. Measure `sim_sign` / `sim_offset_rad`: put the arm at the sim's HOME pose by eye, read degrees, solve.

## 8. Risks and open points

- lerobot on Windows: the Feetech bus uses pyserial, which is fine; lerobot's camera and dataset code pull
  `opencv`, `pyarrow`, `datasets`, which resolved in the dry run but have not been run here. First real import
  is step 1 of bring-up and may surface a Windows-only break. Fallback if it does: a `.venv-hal` the way
  `.venv-isaac` exists, documented in status.md.
- `use_degrees=True` is a lerobot 0.6 config flag; the builder verifies the exact field name against the
  installed package and adjusts `arm.py` if it differs. Units at the HAL boundary stay degrees either way.
- Gripper direction on the real arm (which degree value is open) is checked in `sequence` and the config
  defaults flipped if wrong.

## Decided by the user (2026-10-29)

Leader and follower both available. VLA (MolmoAct2, so101-nexus adapter) is the demo source; imitation policy
is the product; RL optional on top. Record in lerobot's dataset format. All HAL logic under `HAL/`, on branch
`feat/hal`.

## Left to the builder

Exact `SEQUENCE` angles; the rest pose values; the JSON schema details; the stub policies used in tests; the
README wording.

# LeRobot SO-101 joint units vs SO-101 sim joint angles

Written 2026-10-15. Sources: huggingface/lerobot cloned at main `53feaef` (2026-10-14) with all tags
(`scratchpad/lerobot_src`); TheRobotStudio SO-ARM100 `Simulation/SO101` files plus commit history (GitHub API);
MuJoCo Menagerie `robotstudio_so101`; LeIsaac `main` (24d3bcd, 2026-04-15). MuJoCo 3.14 FK runs in
`check_sim_conventions.py` and `derive_old_to_new_calib.py`.
**VERIFIED** means I read the primary source or ran it. **UNVERIFIED** means it is inferred or comes from a secondary source.
Unless a tag or commit is named, file:line refers to lerobot main `53feaef`.

## 0. Answer in one table

| joint | LeRobot value (new API) | sim new_calib angle (deg) | status |
|---|---|---|---|
| arm joints, `use_degrees=True` | v = (tick − mid)·360/4095, mid = (range_min+range_max)/2 | q = +1·v·4095/4096 (+offset) | formula VERIFIED (code); sign +1 VERIFIED for lift/elbow/wrist_flex by data, code-evidence only for pan/roll |
| arm joints, `use_degrees=False` (M100) | v = 200·(clamp(tick) − min)/(max − min) − 100 | q = v/200 · span_deg, span_deg = (max−min)·360/4096 | VERIFIED; span is recoverable from the data's quantisation step (§3) |
| gripper (always RANGE_0_100) | g = 100·(clamp(tick) − min)/(max − min); 0 = closed, 100 = open | q ≈ −10 + 1.10·g (linear onto the sim range) | **APPROXIMATION**: the closed end is backed by sim geometry, the open end is not |
| old API (before 2025-06-05, names `main_*`) | degrees, zero = horizontal pose | q = v + [0, +90, −90, 0, +2.789, 0] | the old_calib→new_calib offsets are VERIFIED by FK; that old LeRobot zero = TRS old_calib is UNVERIFIED |

Default offsets are all 0. Unresolved candidates: **elbow_flex 0 or −5°**, **wrist_roll 0 or +2.789°** (§5).
Frame alignment: dataset row t = (state read *before* action t is sent, action t). Action t first shows up in **state t+1** (§7).

## 1. Encoder, calibration fields, calibration procedure (VERIFIED)

- `"sts3215": 4096` ticks per rev. `motors/feetech/tables.py:190`. The bus runs at `DEFAULT_BAUDRATE = 1_000_000` (`feetech.py:44`). Only `Goal_Position` and `Present_Position` are normalised (`feetech.py:47`).
- `MotorCalibration(id, drive_mode, homing_offset, range_min, range_max)`. `motors_bus.py:175-181`.
- The servo applies the homing offset as "`Present_Position = Actual_Position - Homing_Offset`". `feetech.py:281`.
- `set_half_turn_homings()` "computes and writes a homing offset such that the present position becomes exactly one half-turn (e.g. `2047` on a 12-bit encoder)" (`motors_bus.py:775-797`). The offset is `pos - int(max_res/2)` with `max_res = 4095` (`feetech.py:290-293`).
- `record_ranges_of_motion()` keeps the running min and max of `Present_Position` while the user moves each joint by hand (`motors_bus.py:803-852`).
- SO follower `calibrate()` (`robots/so_follower/so_follower.py`):
  - `:156` prompts "Move {self} to the middle of its range of motion (shown above)". The diagram at `:36-55` shows the upper arm vertical, the forearm horizontal pointing forward, and the gripper forward.
  - `:157` `homing_offsets = self.bus.set_half_turn_homings()`
  - `:160-168` `full_turn_motor = "wrist_roll"`. All other motors are swept, then `range_mins[full_turn_motor] = 0` and `range_maxes[full_turn_motor] = 4095`.
  - `:174` `drive_mode=0` for every motor. `read_calibration()` also hard-codes `drive_mode=0` (`feetech.py:260`).
  - `:180` `write_calibration` writes Homing_Offset, **Min_Position_Limit = range_min and Max_Position_Limit = range_max into the servo** (`feetech.py:268-273`).
- Docs: "First you need to move the robot to the position where all joints are in the middle of their ranges. Then after pressing enter you have to move each joint through its full range of motion." `docs/source/so101.mdx:398`.

## 2. Normalisation code (VERIFIED, `motors_bus.py:854-913`)

```
_normalize:  bounded_val = min(max_, max(min_, val))                                  # :867
  RANGE_M100_100: norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100; -norm if drive_mode  # :868-870
  RANGE_0_100:    norm = ((bounded_val - min_) / (max_ - min_)) * 100; 100-norm if drive_mode       # :871-873
  DEGREES:        mid = (min_ + max_) / 2; max_res = resolution - 1
                  normalized = (val - mid) * 360 / max_res          # NOT clamped, drive_mode NOT applied  # :874-877
_unnormalize:
  RANGE_M100_100: int(((clip(val,-100,100) + 100) / 200) * (max_ - min_) + min_)      # :896-899
  RANGE_0_100:    int((clip(val,0,100) / 100) * (max_ - min_) + min_)                  # :900-903
  DEGREES:        int((val * max_res / 360) + mid)                                      # :904-907
```
- `FeetechMotorsBus.apply_drive_mode = True` (`feetech.py:95`), but SO arms always have drive_mode 0, so **LeRobot applies no sign flip in software**. The direction of each joint is set by how its servo is mounted.
- Mode selection: `norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100` (`so_follower.py:72`). The gripper is always `MotorNormMode.RANGE_0_100` (`:81`). The leader is the same (`teleoperators/so_leader/so_leader.py:63-72`).

## 3. Derived formulas

Let tick be the post-homing Present_Position (0..4095), with mid = (range_min + range_max)/2 and span_t = range_max − range_min.

- DEGREES: v = (tick − mid)·360/4095. So **v = 0 exactly at the middle of the swept range, not at the homing pose** (self-test PASS). Physical angle = (tick − mid)·360/4096 = v·4095/4096. LeRobot's 360/4095 scale overstates angles by 0.024 % (0.024° at 100°). That rests on the assumption that the encoder has 4096 counts per turn (UNVERIFIED on hardware; tables.py says 4096).
- M100: v = 200·(tick − min)/span_t − 100 = 200·(tick − mid)/span_t. **Same zero (mid of range)**. Physical angle = v·span_t/200·360/4096. The values are clamped to ±100, so the state saturates at the calibrated range ends; degrees mode does not clamp.
- 0_100 (gripper): g = 100·(tick − min)/span_t. Physical opening from the closed end = g·span_t/100·360/4096.
- Commands (Goal_Position) are truncated with `int()`, an error under 1 tick (0.088°). The servo itself also holds Min/Max_Position_Limit = range_min/range_max, so goals outside the calibrated range are clamped by the servo. That the firmware clamps is UNVERIFIED (Feetech behaviour); that the registers are written is VERIFIED.

**Unknown per-arm calibration with M100 data.** The zero is still mid-range. The only extra unknown per joint is span_t, and the data reveal it. Present_Position is an integer, so observation.state values lie on a lattice with step 200/span_t (M100) or 100/span_t (gripper). In degrees mode every arm joint has the same step, 360/4095 = 0.087912. `lerobot_units.estimate_tick_step` / `infer_units` recover span_t exactly. On `teacher/results/real_so101/neuroto_ep0/ep0.parquet` (VERIFIED, run output) the spans come out as whole ticks, with lattice residual about 1e-5:

| joint | arm A span (ticks / deg) | arm B span | sim span (deg) |
|---|---|---|---|
| shoulder_pan | 2501 / 219.8 | 2726 / 239.6 | 220.0 |
| shoulder_lift | 2368 / 208.1 | 2550 / 224.1 | 200.0 |
| elbow_flex | 2207 / 194.0 | 2203 / 193.6 | 193.7 |
| wrist_flex | 2361 / 207.5 | 2395 / 210.5 | 190.0 |
| wrist_roll | 3853 / 338.6 | 3833 / 336.9 | 320.0 (swept, so lerobot ≤ v0.4.2) |
| gripper | 1430 / 125.7 | 1448 / 127.3 | 110.0 |

So neuroto_ep0 is **M100 mode**. Assuming the sim spans instead (as LeIsaac does) would give **up to 12 % scale error** on pan, lift and wrist_flex. Always estimate spans from `observation.state`, never from `action`. Teleop actions carry the leader's lattice, but the follower un-normalises the action value with its own calibration, so the **follower span converts both state and action**.

## 4. Version history: dataset → units (VERIFIED from tags; `git merge-base --is-ancestor` checked)

No tags older than `v0.3.2` exist on GitHub (`git ls-remote --tags`). pyproject was `version = "0.1.0"` before and after the redesign. "v0.2.x" was not found as a tag.

| lerobot | date | SO-101 `use_degrees` default | SO-101 wrist_roll range | recorded `action` | gripper torque caps | P/I/D | dataset codebase |
|---|---|---|---|---|---|---|---|
| pre-redesign (≤ b536f47e) | ≤ 2025-06-05 | n/a: old DEGREE/LINEAR calib (§6) | n/a | goal after clip (manipulator.py:478,520) | none | 16/0/32 (manipulator.py:428-442) | v2.1 (`lerobot_dataset.py:77`) |
| e23b41e7 "Hardware API redesign (#777)" | 2025-06-05 | `False` | swept | — | none | 16/0/32 | v2.1 |
| v0.3.2 / v0.3.3 | 2025-08-01/06 | `False` (`config_so101_follower.py:41`) | swept (`so101_follower.py:133` records all motors) | **sent_action** (post-clip) (`record.py:271-274`) | none | 16/0/32 hard-coded | v2.1 (`lerobot_dataset.py:76`) |
| v0.4.0 – v0.4.2 | 2025-10-23 – 11-27 | `False` | swept | **action_values** (pre-clip) (`lerobot_record.py:359` @v0.4.0) | 500/250/25 (commit 882c80d4, 2025-08-29) | 16/0/32 | **v3.0** (`lerobot_dataset.py:79`) |
| v0.4.3 | 2026-01-20 | `False` (`config_so_follower.py:43`) | **0..4095 forced** (ccfd609e, 2026-01-08, `so_follower_base.py:131-139`) | pre-clip | yes | 16/0/32 | v3.0 |
| v0.4.4 – v0.6.0 | 2026-02-27 – 07-06 | **`True`** (6600b60e "always use degrees (#2968)", 2026-02-13) | 0..4095 | pre-clip | yes | 16/0/32 | v3.0 |
| v0.6.1 – main | 2026-08-03 – | `True` (`config_so_follower.py:42`) | 0..4095 | pre-clip (`lerobot_record.py:367`) | yes | config fields 16/0/32 (`:45-47`) | v3.0 |

Notes:
- The default was flipped in commit 6600b60e, but the comment above it still says "Set to `True` for backward compatibility with previous policies/dataset" (`config_so_follower.py:41`).
- Users can override `use_degrees`, so use the quantisation test (§3), not the date alone. The `default` row only tells you what an unmodified CLI produced.
- The SO-100 follower had wrist_roll forced to 0..4095 from at least v0.3.2 (`so100_follower.py:129-137`). The SO-101 got it only from v0.4.3.
- The `max_relative_target` default is `None` in every version (`config_so_follower.py:36`). The pre-clip/post-clip difference only matters if a user set it.

## 5. Sim model conventions (VERIFIED by reading files + FK)

- **TRS so101_new_calib.xml and Menagerie so101.xml are kinematically identical.** Over 200 random configurations the max body-position difference is 0.0000 mm and the rotation-matrix difference is 5e-8. The ranges are identical. Menagerie README: "Copied `so101_new_calib.xml` (commit SHA aec17bbc...)". (`check_sim_conventions.py` [1].)
- Ranges (deg): pan ±110, lift ±100, elbow ±96.83 (±1.69 rad), wrist_flex ±95, wrist_roll −157.21..+162.79 (middle **+2.789**), gripper −10..+100.
- **Sim zero pose** (FK, world +z up): the upper arm (shoulder_lift→elbow anchors) is at +76° elevation, the forearm is horizontal (+2.2°) pointing +x, and the wrist→gripperframe line is horizontal (−2.8°). That is the same shape as LeRobot's calibration diagram (`so_follower.py:36-55`). The SO-101 upper arm is a bent link, which is why its anchor line is not exactly 90°.
- Joint axes at zero (world): pan (0,0,−1); lift/elbow/wrist_flex (0,+1,0); roll (−1,0,0). Effect of +10° on the jaw site: pan → −y (clockwise seen from above); **+lift → forward/down; +elbow → down; +wrist_flex → down**.
- TRS README: "New Calibration (Default): Each joint's virtual zero is set to the **middle** of its joint range." Old Calibration: zero = "fully extended horizontally".
- TRS commit history of `so101_new_calib.urdf` (GitHub API): 4026fe1a 2025-05-19 (added), bbb9734c 2025-05-21, **0c8e2892 2025-06-25 "Updated the new so101 calibration urdf with fixed direction and frame (#106)"** (co-author CarolinePascal), 385e8d7c 2025-07-02 (frame/link-name fixes). PR #106 messages include "shift joint(3) elbow flex by five degrees to adhere to the offset on the real robot", "correct joint 3 offset" and "Renamming joints to match LeRobot middleware naming convention". In that diff:
  - **elbow**: the range went from `[-1.74533, 1.5708]` (−100°..+90°, middle −5°) to `[-1.69, 1.69]`. The `lower_arm` body frame is unchanged. Hence the URDF comment "5-degree calibration offset applied to joint limits".
  - **wrist_roll**: the range went from ±2.79253 (±160°) to −2.74385..+2.84121, and the gripper-body quaternion gained a 2.789° roll. FK confirms the gripper hinge axis is horizontal (gripper upright) at **wrist_roll = +2.789°**, which is the middle of the sim range, not sim zero (`derive_old_to_new_calib.py`).
- **old_calib → new_calib** (FK fit, residual 0.0000 mm, 40 random configurations): `q_new = q_old + [0, +90, −90, 0, +2.789, 0]` deg, all signs +1.
- Offsets I cannot resolve without a real arm:
  - **elbow**: is the true mechanical range symmetric about the CAD frame (offset 0), or −100/+90 as the pre-#106 limits said (offset −5°)? Both neuroto arms have an elbow span of 193.6–194.0°, which matches the sim's ±96.83° (193.66°), but the span does not say where the stops are.
  - **wrist_roll**: is the LeRobot zero (mid of the swept range ≤ v0.4.2, or the homing pose ≥ v0.4.3) at the upright gripper (sim +2.789°) or at sim 0?
  - A constant offset applied to both action and state does not change tracking error directly; it only shifts gravity and contact geometry. Fit it in sysid if needed.

## 6. Old API (before 2025-06-05), for datasets with `main_shoulder_pan`-style names (VERIFIED code @ b536f47e)

- Feature names: `f"{arm}_{motor}"` (`manipulator.py:171`).
- Zero: "We arbitrarily chose our zero target position to be a straight horizontal position with gripper upwards and closed" (`feetech_calibration.py:454`). The rotated position is +90° (`:35-36`). drive_mode comes from the rotated pose: `drive_mode = (rotated_pos < zero_pos)` (`:479`). So per-motor sign flips **were** applied in the old API.
- Value: `if drive_mode: values *= -1; values += homing_offset; values / (resolution // 2) * 180` (old `feetech.py:451-460`). That is 360/4096 per tick. Gripper LINEAR: `(values - start_pos) / (end_pos - start_pos) * 100` (`:478`).
- It is UNVERIFIED that the old "zero" and "rotated" photos match TRS old_calib's zero and positive directions. `to_radians(units='old_degrees')` assumes they do.

## 7. Frame alignment: action vs observation.state (VERIFIED code)

- New API `record_loop` (`scripts/lerobot_record.py`):
  - `:308` `obs = robot.get_observation()`, which does `sync_read("Present_Position")` (`so_follower.py:208`).
  - `:315` builds the observation frame.
  - `:320` `act = teleop.get_action()`, which reads the leader's `Present_Position`.
  - `:362` `robot.send_action(...)`, which does `sync_write("Goal_Position")` (`so_follower.py:254`).
  - `:367-369` the action frame is built from `action_values` and the row is added.
  - So row t holds the state read **before** action t is written. Action t can affect state t+1 at the earliest, and the servo response adds more delay on top. The same holds in v0.3.2 (`record.py:235` observe → `:251` teleop → `:271` send).
- Old API `teleop_step`: the goal is written (`manipulator.py:478`), then the follower is read (`:490`) in the same step, so state t is read a few ms after action t is written. In practice that is the same pairing: action t → state t+1.
- `action` = leader joint value in the leader's own normalisation, reinterpreted by the follower with the follower's calibration. With M100 that makes leader and follower spans differ (neuroto arm A pan: leader step 0.0874 vs follower 0.0800).
- The timestamp is `frame_index/fps`. The camera is read after the motors in `get_observation` (`so_follower.py:214-219`).
- `lerobot_units.align_action_state(action, state, lag_frames=1)` gives the minimum causal pairing. Fit any extra delay as a sim actuator delay.

## 8. Sign evidence (LeRobot value → sim new_calib)

1. LeRobot's own IK/FK feeds degrees straight into `so101_new_calib.urdf`: `joint_pos_rad = np.deg2rad(joint_pos_deg[...])` and then `self.robot.set_joint(joint_name, joint_pos_rad[i])` (`model/kinematics.py:86-90`). There are no signs or offsets, and the examples recommend this URDF (`examples/phone_to_so100/record.py:69-71`). (VERIFIED code.)
2. TRS PR #106 "fixed direction", co-authored by a LeRobot maintainer (VERIFIED commit message).
3. LeIsaac maps real leader `RANGE_M100_100` values linearly onto the USD limits with no flips (`leisaac/utils/robot_utils.py convert_lerobot_action_to_leisaac`; `so101_leader.py:39-44`). (VERIFIED code.)
4. Rest pose. LeRobot's "empirically comfortable pose" `RESET_ORIGIN_DEG` = pan −4, lift −103, elbow 97, wrist_flex 78, roll −65 (`examples/isaac_teleop_to_so101/common.py:192-202`). In the sim with identity signs it gives a compact fold: horizontal reach 12.6 cm, jaw site z −1.7 cm. With lift/elbow/wrist_flex flipped it gives an outstretched arm: reach 35.5 cm, jaw 22 cm up (FK, VERIFIED). This supports identity but is weak evidence.
5. **Gravity sign test on real data** (`check_sim_conventions.py` [6], neuroto_ep0, converted with the inferred M100 spans). At quasi-static frames a P-servo under load satisfies sign(action − state) = sign(sim gravity torque) if the mapping is right. Agreement (n frames) for the joint under test:

| hypothesis | arm A lift | arm A elbow | arm A wrist_flex | arm B lift | arm B elbow | arm B wrist_flex |
|---|---|---|---|---|---|---|
| identity | **87.9 %** (132) | **90.9 %** (176) | n/a (0) | **96.8 %** (277) | **92.3 %** (365) | **100 %** (29) |
| that joint flipped | 55.3 % (132) | 70.5 % (176) | 2.0 % (49) | 4.3 % (277) | 2.3 % (348) | 45.7 % (46) |

Identity wins for every testable joint on both arms. Caveats: this is one episode of unknown provenance (custom 36-dim bimanual layout), and the action is the leader's position, so leader/follower zero mismatch adds noise. shoulder_pan and wrist_roll carry no gravity load, so their signs rest on evidence 1–3 only.

## 9. Gripper

- LeRobot: 0 = range_min = closed, 100 = range_max = open. TRS README "Gripper Note": "`0` = fully closed, `100` = fully open … not yet reflected in the current URDF and MuJoCo files". `examples/isaac_teleop_to_so101/common.py:192-193`: "gripper RANGE_0_100, 100 = open". (VERIFIED docs/comments. Not verified on hardware; with drive_mode 0 the direction depends on servo mounting.)
- Sim: the jaws close toward the **lower** limit. Menagerie jaw primitives (pads + tip spheres) have a min distance of 1.73 mm at −10°, 14.8 mm at 0°, 67 mm at 45° and 99 mm at 100°. They do not touch inside the joint range (FK, VERIFIED). So LeRobot's `0` corresponds to roughly sim −10°, not 0°.
- `to_radians` default `gripper_mode='linear_sim'`: q = −10 + 1.10·g. This is an APPROXIMATION. Real calibrated sweeps were 125.7° and 127.3° on the neuroto arms versus the sim's 110°, so the open end is compressed about 13 %. Use `gripper_mode='span'` with the inferred span to keep the real scale; it can exceed the sim's +100° limit. LeRobot's own kinematics passes the gripper value through as degrees (`'identity'`), which puts "closed" about 15 mm open in the sim.

## 10. Servo settings LeRobot writes on the real follower (VERIFIED code, main)

| register (addr) | value | where | note |
|---|---|---|---|
| Operating_Mode (33) | 0 = POSITION | `so_follower.py:188` | all joints |
| P_Coefficient (21) | **16** | `config_so_follower.py:45`, written `so_follower.py:189` | old comment: "Set P_Coefficient to lower value to avoid shakiness (Default is 32)" (`so_follower.py:164` @v0.6.0) |
| I_Coefficient (23) | **0** | `:46`, `:190` | |
| D_Coefficient (22) | **32** | `:47`, `:191` | old comment: "Set I_Coefficient and D_Coefficient to default value 0 and 32". The BAM STS3215 datasets were recorded with D = 0 (see `sts3215_actuator_models.md`) |
| Return_Delay_Time (7) | 0 (2 µs; factory 250 = 500 µs per comment) | `feetech.py:209-213` | |
| Maximum_Acceleration (85) | **254** (protocol 0) | `feetech.py:214-216` | "to speedup acceleration and deceleration"; units/semantics UNVERIFIED; factory 50 per T-K-233 (see actuator note) |
| Acceleration (41) | **254** | `feetech.py:217` | |
| Phase (18) | bit 4 cleared | `feetech.py:219-225` | position feedback stays in [0, 4095] |
| Homing_Offset (31), Min/Max_Position_Limit (9/11) | from calibration | `feetech.py:268-273` | the servo holds the calibrated range |
| Max_Torque_Limit (16) | **500 on gripper only** ("50% of max torque") | `so_follower.py:194` | since v0.4.0 (882c80d4). Arm joints keep the factory value (not written) |
| Protection_Current (28) | **250 on gripper only** ("50% of max current") | `:195` | |
| Overload_Torque (36) | **25 on gripper only** ("25% torque when overloaded") | `:196` | |
| Torque_Enable (40) / Lock (55) | 1/1 when enabled | `feetech.py:308-311` | |

**Not written** by LeRobot (factory values apply; read them off a real servo before trusting an actuator model): CW/CCW_Dead_Zone (26/27), Minimum_Startup_Force (24), Torque_Limit (48, RAM), Goal_Velocity (46), Goal_Time (44), Protective_Torque (34), Protection_Time (35), velocity-loop P/I (37/39), arm-joint Max_Torque_Limit. Other settings: `max_relative_target=None`, `num_read_retries=2` (`config_so_follower.py:36,53`), and P/I/D 16/0/32 has been the same in every version (§4).

## 11. Not found / open

- Per-joint sign verification for shoulder_pan and wrist_roll from real data. Gravity cannot test them; it would need video + FK or a hand-measured pose.
- Elbow (0 vs −5°) and wrist_roll (0 vs +2.789°) offsets. A sysid fit of a constant offset, or one photo of a real arm at LeRobot zero, would settle them.
- Provenance of `neuroto_ep0` (HF author "neuroto" returned no datasets via `api/datasets?author=neuroto`). Its 36-dim layout is not stock LeRobot. The swept wrist_roll range implies lerobot ≤ v0.4.2.
- Real gripper closed-tick vs jaw contact (does range_min mean the jaws touch, or are they squeezed?) and the real open angle.
- Firmware semantics of P=16/D=32, Acceleration=254 and position-limit clamping: vendor behaviour, not verified here.

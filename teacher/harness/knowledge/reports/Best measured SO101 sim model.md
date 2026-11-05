# Measured servo data beats every SO-101 asset

No SO-101 simulation model was found that has been identified as a whole arm and checked against a real arm with a reported joint error. The best pieces exist only separately. **NVIDIA's Isaac Sim 6.1 `so101_new_calib_SysID` USD** is the only asset with per-joint identified gains, friction and armature (verified), but how NVIDIA collected and fit that data was not found. It reports no error numbers, and its masses are the same CAD values as everyone else's. **MuJoCo Menagerie `robotstudio_so101`** has the most carefully built gripper contact model (verified), but its actuator gains come from a datasheet calculation, not a measurement. **Rhoban BAM's STS3215 servo models** are the only real hardware measurements of the servo: 0.78° mean absolute error (MAE) at firmware P=16 for 7.4 V, and 0.41–0.43° for 12 V (reproduced, in-sample). They were measured on a single servo swinging a pendulum on a test bench, with D=0 and at about 7.3 V. A stock LeRobot follower runs with D=32, acceleration registers set to 254, and a 5 V supply. Every asset also uses unweighed Onshape CAD masses (total 0.632 kg) and has no measured grip force or jaw gap. Reaching the OmniReset-level target of under 2° joint RMSE looks possible in principle but has not been shown on an SO-101. The likely floor is about 1° from servo deadband alone (inference). For cloth grasping with stock jaws, the team should build its own model:

- **Geometry and contact:** Menagerie's (or the NVIDIA USD's for Isaac/Newton).
- **Actuator:** a BAM-style servo model evaluated at P=16.
- **Fitting:** identify the arm on its own LeRobot settings with chirp trajectories (PACE or the Isaac SysID extension).
- **Gripper:** weigh the parts, then measure jaw closed gap and grip force before any reinforcement learning.

## Every asset shares one CAD lineage and none is validated against hardware

Almost every SO-101 asset descends from TheRobotStudio's `Simulation/SO101` files. Those were generated with onshape-to-robot from one Onshape document ([SO101 README](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md)). The per-link masses are identical everywhere I checked: **base 0.147, shoulder 0.100, upper arm 0.103, lower arm 0.104, wrist 0.079, gripper 0.087, moving jaw 0.012 kg, total 0.632 kg** (verified in TRS URDF/MJCF, Menagerie and the NVIDIA SysID `physics.usda`) ([TRS URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf); [NVIDIA physics.usda](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/payloads/Physics/physics.usda)).

NVIDIA's SysID did not change the masses (verified). A published weighing of SO-101 parts was not found. I searched the TRS README and Simulation README, and ran web searches for SO-101 per-part weights and for mass identification. The only density statement in the repo covers the leader arm, whose inertias came from "a rough average printed-part density, not from a measured leader unit" (verified). Vendors list the follower at **704 g** (Hiwonder, vendor listing) ([Hiwonder](https://www.hiwonder.com/products/lerobot-so-101)) and about 800 g (unverified third-party page) ([RoboticsCenter](https://www.roboticscenter.ai/hardware/so-101/specs)). The sim total is therefore 10% or more below the listed figures. Six STS3215 servos at 55 ± 1 g each make up about half the arm (vendor) ([RobotShop](https://www.robotshop.com/products/magnetic-encoding-servo-sts3215-74v-19kgcm)).

Joint and actuator parameters come from three lineages. The TRS/Menagerie values (damping 0.60, frictionloss 0.052, armature 0.028) are "adapted from the Open Duck Mini project" (verified). By my arithmetic, that traces to a BAM identification of a bare 7.4 V servo at firmware kp=32 (inference). The TRS gains kp 998.22 / kv 2.731 were added on 2025-11-27. They are calculated from the datasheet under an assumed P=16 and nothing was measured (verified commit; derivation described in an unverified fetch summary) ([so101_new_calib.xml](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.xml); [Gregory119 README](https://github.com/Gregory119/RBE501-RL-arm-project/blob/main/gymnasium_env/README.md)). The third lineage is NVIDIA's per-joint SysID values, shown in the table below. A third-party survey of about 11 SO-101 sim projects says none reports a quantitative real-vs-sim comparison. That survey also gets the TRS/Menagerie kp lineage wrong, so its other claims should be treated with caution ([survey](https://mcobzarenco-fontaine-blog.static.hf.space/papers/so101-sim-landscape.html)).

| Rank | Asset | Masses | Joint/actuator params | Gripper contact | Real-vs-sim validation |
|---|---|---|---|---|---|
| 1 (Isaac/Newton) | NVIDIA Isaac Sim 6.1 `so101_new_calib_SysID` USD, now Isaac Lab's canonical SO-101 ([PR #7483](https://github.com/isaac-sim/IsaacLab/pull/7483)) | CAD (verified) | **Per-joint identified** stiffness, damping, armature, friction; force limit 3.35 (verified); data source and fit error not found | Convex hull, no gripper primitives (verified) | "Runtime smoke rollouts" only (verified) |
| 1 (MuJoCo) | Menagerie `robotstudio_so101` ([README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/README.md)) | CAD (verified) | Copied from TRS: datasheet kp 998.22 / kv 2.731, Open Duck friction, same on all joints (verified) | **Best:** primitive pads, 0.75 mm tip spheres, condim 6, tuned solref, convex-decomposed jaws (verified) | None stated |
| 3 | TRS `so101_new_calib` URDF/MJCF ([repo](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)) | CAD (verified) | Same as Menagerie; `joints_properties.xml` still has kp 17.8 (verified) | Full CAD meshes (verified) | None found |
| 4 | LeIsaac `so101_follower.usd` ([lerobot.py](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/assets/robots/lerobot.py)) | TRS-derived | 17.8/0.60 reused as PhysX PD gains, effort 10 (verified) | −10…100° | None |
| 5 | NVIDIA Sim-to-Real workshop USD ([so101.py](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop/blob/main/source/sim_to_real_so101/assets/so101.py)) | — | Hand-set PD, effort 30 (about 10x the servo), implausible torque comments (verified) | — | None |
| 6 | ManiSkill SO-100 / Menagerie `trs_so_arm100` ([ManiSkill](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/so100/so_100.py)) | SO-100 kinematics, a different arm | Generic, "not properly identified" (verified) | Friction 2.0 / manual convex jaws | None |

NVIDIA's per-joint values (verified; USD angular units are per degree) are the most informative numbers here:

| Joint | Stiffness (N·m/deg) | Damping (N·m·s/deg) | Armature | Coulomb friction (N·m) |
|---|---|---|---|---|
| shoulder_pan | 0.845 | 0.058 | 0.068 | 0.347 |
| shoulder_lift | 0.834 | 0.045 | 0.028 | 0.345 |
| elbow_flex | 0.257 | 0.006 | 0.038 | 0.411 |
| wrist_flex | 0.761 | 0.072 | 0.051 | 0.248 |
| wrist_roll | 0.958 | 0.052 | 0.055 | 0.222 |
| gripper | 1.191 | 0.049 | 0.078 | 0.084 |

Converted to radians, the arm-joint stiffnesses are about 15–55 N·m/rad (inference) ([physics.usda](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/payloads/Physics/physics.usda)). That is about 20–70x lower than Menagerie's 998. The Coulomb friction is **0.22–0.41 N·m, about 4–8x the 0.052 in TRS/Menagerie**. Two independent sources point the same way. The so101-rt-control project measured a real friction floor of about **18x** the Menagerie value on one run (verified README) ([so101-rt-control](https://github.com/0xshinobii/so101-rt-control)). A jatinsikka Isaac reach policy oscillated on hardware because the sim joints responded much faster than the real servos (verified) ([PR #1](https://github.com/jatinsikka/so101_isaac/pull/1)). Taken together, the Menagerie/TRS actuator is almost certainly too stiff and too frictionless.

How NVIDIA produced its numbers was not found. I searched the Isaac Sim 6.1 release notes, the PR #7483 body and comments, the SysID extension pages (grep for SO-101), and ran a web search for the asset name. Isaac Lab itself keeps a high-gain `SO101_HIGH_PD_CFG` because the SysID gains "may track end-effector targets less tightly" (verified) ([so101.py](https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_assets/isaaclab_assets/robots/so101.py)).

## The only real measurements are of one servo on a pendulum

Two STS3215 hardware identification campaigns with published physical parameters were found, and both use Rhoban's BAM tool:

- **7.4 V set, on BAM `main`.** 100 logs recorded on 2025-02-27 by apirrone for Open Duck Mini, refit to models M1–M6 in August 2026 (verified) ([BAM params](https://github.com/Rhoban/bam/tree/main/bam/params/feetech_sts3215_7_4V); [issue #13](https://github.com/Rhoban/bam/issues/13)).
- **12 V set, in open PR #20 by T-K-233.** 97 logs, firmware 3.10, regulated 12 V, with kp=8 held out (verified) ([PR #20](https://github.com/Rhoban/bam/pull/20)).

A third campaign, **NeuralActuator**, trains a Transformer on real SO-101 teleop data recorded through LeRobot. It reports a worst per-joint MAE of 1.10–1.53° (verified README) ([NeuralActuator](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/main/NeuralActuator)). It is a learned black box, and it needs load, voltage and temperature as inputs, which a simulator would have to make up. Other projects only reuse these data:

- isaac_bam_actuators ships BAM's parameters byte-for-byte (verified).
- PartGap refits BAM's logs (verified).
- arachne-18 extrapolates to 12 V from datasheet points (verified).

Beyond these, I searched GitHub for "sts3215 bam", "sts3215 system identification" and "sts3215 actuator model", and ran two web searches. No other STS3215 identification was found.

| Rank | Actuator model | Error at P=16 | Held-out evidence | Use in sim |
|---|---|---|---|---|
| 1 (12 V servos) | BAM PR #20, M4–M6 | **0.41–0.43°** MAE (reproduced, in-sample) | 0.60° at kp=8 held out (verified + reproduced) | MuJoCo via BAM controller; Newton needs a custom component |
| 1 (7.4 V servos) | BAM main, M3–M6 | **0.78–0.81°** (reproduced, in-sample) | PartGap 80/20 split: 0.88° overall, 0.806° at kp=16 (M6) (verified) ([PartGap](https://github.com/bluvtx09/partgap/blob/main/db/parts/feetech_sts3215_7v4.json)) | Same as above |
| 3 | BAM M1 (Coulomb + viscous) | 1.25° (7.4 V), 1.16° (12 V) (reproduced) | PartGap M1 held-out 1.39° (verified) | **Native** MuJoCo position actuator; Newton 1.6.1 DrivePD + ClampingDCMotor + Delay (inference, untested) |
| 4 | NeuralActuator (learned, SO-101) | 1.10–1.53° worst-joint MAE (verified README) | Train/val/test splits per task | Needs telemetry inputs; trainer supports Newton/MJX/MuJoCo Warp |
| 5 | NVIDIA SysID USD per-joint values | Not reported | None found | Native Isaac/Newton |
| 6 | TRS/Menagerie kp 998 or kp 17.8 | Not measured | None | Native, but likely wrong scale |

All BAM errors come from open-loop replays of 6-second, single-servo pendulum logs, not arm trajectories ([PartGap README](https://github.com/bluvtx09/partgap)). Load-dependent friction (M3 and up) gives the largest gain over M1. M5 and M6 add little beyond that (inference). Two warnings from the primary sources matter for transfer:

- PartGap found that back-driven motion is **off by 1.8x to 16.8x on drop tests**, with torque-off error of about 2.8–3.0° (verified).
- The 7.4 V parameters **do not transfer to 12 V** servos: "they predict the arm falling where the 12 V servo holds it by friction" (verified) ([PR #20](https://github.com/Rhoban/bam/pull/20)).

The fitted kt and R are effective values, not electrical truths. PR #20 notes that only kt·gain/R and kt²/R are identifiable.

Using PartGap's mapping (my computation, unverified on hardware), the 7.4 V M1 model at firmware P=16 becomes a MuJoCo or Newton position drive with **kp 9.38 N·m/rad, damping 0.619, force limit ±3.41 N·m, frictionloss 0.0515, armature 0.0261**, plus a rate limit of about 5.3 rad/s and about 1 ms of delay. The official kp of 998 is about 100x stiffer than this, which points to a units or scale error in the datasheet derivation (inference). Note that NVIDIA's independently fitted stiffnesses (15–55 N·m/rad) sit much closer to BAM's 9.4–18.8 than to 998. The servo armature of about 0.02 kg·m² dominates the small link inertias, which makes armature the highest-leverage parameter to get right (inference).

## LeRobot's real servo settings fall outside every measured dataset

Firmware P=16 is not the problem. BAM recorded at P = 4, 8, 16 and 32 and treats Kp as a runtime input, so **P=16 is inside the measured range** (verified) ([record.py](https://github.com/Rhoban/bam/blob/main/bam/feetech/record.py); [all_record.py](https://github.com/Rhoban/bam/blob/main/bam/feetech/all_record.py)). The other settings are the problem:

| LeRobot follower setting (verified) | What the measured models assume | Consequence |
|---|---|---|
| P=16, I=0 ([config_so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py)) | P in {4, 8, 16, 32} measured | Covered: 0.78° (7.4 V M6) / 0.43° (12 V M6) at kp=16 (reproduced) |
| **D=32** | D=0 in both datasets; BAM has no D term (verified) | Not covered. Sign, scale, and whether D acts in position mode are not documented in anything read. Expect more damping than BAM predicts (inference) |
| **Acceleration=254 (reg 41), Maximum_Acceleration=254 (reg 85)** ([feetech.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py)) | BAM's recorder never writes 41/85. T-K-233 found factory value 50 caps acceleration at about 10–28 rad/s² and set both to 0 (verified) | Not covered. What 254 means in physical units was not found (no primary Feetech register spec for reg 85 retrieved) |
| **5 V supply** for the stock 7.4 V follower ([SO-ARM100 README](https://github.com/TheRobotStudio/SO-ARM100)) | 7.4 V data recorded at a measured 7.1–7.3 V | Extrapolation. Scaling vin to 5 cuts stiffness and saturation by 5/7.4, which is unvalidated (inference) |
| Return_Delay_Time=0; Phase bit 4 cleared | Not stated for 7.4 V data | Unknown |
| Gripper: Max_Torque_Limit 500, Protection_Current 250, Overload_Torque 25 ([so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)) | Not modeled anywhere | Sustained squeeze may drop to about 25% of max, assuming SMS-series register semantics apply (unverified) |
| Calibration: zero at midpoint of swept range; wrist_roll forced to 0–4095 | TRS "new_calib" also puts zero at mid-range (verified) | Per-arm constant offsets and assembly-dependent signs (unverified community reports) ([SO100-to-mujoco](https://github.com/yhchow/SO100-to-mujoco)); the elbow already carries a "5-degree calibration offset" in the URDF (verified) |

No STS3215 identification with D≠0 or with LeRobot's acceleration registers was found. The torque limits in the official models are also too generous for a stock arm. The README rates the 7.4 V servo at 16.5 kg·cm (1.62 N·m) at 6 V, while a vendor says 19.5 kg·cm (1.91 N·m). The two sources conflict, and stall torque at 7.4 V is unconfirmed. Either way, the MJCF's **±3.35 N·m is above even the 12 V servo's 30 kg·cm (2.94 N·m)** rating ([ST-3215-C018 datasheet](https://cdn.robotshop.com/media/F/Fit/RB-Fit-155/pdf/feetech_12v_30kg_cm_magnetic_encoding_servo_sts321_specification_pdf.pdf)). BAM's fitted ±3.41 N·m at 7.4 V is an effective fit limit, not a stall torque (inference).

On data units, LeRobot source sets `use_degrees=True` by default (verified) ([config_so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py)). An unverified docs mirror agreed, and a third-party page claims [−1, 1]. The gripper is always reported as 0–100 between the recorded min and max ticks (verified), so it needs its own measured linear map to jaw angle. TRS says LeRobot's gripper mapping "is not yet reflected" in its sim files (verified).

## Under 2° is a plausible target, but no SO-101 has shown it

OmniReset reports **"under 2 degree RMSE in joint space"** after PACE identification, against about 7° with default parameters (verified). That result is on a UR7e with a Robotiq 2F-85, not an SO-101. It used open-loop chirps, CMA-ES, and fitted friction, armature and motor delay ([OmniReset](https://arxiv.org/html/2603.15789)). A geared actuator with an embedded PD loop, similar in kind to the STS3215, reached 7.54 mrad (0.43°) held-out MAE via differentiable-sim identification (verified abstract) ([arXiv 2604.10351](https://arxiv.org/abs/2604.10351)).

On the SO-101 itself, no matched whole-arm real-vs-sim RMSE was found. Searches covered SO-101/SO-100 sysid, friction and armature identification, and Isaac SysID with LeRobot. LeRobot Discord, theses and HF blogs were not searched, and the full texts of NeuralActuator and the SAGE/GapONet papers were not read. NVIDIA's SAGE/GapONet course compares about 8 h of paired sim and real trajectories, but shows the result only in plots (verified page) ([NVIDIA course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/15-strategy4-sage.html)). so101-rt-control's 1.54° (PD, 90 g) and 0.37° (identified, 90 g) are real-arm tracking errors, not sim fidelity, and the author says the two "are not directly comparable" (verified) ([so101-rt-control](https://github.com/0xshinobii/so101-rt-control)).

The error budget suggests a floor of about 1° (inference). Servos settle 10–16 ticks short under load (verified report) ([PR #1](https://github.com/jatinsikka/so101_isaac/pull/1)). At 651.9 ticks/rad, that is about 0.9–1.4°. The encoder resolution is 0.088° per tick, and backlash is ≤0.5° (datasheet figure, never measured). BAM's single-servo MAE of 0.4–0.8° sits on top of that. A well-identified model that includes deadband, rate limit and delay could plausibly land at 1–2° on free-space arm motion. Contact phases and back-driven phases would be worse.

## What a cloth-grasping team should use and do next

Gripper ground truth is the biggest blind spot for cloth. No source reports measured SO-101 grip force in newtons, jaw closed gap, or pad friction against the real gripper. I searched "SO-101 gripper grip force newtons measured", "SO-101 gripper weak grip torque limit 500 lerobot issue", and "SO-ARM100 gripper jaws do not close fully gap / gripper tape", and checked the mehdihdev force-sensor repo. The sim's −10° lower limit suggests that jaw contact, not a joint stop, sets closure (inference). That links directly to the 2.3 mm jaw gap our Newton grip bench found. Grip force is roughly 0.8–0.95 N·m of servo torque at LeRobot's 50% cap, divided by the jaw lever arm (inference). The lever arm has not been measured.

The recommendation, in order:

| Step | Action | Why |
|---|---|---|
| 1 | **Base asset.** In Isaac/Newton, use the NVIDIA `so101_new_calib_SysID` USD. In MuJoCo, use Menagerie `robotstudio_so101`. For grasping in either, port Menagerie's gripper primitives (pads, tip spheres, condim 6) | Best identified joints plus best gripper contact. Nobody publishes this combination, so it needs its own check |
| 2 | **Weigh** the 6 servos and the printed links; measure jaw lever arm, closed gap (mm) and the jaw angle at LeRobot's recorded gripper min and max | CAD masses are 70–170 g below the listed weights; gripper geometry is unmeasured |
| 3 | **Measure grip force** at LeRobot's gripper settings (Max_Torque_Limit 500, Overload 25), at peak and after the protection time | The torque cap and overload drop are not modeled anywhere |
| 4 | **Record identification data on our own arm with its exact LeRobot settings** (P16/D32, acceleration 254, 5 V or 12 V supply): chirps and random waypoints at the highest bus rate (about 100–200 Hz), fixed base, no contact, plus a held-out set | Covers D=32, acceleration, supply voltage and calibration offsets, which no public data does. 30 fps LeRobot datasets are too coarse for delay |
| 5 | **Fit** with PACE (OmniReset's method) or the Isaac Sim SysID extension (Beta, ingests LeRobot format). Fit per-joint armature, Coulomb and viscous friction, delay, stiffness and a sign/offset per joint. Start from BAM M1 at P=16 (kp 9.38, damping 0.62, frictionloss 0.05, armature 0.026, rate limit about 5.3 rad/s) and NVIDIA's values as bounds | Note that the SysID extension identifies armature and delay only through its MuJoCo-Newton bridge (verified) ([docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.robot_setup.sysid/docs/index.html)) |
| 6 | **Implement in Newton 1.6.1** with DrivePD + ClampingDCMotor + Delay (all experimental) plus a target slew filter. If M1 misses 2°, add a custom load-dependent friction component ([Newton actuators](https://github.com/newton-physics/newton/tree/v1.6.1/newton/_src/actuators)) | M1 maps almost exactly (inference); M3–M6 need per-step friction updates. Whether every Newton solver honors `joint_friction` was not checked |
| 7 | **Gate RL on a held-out joint RMSE.** Report the single-servo floor (0.4–0.8°), the estimated deadband floor (about 1°), and the measured result. Then randomize friction, armature and damping by ±20% and delay by 0–2 steps, as OmniReset did | Matches OmniReset's protocol and DR ranges |

If the arm can take 12 V servos, the 12 V set is the most accurate measured actuator model found (0.41–0.43° in-sample, 0.60° held-out). The 7.4 V parameters must not be used for 12 V hardware, or the reverse.

## Conclusion

The SO-101 community has good geometry and two solid single-servo measurements, but nobody has joined them on a real arm under LeRobot's settings. Every "measured" label in the public assets traces back either to a bench pendulum at D=0 and 7.3 V, or to NVIDIA values whose source was not found. The official kp of 998 is probably wrong by about two orders of magnitude, and the default friction is about 4–18x too low. Policies trained on stock assets will therefore expect a faster, stiffer, more obedient arm than the one on the table. That is consistent with the reach-policy oscillation reported on hardware.

For cloth grasping, joint accuracy matters less than the unmeasured gripper: closed gap, lever arm, and LeRobot's torque caps decide whether thin fabric is held at all. A one-day measurement pass (weighing, gap, grip force, a chirp dataset) would give the team the first published SO-101 model checked against its own hardware, and would show whether under 2° is reachable or limited by the servo's roughly 1° deadband.

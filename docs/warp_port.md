# MuJoCo Warp port: plan and compatibility probe

Branch `warp-port`. Goal: run `ClothFoldEnv` as thousands of parallel worlds on
one GPU so PPO rollouts and world-model data collection stop being bound by
CPU cores. This document records what the Warp source and docs say the port
needs, what has to be rewritten, and the order of work with a check after each
step. Written 2026-09-17 against mujoco-warp 3.13.0 (wheel unpacked and read;
the README's compatibility paragraph is too coarse to plan from).

## 1. Why the estimate is what it is

One control step today is 100 substeps at 0.5 ms (`ARM_TIMESTEP`, "1 ms
explodes on contact"), 375 DoF, `implicitfast`. Both task wrappers run the env
in `joint_delta` mode, so the loop is 100 x `mj_step` with `ctrl` written once
per control step; the per-substep Python IK exists only in `ee_delta` mode,
which no task uses. Measured: 9 control steps/s single env (0.84 ms per
substep, all MuJoCo), 45/s across 8 envs, 1M PPO steps in 6 hours
(`cloth_fold_rl/README.md`). Warp's throughput is
`worlds x per-world step rate / substeps`. The only published anchor is the
humanoid benchmark (27 DoF, 64 constraint rows, 2.7M steps/s at 8,192 worlds);
this cloth has 375 DoF and 800 to 900 constraint rows when flat on the table,
so per-world work is 15x to 50x higher. On a shared RTX 3090 shard that gives
20k to 100k substeps/s, or 200 to 1,000 control steps/s against 45 today:
4x to 20x, likely near 10x. The number moves up if the Warp solver holds at a
larger timestep than the CPU one does, and collapses if any per-world Python
stays in the substep loop. Section 5 measures it before anything else is
built on it.

## 2. Compatibility probe (mujoco-warp 3.13.0 source)

Every feature the compiled model uses, checked against `_src/io.py`
(`put_model` raises `NotImplementedError` for unsupported features),
`_src/types.py` (enums and array shapes) and `_src/collision_flex.py`.

| Model feature | Where it comes from | Warp 3.13 | Note |
|---|---|---|---|
| `integrator="implicitfast"` | cloth XML | supported | `IntegratorType.IMPLICITFAST`; `forward.implicit` differentiates dof damping, springs, actuators. `derivative.py` has no flex code, so `<edge damping="0.2">` integrates explicitly. Per-vertex `dof_damping = 0.3` is implicit. |
| Newton solver (default) | cloth XML | supported | CG also supported; Warp's cloth benchmark uses CG. PGS and noslip are not. |
| `jacobian="auto"`, nv = 375 | default | sparse | Auto picks sparse above nv 32; dense is refused above nv 60. Sparse is the only option. |
| flexcomp grid `dim="2"` | cloth XML | supported | `has_2d_flex`; `flex_interp` 0 (direct vertices). Quadratic interpolation is the unsupported variant. |
| `<edge equality="true">` | cloth XML | supported | `EqType.FLEX`. Refused only with sleeping enabled; sleeping is off. |
| `selfcollide="none"`, `internal="false"` | cloth XML | supported | Flex internal collision is the unsupported case, and it is off. |
| flex vs box (table, finger pads, grabber paddles) | cloth XML, so101, `prove_grabber.py` | supported | `_flex_geom_vertex_narrowphase` handles sphere, capsule, box, cylinder, ellipsoid, mesh. |
| flex vs mesh (gripper collision meshes) | so101 `collision_gripper_mesh` | supported | `test_mesh_cloth_collision` exists. |
| flex vs plane (floor) | cloth XML | supported | `_flex_plane_narrowphase`. |
| WELD equality, inactive at compile | `compile_model` grasp welds | supported | `EqType.WELD`. |
| `data.eq_active` toggled per episode | `set_gripper` | supported | `Data.eq_active: (nworld, neq)`. |
| `model.eq_data[eqid, 3:6]` set at grasp time | `set_gripper` | supported, batched | `Model.eq_data: array("*", "neq", vec11)`; pass `put_model(mjm, batch_sizes={"eq_data": nworld})` so each world holds its own relpose. |
| `body_mass`, `geom_friction`, `dof_damping` randomized per episode | `reset` | supported, batched | All three are `array("*", ...)` fields; batch them the same way. |
| position actuators (`kp`, `kv`, `forcerange`) | so101 | supported | Standard gain/bias types. |
| plugins, sensors, tendons | none used | n/a | Body, actuator and sensor plugins are the refused ones. |
| `mj_jacSite` (IK) | `ik_substep`, experts | replaced | `mujoco_warp.jac(m, d, ...)` is public and batched over worlds. |
| `mj_objectVelocity` (obs) | `_get_obs` | replaced | `Data.cvel: (nworld, nbody)` and `cdof` are available. |
| `mju_mat2Quat`, `mju_subQuat`, `mju_quatIntegrate` | IK, obs | replaced | Torch ops on `site_xmat`. |
| `data.qacc` (instability check) | `_failed` | available | `Data.qacc: (nworld, nv)`. |
| partial reset | new | supported | `reset_data(m, d, reset=mask)`. |
| CPU execution | development on the Mac | supported | README: CPU execution is supported for development and debugging. warp-lang 1.17.0 ships a `macosx_11_0_arm64` wheel and installs on Python 3.14. |
| `mujoco.Renderer` | videos, pixels mode | unchanged | `get_data_into(mjd, m, d, world)` copies one world back to `MjData` for the CPU renderer. Warp's own batch renderer is a later option. |

Version constraint: `mujoco-warp 3.13.0` requires `mujoco>=3.12.0`. The repo
pins 3.10.0 (physical grabber) and 3.11.0 (weld checkpoints), and those two
already disagree by 1 cm on where the cloth settles. The port runs on a third
MuJoCo, and the CPU reference it is compared against must be the same 3.13,
compiled from the same `MjSpec`. `so101-nexus` does not pin mujoco; 0.5.1 is
the version the weld checkpoints used and is the first thing to try.

Nothing in the model is refused by `put_model`. The open questions are
numeric, and only the parity run in section 5 answers them:

1. Stability at 0.5 ms with edge damping integrated explicitly.
2. Contact behaviour of the finger pads and grabber paddles under Warp's
   solver (float32, padded contact arrays, its own narrowphase).
3. `nconmax` / `njmax` sizing. A flat 11x11 cloth touches the table at up to
   121 vertices; Warp drops contacts past the cap and reports it through
   `OverflowType`, so `opt.warn_overflow` stays on during development.

## 3. What has to be rewritten

`compile_model` stays as it is: the `MjSpec` compiles on the CPU and
`put_model` uploads the result. Everything that touches `self.data` per
episode becomes a batched tensor operation over `nworld`.

| Layer | File | Today | Port |
|---|---|---|---|
| model build | `mujuco/sim_main.py:compile_model` | MjSpec, welds, pad softening | unchanged, plus `put_model` with batched `eq_data`, `body_mass`, `geom_friction`, `dof_damping` |
| reset | `ClothFoldEnv.reset` | per-env DR, `mj_resetData`, cloth shift, 2,000-step settle | masked `reset_data`; DR written into the batched model rows; settle once per DR sample and cache the settled `qpos`, since 2,000 substeps per reset is 10 control steps of budget |
| substep loop | `ClothFoldEnv.step` (`joint_delta`) | clamp joint targets into `ctrl`, 100 x `mj_step` | same, with the 100 `mujoco_warp.step` calls captured as one CUDA graph; `ee_delta` mode (IK per substep) is not ported |
| grasp | `set_gripper`, `grasp_active` | Python per weld | gap test on `site_xpos` vs corner `xpos` per world, masked writes to `eq_data[world, eqid, 3:6]` and `eq_active` |
| observation | `_get_obs`, `StateOnlyWrapper` | numpy per env | one gather from `qpos`, `qvel`, `ctrl`, `site_xpos`, `site_xmat`, `cvel`, `xpos` into the 393-value state `cloth_angles/tasks.py` already defines |
| reward, termination | `_reward`, `_failed`, `SingleCornerFoldEnv` | scalar Python | potential shaping, anchor drift, workspace and `qacc` checks as tensor ops with per-world `success_steps` |
| stages | `QuarterFoldEnv` | Python stage machine | per-world stage index and weld mask; second priority |
| experts | `expert.py`, `quarter_fold_expert.py` | phase machine per env, `solve_ik` once per control step on a scratch `MjData` | per-world phase tensor and a batched `solve_ik` on `mujoco_warp.jac`; needed for collection at scale, single-corner first |
| physical grasp | `physical_env.py`, `physical_expert.py` | contact predicate, scoop IK with nullspace | out of scope for the first pass |
| vec env | `cloth_fold_rl/train.py` (SB3 `SubprocVecEnv`) | 8 processes | a `VecEnv` adapter over the batched env, device-to-host copy per control step (a few MB, negligible against 100 substeps) |
| collection | `scripts/collect_fold_state_episodes.py` | spawn pool | batched env writes the same `episode_*.npz` and `manifest.json`, so the world-model stack does not change |

Line count in scope: `sim_main.py` (882), `fold_env.py` (235),
`quarter_fold_env.py` (255), `expert.py` (197), `quarter_fold_expert.py`
(140), and adapters. The new code lives beside the old under
`mujuco/warp_sim.py` and `cloth_fold_rl/warp_*.py`; the CPU path stays
runnable for parity checks and rendering.

## 4. Order of work

Each phase ends with a check that decides whether the next one starts.

### Phase 0: environment

- `requirements-warp.txt`: mujoco 3.13.0, mujoco-warp 3.13.0, warp-lang 1.17.0,
  so101-nexus 0.5.1, torch, gymnasium, numpy. A separate venv; the existing
  pins stay for the CPU results.
- `scripts/warp_probe.py`: `compile_model(ARM_TIMESTEP)` on 3.13, then
  `put_model` and `put_data(nworld=2)` on the CPU device, one `step`, print
  `nefc`, `nacon`, any overflow flags, and `flexvert_xpos` deltas.

Check: the probe runs on the Mac with no exception. This is the only step that
can fail for a reason the source reading did not predict.

### Phase 1: batched core

`mujuco/warp_sim.py`: model upload with batched fields, masked reset with the
settled-state cache, the 100-substep loop with batched IK and weld logic,
observation gather. Single world first, on the CPU device.

Check: `scripts/warp_parity.py` replays the recorded actions of an expert
episode from `outputs/cloth_angles/fold_state_v2` through both `ClothFoldEnv`
(mujoco 3.13 CPU) and the Warp env, and reports vertex position error per
control step, grasp step agreement, and the reward trace. Tolerance: the
Warp-vs-CPU error at 10 control steps must sit well under the world model's
own 1.47 mm at 10 steps, or the port is training on a different cloth than the
one it is measured against. Run at 0.5 ms, then 1 ms and 2 ms to see where
Warp's solver stops holding.

### Phase 2: throughput on WATcloud

Needs the minimal job wrapper: `git archive HEAD`, one sbatch with
`--gres=shard:rtx_3090:8192`, the CUDA torch wheel, a command string, pull the
output directory. No data staging.

Check: substeps/s at `nworld` 256, 1,024, 4,096 with the graph-captured loop,
against the CPU figures (4,500/s at 8 envs, ~7,000/s at 12). Go/no-go: at
least 35,000 substeps/s (5x the Mac). Below that the port's payoff is under
the section 1 floor and the work stops here with a measured answer.

### Phase 3: single-corner task and expert

`SingleCornerFoldEnv` reward, termination and the 6-dim action expansion as
tensor ops; `FoldExpert` as a per-world phase machine.

Check: batched expert success on 100 worlds against the CPU expert's 100% on
the same seeds; the PPO `run2` checkpoint evaluated through the Warp env
against its 59/60.

### Phase 4: PPO on the batched env

SB3 through the `VecEnv` adapter first, `n_envs` raised in steps (64, 512,
2,048) with `n_steps` lowered to keep the rollout size near `run2`'s
`512 x 8`; the hyperparameters do not carry across a 100x change in batch.

Check: reach `run2`'s success rate on the single-corner task, and record wall
time per 1M steps. Then the first quarter-fold PPO baseline, which is the
result the project has not been able to afford.

### Phase 5: collection for the world model

`collect_fold_state_episodes.py` over the batched env and expert, same output
format.

Check: `scripts/benchmark_state_predictor.py` runs unchanged on the new
directory; compare the 300-episode result against `quarter_benchmark`
(0.181 / 1.470 mm), then scale the episode count and see whether the test
error moves, which is the data-vs-capacity question from `world_model.md`.

Out of the first pass: the physical grabber (contact-only grasp is the part
of the sim most sensitive to solver differences; port it after the weld tasks
match), `QuarterFoldEnv` stages (phase 3b once single-corner matches), pixel
observations and the Warp renderer.

## 5. Risks, ordered by how early they can end the project

1. **Explicit edge damping under `implicitfast`.** If the 0.5 ms cloth is
   unstable on Warp, the fixes are lowering `<edge damping>` and leaning on the
   implicit per-vertex `dof_damping`, or switching to CG. Either changes the
   cloth, and the phase 1 parity numbers say by how much.
2. **Throughput below the floor.** Per-world cost at 375 DoF is the unknown
   the humanoid anchor cannot pin. Phase 2 measures it before any task code
   is written.
3. **Contact parameters.** Finger pads were softened to stop the CPU cloth
   exploding on touch; Warp's solver will want its own values. Affects grasp
   radius behaviour and the physical grabber most.
4. **Silent contact overflow.** `nconmax` and `njmax` too small drop contacts
   with only a flag set. Keep `warn_overflow` on and assert on it in the
   parity script.
5. **Kernel launch overhead.** 100 `step` calls per control step is hundreds
   of launches; without CUDA graph capture the GPU idles. `WarpClothSim`
   captures the substep loop once per loop length and replays it.
6. **Third MuJoCo version.** Everything measured so far is on 3.10 or 3.11.
   The parity reference is 3.13 CPU, and the CPU expert's own success rate on
   3.13 is re-measured in phase 3 before it is used as a yardstick.
7. **Float32.** The CPU sim is float64. The parity tolerance in phase 1 is the
   place this shows up; the world model trains in float32 anyway.

## 6. Phase 0 results (2026-09-17, Mac CPU device)

`requirements-warp.txt` installs on Python 3.14 (mujoco 3.13.0, mujoco-warp
3.13.0, warp-lang 1.17.0, so101-nexus 0.5.1, torch 2.13.0). `compile_model`
compiles unchanged on 3.13; `put_model` accepts the model with `eq_data`,
`body_mass`, `geom_friction` and `dof_damping` batched. `scripts/warp_probe.py`
steps it. Findings:

- **Contact capacity.** Warp's defaults (512 contacts, 512 rows in total)
  overflow as soon as the cloth lies on the table. Measured per world once
  settled: 398 contacts, 1,924 constraint rows (320 edges + 398 x 4 + 12).
  `WarpClothSim` allocates 1,024 contacts and 4,096 rows per world.
- **Resting height differs.** With identical `solref`/`solimp`, CPU MuJoCo
  (3.10 and 3.13 alike) generates 50 element contacts for the flat cloth and
  lets it sink 4.4 mm into the table; Warp generates 398 element contacts and
  holds it at 0.4 mm penetration. Eight times the contacts means a stiffer
  aggregate contact. After the 2,000-step settle the mean vertex error is
  2.0 mm and the maximum 4.4 mm, nearly all of it a constant z offset. This
  is risk 3 showing up before any arm moves. It is inside the task's 3 cm
  grasp radius and 5 cm placement tolerance and outside the world model's
  1.47 mm figure, so Warp data is Warp physics: a model trained on it is
  measured on it, and the CPU datasets are not mixed in.
- **Version drift for scale.** Replaying episode 0 of `fold_state_v2`
  (recorded on 3.11) through the 3.13 CPU env gives 0.6 mm mean / 2.7 mm max
  at reset from the version change alone.
- **No instability.** `implicitfast` with the explicit edge damping holds at
  0.5 ms through the settle and the first replayed control steps; `qacc` is
  lower on Warp (3) than on CPU (28).
- **Warp on the CPU device is for correctness only.** 61 substeps/s for one
  world against 1,190/s for CPU MuJoCo; a control step is 1.75 s. Parity runs
  fit (an episode is a few minutes), throughput needs the GPU.
- **Parity harness.** `scripts/warp_parity.py` replays a recorded episode's
  actions through `SingleCornerFoldEnv` (3.13 CPU) and `WarpClothSim` from the
  same seed (the CPU reset reproduces the recording's physics scales exactly)
  and reports vertex error, the moving corner's error, the grasp step and the
  z offset per step, three ways: Warp vs CPU, CPU vs recording.

## 7. Phase 1 results (2026-09-17, Mac CPU device)

`mujuco/warp_sim.py` runs the joint_delta env batched. `scripts/warp_parity.py`
on episode 0 of `fold_state_v2` (expert, seed 30000, recorded on 3.11):

**Open-loop replay of the recorded actions.** Warp vs CPU 3.13 stayed at
2.1 mm mean / 4.8 mm max for all 79 steps with the arm moving; the moving
corner agreed within 0.5 mm; the gap is the 2 mm z offset from section 6.
The 3.11 recording grasped at step 27; the 3.13 CPU env replaying the same
actions never grasped, and Warp agreed with the 3.13 env. Open-loop replay
across MuJoCo versions is not a usable test once a grasp is involved.

**Closed loop.** `FoldExpert` drove the 3.13 CPU env and Warp replayed its
actions step by step:

| step | Warp vs CPU 3.13 vertices (mean / max) | moving corner | grasp |
|---|---:|---:|---|
| 10 | 2.1 / 4.5 mm | 0.4 mm | neither |
| 25 | 3.4 / 34.9 mm | 8.0 mm | both, at step 20 |
| 100 | 4.3 / 31.6 mm | 9.1 mm | both |
| 200 | 7.6 / 26.3 mm | 8.9 mm | both |

Grasp on the same control step in both sims. Through the carry the moving
corner stays within 9 mm; the maximum vertex error of 26 to 35 mm is in the
hanging part of the cloth, which drapes differently. At step 200 the corner
sits 118.4 mm from the goal on Warp and 116.7 mm on CPU 3.13: the two sims
agree on the outcome to 2 mm.

That outcome is a failure. On 3.11 this seed succeeded in 79 steps; on 3.13
the same expert, closed loop, grasped earlier (step 20 vs 27) and stalled
117 mm short of the goal. The MuJoCo upgrade changed the episode's result
where the Warp port did not, so the Warp-vs-CPU gap is smaller than the
version drift the project already carries. Consequence for phase 3: the CPU
expert's success rate on 3.13 has to be re-measured before it is used as a
yardstick (risk 6), and Warp datasets stay separate from the 3.11 ones.

Phase 1 passes on the terms set in section 4, read as relative geometry: same
grasp step, corner within 9 mm through the carry, same end state. Next is
phase 2, throughput on the cluster (`scripts/warp_throughput.py`).

## 8. Phase 2 log

**Expert on MuJoCo 3.12 and 3.13 (CPU, 10 episodes each):** 0/10 success on
both, grasp 80%, the carry stalls 11 to 13 cm from the goal, best fold score
0.650; the two versions give identical numbers line for line. On 3.11 the same
expert is at 100%. The regression enters with 3.12, the oldest MuJoCo that
mujoco-warp accepts, so pinning 3.12 changes nothing and the expert is
re-tuned on the new physics in phase 3 whichever backend runs it.

**Job kit.** `cloud/watcloud/` is a trimmed copy of the mcrl kit: `git archive
HEAD` plus optional `--in` paths, a uv venv (Python 3.12) on tmpdisk, `--gpu`
adds the 3090 shard, `--out` paths come back in place. First two jobs failed
on the kit itself: `job.env` was written unquoted, so sourcing it executed the
script path (fixed with `%q`), and the uv installer appended a line to the
cluster `~/.profile` (removed; the installer now runs with
`INSTALLER_NO_MODIFY_PATH=1`). Setup on the node takes about five minutes
(torch cu130 is the bulk); Warp's first-collision kernel compile adds about
three more (the flex element narrowphase alone is 55 s, each CCD variant 20 s).

**mujoco-warp 3.13.0 + warp-lang 1.17.0 on CUDA fails before the first step**
in `collision_convex._ccd_grid_size`: the CCD kernel's occupancy query
compiles a second module variant (Warp's default 256-thread block against the
launch's 64) and `wp.get_suggested_block_size` then reports "CUDA error 500:
named symbol not found". Upstream fixed it in mujoco_warp PR #1675 (merged
2026-09-17, after the 3.13.0 release of 2026-09-09; warp 1.17.0 is from
2026-09-02). `warp_sim.py` sidesteps it by sizing the CCD grid at `naconmax`,
which is what the function's own CPU branch returns; the kernel strides over
the real candidate count, so this costs idle threads and nothing else. Drop
the patch when a mujoco-warp release includes #1675.

## 9. Phase 2 results (2026-09-18, WATcloud RTX 3090 shard)

Six jobs to get here; the log of what broke is in section 8 and below. The
numbers, `scripts/warp_throughput.py`, 10 timed control steps after a full
settle, 1,024 contacts / 4,096 rows / 64 CCD slots per world:

| solver | timestep | worlds | graphs | world-substeps/s | control-steps/s | Warp memory |
|---|---|---:|---|---:|---:|---:|
| Newton | 0.5 ms | 256 | yes | 9,800 | 98 | |
| Newton | 0.5 ms | 1,024 | yes | 15,300 | 153 | |
| CG | 0.5 ms | 1,024 | yes | 22,700 | 227 | 0.8 GiB |
| CG | 0.5 ms | 1,024 / 2,048 / 4,096 | no | 11,800 / 13,900 / 15,200 | 118 / 139 / 152 | 1.7 / 3.4 / 6.8 GiB |
| CG | 1 ms | 1,024 / 2,048 / 4,096 | no | 11,000 / 13,000 / 14,300 | 221 / 260 / 285 | |
| **CG** | **1 ms** | **2,048** | **yes** | **23,000** | **460** | 1.6 GiB |
| CG | 1 ms | 4,096 | yes | 23,700 | 474 | 3.3 GiB |
| Newton | 1 ms | 2,048 | yes | 14,700 | 295 | 3.3 GiB |
| Newton | 1 ms | 4,096 | yes | 15,100 | 301 | |

Reading it:

- **Go.** The floor in section 4 was 35,000 world-substeps/s at 0.5 ms, which
  is 350 control-steps/s. CG at 1 ms with graphs gives 460 at 2,048 worlds:
  10x the Mac's 8-env PPO (45/s), 7x its 12-worker collection, about 4x a
  48-core CPU job. 1M PPO steps go from 6 h to about 36 min. Newton at 1 ms
  gives 295, still 6.5x.
- **Per-substep cost is flat in timestep** (11.8k vs 11.0k eager at 0.5 and
  1 ms), so the timestep is a pure multiplier on control steps. 1 ms passed
  closed-loop parity on the CPU device (section 7 harness: grasp at the same
  step, corner within 5 to 7 mm through the carry, end state 117.2 vs
  116.7 mm), where CPU MuJoCo needs 0.5 ms. 2 ms is untested.
- **World count saturates past 2,048** (+3% from 2,048 to 4,096 with graphs).
  The 375-DoF cloth fills the card; 2,048 worlds at 1.6 GiB is the working
  size.
- **CUDA graphs are worth 2x**, eager stepping runs the solver's convergence
  loop on the host. Graphs are captured per control step (a whole-settle graph
  does not instantiate) and fall back to eager on failure.
- **CG vs Newton.** CG is 1.56x faster and reports `ITERATIONS` overflow (its
  100-iteration cap reached in some worlds on some substeps); Newton converges
  clean. CG matched Newton's closed-loop trajectory on the CPU device (same
  grasp step, corner within 9 mm, end state 118.4 vs 116.7 mm). Phase 3 runs
  Newton for the expert-parity check and CG for volume once the batched
  expert reproduces its CPU success rate under both.

What cost the jobs: the CCD work arrays. mujoco_warp sizes eleven `multiccd_*`
arrays by `nccdmax x` mesh degree and defaults `nccdmax` to `nconmax`, so
1,024 worlds x 2,048 contacts allocated ~21 GB lazily on the first step and
every later allocation (a 1.26 GB buffer, or a graph executable) failed while
`cudaMemGetInfo` reported 22 GiB free a moment earlier. Only the arm geoms use
CCD here; `WarpClothSim` caps it at 64 per world. `scripts/warp_gpu_diag.py`
showed the "shard" is a scheduler share, not a memory cap: one process
allocated 18 GiB. A red herring on the way: eager stepping was made the
default when graphs OOMed, and measured at half the throughput.

Next: phase 3, `SingleCornerFoldEnv` and `FoldExpert` batched, checked against
the CPU expert re-measured on MuJoCo 3.13 (0/10 as tuned for 3.11; section 8).

## 10. Source references

- `mujoco_warp/_src/io.py`: `put_model` checks (lines 255 to 310, 401 to 408),
  `put_data` signature (1864), `reset_data` (2410), `get_data_into` (2159).
- `mujoco_warp/_src/types.py`: `IntegratorType` (471), `SolverType` (532),
  `EqType` (713), batched `Model` fields (`body_mass` 1621, `dof_damping` 1656,
  `geom_friction` 1681, `eq_data` 1826), `Data` fields (`eq_active` 2326,
  `qacc` 2329, `flexvert_xpos` 2352, `cvel` 2377, `site_xpos` 2343).
- `mujoco_warp/_src/collision_flex.py`: geom types accepted by the flex
  narrowphase (lines 931 to 936).
- `mujoco_warp/_src/forward.py`: `implicit` (986), `fwd_kinematics` calls
  `smooth.flex` (1047).
- `mujoco_warp/_src/derivative.py`: no flex terms.
- `mujoco_warp/_src/support.py`: `jac` (583).
- `mujoco_warp/_src/flex_test.py`: plane, sphere, cylinder, mesh vs cloth
  tests (887 to 1030).
- Benchmarks: `benchmarks/cloth/README.md` (918 bodies, 2,706 DoF, CG, Euler,
  5 ms, 2,048 worlds), `benchmarks/README.md` (humanoid 2.7M steps/s).

# V — vectorised Isaac env: design (2026-10-04)

Milestone V of Phase W (roadmap): B half-fold envs in ONE Isaac process on the GPU weld profile
(`IsaacClothFoldEnv(profile="weld")`, backend `isaac_weld`), cloth kept 101×101. Opt-in: with B = 1 (the
default, `WORLDFOLD_ISAAC_ENVS_PER_PROC` unset) every code path is today's.

V0 (`isaac/probes/vec_probe.py`, results.md "V0 vectorisation probe") showed B independent *copies* of the scene
work in one process: each copy is a LeHome `GarmentObject` (own particle system and material), an arm pair and a
table. The plan text's alternative (one regex `ClothPrim` over `/World/envs/env_i`) is not used: the copies are what
V0 validated, they keep copy 0's prim paths unchanged, and per-copy materials give per-env DR for free.

## 1. Scene: `isaac/lab_scene.py`

- `SceneCfg.n_copies` (default 1). `make_cfg(..., n_copies=1)`.
- **Copy 0 is today's scene, prim for prim** (`/World/Robot/Left_Robot`, `/World/Object/cloth`, `/World/table`,
  `/World/rig_main`, ...). Copies i ≥ 1 live under `/World/Copy{i}/` (`Robot/Left_Robot`, `Robot/Right_Robot`,
  `table`, `cloth`, `rig_main`, wrist cams under their own grippers), translated by `COPY_OFFSETS[i]`.
- **Layout**: a 3×3 grid at `COPY_SPACING = 2.5 m` around copy 0 (offsets (0,0), (±2.5,0), (0,±2.5), (±2.5,±2.5)),
  so every copy sits on the one 8×8 m floor (its grid texture is scaled with its size, so enlarging or duplicating
  the floor would change copy 0's pixels) and B ≤ 9. 2.5 m vs V0's 1.5 m: the main camera sees
  x ∈ [-0.80, 0.33], y ∈ [-0.6, 0.6] m of the floor around its copy, so no copy sees a neighbour, and a cloth that
  blows up has room before it reaches another copy (and is parked at once, below). Physics coupling between copies
  is zero by construction: separate particle systems (cloth–cloth across systems doesn't collide) and no geometry
  within reach of another copy.
- Everything per-copy is a `_Copy` record in `SceneEnv.copies[i]`: `arms`, `joint_ids`, `gripper_body`, `cloth`,
  `cloth_pose` (world), `cloth_rest` (world), `pin_floor`, `pins`, `mass_scale`, `rest_masses`, `rest_material`,
  `rig_cameras`, `offset`. The old attributes (`lab.arms`, `lab.cloth`, `lab.pins`, `lab._rest_masses`, ...)
  stay as properties for copy 0, so probes and checks (weld_check, profile_check, overshoot_measure) are unchanged.
- `targets` is `(1, 12·B)`; `_apply_action` sets each copy's arms from its 12-column slice and drives each copy's
  pins. With B = 1 the slices are today's `[:, :6]` / `[:, 6:]`.
- Per-copy methods take `c` (default 0): `particle_positions(c)` (copy-local frame: world minus the copy offset,
  skipped for copy 0 so B = 1 is bit-identical), `reset_cloth(..., c)`, `set_dynamics(..., c)`,
  `set_pinned_masses(c)`, `_drive_pins(c)`, `site_pose(prefix, c)`.
- `reset_copy(c)`: that copy's arms back to their default root pose and HOME joints (zero velocity), their
  actuator buffers reset, its pins cleared, its rig cameras reset, then `sim.forward()` (kinematics only, no
  physics step). Other copies are not touched. `lab.reset()` (DirectRLEnv's global reset) still resets every copy,
  and is what the B = 1 env keeps calling.
- The lehome profile (CPU device) is not vectorised: `n_copies > 1` needs the GPU pipeline (raises otherwise).

## 2. Env: `isaac/isaac_env.py`

`IsaacClothFoldEnv` keeps its constructor and behaviour. Internally it is split into `_configure()` (spaces,
constants, per-env state) and `_attach(lab, c)` (bind to copy c of a scene), and all its scene access goes through
copy c. Positions it reports (cloth, gripper, link) are in the copy's local frame, i.e. identical to a B = 1 env;
weld offsets are differences, so frame-free.

One control step is `_advance() = _submit(targets); _physics(); _refresh_cloth()`. For the standalone env
`_physics()` is today's `lab.step(...)` + Kit tick (same calls, same order: B = 1 is today's code path).

**`IsaacClothFoldBatch(n, **IsaacClothFoldEnv kwargs)`** builds one `SceneEnv(n_copies=n)` and `n`
**`IsaacSubEnv`** views (`batch.envs[i]`, a subclass of `IsaacClothFoldEnv` attached to copy i). A sub-env has the
full public API and attributes of `IsaacClothFoldEnv` (it *is* one: `observation_space`, `action_space`, `reset`,
`step`, `cloth_positions`, `gripper_position`, `gripper_pose`, `joint_positions`, `grasp_active`,
`corner_positions`, `weld_mask`, `_domain_params`, `_particles`, `profile`, `grasp_mode`, `prefixes`, `lab`,
`render_rig`, `keep_alive`, ...), so `HalfFoldEnv(base_env=sub_env)`, `ScriptedTeacher` and `IsaacFoldExpert`
run on it unchanged. Differences, all inside the class:

- `_physics()` → `batch.sync(i)`: the sub-env's share of one global control step (below).
- reset uses `lab.reset_copy(i)` instead of the global `lab.reset()`; `keep_alive()` / `close()` go to the batch.
- `step_submit(action)` / `step_collect()` split `step()` around the physics for callers that drive the batch by
  hand (`step = submit; sync; collect`).

**Physics advances only in `batch.advance()`**: one `lab.step` with every copy's targets (copies without a pending
step hold their last targets), then the Kit tick. Who calls it:

- no scheduler (default): `sync(i)` calls `advance()` at once (every copy steps; fine for B = 1 and hand-driven
  tests);
- with a `Lockstep` scheduler attached (`batch.scheduler`, the rollout worker), `sync(i)` parks the calling slot
  thread and the scheduler calls `advance()` when every *active* slot is parked (section 3).

Per-env reset = `reset_copy(i)` + cloth soft reset + `settle_steps` calls of `_advance()`; in a batch those settle
steps are the env's share of global steps that the other envs spend on their normal steps. The DR draw order,
`set_dynamics` after the first settle step, and the np_random use are today's (`np_random` is per sub-env and
seeded by its own `reset(seed)`).

USD-writing scene calls (`reset_copy`, `reset_cloth`, `set_dynamics`) and every Kit/physics call run on the
process's main thread: `batch.on_main(fn)` hands them to the scheduler (direct call without one). Slot threads do
only tensor reads/writes and Python.

**Parking**: when a sub-env's episode ends with its cloth failed (`_failed()`: non-finite, off the workspace or
> 20 m/s) the worker parks it (`reset_copy` + flat cloth, no settle) so an exploded cloth idling while the others
step can't drift anywhere. A parked copy is reset again at its next episode anyway.

## 3. Rollout: `imitation/rollout.py` + `imitation/lockstep.py`

`WORLDFOLD_ISAAC_ENVS_PER_PROC = B` (default 1; read by `EnvPool` for Isaac backends; B > 1 only for
`isaac_weld`). `EnvPool(n, kw)` starts n processes with B pipes each: `len(pool) = n·B` logical slots,
`pool.pipes[k]` is slot k, and the driver protocol per slot is unchanged (`ready`, `reset`, `step`, `label`,
`resync`, `close`). One pipe per slot (no multiplexer): `_rollout` and `EnvPool.call` work as they are.

The command handling of today's `_worker` becomes `_SlotServer.handle(cmd, arg)`; the B = 1 worker is the old loop
around it (same idle Kit tick, same exits). A B > 1 worker builds `make_env_batch(backend, B, ...)` (B
`HalfFoldEnv`s over one batch), starts one **slot thread** per pipe running the same handler, and runs the
`Lockstep` scheduler on the main thread.

**Lockstep** (pure Python, no Isaac): one condition variable whose lock is the *baton*: a slot thread holds it
while it runs env code, so exactly one thread touches the scene at a time (the GIL would serialise them anyway),
and the order slot threads run in between two global steps doesn't matter (each touches only its own copy; physics
happens only in `advance`). State: `active` (slots with an episode in flight) and `parked` (slots waiting inside
`sync`).

- slot thread: `recv` (baton free) → take baton → `reset` marks the slot active → run the handler; each
  `_advance` inside it calls `sync(i)`: add to `parked`, notify, wait (baton released) until the generation
  changes → after the reply: a terminal `step` reply (term or trunc), `close` or an error marks it inactive → send
  the reply → release the baton.
- main thread: waits until `parked ≠ ∅ and active ⊆ parked`, then `advance()` (baton held), clears `parked`,
  bumps the generation, notifies. Main-thread calls (`on_main`) queued by a slot are run as they come. Idle for
  `KIT_TICK_S` → `keep_alive()`. An exception in `advance` is re-raised in every parked slot (each replies
  `error`).

Micro-steps per command: `step` = 1, `reset` = settle_steps (the obs comes from the last), `label(K)` = whatever the
teacher steps (on Isaac labels are takeovers made of `step`s; `label` is only used on MuJoCo), `resync` = 0,
`close` = 0. An inactive slot never blocks; an inactive slot that is sent a `step` anyway is stepped with the next
global step (`parked ≠ ∅` suffices).

### Why a strict barrier can't deadlock

Claim: under `_rollout`, every global step eventually happens, i.e. whenever some slot is parked, every active slot
of that worker eventually parks too (or becomes inactive).

Take an active slot s of the worker that is not parked. Either
1. s holds or waits for the baton: it runs Python until it parks, replies or raises — finite, because the only
   blocking call inside env code is `sync` (parks) and `on_main` (served by the main thread, which is waiting on
   the condition, not on s); or
2. s is idle in `recv`, with an episode in flight. In `_rollout` an in-flight slot with no command outstanding is
   always in `awaiting` (every reply handler calls `advance(i)`, which either sends a command at once or puts i in
   `awaiting`; `start` sends `reset` at once). `plan_batch` runs when `awaiting` is non-empty and either nothing is
   in flight or `max_wait` (10 ms) has passed since the first one parked — it does **not** wait for replies from
   any other slot. So within max_wait plus one `plan` call s is sent a command and goes to case 1.

No slot's next command waits on a reply from another slot of the same worker, so no cycle exists. The same holds
for any driver with that property.

`EnvPool.call(idx, cmd)` does **not** have it: it sends one command per slot and then receives in slot order. Slots
drift out of phase (a slot that resets while its neighbour has no episode gets a head start), so the slot that
finishes first sits idle mid-episode and holds the others' remaining steps while `call` waits on one of them — a
deadlock (the first version of the fake-batch test hit exactly this, 1 run in 6). `call` therefore refuses
`reset`/`step`/`label` when B > 1 (no pipeline code uses it; only the B = 1 Isaac backend test does).

Fast tests (`tests/imitation/test_lockstep.py`) drive a pure-numpy fake batch through the real worker code and
`_rollout`: mixed `step`/`resync`/`reset`/takeover sequences, episodes of different lengths, B = 1 vs B = 3 with
identical per-seed episodes, and random command orders under a timeout.

### Per-episode determinism

An episode's inputs are its seed only: the DR draws and the drop tilt come from the sub-env's `np_random`, seeded by
`reset(seed)`; the cloth jitter from HalfFoldEnv's `_rng` (reset seed); the teacher's rng is re-seeded per episode
(`QuarterFoldExpert.reset`); the driver's rngs are keyed by seed; perturbations likewise. Physics: an active copy is
stepped exactly once per control step it asks for (the barrier waits for it), and an idle copy is only stepped
between episodes, which reset overwrites (arms by `reset_copy`, cloth by the soft reset whose USD pose write makes
PhysX re-parse the cloth on the next step — the same path B = 1 relies on between its episodes). What remains is GPU
float nondeterminism, which differs between B = 1 and B = 4 the same way it differs between two B = 1 runs; hence
parity is statistical (Wilson), not bitwise.

## 4. Verification

- fast: `test_lockstep.py` (above), plus the existing suite.
- Isaac: `test_isaac_weld.py` / `test_isaac_profile.py` (B = 1 regression); `isaac/vec_check.py` via
  `tests/imitation/test_isaac_vec.py`: B = 3 copies, per-copy cloth independence, per-copy DR read back, a reset of
  one copy leaves the others' particles, masses and arms alone through the next steps, sub-env readings in the local
  frame equal copy 0's for the same seed at reset.
- parity: `imitation.evaluate --backend isaac_weld --ckpt expert --sets id_easy --n 40 --workers 1` with
  `WORLDFOLD_ISAAC_ENVS_PER_PROC=4` vs `1`; Wilson CIs overlap.
- throughput: `isaac/bench_vec.py B`: expert episodes/hour in one process for B ∈ {1, 2, 4, 8}, plus GPU memory.

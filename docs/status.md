# Status

**Updated:** 2026-09-23 · **Branch:** `feature/imitation` · **Phase:** 1 (fix the silent
breakers, then freeze v1)

One-screen answer to "where are we". Update at the end of **every work pass** (see
`CLAUDE.md`), and add a line to the pass log at the bottom. Full plan in
[roadmap.md](roadmap.md); spec in [imitation.md](imitation.md); numbers in
[results.md](results.md).

---

## Where we are

The `imitation/` package is written and unit-tested but **has never run end to end**. The
immediate goal is a first credible half-fold imitation result, which is blocked on freezing
a dataset. The observation is fixed (M1.1); the remaining write-once-schema and collection
defects (M1.2-M1.4) must land before that dataset is collected.

## Next action

**M1.7 — gate: expert ceiling** on `id_easy` / `id_hard` / `recovery`, n≥100, plus the
`check_resync` artifact. M1.6 is done: collect → train → evaluate → 1 DAgger round runs
green on 20 episodes (`outputs/imitation/smoke/`, smoke numbers are not results).

## Environment

| | |
|---|---|
| pins | `imitation/requirements.txt` — mujoco **3.10.0**, so101-nexus **0.4.8**, numpy 2.5.1, gymnasium 1.3.0 |
| torch | 2.13.0+cu130, **CUDA available** |
| tests | 61 fast + 7 slow, all passing (`pytest -m "not slow"` / `-m slow`) |

⚠️ `cloth_fold_rl/requirements.txt` pins mujoco 3.11.0 / so101-nexus 0.5.1 for its own
committed checkpoint. Do not "unify" these without re-running the expert benchmark — cloth
grasping is contact-dominated and MuJoCo minors change results.

## Frozen datasets

| version | episodes | source | hash | notes |
|---|---|---|---|---|
| — | — | — | — | **none frozen yet** |

`v1` exists as an empty directory only: a 400-episode collection died at episode 215 and
wrote nothing, because `collect.py` returned every episode before writing any (fixed in
M1.2). `outputs/imitation/collect_v1.log` is retained as the record; the empty `v1/`
directory was deleted in M1.2.

## Best checkpoint

None. No policy has been trained on real data.

## Open blockers and known defects

Ordered by what they block. Each is a roadmap milestone.

| # | defect | blocks | milestone |
|---|---|---|---|
| 7 | Expert ceiling on eval sets unknown; teacher recovery rate unknown | **gates the recovery/DAgger arm** | M1.7 |
| 8 | DAgger val split drifts every round; chunk targets contaminated; keep/stop rule is noise at n=48; not resumable | DAgger validity | M3.1 |
| 9 | `failure_code` returns R1 for all perturbed failures → recovery histogram uninformative | failure analysis | M2.2 |
| 10 | `cloth_angles/tasks.py::QUARTER` diverged from `quarter_fold_env.STAGES` (3 ways) | world-model work only (parked) | M7.3 |

## Parked

- **World model** (`cloth_angles/`) — off the critical path by decision. Its task
  definition has diverged and must be reconciled or deleted before any result from it
  counts.
- **VLA** — enters at Phase 6 as a subgoal proposer, not an action labeler. MolmoAct emits
  8×6 absolute joint degrees, single arm; the env takes 12-D normalized dual-arm deltas.
- **PPO** — `cloth_fold_rl/train.py` and `bc.py` are superseded and deleted in M7.2. The
  unrelated `MuJoCoTouch-v1` PPO baseline stays.

## Pass log

Newest first. One line per work pass: date · what changed · commit.

- 2026-09-23 · M1.6 done: 20-ep smoke chain green (collect 19+1 fail split, train 600 steps, evaluate n=14×3, dagger 1 round froze smoke_dagger_r1 with 332 labels); no code change needed · COMMIT
- 2026-09-23 · M1.5 done: cached IK scratch (no speedup: 81-83 s → 82-93 s, identical hash); fixed cross-episode IK-rng leak that made collection nondeterministic; benchmark rows identical 48/50; 61 fast + 7 slow green · c95c431
- 2026-09-23 · M1.4 done: failures → `<version>_failures` in collect, `bc_episodes` filter in train (`--allow-failures`); 28-ep all-perturbed collect split 25/3; 61 fast green · c64e9ce
- 2026-09-23 · M1.3 done: `terminated`/`truncated`/`discount` arrays in schema + validator (only-last-step, exactly-one, discount mask); `unstable` → truncated; 14-ep real collect validates; 61 fast + 7 slow green · 1fb0e0e
- 2026-09-23 · M1.2 done: streaming/resumable `DatasetWriter` + `collect --resume`, seed-named npzs, guard on unfrozen versions; real 20-ep collect killed at 60 s kept 18, resume finished 20/20 unique, hashes verify; 59 fast green · fa31797
- 2026-09-23 · M1.1 done: 139-D obs (per-stage goals, no one-hot, +stage/settle), `cloth_offset_xy` recorded, goals in snapshot; 63 tests green; expert benchmark unchanged 48/50 · e847847
- 2026-09-23 · Sim verified against docs before M1.1: pins, 57+2 tests, benchmark 48/50 reproduced, all §2.2 defects confirmed · no code change
- 2026-09-23 · Phase 0: imitation + DAgger pipeline committed, env pinned, tracking docs · a5aa63f

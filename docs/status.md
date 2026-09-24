# Status

**Updated:** 2026-09-23 · **Branch:** `feature/imitation` · **Phase:** 2 (imitation baseline)

One-screen answer to "where are we". Update at the end of **every work pass** (see
`CLAUDE.md`), and add a line to the pass log at the bottom. Full plan in
[roadmap.md](roadmap.md); spec in [imitation.md](imitation.md); numbers in
[results.md](results.md).

---

## Where we are

Phase 1 is complete. The pipeline runs end to end, collection is streaming, resumable and
deterministic, the schema carries transition flags, and **v1 is frozen** (384 successful
expert episodes; 16 failures in `v1_failures`). The expert ceiling is measured
(id_easy 100%, id_hard 97%, recovery 96%).

## Next action

**M2.1 — chunk-MLP BC on v1**, K=16 / replan 8, ≥2 seeds, eval n≥200 per set, Wilson
intervals into `results.md`.

## Environment

| | |
|---|---|
| pins | `imitation/requirements.txt` — mujoco **3.10.0**, so101-nexus **0.4.8**, numpy 2.5.1, gymnasium 1.3.0 |
| torch | 2.13.0+cu130, **CUDA available** |
| tests | 71 fast + 7 slow, all passing (`pytest -m "not slow"` / `-m slow`) |

⚠️ `cloth_fold_rl/requirements.txt` pins mujoco 3.11.0 / so101-nexus 0.5.1 for its own
committed checkpoint. Do not "unify" these without re-running the expert benchmark — cloth
grasping is contact-dominated and MuJoCo minors change results.

## Frozen datasets

| version | episodes | source | hash | notes |
|---|---|---|---|---|
| v1 | 384 (+16 in `v1_failures`) | expert, seeds 0-399, 30% perturbed | `769dd372719c` | successes only; 38,136 steps; failures hash `699a1dc71c71` |

Collection log: `outputs/imitation/collect_v1_m1_8.log`. The earlier failed attempt is
kept as `outputs/imitation/collect_v1.log`.

## Best checkpoint

None. No policy has been trained on real data.

## Open blockers and known defects

Ordered by what they block. Each is a roadmap milestone.

| # | defect | blocks | milestone |
|---|---|---|---|
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

- 2026-09-24 · M2.2 code (mechanistic failure codes + `perturbed_failures`, collision-free eval versions), M2.3 plumbing (`--obs-subset`, `obs_mask` buffer), M3.1 done (hashed val split, expert-only dense targets, terminal-only padding, SE keep/stop at n=200, `--resume`, clamp logging), `imitation/demo.py`; 71 fast green · COMMIT
- 2026-09-23 · M1.8 done, Phase 1 complete: froze v1 (384 eps, `769dd372719c`) + v1_failures (16); clean 271/279, recovery 113/121 · 212469d
- 2026-09-23 · M1.7 gate passed: expert id_easy 100/100, id_hard 97/100, recovery 96/100; check_resync 92/94/93 of 100 · 08eeda3
- 2026-09-23 · M1.6 done: 20-ep smoke chain green (collect 19+1 fail split, train 600 steps, evaluate n=14×3, dagger 1 round froze smoke_dagger_r1 with 332 labels); no code change needed · a0f92ce
- 2026-09-23 · M1.5 done: cached IK scratch (no speedup: 81-83 s → 82-93 s, identical hash); fixed cross-episode IK-rng leak that made collection nondeterministic; benchmark rows identical 48/50; 71 fast + 7 slow green · c95c431
- 2026-09-23 · M1.4 done: failures → `<version>_failures` in collect, `bc_episodes` filter in train (`--allow-failures`); 28-ep all-perturbed collect split 25/3; 61 fast green · c64e9ce
- 2026-09-23 · M1.3 done: `terminated`/`truncated`/`discount` arrays in schema + validator (only-last-step, exactly-one, discount mask); `unstable` → truncated; 14-ep real collect validates; 71 fast + 7 slow green · 1fb0e0e
- 2026-09-23 · M1.2 done: streaming/resumable `DatasetWriter` + `collect --resume`, seed-named npzs, guard on unfrozen versions; real 20-ep collect killed at 60 s kept 18, resume finished 20/20 unique, hashes verify; 59 fast green · fa31797
- 2026-09-23 · M1.1 done: 139-D obs (per-stage goals, no one-hot, +stage/settle), `cloth_offset_xy` recorded, goals in snapshot; 63 tests green; expert benchmark unchanged 48/50 · e847847
- 2026-09-23 · Sim verified against docs before M1.1: pins, 57+2 tests, benchmark 48/50 reproduced, all §2.2 defects confirmed · no code change
- 2026-09-23 · Phase 0: imitation + DAgger pipeline committed, env pinned, tracking docs · a5aa63f

# Status

**Updated:** 2026-09-23 · **Branch:** `feature/imitation` · **Phase:** 5 done → 6 (VLA, not started)

One-screen answer to "where are we". Update at the end of **every work pass** (see
`CLAUDE.md`), and add a line to the pass log at the bottom. Full plan in
[roadmap.md](roadmap.md); spec in [imitation.md](imitation.md); numbers in
[results.md](results.md).

---

## Where we are

Phases 1-5 have run end to end. The half-fold demo is rendered from pipeline-trained
policies (`outputs/imitation/demo/half_fold_dagger.mp4`, `half_fold_vision.mp4`).

| policy | id_easy | id_hard | recovery | where |
|---|---|---|---|---|
| expert (ceiling) | 100% | 97% | 96% | M1.7 |
| BC chunk-MLP | 98.0 | 69.5 | 38.5 | M2.1 |
| BC diffusion | 99.5 | 80.0 | 51.0 | M2.4 |
| **DAgger (privileged)** | 97.0 | 64.5 | **65.0** | M3.2 |
| IQL from DAgger | 94.0 | 67.5 | 45.5 | M5.3 ❌ |
| vision BC | 97.0 | 94.5 | 21.5 | M4 |
| **vision distilled** | 97.5 | **90.5** | 31.5 | M4.2 |

Not met: M5.3 (IQL worse than DAgger: the critic can't rank actions on single-policy
data) and the recovery half of M4.2 (the vision student is 33 pp behind its teacher).
All in `results.md`.

## Next action

Phase 6 (VLA subgoal proposer) is out of this session's scope. Before it, the cheapest
wins found here:
1. **DAgger from the diffusion checkpoint.** Diffusion BC already beats chunk-MLP BC by
   10-12 pp under shift.
2. **DAgger / distill rollouts from id_hard-like poses** on a disjoint seed range.
   Neither loop ever visits shifted starts.
3. **Vision recovery:** more perturbed distill rollouts, and higher-res main camera.

## Environment

| | |
|---|---|
| pins | `imitation/requirements.txt` — mujoco **3.10.0**, so101-nexus **0.4.8**, numpy 2.5.1, gymnasium 1.3.0 |
| torch | 2.13.0+cu130, **CUDA available** |
| tests | 79 fast + 9 slow, all passing (`pytest -m "not slow"` / `-m slow`) |

⚠️ `cloth_fold_rl/requirements.txt` pins mujoco 3.11.0 / so101-nexus 0.5.1 for its own
committed checkpoint. Do not "unify" these without re-running the expert benchmark — cloth
grasping is contact-dominated and MuJoCo minors change results.

## Frozen datasets

| version | episodes | source | hash | notes |
|---|---|---|---|---|
| v1 | 384 (+16 in `v1_failures`) | expert, seeds 0-399, 30% perturbed | `769dd372719c` | successes only; 38,136 steps; failures hash `699a1dc71c71` |
| v1_img / v1_failures_img | 384 / 16 | v1 replayed + cameras | `769dd372719c` / `699a1dc71c71` | same hash as source: froze before images entered the digest |
| v1_dagger_v2_r1..r3 | +128 each | DAgger (shadowing teacher) | see manifests | parent chain on v1 |
| harvest_v1 | 1000 | dagger_v2/round_3, σ=0.1 | `3b5d331bf404` | 81% success |
| v1_img_distill_v1_r1..r2 | +128 each | vision student rollouts, images | see manifests | parent chain on v1_img |

Collection log: `outputs/imitation/collect_v1_m1_8.log`. The earlier failed attempt is
kept as `outputs/imitation/collect_v1.log`.

## Best checkpoint

- privileged: `outputs/imitation/runs/dagger_v2/round_3/final.pt` (97.0 / 64.5 / 65.0)
- sensor-only: `outputs/imitation/runs/distill_v1/round_2/final.pt` (97.5 / 90.5 / 31.5)
- success detector: `outputs/imitation/runs/success_v1/detector.pt` (91.8% agreement)

Weights are gitignored; each run's `run.json` / `history.json` is committed.

## Open blockers and known defects

Ordered by what they block. Each is a roadmap milestone.

| # | defect | blocks | milestone |
|---|---|---|---|
| 11 | `v1_dagger_v1_r1`/`_r2` frozen with bad teacher labels — never train on them | DAgger | — |
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

- 2026-09-24 · Phase 4 done: vision distill (97.5/90.5/31.5, recovery margin not met), detector 91.8% agreement; half-fold demos rendered (DAgger + vision, 4/4 each); status rewritten; 79 fast + 9 slow green · 3552e37
- 2026-09-24 · M3.2 done: dagger_v2 3 rounds, recovery 38.5→65.0%, ID 97%, id_hard flat; detector trained (95% val frame acc) · 4e22261
- 2026-09-24 · Phase 4/5 code: vision success detector, RL transitions + IQL + harvest, vision-aware demo, image-inclusive version hash; vision BC trained (val 0.066), v1_failures_img rendered; driver `runs/phase3to5.sh` chained after dagger_v2; 79 fast green · b72fe71
- 2026-09-24 · Found + fixed DAgger teacher label bug (shadowing teacher) and eval nondeterminism (padded predict); dagger_v1 marked invalid; Phase 4 plumbing (rig, image store, v1_img, vision policy, distill loop); 75 fast + 9 slow green · 91254d9
- 2026-09-24 · M2.1 (BC 2 seeds, n=200) + M2.2 (mechanistic histograms) + M2.3 (obs ablation) done; docs/pipeline.md added; DAgger + diffusion launched · 5b02a77
- 2026-09-24 · M2.2 code (mechanistic failure codes + `perturbed_failures`, collision-free eval versions), M2.3 plumbing (`--obs-subset`, `obs_mask` buffer), M3.1 done (hashed val split, expert-only dense targets, terminal-only padding, SE keep/stop at n=200, `--resume`, clamp logging), `imitation/demo.py`; 71 fast green · 33067c6
- 2026-09-23 · M1.8 done, Phase 1 complete: froze v1 (384 eps, `769dd372719c`) + v1_failures (16); clean 271/279, recovery 113/121 · 212469d
- 2026-09-23 · M1.7 gate passed: expert id_easy 100/100, id_hard 97/100, recovery 96/100; check_resync 92/94/93 of 100 · 08eeda3
- 2026-09-23 · M1.6 done: 20-ep smoke chain green (collect 19+1 fail split, train 600 steps, evaluate n=14×3, dagger 1 round froze smoke_dagger_r1 with 332 labels); no code change needed · a0f92ce
- 2026-09-23 · M1.5 done: cached IK scratch (no speedup: 81-83 s → 82-93 s, identical hash); fixed cross-episode IK-rng leak that made collection nondeterministic; benchmark rows identical 48/50; 79 fast + 9 slow green · c95c431
- 2026-09-23 · M1.4 done: failures → `<version>_failures` in collect, `bc_episodes` filter in train (`--allow-failures`); 28-ep all-perturbed collect split 25/3; 61 fast green · c64e9ce
- 2026-09-23 · M1.3 done: `terminated`/`truncated`/`discount` arrays in schema + validator (only-last-step, exactly-one, discount mask); `unstable` → truncated; 14-ep real collect validates; 79 fast + 9 slow green · 1fb0e0e
- 2026-09-23 · M1.2 done: streaming/resumable `DatasetWriter` + `collect --resume`, seed-named npzs, guard on unfrozen versions; real 20-ep collect killed at 60 s kept 18, resume finished 20/20 unique, hashes verify; 59 fast green · fa31797
- 2026-09-23 · M1.1 done: 139-D obs (per-stage goals, no one-hot, +stage/settle), `cloth_offset_xy` recorded, goals in snapshot; 63 tests green; expert benchmark unchanged 48/50 · e847847
- 2026-09-23 · Sim verified against docs before M1.1: pins, 57+2 tests, benchmark 48/50 reproduced, all §2.2 defects confirmed · no code change
- 2026-09-23 · Phase 0: imitation + DAgger pipeline committed, env pinned, tracking docs · a5aa63f

# Status

**Updated:** 2026-09-23 · **Branch:** `feature/imitation` · **Phase:** 0 (tracking and
reproducibility)

One-screen answer to "where are we". Update at the end of every milestone. Full plan in
[roadmap.md](roadmap.md); spec in [imitation.md](imitation.md); numbers in
[results.md](results.md).

---

## Where we are

The `imitation/` package is written and unit-tested but **has never run end to end**. The
immediate goal is a first credible half-fold imitation result, which is blocked on freezing
a dataset — and the observation defects must be fixed before that dataset is collected,
because the store is write-once.

## Next action

**M1.1 — fix the observation** (`docs/imitation.md` §2.2): unswap the goal channels, drop
the dead task one-hot, add `stage` and `settle_steps`. Then M1.2 (streaming/resumable
collect) before any 400-episode run.

## Environment

| | |
|---|---|
| pins | `imitation/requirements.txt` — mujoco **3.10.0**, so101-nexus **0.4.8**, numpy 2.5.1, gymnasium 1.3.0 |
| torch | 2.13.0+cu130, **CUDA available** |
| tests | 57 fast + 2 slow, all passing (`pytest -m "not slow"` / `-m slow`) |

⚠️ `cloth_fold_rl/requirements.txt` pins mujoco 3.11.0 / so101-nexus 0.5.1 for its own
committed checkpoint. Do not "unify" these without re-running the expert benchmark — cloth
grasping is contact-dominated and MuJoCo minors change results.

## Frozen datasets

| version | episodes | source | hash | notes |
|---|---|---|---|---|
| — | — | — | — | **none frozen yet** |

`v1` exists as an empty directory only: a 400-episode collection died at episode 215 and
wrote nothing, because `collect.py` returns every episode before writing any (fixed in
M1.2). The partial run's compute is lost. `outputs/imitation/collect_v1.log` is retained as
the record. Delete `outputs/imitation/datasets/v1/` before re-collecting under that name.

## Best checkpoint

None. No policy has been trained on real data.

## Open blockers and known defects

Ordered by what they block. Each is a roadmap milestone.

| # | defect | blocks | milestone |
|---|---|---|---|
| 1 | Goal channels are swapped and ~5 obs dims are dead (`imitation.md` §2.2) | dataset quality | M1.1 |
| 2 | `stage` / `settle_steps` absent from obs → reward non-Markovian | offline RL | M1.1 |
| 3 | `cloth_offset_xy` unrecorded in metadata (the largest per-episode factor) | failure analysis | M1.1 |
| 4 | `collect.py` is all-or-nothing, no resume; unfrozen versions silently orphan npzs | any long collection | M1.2 |
| 5 | No terminal/truncation flags in the write-once schema | offline RL, permanently for v1 | M1.3 |
| 6 | Failed episodes train as demos (~17% of frames, tail-padded to "freeze in place") | BC quality | M1.4 |
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

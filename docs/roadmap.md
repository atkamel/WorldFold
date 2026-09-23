# Roadmap

What we are building, in order, with an exit criterion per milestone. Current position:
[status.md](status.md). Spec: [imitation.md](imitation.md). Numbers:
[results.md](results.md).

**Rule: every milestone exits with a committed artifact, not a claim.**

---

## The arc

```
fold variants + instruction                      (task specification)
        ↓
VLA: (instruction, image) → which Stage/subgoal   Phase 6 — upstream planner
        ↓
scripted expert: IK + phase machine → actions     the executor (works: 96% on train seeds)
        ↓
IMITATION (chunk policy)  ← the backbone          Phases 2-3
        ↓
DAgger (on the states the student visits)         Phase 3
        ↓
privileged → sensor-only distillation             Phase 4
        ↓
offline RL polish (IQL/AWAC)                      Phase 5
        ↓
lightweight deployed policy → robot               Phase 7
```

The imitation policy is the backbone. The VLA sits **upstream** as a planner, not in the
control loop. The world model sits **beside** it and is parked.

DAgger and distillation are the same machinery — DAgger is distillation where the student
picks the states. That is why one `Teacher` ABC (`imitation/teachers/base.py`) serves the
scripted expert (Phase 3), the privileged policy teaching the sensor-only student (Phase 4),
and the VLA (Phase 6) with the loop unchanged. The ABC is defined over `env` rather than
over observations, which is what makes the privileged-teacher case fall out for free.

Two corrections to the reference 13-milestone plan, both addressed in Phase 4: milestone 13
("compress and deploy") is vacuous while the policy input is a privileged state vector, and
milestone 12 ("real-robot fine-tuning") is blocked on a vision-based success detector that
the reference sequence never introduces.

---

## Phase 0 — Tracking and reproducibility

| | milestone | exit | status |
|---|---|---|---|
| M0.1 | Commit and protect the work; gitignore datasets before staging | `imitation/`, `tests/`, the 3 `cloth_fold_rl` edits and the benchmark are committed | ✅ |
| M0.2 | Pin the env that produced the benchmark (mujoco 3.10.0) | `imitation/requirements.txt` | ✅ |
| M0.3 | Make the tests runnable and marked | `pytest.ini`, root `conftest.py`; 57 fast + 2 slow passing | ✅ |
| M0.4 | The four tracking docs | `imitation.md`, `roadmap.md`, `status.md`, `results.md` | ✅ |

## Phase 1 — Fix the silent breakers, then freeze v1  *(ref. milestones 2, 4)*

Everything touching the **write-once** schema or the observation happens before 400 episodes
are collected.

| | milestone | exit | status |
|---|---|---|---|
| M1.1 | Observation cleanup: unswap goal channels, drop dead one-hot, add `stage` + `settle_steps`, record `cloth_offset_xy` | obs spec in `imitation.md` §2 updated; expert benchmark re-run | ✅ |
| M1.2 | Streaming + resumable collection; refuse/guard non-empty unfrozen versions | a killed collection loses nothing | ✅ |
| M1.3 | Transition-level schema fields (`terminated`/`truncated`, discount, terminal flag) | present in v1, since the store is write-once | ✅ |
| M1.4 | Separate failures from demos (`--success-only` default; failures as their own version) | BC trains on successes only | ✅ |
| M1.5 | Cache the IK scratch `MjData` (`cloth_fold_rl/expert.py:37`) | measured speedup on a 20-episode collect | ✅ none measurable (results.md); fixed IK-rng nondeterminism |
| M1.6 | Smoke the whole chain on 20 episodes | collect → train → evaluate → dagger 1 round, green | ✅ |
| M1.7 | **Gate:** expert ceiling on `id_easy`/`id_hard`/`recovery`, n≥100 + `check_resync` artifact | table in `results.md`. If teacher recovery is poor, the recovery arm is redesigned before Phase 3 | ☐ |
| M1.8 | Freeze v1 | 300-500 episodes, hash in `status.md` | ☐ |

## Phase 2 — Imitation baseline and what it cannot see  *(ref. milestones 5, 6)*

| | milestone | exit | status |
|---|---|---|---|
| M2.1 | Chunk-MLP BC, K=16 / replan 8, ≥2 seeds, eval n≥200 | checkpoint + Wilson intervals in `results.md` | ☐ |
| M2.2 | Failure map; fix the R1 short-circuit and the `eval_{name}` collision | mechanistic histogram per set | ☐ |
| M2.3 | Privileged-features ablation: proprio-only / +corners / full 139 | table sizing the Phase 4 vision work | ☐ |
| M2.4 | Diffusion head-to-head, same protocol | keep only if it wins outside the intervals | ☐ |

M2.3 is three index-sliced trainings on data already in hand and is the cheapest informative
experiment in the plan — the real split is 48 sensor-available / 91 privileged dims, not 139
privileged.

## Phase 3 — DAgger  *(ref. milestone 7)*

| | milestone | exit | status |
|---|---|---|---|
| M3.1 | Fix the loop *before* running it: deterministic val split, no contaminated chunk targets, n≥200 scoring in standard errors, `--resume`, normalizer-clamp logging | each fix has a test | ☐ |
| M3.2 | Run the rounds (β decay, label at replan points, recovery starts, additive frozen versions) | ≥80% ID and ≥60% recovery, gains in standard errors | ☐ |

## Phase 4 — Sensor-only student  *(new; supplies what ref. milestones 12-13 assume)*

| | milestone | exit | status |
|---|---|---|---|
| M4.1 | Image obs plumbing: dict passthrough, ≥128² + wrist cam, visual DR, separate image store, lazy DataLoader, per-modality norm | image-conditioned policy trains | ☐ |
| M4.2 | `PrivilegedPolicyTeacher` → on-policy distillation through the Phase 3 loop unchanged | sensor-only student within a stated margin of privileged | ☐ |
| M4.3 | Vision success / fold-score detector (labels free in sim) | agreement rate vs the sim metric | ☐ |

The data path must go lazy here at the latest: `build_samples` materializes every window
densely and `train.py:90` uploads the whole dataset to the GPU. ~80 lines to convert now, a
rewrite later.

## Phase 5 — Reward and offline RL  *(ref. milestones 8, 9)*

| | milestone | exit | status |
|---|---|---|---|
| M5.1 | Student rollout harvest, 1000-2000 rollouts stratified by failure code | reward variance exists to learn from | ☐ |
| M5.2 | RL-usable reward: sparse critic label, clipped toggle spikes, `unstable` tagged, `build_transitions()`, chunk-as-macro-action | documented in `imitation.md` §7 | ☐ |
| M5.3 | IQL or AWAC warm-started from the DAgger policy | ≥ the imitation student on ID and recovery | ☐ |

## Phase 6 — Language-conditioned folds and the VLA  *(ref. milestone 3, reframed)*

| | milestone | exit | status |
|---|---|---|---|
| M6.1 | Fold variants via the existing `Move`/`Stage` parameterization + text instructions | a held-out fold variant = the generalization axis the plan otherwise lacks | ☐ |
| M6.2 | `SubgoalTeacher`: VLA maps (instruction, image) → Stage/subgoal; IK phase machine executes | VLA-specified folds succeed above chance on held-out variants | ☐ |
| M6.3 | Distill subgoal choice into the deployed policy | VLA leaves the runtime | ☐ |

Reframed from "VLA as teacher" to **VLA as task specification**. `FoldExpert` already takes
`goal` as a callable, so M6.2 is a closure swap, and pixel→3D lift already exists.
Open risk: `_maybe_retry` closes the last centimetre using the measured miss from sim vertex
positions; it must become vision-based or be dropped, and dropping it costs success rate.

## Phase 7 — Deploy and clean up

| | milestone | exit | status |
|---|---|---|---|
| M7.1 | Deployment budget check against `imitation.md` §8 | budget met with no VLA in the stack | ☐ |
| M7.2 | Delete the dead PPO path (`cloth_fold_rl/train.py`, `bc.py`); keep the MuJoCoTouch baseline | SB3 dep still justified | ☐ |
| M7.3 | Reconcile or delete `cloth_angles/tasks.py::QUARTER` | one source of truth for the task | ☐ |

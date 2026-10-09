# Roadmap

What we are building, in order, with an exit criterion per milestone. Current position:
[status.md](status.md). Spec: [imitation.md](imitation.md). Numbers:
[results.md](results.md).

**Rule: every milestone exits with a committed artifact, not a claim.** A milestone closes
either ✅ (exit met) or ❌ (closed with evidence in results.md and a named follow-up
milestone if a cheap next lever exists). "Complete before the VLA" means no ☐ or ◐
remains above Phase 6.

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
offline RL polish (IQL/AWAC)                      Phase 5 — dropped 2026-09-25 (M5b.4)
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
| M1.7 | **Gate:** expert ceiling on `id_easy`/`id_hard`/`recovery`, n≥100 + `check_resync` artifact | table in `results.md`. If teacher recovery is poor, the recovery arm is redesigned before Phase 3 | ✅ |
| M1.8 | Freeze v1 | 300-500 episodes, hash in `status.md` | ✅ |

## Phase 2 — Imitation baseline and what it cannot see  *(ref. milestones 5, 6)*

| | milestone | exit | status |
|---|---|---|---|
| M2.1 | Chunk-MLP BC, K=16 / replan 8, ≥2 seeds, eval n≥200 | checkpoint + Wilson intervals in `results.md` | ✅ |
| M2.2 | Failure map; fix the R1 short-circuit and the `eval_{name}` collision | mechanistic histogram per set | ✅ |
| M2.3 | Privileged-features ablation: proprio-only / +corners / full 139 | table sizing the Phase 4 vision work | ✅ |
| M2.4 | Diffusion head-to-head, same protocol | keep only if it wins outside the intervals | ✅ wins (id_hard +10.5, recovery +12.5 pp) |

M2.3 is three index-sliced trainings on data already in hand and is the cheapest informative
experiment in the plan — the real split is 48 sensor-available / 91 privileged dims, not 139
privileged.

## Phase 3 — DAgger  *(ref. milestone 7)*

| | milestone | exit | status |
|---|---|---|---|
| M3.1 | Fix the loop *before* running it: deterministic val split, no contaminated chunk targets, n≥200 scoring in standard errors, `--resume`, normalizer-clamp logging | each fix has a test | ✅ |
| M3.2 | Run the rounds (β decay, label at replan points, recovery starts, additive frozen versions) | ≥80% ID and ≥60% recovery, gains in standard errors | ✅ |

## Phase 4 — Sensor-only student  *(new; supplies what ref. milestones 12-13 assume)*

| | milestone | exit | status |
|---|---|---|---|
| M4.1 | Image obs plumbing: dict passthrough, ≥128² + wrist cam, visual DR, separate image store, lazy DataLoader, per-modality norm | image-conditioned policy trains | ✅ full scope done (dict obs, 128² + wrists, visual DR, image store, lazy sampler, per-camera norm); vision BC on it 94.5 / 91.0 / 26.5 (results.md) |
| M4.2 | `PrivilegedPolicyTeacher` → on-policy distillation through the Phase 3 loop unchanged | sensor-only student within a stated margin of privileged | ❌ closed with the margin stated: vs the best privileged teacher, id_easy −3 pp, id_hard +20 pp, recovery −42.5 pp (M5b.3); now runs through the Phase 3 loop (PolicyTeacher) |
| M4.3 | Vision success / fold-score detector (labels free in sim) | agreement rate vs the sim metric | ✅ 91.8% [87.8, 94.6] |

The data path must go lazy here at the latest: `build_samples` materializes every window
densely and `train.py:90` uploads the whole dataset to the GPU. ~80 lines to convert now, a
rewrite later.

## Phase 5 — Reward and offline RL  *(ref. milestones 8, 9)*

| | milestone | exit | status |
|---|---|---|---|
| M5.1 | Student rollout harvest, 1000-2000 rollouts stratified by failure code | reward variance exists to learn from | ✅ `harvest_v1`, 1000 eps, 81% success |
| M5.2 | RL-usable reward: sparse critic label, clipped toggle spikes, `unstable` tagged, `build_transitions()`, chunk-as-macro-action | documented in `imitation.md` §7 | ✅ |
| M5.3 | IQL or AWAC warm-started from the DAgger policy | ≥ the imitation student on ID and recovery | ❌ closed: not met twice more in M5b.4 (iql_v4 89/34/34.5 vs 98/72/72.5); **RL dropped from the plan** (results.md) |

## Phase 5b — Close the gaps found in Phases 2-5

Added 2026-09-24 from the measured gaps ([reports/2026-09-24-phase1-5.md](reports/2026-09-24-phase1-5.md)).
Ordered by expected payoff. Phase 6 starts after M5b.1-M5b.3.

| | milestone | exit | status |
|---|---|---|---|
| M5b.1 | DAgger from the diffusion checkpoint (`diff_v1_s0`); diffusion BC is already +10-12 pp over chunk-MLP under shift | beats `dagger_v2/round_3` on id_easy + recovery by > 1 SE at n=200 | ✅ round 1: 98.0 / 72.0 / 72.5, +1.74 SE |
| M5b.2 | Shifted-pose coverage: DAgger and distill rollouts from id_hard-like poses on a new disjoint `SHIFT_SEED_BASE` (with a disjointness test); no loop visits shifted starts today | privileged id_hard ≥ 85% without losing recovery | ❌ not met: 2 rounds, id_hard 72 → 61-64.5%; the failures are fine-placement stalls (S1), not missing coverage (results.md) |
| M5b.3 | Vision recovery: ≥ 60% perturbed distill rollouts, 128² main camera, keep rule that also scores id_hard | sensor-only recovery within 15 pp of its teacher | ☐ |
| M5b.4 | Offline RL retry, only with action diversity (multi-policy or high-σ harvest, per-episode subsampling so stalls don't dominate) | ≥ DAgger on ID and recovery, else drop RL from the plan | ❌ closed 2026-09-25: harvest_v2 tripled the advantage spread but not the success-vs-failure gap (+0.003); iql_v4 89.0/34.0/34.5 @8, 69.0/18.5/25.5 @4 vs 98/72/72.5 → **RL dropped** (results.md) |
| M5b.6 | Fine placement (follow-up to M5b.2's diagnosis: shifted-pose failures are S1 stalls 4-12 cm off goal, where the student under-imitates the expert's slow measured-miss corrections): sweep the replan interval (8 / 4 / 2) at eval, then retrain with placement-phase oversampling if the sweep helps | privileged id_hard ≥ 85%, or the sweep shows replanning isn't the lever (closed with evidence) | ❌ closed: replanning *is* a lever (privileged @4: 99.5 / 77.0 / 79.5; vision @2 recovery 30 → 41%) but id_hard peaks at 77%; operating points adopted (results.md) |
| M5b.5 | Hygiene: re-run M2.1-M2.3 under deterministic eval; re-freeze image versions with image-inclusive hashes if they're used for a result | numbers reproduce exactly | ✅ 2026-09-25: repeat eval identical; all four re-evals inside the original intervals; image results use `v1_img128` (images in the digest) (results.md) |

## Phase 5c — Throughput: make real use of the GPU

Added 2026-09-25. Measured during Phase 5b: **training got faster** (distillation 2,000
steps 226 s → 160 s with cached teacher targets; image frames now stay on the GPU), but
**end-to-end wall-clock didn't move much**. Rollouts and evaluation are ~80% of the time
and bound by CPU MuJoCo physics. During them the GPU sits at **2-17% utilization**. One
n=200 × 3-set eval takes ~30-40 min, and a DAgger round ~1 h, of which training is ~15 min.
The CPU-bound harvest ran ~7 s/episode on 6 workers.

| | milestone | exit | status |
|---|---|---|---|
| M5c.1 | Profile one DAgger round and one eval: time split into physics step / rendering / policy inference / teacher labels / pipe IPC / training, per process | a table in results.md that says where the hours go | ✅ 2026-09-26: table in results.md — physics ~50% of worker time; expert label look-ahead 43% of DAgger (2.4× slower per step); inference 1.7-6.0% |
| M5c.2 | Cheap GPU wins in the loop: CUDA graphs / `torch.compile` for batched inference (the 10-step diffusion sampler is latency-bound: 104 ms for a batch of 14), fewer DDIM steps or a distilled one-step head; batch MLP teacher labels on the GPU in the main process | inference + labels < 5% of rollout time, with identical actions (determinism test) | ❌ closed narrowly 2026-09-26: CUDA-graph sampler 12.7×, bit-identical; share < 5% in 3 of 4 loops, 6.0% for diffusion state eval (results.md) |
| M5c.3 | Overlap GPU and CPU inside a run: evaluate round r on the CPU while training round r+1 candidates on the GPU; run CPU-bound queues (harvest, re-evals) during every training phase | GPU busy > 50% across a DAgger run | ❌ closed 2026-09-25: CPU slot works (one sim pool across 3 lanes), but GPU kernel-active 27% across a DAgger run (32% with overlapping lanes); physics-bound, and M5c.4 ruled out GPU physics (results.md) |
| M5c.4 | GPU physics feasibility: MuJoCo Warp / MJX with this cloth (flex) model. Port the half-fold env, then re-run the expert benchmark and the M1.7 ceilings | expert ceiling within intervals of the CPU sim, or a written reason it can't be. Only then: thousands of parallel envs | ❌ closed: loads (flex OK) but ~2× slower than one CPU core at best (274 vs 576 steps/s) and drifts > 1 cm from the CPU trajectory by step 15 (results.md) |
| M5c.5 | Batched GPU rendering of the three cameras (MuJoCo Warp / Madrona-style) once M5c.4 holds | vision rollouts no slower than state rollouts | ❌ closed: gated on M5c.4, which failed |

M5c.1-M5c.3 are safe any time: they don't change the simulator. M5c.4 changes the
physics every result so far was measured on, so it gates on matching the CPU ceilings,
and its numbers would start a new results baseline.

## Phase I — Isaac Sim: make the pipeline run on the new simulator

Added 2026-10-02.
- `origin/main` (PR #15) brought `isaac/`: the cloth-fold env on LeHome's Isaac Sim 5.1
  stack, with a friction grasp and no weld.
- This phase is a **viability pass**. Every stage of the imitation pipeline runs on Isaac
  through the real CLIs at pilot scale.
- It runs locally on Windows, in a separate gitignored `.venv-isaac` (Python 3.11). Installs
  are reviewed and pinned first.
- Grasp tuning and full-scale retraining are later milestones, listed below.
- Plan: [superpowers/plans/2026-10-02-isaac-viability.md](superpowers/plans/2026-10-02-isaac-viability.md).
- Every milestone closes on `python -m imitation.verify <ID>` plus committed evidence.

| | milestone | exit | status |
|---|---|---|---|
| I0.1 | Pull: merge `origin/main` into `feature/imitation`; fix the merge silent-breakers (`stages` shadowing, hard-coded one-hot offset, isaac test stub) | fast + slow tests green; MuJoCo expert benchmark 48/50 with rows identical to M1.5 | ✅ 2026-10-02: 116 fast + 12 slow; rows identical |
| I0.2 | Install review: pinned sources, hashed lock, source review of the LeHome repos, asset audit | `isaac/INSTALL_REVIEW.md`; user OK on artifact list + EULA | ✅ 2026-10-02: SAFE; user OK + EULA |
| I0.3 | **Gate:** scoped install in `.venv-isaac` + `isaac/smoke_test.py` state and hybrid on Windows | `SMOKE OK` ×2, CUDA torch, nothing written outside documented paths. Fails → stop and report | ✅ 2026-10-03: SMOKE OK ×2, 3.4 / 4.97 steps/s |
| I0.4 | Verifier `python -m imitation.verify` | its tests green | ✅ 2026-10-03 |
| I1.1 | Backend switch (`make_env(backend=)`, EnvPool, CLIs, lazy MuJoCo imports, Isaac worker lifecycle) | tests green in both venvs; MuJoCo unchanged | ✅ 2026-10-03: hash-identical MuJoCo, Isaac pool green |
| I1.2 | Isaac seed sets (id_easy / id_hard ring / scaled recovery) | disjoint; id_hard reachable 200/200 | ✅ 2026-10-03: R_max 1.0 cm (id_hard = edge of jitter, not OOD) |
| I1.3 | Isaac camera rig: main 128² + wrists 64² | shapes; wrist pose within 1 mm / 0.5° | ✅ 2026-10-03: wrist mount 0.014 mm / 0°, 2.84 steps/s |
| I2.1 | Isaac expert (grasp as built) as the scripted teacher | runs; pilot rates recorded (no threshold) | ✅ 2026-10-03: runs as teacher; pilot 2/20 id_easy, 0/10 recovery (grasp as built) |
| I2.2 | DAgger labels on Isaac: executed-expert takeover labels (user-approved 2026-10-02) | pilot sanity bar | ✅ 2026-10-03: 204/208 takeovers full chunks (98.1%) |
| I3.1 | End-to-end micro chain test (Isaac M1.6) | green | ✅ 2026-10-03: 6/6 in 32 min |
| I3.2 | Pilot run: collect 80, BC ×2, vision, DAgger 1 round, detector, demos | all artifacts valid; pilot rows in results.md | ✅ 2026-10-03: every stage ran; pilot numbers in results.md |
| I3.3 | Close-out | `verify --all` green in both venvs | ✅ 2026-10-03: verify --all PASS in both venvs |

**Later milestones (not in the viability pass)**

| | milestone | exit |
|---|---|---|
| IG.1 | Grasp bench (per-arm acquired / held / placed / released, anchor drift) | baseline on tuning seeds 600 000+ — ✅ 2026-10-03 (track G): `isaac/grasp_bench.py`, n = 20; left held 18/20, right held 1/20 |
| IG.2 | Grasp reliability loop (one knob per iteration; adhesion 0, friction ≤ 2.0, no attachments) | per arm, n = 100 on two blocks plus a fresh block: acquired / held / released ≥ 98, placed ≥ 95. "released" (redefined, user-approved 2026-10-04): the corner was held and by the end of the retreat is back on the cloth, not carried up by the gripper (within 2.5 cm of its rest height); where it lands is "placed". The old < 3 cm move test is kept as `released_strict`, a record only |
| IG.3 | Expert ceiling gate on Isaac | id_easy ≥ 95, id_hard ≥ 90, recovery ≥ 90, check_resync ≥ 90 (n = 100; 92–94 extends to n = 200) |
| IS.1–6 | Full retrain at scale (`isaac_v1` 400 eps, BC ×2 seeds, replan sweep, DAgger, vision ×2, detector ≥ 90%) | n = 200 × 3 sets, Wilson, gains in SE |
| I-DR | Visual DR in the Isaac rig + dynamics DR | robustness on held-out DR draws |

## Phase W — MuJoCo weld baseline on Isaac (+ parallel friction-grasp track G)

Added 2026-10-03.
- **Why:** Phase I showed the pipeline runs on Isaac, but with the friction grasp as built the
  expert collects 2/80 successes and every student trains on 2 demos.
- **The baseline:** move the full MuJoCo setup into Isaac (weld grasp, cloth pose/jitter, arm
  drives, dynamics DR, expert, eval sets) and retrain on it.
- **Track G:** friction-grasp work (IG.1–IG.3) runs in parallel as a separate session on
  `feature/isaac-grasp`, with 1 Isaac worker against the baseline's 2.
- **Plan:** [superpowers/plans/2026-10-03-isaac-weld-baseline.md](superpowers/plans/2026-10-03-isaac-weld-baseline.md).

| | milestone | exit | status |
|---|---|---|---|
| W1 | Weld grasp in Isaac (`grasp_mode="weld"`): MuJoCo semantics, particle patch pinned each substep | weld_check: tracking < 3 mm while lifted/carried, release drops, mask and hysteresis honoured | ✅ 2026-10-03: GPU pipeline (zero-mass pins); lift 1.92 mm, carry 2.42 mm |
| W2 | Weld profile (MuJoCo setup carried over; profile renamed `mujoco` → `weld` 2026-10-04): cloth (0,0) ±2.5 cm, no tilt, arm drives kp ≈ 998 / kv ≈ 2.73, dynamics DR ×U(0.7,1.3), cap 250, MuJoCo eval sets | FoldExpert waypoints on MuJoCo poses (n = 200 per set) reachable at parity with MuJoCo's own arm (≤ 5 pp gap; id_easy 200/200); arm reaches ≥ 90% of a 0.05 rad step in one control step; DR deterministic per seed and read back from the solver | ✅ 2026-10-03: reach 200/200 and 122/200 vs MuJoCo 200/130 (gap 4.0 pp); profile 9/9 |
| W3 | MuJoCo FoldExpert on Isaac (backend split, MuJoCo byte-identical), overshoot re-measured | expert gate n = 100: id_easy ≥ 95, id_hard ≥ 90, recovery ≥ 90 | ✅ 2026-10-03 with an accepted exception: 100 / 100 / **81** (recovery below 90, accepted by decision as the Isaac ceiling) |
| W4 | Pilot on the weld baseline (as I3.2) | artifacts valid; diffusion id_easy Wilson LB > 0 | ✅ 2026-10-04: collect 76/80; diffusion 20/20 easy, 8/10 hard, 1/10 recovery; vision 10/10; DAgger + detector run |
| W3b | Expert recovery on the weld profile (Isaac expert only; cap 250 and eval sets unchanged) | recovery ≥ 90 at n = 100 (≥ 85 if two attempts in a row gain < 2 pp); id_easy/id_hard not regressed | ❌ closed 2026-10-06: plateau 81.7% pooled (147/180) after attempts A–D; user accepted the W3 expert for W5 |
| Z1 | Zero-shot MuJoCo-trained checkpoints on `isaac_weld` (diffusion DAgger r4, best BC, vision r2) | n = 20 × 3 sets recorded; W5 fine-tunes if id_easy ≥ 50 | ☐ |
| V | Vectorised Isaac env: B cloth envs per process (GPU weld profile, 101×101), sub-env views, worker step barrier | expert parity B=4 vs B=1 (id_easy n = 40, Wilson); DR deterministic per seed; ≥ 3× episodes/h (kept if > 1.5×) | ☐ artifacts in (`feature/isaac-vec`): parity 40/40 vs 40/40, DR per seed ✓, 2.15× at B = 8 (kept, 3× not met); lead-verified (fast 183, slow 12), merged 2026-10-04 |
| W5 | Slimmed weld retrain (`isaac_v1_weld` 400 ep on the W3b expert + vectorised env; diffusion ×2 seeds (fine-tuned from MuJoCo weights if Z1 says so), vision ×1, DAgger ×2, detector, demos, report; chunk-MLP dropped; cloth stays 101×101) | n = 200 × 3, Wilson, SE rules; MuJoCo-vs-Isaac side-by-side | ✅ 2026-10-07 (fast track): privileged DAgger r1 @4 99.0 / 97.5 / 54.5, sensor-only @2 99.0 / 82.5 / 31.0 (n = 200); detector 98.0%; 6 demos; report `docs/reports/2026-10-07-isaac-half-fold.md`; recovery target missed (expert ceiling 81.7%) |
## Phase F — Physical grasp on LeHome's physics, retrain, raise the benchmarks

Added 2026-10-07.
- **Why:** W5's grasp is a weld. Particles are pinned once the gripper site is within 3 cm of the corner, and the jaws
  are forced open. The demo shows an arm hovering while the cloth follows it.
- **Goal:** a grasp where the jaws close and only contact forces move the cloth. Then retrain the pipeline on it.
- **Rules** (user, 2026-10-07): LeHome-style friction + adhesion only.
  - Adhesion is capped at 0.3, with a release test.
  - Pads, jaw gap, drive gains and physics rate may change.
  - No attachments, pins, welds, or kinematic writes to particles.
- **Ruby's anchor grasp** (`origin/feat/isaac-half-fold`) is a distance-gated attachment. Her Markov expert ideas are
  borrowed. The anchor comparison was dropped (user, 2026-10-07).
- **The comparison row is Adam's Isaac friction grasp as built** (PR #15: the lehome profile with the I2.1 expert),
  on the same LeHome task sets.
- **Weld results are frozen** as the reference. The benchmark push happens on the friction profile only.
- **Plan:** [superpowers/plans/2026-10-07-physical-grasp.md](superpowers/plans/2026-10-07-physical-grasp.md).

| | milestone | exit | status |
|---|---|---|---|
| F0 | Track G merged forward. `friction` profile = weld profile's setup (GPU, vectorised, DR, MuJoCo eval sets) + friction grasp with IG.2 #9 knobs as real defaults | grasp bench on the friction profile (B = 4, GPU) within the interval of #9, or delta documented + re-tune in F2; fast tests green | ◐ 2026-10-07: delta documented (held 8/20, 10/20: the stiff MuJoCo drive on the jaw); re-tuned in F2 (LeHome gripper drive) |
| F1 | Grasp honesty and observability: no pins or attachment prims, jaws not forced open; physical `grasp_active` (jaw closed + settled, corner in the pad pocket); G2 slip code; demo overlay | honesty check passes on 20 friction expert episodes and fails on weld (negative control) | ✅ 2026-10-07: friction 0 violations (n = 20, 18/20 successes); weld 251 pinned / 496 forced-open steps; `verify F1` PASS |
| F2 | Grasp reliability loop (IG.2 continued): one knob per iteration, in order: adhesion, adhesion offset, gravity scale, jaw gap, gripper drive, pads, 200 Hz, particle friction; plus Ruby's lift-on-settled rule and carry speed | ~~per arm, n = 100 × 3 blocks~~ → fast path (user, 2026-10-07): the kept config's bench run (n ≥ 40) holds ≥ 97.5% per arm, and F3's full-task expert eval (n = 100 per set) has grasp success ≥ 95% and G2 slips ≤ 5% on every set | ◐ config M kept: held 40/40 both arms; gate 1 (config I, n = 300) right arm 93% → fixed by ALIGN_TOL; awaiting F3 |
| F3 | Friction expert with Markov re-grasp (jaw-settle lift, physical hold check → re-grasp, replan on corner move, carry speed, pre-close alignment); Adam's as-built grasp as the comparison row (`scripts/adam_baseline.ps1`) | expert n = 100 on LeHome's task sets (the pinch reaches ~1.5 cm of cloth offset; MuJoCo's id_hard is 0/200 reachable): id_easy ≥ 95, id_hard ≥ 90, recovery ≥ 90 (≥ 85 plateau), check_resync ≥ 90; weld / Adam as-built / friction rows side by side | ☐ |
| F3b | **Expert first** (user, 2026-10-08: F4 paused). Perturbation suite redefined: `knock_arm`, `drop`, `joint_noise`, `overshoot`; the legacy all-dims knock is reported only. Recovery traced, then fixed one mechanism at a time (tune seeds, n = 40, paired). Plan: [superpowers/plans/2026-10-08-expert-first.md](superpowers/plans/2026-10-08-expert-first.md) | expert n = 100: id_easy ≥ 95, id_hard ≥ 90, G2 ≤ 5%; each perturbation set ≥ 60% (aim ≥ 80 on `drop` / `knock_arm`); plateau → user | ✅ 2026-10-08: expert defaults RESYNC_IK_HOME + DESCEND_RETRY + CART_ARRIVE; gate id_easy 92 / id_hard 94 / knock_arm 75 (n = 100; clean bar 90 by user); forced drop skipped (user); `verify F3` PASS |
| F4 | (paused 2026-10-08 until F3b; a partial collection at 281/400 was discarded) Retrain on friction: `isaac_v1_friction` 400 eps, 50% knocked; diffusion × 2 seeds; DAgger × 2 at takeover p = 0.6; replan sweep; vision student; detector | n = 200 × 3 sets, Wilson; side by side with weld W5 | ☐ 2026-10-08: BC-only finals (3.5 h cut: 200 eps, no DAgger, n = 100/50) id_easy 81 / id_hard 71 / knock_arm 35; DAgger pending |
| F5 | Benchmark push, one A/B at a time (SE rules, n = 200). Teacher: intervention upweighting, DAgger to plateau, seed ensemble. Student: student DAgger, image aug, recovery weighting, corner auxiliary head | teacher recovery ≥ 75, id_hard ≥ 95; student id_hard ≥ 90, recovery ≥ 45; otherwise ❌ with the margin | ☐ |
| F6 | Report + demos (6 friction videos + a weld vs friction clip), HTML page | `verify F0..F6` PASS | ☐ |

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

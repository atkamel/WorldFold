# The imitation pipeline, end to end

How a half-fold policy is made in this repo, from the scripted expert to a rendered demo.
This is the "how to run it and why it is shaped this way" guide. The spec (observation,
schema, metrics) is [imitation.md](imitation.md); the plan is [roadmap.md](roadmap.md);
every number is in [results.md](results.md).

```
 scripted expert (IK + phase machine)          cloth_fold_rl/quarter_fold_expert.py
        │  collect: 400 seeds, 30% knocked off course mid-episode
        ▼
 frozen dataset v1  (+ v1_failures)             outputs/imitation/datasets/
        │  train: chunk-MLP, obs history → 16-action chunk
        ▼
 BC student                                     outputs/imitation/runs/bc_v1_s*/
        │  DAgger: student drives, expert labels the states it reaches
        ▼
 DAgger student  (v1_<tag>_r1, _r2 … aggregated) outputs/imitation/runs/dagger_*/
        │  evaluate on held-out seed sets (n=200 each)  ·  demo → mp4
        ├──────────────────────────────┐
        ▼                              ▼
 offline RL (IQL on harvested      sensor-only student: cameras + proprio,
 student rollouts)                  distilled from the privileged policy
        │                              │   + vision success detector
        ▼                              ▼
 half-fold demo                                 outputs/imitation/demo/
```

Everything runs with the pinned env: `.venv/Scripts/python.exe` (Windows) with
`imitation/requirements.txt` (mujoco 3.10.0).

---

## Where it stands (2026-09-24)

| stage | id_easy | id_hard | recovery |
|---|---|---|---|
| expert (ceiling) | 100% | 97% | 96% |
| BC | 98.0% | 69.5% | 38.5% |
| DAgger (privileged state) | 97.0% | 64.5% | 65.0% |
| sensor-only student (cameras) | 97.5% | 90.5% | 31.5% |

n = 200 per set; intervals and details in [results.md](results.md). Offline RL (IQL) was
tried and did *not* beat DAgger, see §8.

## 1. The task and the teacher

**Half fold** (`imitation/tasks/half_fold.py`): two SO-101 arms carry the cloth's two north
corners onto the south corners, release, and the corners must stay placed for 20 steps
(1 s). The step cap is 250 steps (12.5 s).

- **Observation:** 139-D state vector (`imitation.spec.OBS_DIM`): proprioception (50),
  cloth state (69), task/goal (18), stage + settle counter (2). Most of it is privileged
  (read from the sim), see imitation.md §2.1.
- **Action:** 12-D in [-1, 1]: per arm 5 joint deltas + gripper (`ACTION_DIM`).

**The teacher** is the scripted expert: damped-least-squares IK plus a phase machine
(approach → descend → grasp → lift → carry → place → hold → release). It reads sim state
directly. The `Teacher` interface (`imitation/teachers/`) lets it:

- `act()`: drive the episode itself (data collection);
- `label_chunk()`: from *any* state, snapshot the sim, run itself K steps, restore, and
  return those K actions (DAgger labels);
- `resync()`: re-infer its phase after someone else was driving (recovery).

Its ceiling on the held-out sets is 100% / 97% / 96% (id_easy / id_hard / recovery). No
student can be expected to beat that.

## 2. Collect: `imitation.data.collect`

```bash
.venv/Scripts/python.exe -m imitation.data.collect --episodes 400 --workers 14 --version v1
```

- Runs the expert on seeds `0..N-1` across a pool of worker processes (`imitation/rollout.py`).
- 30% of episodes are **recovery demos**: a few random actions knock the expert off
  course, then it resyncs and finishes. These teach the student what to do from states
  the clean expert never visits.
- **Streaming and resumable:** every finished episode is written at once (atomic npz +
  a journal line). If a run dies, rerun the same command with `--resume` and it continues.
- **Failures go elsewhere:** successful episodes → `v1`, failed ones → `v1_failures`.
  A failed expert episode isn't a demonstration; it's kept for failure analysis and
  offline RL.
- **Deterministic:** the same command gives a byte-identical dataset (same hash).

## 3. The dataset store: `imitation/data/schema.py`

A **version** is a directory with one npz per episode and a `manifest.json` holding the
sha256 of every file. It is **write-once**: after `freeze()` it can't be added to, and
`load_dataset` re-verifies every hash on load. DAgger versions name a `parent` and list
the parent's episodes plus their own, so aggregation never copies or edits old data.

Per step an episode stores `obs, actions, rewards, stage, fold_score, grasped, actor`
(who chose the action: teacher / student / perturbation) and the RL transition flags
`terminated, truncated, discount`; DAgger episodes also store `label_steps` and `labels`.

Frozen so far: `v1` (384 expert successes, hash `769dd372719c`) and everything derived
from it; see the table in [status.md](status.md).

## 4. Train: `imitation.train`

```bash
.venv/Scripts/python.exe -m imitation.train --dataset v1 --run outputs/imitation/runs/bc_v1_s0 --seed 0
```

- **Policy:** `chunk_mlp` (`imitation/policies/chunk_mlp.py`): the last 2 observations
  go in, a chunk of K = 16 future actions comes out. At run time only the first 8 are
  executed, then it re-plans (the "replan 8" in every eval). `diffusion` is the alternative head.
- **Samples** (`imitation/data/dataset.py::build_samples`): every expert step t gives
  (obs history at t → actions t..t+15). Steps the expert didn't choose (perturbations)
  are masked out of the loss. Every DAgger label gives (obs history at t → the teacher's
  chunk from that exact state).
- **Only successes are imitated:** failed expert episodes are dropped unless you pass
  `--allow-failures`.
- **Normalizer** is fit on training episodes only and saved inside the checkpoint.
- **Split:** each seed is hashed to train or val, so adding data never reshuffles it.
- Output: `final.pt` (EMA weights) + `run.json` (git commit, dataset hash, config,
  checkpoint hash). A run is auditable from `run.json` alone.

`--obs-subset proprio | proprio_corners | full` hides observation dims (M2.3 ablation):
it measures how much the policy depends on privileged cloth state, which sizes the
vision work in Phase 4.

## 5. Evaluate: `imitation.evaluate`

```bash
.venv/Scripts/python.exe -m imitation.evaluate --ckpt outputs/imitation/runs/bc_v1_s0/final.pt --n 200
```

Closed-loop rollouts on three **held-out** seed sets (`imitation/seeds.py`). These never
overlap training or DAgger seeds:

| set | what it tests |
|---|---|
| `id_easy` | nominal start poses |
| `id_hard` | cloth shifted 2.5–4 cm off nominal |
| `recovery` | knocked off course by 8–15 random actions mid-episode |

It reports the success rate, mean fold score, grasp rate, termination reasons and a
**failure code** per failed episode: G1 grasp, M1 motion (drag / sim blow-up),
F1 misplaced, S1 stalled. It also counts failures that followed a disturbance
(`perturbed_failures`). Use n ≥ 200: at n = 48 the standard error is ~7 pp, which is
wider than the effects we care about. Report Wilson intervals.

`--ckpt expert` evaluates the teacher itself (the ceiling).

## 6. DAgger: `imitation.dagger`

```bash
.venv/Scripts/python.exe -m imitation.dagger --init outputs/imitation/runs/bc_v1_s0/final.pt \
    --dataset v1 --out outputs/imitation/runs/dagger_v1 --rounds 4
```

Behaviour cloning only sees expert states; once the student drifts, it's somewhere it
has never been taught. DAgger fixes that. Each round:

1. The current best student drives on fresh seeds. At every replan point the teacher
   labels the state the student is actually in. With probability β (decaying 0.3, 0.15, …)
   the teacher's chunk is executed instead, which keeps early rounds on track.
   30% of episodes are knocked off course, so recovery states get labelled too.
2. Those episodes are frozen as a new version on top of the last one (`v1_dagger_v1_r1`, …).
3. The student is retrained from the best checkpoint on the aggregated data. DAgger
   labels are oversampled ×4, because they are few and exactly where it goes wrong.
4. It's evaluated at n = 200. It's kept if it scores higher (id_easy + recovery), and the
   loop stops once a round gains less than 1 standard error.

**The teacher shadows the student.** While the student drives, the worker calls
`ScriptedTeacher.observe()` before every step, so the expert's phase machine, IK targets
and timers evolve exactly as if it were acting. A label keeps that state unless the arm's
phase *group* changed (pre-grasp / holding / released), e.g. the student dropped a corner.
The first version re-inferred phases from geometry at every label. It disagreed with the
expert on the expert's own trajectory, and DAgger regressed (results.md, "dagger_v1").

`--resume` continues a killed run from `history.json`. The log reports how many training
observations hit the normalizer's clamp. A warm-started student keeps its first
normalizer, so DAgger states far from the expert data show up there.

## 7. Sensor-only student: `imitation.vision.*` (Phase 4)

The state policies read cloth vertex positions straight from the simulator; a robot can't.
The sensor-only student sees what a robot has: the `main` camera (96²) and both wrist
cameras (64²), plus the 48 sensor-available proprio dims.

1. **Render the demos:** `python -m imitation.vision.replay --version v1` replays every
   frozen episode's actions (exact: each replayed observation is checked against the
   stored one) and writes `v1_img`, the same episodes plus images. Visual domain
   randomization (cloth/table colour, light, camera jitter) is drawn per episode and only
   touches rendering.
2. **Distill** is the Phase 3 DAgger loop with a *policy* as the teacher (M4.2):
   `python -m imitation.dagger --init <vision checkpoint> --dataset v1_img128 --teacher <best state policy> --relabel`.
   The worker hosts `PolicyTeacher` (`imitation/teachers/policy.py`) in place of the
   scripted expert. Every stored episode already holds the privileged observation, so
   `--relabel` retrains on every visited step labelled by the teacher (`train --teacher`).
   An image student gets a camera-rendering pool automatically. The first run
   (`distill_v1`) used a separate copy of this loop, which was retired in Phase 5b.
3. **Success detector:** `python -m imitation.vision.success train|agree` is a CNN on the
   main camera. It predicts "folded" (corners on goal, grippers open) and the fold score
   from one frame, with labels free from sim. `agree` scores it against the sim's success
   flag on held-out student rollouts. Hardware evaluation needs this.

## 8. Offline RL: `imitation.rl.*` (Phase 5)

1. **Harvest:** `python -m imitation.rl.harvest --ckpt <DAgger best> --episodes 1000`.
   The student runs on its own seed range, 30% knocked off course, with light action
   noise. Successes *and* failures are kept, since the critic needs reward variance.
2. **Transitions** (`imitation/rl/transitions.py`): one transition is one replan
   interval, with the 8 executed actions as the macro-action. The reward is +1 on a
   successful terminal plus Δfold-score clipped to ±0.05, which kills the grasp-toggle
   spikes of the env reward. Truncations bootstrap, and `unstable` episodes are dropped.
3. **IQL** (`python -m imitation.rl.iql --init <DAgger best> --versions ...`): expectile V,
   twin Q over macro-actions, and advantage-weighted regression into the same chunk MLP.
   The result is an ordinary checkpoint for `evaluate` and `demo`.

**Status: it didn't help here.** On single-policy harvest data the critic ranks states
well but can't rank actions (advantages ≈ 0 ± 0.02 on both successful and failed
trajectories). The update then reduces to BC over everything, stalls included, and
recovery fell 16-20 pp (results.md, M5.3). It needs more action diversity before it can
beat DAgger.

## 9. Demo: `imitation.demo`

```bash
.venv/Scripts/python.exe -m imitation.demo --ckpt <best checkpoint> --seeds 100000 100001 100002
```

Runs the checkpoint exactly as `evaluate` does (chunk of 16, replan every 8) on held-out
seeds, renders offscreen with MuJoCo, and writes an mp4 with a fold-score/grasp overlay
plus a JSON of outcomes. `--ckpt expert` renders the teacher for comparison. A vision
checkpoint gets its cameras rendered each replan, exactly as in training.

## 10. Where the GPU is used, and where it can't be

The sim is MuJoCo 3.10 on the **CPU**: worker processes, one per core. Rollouts, DAgger
labelling and evaluation are bound by physics at ~50-250 ms per control step. Moving
physics to the GPU would need MJX or MuJoCo Warp. That's a different simulator from the
pinned one every number was measured on, with its own cloth (flex) support, so it's a
milestone of its own, not a speed switch. Everything learned runs on the **GPU**:
- training (bf16 autocast; image frames kept on the GPU when they fit, else pinned host memory);
- policy inference for every rollout, batched across workers and padded to a fixed shape;
- teacher targets for distillation, computed once per dataset in large GPU batches
  (`WindowSampler._cache_teacher_targets`) instead of inside every training step:
  2,000 steps took 226 s → 160 s with a diffusion teacher, before frames moved on-GPU.

Independent CPU-bound and GPU-bound jobs run side by side. Phase 5b used three
tracks: vision distillation, offline RL + re-evaluations, and the final
figures/demos, each with a worker budget summing to the 16 cores.

## 11. Reproducing a result

A number in `results.md` is reproducible from three things: the dataset `manifest.json`
(content hash), the checkpoint's `run.json` (git commit, config, checkpoint hash), and
the eval command with its seed set and n. Collection is deterministic per seed,
evaluation seeds are fixed by `imitation/seeds.py`, and policy inference is padded to a
fixed batch shape (`rollout.padded_predict`). GPU results otherwise shift ~1e-6 with batch
size, the cloth sim amplifies that, and the same checkpoint scored 145 then 134/200.

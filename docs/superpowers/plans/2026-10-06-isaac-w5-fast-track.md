# W5 fast track: make the Isaac (weld) half fold good with the MuJoCo BC → DAgger → vision pipeline

## Context
`origin/main`'s Isaac sim is merged. The weld baseline pilot (W4) learns: diffusion 20/20 easy, 8/10 hard,
1/10 recovery. The vectorised env (V) is merged. The user wants the Isaac half fold "super good", quickly,
with the same pipeline that produced the MuJoCo results. Decisions (2026-10-06):
- **Track G is paused.** Leave branch `feature/isaac-grasp` and its worktree as they are. That frees a third
  Isaac slot for W.
- **W3b stops; collect now.** The current expert (retries on) stays: 81.7% pooled recovery. Attempt D, with
  retries off, scored 58/80 = 72.5%. Record it as a plateau the user accepted.
- **Scope:** a privileged diffusion teacher (BC → DAgger), then a vision student. The vision student keeps
  today's sensor set: 3 cameras plus the 48 sensor proprio dims (`OBS_SUBSETS["proprio"]`), with no cloth
  state and no weld flags. No spec change.
- **Eval:** n = 100 per set to choose rounds and seeds; n = 200 × 3 only on the final checkpoints.
- **No fine-tuning from MuJoCo weights.** Z1 scored 0–1/20, so training starts from scratch.

Targets (the MuJoCo bests, scaled to an expert whose recovery ceiling is about 82%):
- privileged: id_easy ≥ 95, id_hard ≥ 75, recovery ≥ 65
- vision: id_easy ≥ 90

These are targets that guide decisions, not hard stops.

## Step 0: housekeeping (about 15 min, no sim)
- Record the attempt D result in `docs/results.md`, with n, seeds 610000–610079 and Wilson intervals. Close
  W3b in the roadmap as "plateau accepted by user at 81.7%".
- Mark track G as paused in `status.md`.
- Remove stale blocker 14.
- Revert or commit the two modified `outputs/.../verify/I3.2.json` / `I3.3.json` files after looking at the
  diff.

## Step 1: smoke the vectorised env with cameras (about 20 min)
- Check whether the rig cameras work at B > 1. `IsaacClothFoldBatch` was benchmarked without images.
- Run a 4-episode collect with `WORLDFOLD_ISAAC_ENVS_PER_PROC=4 --render --cameras main=128,...`. Check that
  the frames differ per sub-env and match a B = 1 frame for the same seed.
- **Fallback:** if the images are wrong, render stages run at B = 1 with 3 workers, and state-only stages
  (DAgger state rollouts, privileged evals) stay at B = 8.

## Step 2: driver script `scripts/isaac_retrain.ps1`
- **Pattern:** copy `scripts/isaac_pilot.ps1`. Keep its resumable `Stage` function, the STEP / DONE / FAIL
  markers and the per-stage logs.
- **Settings:** `-Backend isaac_weld`, `$tag = isaac_v1_weld`, 3 workers, env `WORLDFOLD_ISAAC_ENVS_PER_PROC`
  (8, or 4 if VRAM > 14 GB).
- **Run it** as a visible background task with a Monitor on its log (memory rules: at most 3 Isaac
  processes; check for orphaned processes first).

Stages:

| # | stage | command (existing CLIs) | est. |
|---|---|---|---|
| a | collect | `imitation.data.collect --backend isaac_weld --episodes 400 --recovery-fraction 0.3 --render --cameras main=128,left_wrist_cam=64,right_wrist_cam=64 --version isaac_v1_weld --resume` | ~2–2.5 h |
| b | diffusion BC ×2 | `imitation.train --policy diffusion --steps 30000 --seed {0,1}` (the M2.4 protocol) | ~35 min each (the second overlaps c) |
| c | BC eval | `imitation.evaluate --sets id_easy id_hard recovery --n 100 --replan-every {8,4}` | ~1 h per checkpoint |
| d | diffusion DAgger | `imitation.dagger --init <best BC> --labels takeover --takeover-p 0.3 --episodes 128 --rounds 3 --train-steps 15000 --eval-n 100 --eval-sets id_easy id_hard recovery --score-sets id_easy id_hard recovery --min-gain-se 1 --resume` | ~2.5 h per round |
| e | vision BC | `imitation.train --policy vision --dataset isaac_v1_weld` (M4.1 settings) | ~1 h |
| f | vision distil | `imitation.dagger --teacher <best privileged> --relabel --recovery-fraction 0.6 --shift-fraction 0.3 --replan-every 2 --rounds 2 --eval-n 100` (the M5b.3 protocol) | ~3 h per round |
| g | detector | `imitation.vision.success train/agree` on `isaac_v1_weld` + `_failures` (bar ≥ 90%) | ~10 min |
| h | final evals | best privileged at replan 4 and best vision at replan 2: n = 200 × 3 sets | ~3 h |
| i | demos | the same demo command as the pilot's `demo_diffusion` / `demo_vision` → `docs/reports/media/half_fold_isaac_{privileged,sensor}.mp4` | ~15 min |

- **Concurrency:** b-seed1 and e train on the GPU while the c/d sims run. Never more than 3 Isaac processes.
- **Wall clock:** about 14–18 h total. The privileged result is ready after about 8 h.

## Step 3: check-in points
- **After c:** if BC recovery < 30% at n = 100, look at the failure codes before DAgger. G1 means grasp, S1
  means stall. DAgger fixed recovery on MuJoCo (+17 pp), so the default is to proceed.
- **After d:** report the best round. If id_hard drops round over round (the M5b.1 keep-rule flaw), carry the
  best round on all three sets forward (`imitation.viz.pick winner --carry-sets`).
- **After f:** distillation did not help on MuJoCo (round 0 was kept), so the keep rule decides. Round 0 is an
  acceptable result.

## Step 4: verify and docs
- Add `@check("W5")` to `imitation/verify.py`, modelled on `_w4`:
  - artifacts exist
  - final n = 200 evals with Wilson intervals
  - detector ≥ 90
  - `results_md_has("W5")`
- Docs, in this order:
  - `results.md`: every number, with n, seeds, dataset hash and Wilson interval
  - `roadmap.md`: W3b and W5 ticked
  - `status.md`: phase, next action, pass log
  - `imitation.md` §10: Isaac protocol differences (takeover labels, n = 100 selection)
  - `subagents.md`: if anything is delegated
- Tests and commit:
  - `pytest -m "not slow"` in `.venv`, `-m slow` plus the Isaac weld tests in `.venv-isaac`
  - commit code and docs together on `feature/isaac-imitation`
- A short dated report: `docs/reports/2026-10-xx-isaac-w5.md`, with a MuJoCo-vs-Isaac table.

## Critical files
- **New:** `scripts/isaac_retrain.ps1`
- **Modify:** `imitation/verify.py` (W5 check); docs (`status`, `roadmap`, `results`, `imitation`)
- **Reuse, unchanged:**
  - `imitation/data/collect.py`, `imitation/train.py`, `imitation/evaluate.py`, `imitation/dagger.py`
    (takeover labels, policy teacher with `--relabel`)
  - `imitation/vision/success.py`
  - the env-var B switch in `imitation/rollout.py:269`
  - `imitation/lockstep.py`
  - `scripts/isaac_pilot.ps1` (template)

## Verification
- `python -m imitation.verify W5` passes in `.venv-isaac`, and I re-run it myself.
- Final n = 200 × 3 numbers in `results.md`, compared with the targets above and the MuJoCo table in
  `status.md`.
- Watch the demo videos before calling it done.
- Fast and slow test suites green and recorded in the pass log.

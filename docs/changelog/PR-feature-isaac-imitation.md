**Title:** Half fold on Isaac Sim: imitation pipeline retrained on PR #15's env (weld baseline, W5) + MuJoCo pipeline

**Base:** `main` ← **compare:** `feature/isaac-imitation`

---

## Summary

- **Imitation pipeline.** This brings the half-fold imitation pipeline (`imitation/`) to `main`, retrained on the
  Isaac Sim / LeHome env from #15. The pipeline is expert demos → diffusion BC → DAgger → a sensor-only student
  → success detector.
- **Isaac results** (weld grasp profile, n = 200 per set, held-out seeds):

| policy | clean | shifted cloth | knocked mid-fold |
|---|---|---|---|
| Privileged (diffusion + DAgger, replan 4) | **99.0%** [96.4, 99.7] | **97.5%** [94.3, 98.9] | 54.5% [47.6, 61.3] |
| Sensor-only (3 cameras + joints/gripper, replan 2) | **99.0%** [96.4, 99.7] | 82.5% [76.6, 87.1] | 31.0% [25.0, 37.7] |

- **Detector:** the success detector agrees with the sim's success check on 98.0% of episodes.
- **Videos:** 6 demo videos, one per policy × benchmark, in `docs/reports/media/half_fold_isaac_*`.
- **Report:** `docs/reports/2026-10-06-isaac-w5.md`.
- **Full change log:** `docs/changelog/2026-10-07-feature-isaac-imitation.md`.

## What's in it
- **`imitation/`:** the MuJoCo pipeline (Phases 1–5c, previously only on `feature/imitation`).
- **Backend switch** (`--backend mujoco|isaac|isaac_weld`). The MuJoCo collection hash is unchanged.
- **`isaac/` additions on #15's env:**
  - a MuJoCo-semantics weld grasp profile (`isaac_weld`)
  - vectorised scene copies (B envs per Kit process, 2.15× throughput)
  - the camera rig
  - the expert port
  - Windows install locks
- **`scripts/isaac_retrain.ps1`:** a resumable end-to-end driver.
- **`scripts/thermal_guard.py` and `imitation/thermal.py`:** cooperative GPU thermal pause.
- **`imitation/verify.py`:** the milestone checks. `python -m imitation.verify W5` passes.

## Not in it / open
- **Friction grasp:** the weld profile stands in for it. The friction-grasp track (`feature/isaac-grasp`) is
  paused.
- **Recovery** after a knock is the weak set. The expert itself recovers only 81.7%.
- **`feat/isaac-half-fold` (Ruby)** is a parallel Isaac half-fold port: anchor grasp, BC 9/10 at n = 10.
  - It touches the same files; a trial merge conflicts in 9 of them.
  - Coordinate which lands first. The two grasp modes can coexist behind flags.

## Test plan
- [x] `pytest -m "not slow"`: 187 passed, 4 skipped
- [x] `pytest -m slow`: 12 passed
- [x] `python -m imitation.verify W5`: PASS (evidence `outputs/imitation/isaac/verify/W5.json`)
- [x] MuJoCo expert benchmark unchanged after the backend switch (I0.1)
- [ ] Reviewer: watch `docs/reports/media/half_fold_isaac_{privileged,sensor}_{id_easy,id_hard,recovery}.mp4`

🤖 Generated with [Claude Code](https://claude.com/claude-code)

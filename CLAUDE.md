# Agent instructions

Read before doing any work in this repo. Applies to every agent and every session.

## Start of a pass
1. Read `docs/status.md` (where we are, next action, blockers), then the relevant milestone
   in `docs/roadmap.md`. Do not start work that isn't the next action without saying why.
2. Use the pinned env: `.venv/Scripts/python.exe` (Windows) / `.venv/bin/python`. Pins are in
   `imitation/requirements.txt`; do not change them without re-running the expert benchmark.

## End of a pass — mandatory, even if the pass was only investigation
1. **Update `docs/status.md`:**
   - bump the `Updated:` date and `Phase:`
   - rewrite "Where we are" / "Next action" if they changed
   - remove fixed rows from "Open blockers", add newly found defects
   - update the test counts in "Environment" if they changed
   - add one line to the **Pass log** (newest first): date · what changed · commit hash
2. Tick finished milestones in `docs/roadmap.md` (✅) — only when the exit artifact exists.
3. Any number you report goes into `docs/results.md` first (append-only, with n, seeds,
   dataset version and Wilson interval). Spec changes go into `docs/imitation.md`.
4. Run `pytest -m "not slow"` (and `-m slow` if you touched the sim/env) and record the
   result in the pass log line.
   Failing on Windows before any change (2026-11-27): `test_rlpd.py::test_learns_toy_placement`, three in
   `test_replay.py` (two slow encoder reads, the grasp-check refusal), one in `test_dagger.py` (renames a rows
   file it still has open).
5. Commit code **and** the doc updates together, so the status never lags the code.

A pass is not finished until status.md reflects it.

## Subagents — use them, sized to the task
Delegate tedious, well-scoped work to subagents, and pick the cheapest model whose reasoning
matches the task's complexity. The lead agent keeps design, merges, debugging of unclear
failures, and the final verification of every milestone. It re-runs the verify commands
itself and never accepts a subagent's "it passes" on trust.

| task shape | model |
|---|---|
| mechanical edits with exact instructions, renames, file listings, log/grep sweeps, doc table updates | `haiku` |
| scoped code + tests against a written spec, read-only code exploration, web research | `sonnet` |
| architecture, cross-module design, ambiguous debugging, code review of risky changes | lead agent / `opus` |

- **Prompt contract.** Every subagent prompt states:
  - the files it may touch
  - what it must not do (no git commit/push, no installs, no long sim jobs unless told)
  - the commands to run
  - the exact output to report
- **Log every delegation** in `docs/subagents.md` (newest first): date · milestone · model ·
  task · outcome · whether the lead re-verified.

## Rules
- The dataset store is write-once. Anything touching the observation or episode schema
  lands before a version is frozen.
- Observation/action sizes live in `imitation/spec.py`; never hard-code them.
- Run outputs (logs, episode records, checkpoints, videos) stay out of git. Only a few showcase videos go in
  `docs/reports/media/`. Delete superseded code in the pass that replaces it.

## Hardware rig (`hardware/so101`)
- Read the run before changing a setting: `runs/<time>/plan.json` and `steps.jsonl` give where each arm was
  sent, the claw's lean, whether the table stopped the fingertip (it ends above the height it was sent to), and
  whether the jaw holds cloth (about 1.5 deg more open than its empty reading).
- A fold from the panel or from `dagger_run.py` saves pictures in `runs/<time>/frames/` (5 a second, times in
  `frames.jsonl`): look at the frame at the pinch row before changing a setting. With `ui.py` stopped,
  `rgbd.Phone()` gives a frame of the table as it is. The phone cannot see the jaws: do not ask for a description.
- The jaw reading is not proof of a hold: on the blue waffle cloth a jaw carrying a corner read within 0.2 deg of
  empty. The pictures show whether the corner lifted.
- Check the cloth fits before tuning. A corner is grabbed only with the claw within about 10 deg of upright
  (inside about 34 cm of the base), landing under 2 cm inside the far edge, on the table. From the base's front
  edge (6.5 cm) that leaves about 27.5 cm of cloth depth.
- The cloth has a window of 3 to 4 cm. Nearer than 7 cm in front of either base is not searched: a near corner
  that reads the same number on every look (9.6 cm on 2026-11-27) is cut off and the place point is wrong. Far
  corners past about 42 cm get the pinch pulled in and missed. A near right corner tucked in (Y near -24 in the
  left frame, not -27) gives `release_short_mm` 0 on the right arm and a crooked fold.
- The jaws catch only at the cloth's edge, fingertips touching it: a pinch landing inside flat cloth, or 2 mm
  high, closes on nothing. A shorter reach sags less and ends higher, so moving the pinch in needs a lower
  `--pinch-height`.
- The arm model's table is not the real one and differs between the arms: `--pinch-height LEFT RIGHT`.
- A moved phone: check each touch picture shows the fingertip before clicking (`--outer 0` when the arm's outer
  side is out of view), and expect `--inset`, `--overshoot` and `--pinch-height` to need setting again.
- `dagger_run.py`: every takeover that changes the offsets (`runs/dagger/nudges.json`) puts the earlier rows
  aside, so a correction on every fold leaves the network one fold of rows. A takeover before the fingertips are
  at the cloth stores a useless offset; one after the pinch is stored as the carry offset, and a carry offset
  unlike the pinch offset is a jump after the grasp check. Put a steady error in the flags, not in nudges.
- `fold_camera.py calibrate`, and the panel's Calibrate, put every saved setting they are not given back to its
  default.
- Stop `ui.py` before running a script: it holds the phone and both serial ports. A moved phone or base means
  new touches (`arm_calibration.py`).
- `pytest hardware/so101` takes about 7 minutes: run the files you touched first.

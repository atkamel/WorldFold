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
- Only a fold started from the panel (`ui.py`, Fold) saves pictures. The phone looks from behind the arms and
  cannot see the jaws: work the jaw out in the arm model, do not ask for a description.
- Check the cloth fits before tuning. A corner is grabbed only with the claw within about 10 deg of upright
  (inside about 34 cm of the base), landing under 2 cm inside the far edge, on the table. From the base's front
  edge (6.5 cm) that leaves about 27.5 cm of cloth depth.
- The arm model's table is not the real one and differs between the arms: `--pinch-height LEFT RIGHT`.
- `fold_camera.py calibrate`, and the panel's Calibrate, put every saved setting they are not given back to its
  default.
- Stop `ui.py` before running a script: it holds the phone and both serial ports. A moved phone or base means
  new touches (`arm_calibration.py`).
- `pytest hardware/so101` takes about 7 minutes: run the files you touched first.

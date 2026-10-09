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
- Don't merge `cloth_fold_rl/requirements.txt` pins with `imitation/requirements.txt`.

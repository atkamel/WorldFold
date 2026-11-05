# Lessons already paid for

## Working
- **Separate measured, published and interpretation.** Overclaiming has cost us real time here.
- **Check what exists before writing a new script.** `grep` this folder first, a lot is already built.
- **Say so at the first fix-of-a-fix.** If a fix needs a fix, stop and rethink.
- **Before paid runs, give floors:** what one stream can do, what parallel can do, where we are now. Real numbers only.
- **Report results as they land.** Have scripts print result lines and have watchers fire on those, not on process exit.
- **A failed search isn't evidence that something doesn't exist.** Follow citations before saying so. A completeness
  check found a cloth solver our earlier survey missed.
- **Show videos, not just scores.** A human watching caught things the numbers hid (trousers).
- **Don't trust one metric's first definition.** Our 2 cm "held" rule said 2/65 grasps held. A lift measure on the same
  data said 64/65 lifted the cloth. Heavy cloth had sagged out of the radius while still on the jaw.

## Physics facts
- **Don't reuse cached cloth poses in Isaac.** PhysX moves the garment prim's frame as cloth moves, so read it live.
  A stale frame gave us wrong hold numbers once.
- **LeHome's shirts are modelled worn,** about 10 cm thick. Stiffening them makes a heap unless you flatten them first
  (`bash docker_run.sh flatten`).
- **SO-101 on the team: 12 V servos.** Jaws close tip-first, only the last 6 to 10 mm can pinch thin cloth, jaw-cotton
  friction is about 0.15 to 0.25 (a proxy value).
- **RL with the fitted arm needs ≥ 10/20 solver iterations,** and the command delay in the env. At 5/2 the gain vanishes.
- **Isaac cloth grasping is widely reported as unreliable.** The usual fixes are an attach volume, barrier/IPC contact,
  or tuned stickiness. Sticky isn't acceptable for demos.
- **Real SO-101 towel folding (arXiv 2607.04591):** 19 of 21 failures were grasps. Grasps are the likely problem on real
  arms too.

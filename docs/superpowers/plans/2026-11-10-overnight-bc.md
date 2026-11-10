# Overnight behaviour cloning on the half fold (2026-11-10)

User, 2026-11-09 night: "We right now need to get working policies as FAST as possible." Working is a gate-valid
fold within 2 cm mean / 3 cm edge, on the stiff cloth (`bend_stiffness` 5e5); GPUs as many as WATcloud allows (the
account caps at 3 at once). Real teleop data is proof of concept only and not used.

## Why behaviour cloning

The scripted place-servo expert on the stiff cloth lands 49% of folds within 2 / 3 cm (gate minus stillness, 160 runs
of batches S-J), so its successes are demonstrations. The imitation pipeline (`imitation/`) already trains diffusion,
chunk-MLP and camera students; its Isaac teacher was the older expert, so the bridge records the place-servo expert's
episodes in the batched precision env instead.

## Pieces (code 1846135 onward)

- `PrecisionBatch` recording hook: each copy's 139-D observation and rig frames before a step and the 12-D action it
  executes, from reset to its snapshot and from the restore to the end of its window episode; the snapshot's frame
  stands in for the first window frame. `drive()` steps copies with a student's actions.
- `isaac/tools/bc_collect.py`: demos are settled, gate-valid (minus stillness) folds within 2 / 3 cm. The expert's
  gentle release sets the jaw's open cap through a side channel its action doesn't carry, so for demos it only commands
  "open" and the env's `jaw_open_rate` 0.02 ramps the jaw, as it will for a student. Rolling frozen versions every 4
  chunks, `--resume`, `--parallel` children per GPU, crash-safe freezing.
- `isaac/tools/bc_train.py`: merges the frozen versions, trains several students in parallel on one GPU.
- `isaac/tools/bc_eval.py`: a student acts alone from reset for a fixed horizon (chunk replan every 4) and is scored
  like the expert, with the full gate; blown-up copies are parked; score trace every 50 steps.
- Cameras: the overhead `top` camera at 128 px plus both wrist cameras at 64 px (the user's planned rig: an overhead
  iPhone and wrist cameras). The old oblique `main` view hid the cloth behind the left arm.

## Schedule

| phase | what |
|---|---|
| smoke | 744887: 8 demos recorded end to end (3 within 2 / 3 cm with the full gate), about 9 min per 8-episode chunk with cameras |
| collect | 2-3 jobs x 2 collectors, seeds 1,001,000-1,002,279 |
| train | from about 02:30: state diffusion, state chunk-MLP, vision (top + wrists), 30k steps each |
| evaluate | eval seeds 2,000,000+, students and the expert on the same seeds, all bars, videos |
| quarter (track Q) | a subagent steers fold 2's right-arm pinch to 25-35° tilt (offline, then one Isaac run) |

Reviewed before launch by an opus subagent (subagents.md); its blocker (the jaw side channel) and should-fixes are in.

## Outcome (2026-11-10 10:30)

1,352 expert episodes recorded (433 training demos), four training rounds, every student evaluated on the same 64
held-out seeds as the expert. Best: the state chunk-MLP on all 433 demos, 30/64 within 2 / 3 cm with the full gate
(46.9% [35.2, 58.9]) against the expert's 21/64; the camera student 11-13/64. Training on the expert's failed episodes
taught the release but not the fold. Details and tables in results.md.

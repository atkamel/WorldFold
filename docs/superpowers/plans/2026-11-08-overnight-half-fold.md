# Overnight half-fold optimisation (2026-11-08)

Goal. Raise the scripted half fold's pass rate at the bar (mean 1.0 cm, edge 1.5 cm, strict gate) on the friction
profile, starting from the audit's kept-jaw base r0 (2.07 / 3.65 cm, 1/32). Then confirm the best on fresh seeds with
video. In parallel, unblock `quarter2` fold 2.

Where the error is (r0, 32 seeds, results.md round 2a). The fold line still forms short: the interior top layer sits a
median 1.8 cm past the bottom in −y (IQR 0.7-2.8), and removing it takes the median mean from 2.14 to 0.92 cm. The worst
edge pair is the open-edge row in 31 of 32 seeds, mostly the right arm's side. The top corners land outside their
partners in x (left −1.5 cm, right +2.3 cm median).

## Rules for every batch

- User, 2026-11-07 22:40 EST. Run until 10:00 EST on 2026-11-08, batch after batch, always the highest-yield next experiments.
  Before launching each new batch, a subagent reviews its proposal and the current state for new errors and unfounded
  assumptions, and the lead acts on what it finds.

- Tuning seeds 910000-31 (paired, 8 copies, window 450). Confirmation on fresh seeds 920000-63. The headline numbers are
  the confirmation ones.
- Score, gate and bar never change. Tables report "scoreable" (strict gate minus stillness) and strict passes with
  Wilson intervals, and paired mean / edge differences with SE.
- A config wins when its paired mean is at least 2 SE better and its edge is no more than 1 SE worse, or the reverse.
  The winner becomes the base of the next batch. With no winner the base stays.
- Each batch includes the current base as a control.
- After each batch, append results.md, update status.md, commit on `fix/half-fold-audit`. No push.
- At most 3 WATcloud jobs at once, per-child timeout 120 min. If SSH fails, retry every 15 min for 2 h, then stop and
  write the summary.
- A config that crashes in our code is fixed (with a test) when the cause is obvious, otherwise dropped.

## Batch 0. Defaults (code, before batch A)

- `KEEP_JAW` behaviour permanent (knob removed). Reset velocity zeroing kept. MuJoCo's jaw armature kept.
- `PLACE_STALL_EXIT` deleted (neutral in both rounds). `Z_FLOOR` stays 5 mm (17 mm cost 1.4 cm).
- `PINCH_HEIGHT` per r6 (742670). The layer-shift correction stays as an off-by-default knob.
- New knob `PLACE_OFFSET = (along, inward)` m: the place target moves `along` toward the crease (back along the carry)
  and `inward` toward the cloth's centre line, per arm. Sim-free tests.

## Batch A. Placement and lay-down shape (2 jobs, 9 configs, about 1.5 h)

On r0. Control, then `PLACE_OFFSET` along 1 / 2 cm, inward 1.5 cm, along 2 + inward 1.5, `GFOLD_SLOPE` 1.0 / 1.6 /
2.0, and layer shift (gain 0.5) with inward 1.5.

## Batch B. Combine (2 jobs, about 8 configs, about 1.5 h)

The batch-A winners together. Then on top of them the layer shift at gain 0.25 with a 1.5 cm cap and at gain 0.5
with a 3 cm cap, one touch-up round, and the r6 pinch height if it won.

## Batch C. Confirmation (2 jobs, about 2.5 h)

The best two configs and r0 on seeds 920000-63 with video. Report pass rates with Wilson intervals and copy the best
passing videos to the Mac (not git).

## Track Q. `quarter2` fold 2 (parallel)

- Q1, offline on the Mac. Re-solve the right arm's fold-2 pinch at the S positions recorded in the traces (742655)
  with PinchIK. Find why the close lands 10-29 cm from S (seed, flipped branch, weights) and fix it, with a test.
- Q2. Rerun the 8-seed trace job on `quarter2` with the fix and the best fold-1 base available by then.

## Morning summary

What won and by how much, the confirmation pass rate, videos, the fold-2 diagnosis, what to try next.

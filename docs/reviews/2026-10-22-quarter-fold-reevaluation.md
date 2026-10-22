# Quarter fold: re-evaluation (2026-10-22)

The user asked for a step back with independent evaluation after the lead twice reported artifacts as near-successes.
Inputs: an independent audit (opus, read-only, re-checked raw grids, code and video frames), a read of Roy's branches
(sonnet), web research (sonnet). The lead re-ran the audit's two key analyses (the fold-1 bound, the two-arm
collision decomposition) and the p8 / base hunt statistics; the numbers below are from those re-runs.

## What holds

- **The two recorded half-fold passes are real flat two-layer folds** (930005 scripted + touch-up: 0.95 / 1.45 cm;
  980008 RL policy p8: 0.43 / 1.13 cm). Grids flat (z 0.424-0.441 m), neighbour spacing at rest length, videos
  continuous across restores. Defects the metric can't see: a 5 mm tuck and a 1.7 cm bulge in 930005, a 1.7-2.7 cm
  ridge on 980008's top open edge.
- Snapshot copy-frame fix, vertex indexing, the half fold's 55 pairs / 19 outline pairs, the RLPD target: correct.
- Fold 1 (two arms) is reliable on the friction layout (8/8 in every quarter run).

## What was wrong

| claim | status | why |
|---|---|---|
| "no two-armed fold 2 is collision-free on any layout" | **refuted** | the search counted both arms' simultaneous home-to-above approach as the fold; for its best C/S layout the fold paths clear the other arm's parked pose by 34 / 68 mm (lead re-run). Those layouts put one arm on the far (north) side of the cloth. |
| one-arm fold 2 slips because of the 38° pinch | **uncertain** | in that run the idle left arm sat ~6 cm from the grasp point and the right gripper pushed it (frames); nothing was lifted (stack rise ≤ 2 cm) |
| "fold 2 drops 5-6/8" | counts right, label wrong | `dropped` fires when the mean of the two grasped vertices is 5 cm from the site: a firmly held top layer with the bottom one left behind, and a missed grasp, both count |
| the drag makes fold 2 near-vertical | only on paper | in Isaac the sheet compressed 3.6-7.3 cm (median), the west middle moved +2-17 cm vs the planned +12, fold 1 degraded 3.0 → 4.1 cm |
| the RL policy's pass | real fold, **no RL gain** | p8 hunt: mean 2.65 cm, 1/64, 23/64 truncated; the scripted base's hunt: 2.62 cm, 1/64, 6 truncated. A best-of-64 draw under 0.5-1 cm per-fold noise |
| sweep 4 "rounds lower the mean 0.39 cm (2 SE)" | overstated | summaries averaged settled, truncated and dragged episodes; settled only: 3 rounds −0.20 (SE 0.19), 5 rounds −0.79 (SE 0.20) on 14/32 |
| Roy's environment shows the quarter fold is possible | **not in Isaac** | his Isaac branch is a teleop rig for the LeHome shirt (single fold + drag); the quarter fold there is our MuJoCo task with a weld grasp |

Also: **no score had a validity gate** (cloth on the table, jaws released, cloth still); a cloth hanging from a gripper
scores 0.0 cm (synthetic test), which is how carried-off sheets scored 4.5-6.3 cm. **Adhesion 0.1** was kept from a
grasp test and never A/B'd for release, while cloth sticking to the fingers keeps recurring.

## What the bar implies

Every quarter-fold group contains two fold-1 pairs, so the quarter score can't beat fold 1's. Over the 688 stored
settled half folds, the share whose fold 1 alone still allows a quarter-fold pass (lead re-run of the audit's
`q5_bound.py`):

| quarter bar (mean / edge, cm) | 1.0 / 1.5 | 1.5 / 2.5 | 2.0 / 3.0 | 2.0 / 4.0 | 2.5 / 5.0 |
|---|---|---|---|---|---|
| fold 1 allows it | 0.4% | 3.5% | 6.8% | 12.2% | 25.4% |

So the strict bar needs a near-perfect fold 1 and a near-perfect fold 2 in the same episode; a "very slight"
relaxation (1.5 / 2.5) still leaves 3.5% before fold 2 adds any error. The literature has no comparable cm bar for a
friction-gripped two-fold (SoftGym uses 1.25 cm mean vertex error with welded pickers).

## Revised plan (for the user's sign-off)

1. **Measurement first.** Gate every score: grid within 2.5 cm of the table, jaws open and grippers ≥ 3 cm clear,
   cloth still (< 1 mm per step), cloth on the table; re-score stored data; per-step stage / grasp traces in quarter
   runs; report settled-only statistics with the ending counts.
2. **Two cheap checks:** adhesion 0 vs 0.1 on the half fold (release and alignment); the one-arm fold 2 without the
   drag and with the idle arm really parked, to settle whether the "slip" was the arm contact.
3. **Main route: a two-arm, two-point fold 2** (the crease end and the corner stack), sequenced approach, on a layout
   from the corrected search (one arm moves to the far side of the cloth), with fold 1 re-verified on that layout. The
   research agrees: every system that does reliable multi-folds re-grasps two points for each fold; a one-point fold 2
   leaves the edge's other end trailing 8-9 cm. Target: scripted fold 2 landing within 3 cm, with videos, before RL.
4. **Then RL** (residual or corrector) on fold 2 from fold-1 snapshots, judged on settled, gated episodes against the
   scripted base on paired seeds; demo hunts with video and the gates.
5. **Bar:** the user picks from the table above.

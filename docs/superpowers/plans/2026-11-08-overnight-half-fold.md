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

## Batch A design (reviewed 2026-11-07 22:50, launched 742699-742701, killed by the WATcloud DNS outage)

On r0. Control, then `PLACE_ALONG` 1 / 2 cm, `PLACE_INWARD` 1.5 cm, along 2 + inward 1.5, `GFOLD_SLOPE` 1.0 / 1.6 /
0.8, and layer shift (gain 0.5) with inward 1.5.

## Batch S. A stiffer cloth (user, 2026-11-08, reviewed 11:35, launched 2026-11-09 as 743600-743603)

The user's hypothesis is that the cloth is too fine, so it wrinkles and folds imprecisely. Isaac's sheet is 101x101
particles at 3 mm (`CLOTH_SUBDIV` 10) with LeHome's springs (bend 5000, shear 5000, damping 10, stretch 1e8) and 50 g in
total, so each particle weighs 4.9 mg. MuJoCo's cloth, the one that folded well with the weld, is the 11x11 grid itself
at 3 cm.

The springs aren't saturated (review, from PhysX 5.6.1's particle solver). PhysX solves each spring as an implicit
spring-damper once per 0.31 ms substep. The share of a length error it removes per substep is about k / (k + c/dt),
32,000 N/m at damping 10, almost independent of the mass: 0.015 at k 500, 0.135 at 5000, 0.61 at 5e4, 0.94 at 5e5.
Damping already removes 99.9% of the relative velocity per substep, so more damping only softens the springs. A rough
static estimate puts curvature under a load at (k h^4)^(-1/3) for spring stiffness k and spacing h, so the batch is one
ordered axis. The 6 mm mesh at LeHome's k sits near the 3 mm mesh at 16x k. Stiff shear on top of stretch 1e8 makes a
paper-like sheet that may wrinkle more when a corner lifts, so the spring configs change bend alone, and two of them
change both. The 6 mm mesh keeps the particle offsets. Neighbours at rest leave 0.76 mm of margin against a particle
passing through a cell (2.88 mm at 3 mm), so the press in place and hold has less room. A 1 cm mesh would let particles
through. Total mass stays 50 g, so each particle weighs 19.2 mg at 6 mm.

Configs on r0 (`LATCH_GOAL=1,GFOLD_SLOPE=1.2`, window 450, seeds 910000-31, 8 copies), knobs in the sweep's 8th field.

- s0 r0 repeat (control, and r0's run-to-run noise), s1 bend 500, s2 bend 5e4
- s3 bend 5e5, s4 bend + shear 5e5, s5 `cloth_subdiv=5`
- s6 `cloth_subdiv=5` + bend 5e5, s7 bend + shear 5e4, s5r s5 repeated

Drape probe (`isaac/tools/drape_probe.py`, a 4th short job, over the 3-job cap for about 40 minutes). Each cloth above
plus bend 5e6 is laid flat with 8 cm over the table edge. The probe measures the free edge's droop after 4 s and reads
back the authored springs, the cooked springs, the particle count and the mass. That tells "the knob didn't take" apart
from "stiffness doesn't matter", and the user can repeat the ruler test on the real cloth.

Judged against s0 only, by the usual rule, with the scoreable counts compared as well as the paired seeds (pairing
hides a config that leaves more cloth on a finger). The paired SE is about 0.13 cm, so the batch sees about 0.35 cm.
With 8 configs a false winner has about a 25% chance, so no cloth is promoted on this batch alone. It needs its repeat
or fresh seeds, and a drape match to the real cloth, because a sim cloth only helps the hardware if the real one is as
stiff. Batch B stays on r0's cloth until then. Extra columns: the folded layers' z gap and the share under 1 mm (4-13%
on round 2a's fine mesh, where the top layer slid off its partner), how much each half's grid edges shrank in plan
view, the reset's flatness, the sheet's move before the carry (3.3 cm on round 2a), drops, interior top-minus-bottom y
and folded corners. The expert was tuned on the current cloth, so a stiffer sheet that loses on this pass may still win
after re-tuning the pinch and place.

## Batch G. The grip on the stiff cloth (2026-11-09, reviewed 10:55, revised)

Batch S: bend 5e5 (s3) passes 10/32 against the control's 0/32, and when both top corners are held the fold is
0.56 / 1.01 cm median. The right arm loses its corner in about half the seeds. Every row starts its carry with both
corners within 5 cm of their sites, so nothing is lost at the close or lift. Ten lost right corners end on the table
15-28 cm from their partners, let go early in the carry. On stiff cloth the right corner rides 2.8 cm up the pads in
the arm's tight close mode (site 12 mm outside the corner), so the 5 cm drop rule may abort carries with the corner
still pinched. The rows can't tell that from a slip, so drops now log their phase, carry waypoint, and the
corner-to-site distance and height (`drop_at`).

Configs on s3's settings, seeds 910000-31: g0 s3 repeated (pooled with s3 as the control, n 64), g1 `HOLD_DIST=0.08`,
g2 `gripper_closed=0.02`, g3 `adhesion=0.2` (also sticks the cloth to the table), g4 `PINCH_INSET=0.008` (outward:
the right arm's tight mode loses fewer corners, 6/18 against 9/14 on s3), g5 `CARRY_SPEED=0.6` (also slows the lift).
`pad_friction` is out, because PhysX takes particle-rigid friction from the particle material only. Videos of 8 seeds
and a cantilever probe follow, then a lean batch B on r0's cloth (b0, b2, b6, b2r), since S doesn't show the stiff
cloth is the real one.

Decision rule, fixed before the results:

- Primary: seeds with both top corners held, where a corner counts as lost at 8 cm or more from its partner or when it
  lies on the table off it. Alongside it, strict passes with Wilson intervals.
- Guard: the median mean and edge over the held seeds.
- A knob is a candidate only at a one-sided Fisher p < 0.01 against the pooled control, then goes to fresh seeds.
  (At n 32 one of five null knobs beats a single control by 4+ passes 45% of the time.)
- Split by the right arm's close mode (tight about −12 mm, loose −3 to −6 mm).

Cantilever probe (`isaac/tools/drape_probe.py`, rewritten): the 8 cm drape in 743603 measured cloth lying on the
table's side face, held there by adhesion 0.1, so it said nothing about stiffness. Now flaps of 1-6 cm with adhesion 0,
read from the full-resolution free edge, count only while they hang 6 mm or more clear of the face, and give the Peirce
bending length c. The same ruler test on the real cloth gives its c.

## Batch B. Floor plus placement offset (3 jobs, 9 configs, about 1.5 h). Runs before batch A

Changed after review (2026-11-07 23:20). Round 2's 17 mm floor (r1) leaves an almost rigid translation of the top layer
(interior −3.44 cm y, corners −2.7 / −3.0, partners barely moving, seed IQR −3.70 to −3.11). Removing one constant
−3.4 cm from r1's grids gives 0.63 / 1.85 cm with 12/32 passing, an upper bound that needs the interior to follow the
corners. On the floor it did follow by about 81% in r3, against 34% on r0. On r0 no constant passes any fold, and the
reviewer's ceiling for batch A's r0 knobs is 1/26. Configs, all on `LATCH_GOAL=1,GFOLD_SLOPE=1.2`:

- b0 `Z_FLOOR=0.017` (r1 repeat, the floor's own noise), b1 + `PLACE_ALONG=0.045`, b2 + 0.035
- b3 + 0.03, b4 + 0.04, b5 `CARRY_FLOOR=0.017` (floor in carry and place only, hold lowers before release)
- b6 `CARRY_FLOOR=0.017` + 0.035, b7 `CARRY_FLOOR=0.017` + 0.03, b2r (b2 repeated)

Winners are judged against r0 (not b0), from the shape of the response across ALONG. Extra report columns: corners
folded back on themselves, the open row's offset next to the interior rows', z / clearance failures, drops, paired n.

## Batch A. Placement and lay-down shape on r0 (reviewed, after B)

As above, if B leaves time; a0 against r0 gives the run-to-run noise.

## Batch C. Confirmation (2 jobs, about 2.5 h)

The best two configs and r0 on seeds 920000-63 with video. Report pass rates with Wilson intervals and copy the best
passing videos to the Mac (not git).

## Track Q. `quarter2` fold 2 (parallel)

- Q1, offline on the Mac. Re-solve the right arm's fold-2 pinch at the S positions recorded in the traces (742655)
  with PinchIK. Find why the close lands 10-29 cm from S (seed, flipped branch, weights) and fix it, with a test.
- Q2. Rerun the 8-seed trace job on `quarter2` with the fix and the best fold-1 base available by then.

## What happened (2026-11-08 10:00)

Nothing ran. WATcloud's internal DNS failed at about 22:55, which killed batch A in the image build (742699-742701).
PyPI resolved again from about 00:15, but `squeue` never reached the SLURM controller (polls every 2-4 min from 02:18 to 09:58,
all `connect failure`). Batch B and the fold-2 rerun are ready to launch on code a528405, from `~/WorldFold` on the
login node after syncing the repo.

```bash
B="LATCH_GOAL=1,GFOLD_SLOPE=1.2"
COMMON=(isaac/rl/precision_rl.py sweep --seeds 910000:910032 --copies 8 --parallel 2 --child-minutes 120)
G=gpu:rtx_3090:1,tmpdisk:200000
sbatch --job-name=wf-nb1 --gres=$G isaac/watcloud.sbatch "${COMMON[@]}" --out outputs/isaac/rl/night_b1 \
    "b0_floor||$B,Z_FLOOR=0.017|450||||" \
    "b1_floor_along45||$B,Z_FLOOR=0.017,PLACE_ALONG=0.045|450||||" \
    "b2_floor_along35||$B,Z_FLOOR=0.017,PLACE_ALONG=0.035|450||||"
sbatch --job-name=wf-nb2 --gres=$G isaac/watcloud.sbatch "${COMMON[@]}" --out outputs/isaac/rl/night_b2 \
    "b3_floor_along30||$B,Z_FLOOR=0.017,PLACE_ALONG=0.03|450||||" \
    "b4_floor_along40||$B,Z_FLOOR=0.017,PLACE_ALONG=0.04|450||||" \
    "b5_carryfloor||$B,CARRY_FLOOR=0.017|450||||"
sbatch --job-name=wf-nb3 --gres=$G isaac/watcloud.sbatch "${COMMON[@]}" --out outputs/isaac/rl/night_b3 \
    "b6_carryfloor_along35||$B,CARRY_FLOOR=0.017,PLACE_ALONG=0.035|450||||" \
    "b7_carryfloor_along30||$B,CARRY_FLOOR=0.017,PLACE_ALONG=0.03|450||||" \
    "b2r_floor_along35||$B,Z_FLOOR=0.017,PLACE_ALONG=0.035|450||||"
# Track Q2, once a slot frees: the 742655 job with TWO_IK
WORLDFOLD_PLACE_PARAMS=$B sbatch --job-name=wf-q2-twoik --gres=$G isaac/watcloud.sbatch isaac/tools/quarter_batch.py \
    --copies 8 --seeds 0:8 --profile quarter2 --fold2 two --drag none --video --out outputs/isaac/quarter/q2_twoik
```

Batch B takes about 1.5-2 h with the image build, the fold-2 rerun about 1 h. r0's rows for the paired comparison
are in `outputs/isaac/rl/audit_r2a/r0_base/`.

## Morning summary

What won and by how much, the confirmation pass rate, videos, the fold-2 diagnosis, what to try next.

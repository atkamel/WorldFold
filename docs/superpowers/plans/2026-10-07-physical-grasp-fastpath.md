# Phase F — faster path to the retrain (re-plan, 2026-10-07 21:10)

## Context

The user asked for an honest roadmap, and for the retrain to start as soon as possible.

### Where we are

- **F1 ✅ — the grasp is physical.** Honesty check: 0 violations over 20 expert episodes. The weld negative control
  fails every check.
- **F2 grasp tuning is essentially done.** Config M (n = 40):
  - held 40/40 on both arms;
  - placed 40/40 and 38/40;
  - 36/40 bench successes.
- **Defaults now:** LeHome gripper drive, jaw +0.05, adhesion 0.1, pad friction 2.0, `ALIGN_TOL` 8 mm.
- **Gate 1** (the previous config, n = 300) held the right arm 93%. Its cause is fixed in M.
- **Running now:**
  - F2 gate 2, 3 × 100 single-env bench episodes at about 50 s each: **about 2.5 h on both Isaac lanes**. It just
    started, at 3 rows.
  - The anchor comparison smoke test on the CPU, n = 10, about 17 min.
  - Nothing important is mid-way; both can be stopped safely, and both resume.

### Honest cost model

- **Friction episodes are about twice as long as the weld's.** Successes run 160–205 steps against about 90. Failures
  run to the 400-step cap.
- **W5's stage times on the weld** (2 × 4 envs):
  - collect 75 min;
  - BC 36 min;
  - 1 DAgger round 64 min;
  - privileged finals 102 min;
  - vision finals 125 min.
- **Scaled for friction:** about **15–17 h of wall time for F4** if run as written. Most of it is evaluation.

### Honest risks

- **The full-task expert is measured only at n = 20:** 18/20 id_easy. Recovery with a physical re-grasp may land
  below the weld's 81.7%, and that ceiling caps every student.
- **The sensor-only student's re-grasp is the hardest part:** 115/138 of its W5 failures were G1.
- **The friction numbers use LeHome's task sets.** That's forced by the pinch's reach. They are not directly
  comparable to the weld's W5 numbers.

## Plan: cut gates to what informs decisions, overlap everything, cheaper selection

1. **Stop F2 gate 2 now.**
   - The grasp gate moves into F3's expert evaluation, which is the real task, vectorised and about 3× faster.
   - F2 closes on M's n = 40 plus the F3 evaluation's grasp-success and G2 rates. The verify F2 check is rewritten
     to read those. The roadmap records the trade.
   - Gate 2's partial rows are kept.
2. **Run the F3 expert gate immediately** with `scripts/f3_gate.ps1 -Lane eval` (2 × 4, n = 100 per set, about
   70–90 min).
3. **Start F4 collection as soon as F3's id_easy and id_hard are in.** Don't wait for the recovery set.
   - If the expert's recovery is low, the collection's 50% knocked episodes still teach recovery.
   - Failures go to the failures dataset either way.
4. **Fill idle Isaac lanes during GPU-only training:**
   - check_resync (sequential, about 4 h) runs only in idle windows. It no longer gates F3, because the recovery set
     measures the same resync behaviour.
   - The anchor rows run at n = 20 per set.
   - Demo rendering.
5. **Cheaper selection in F4:**
   - **BC:** one seed only. W5 has already measured seed variance at 12 pp.
   - **DAgger selection:** n = 100 on id_hard and recovery only. id_easy is saturated at about 99%, so it is checked
     once at the end.
   - **DAgger rounds:** run round 2 only if round 1 gains 1 SE or more.
   - **Finals** stay at n = 200 × 3, the honest benchmark.
6. **F5 screening:** each A/B screens at n = 100 on id_hard and recovery. Only winners get the n = 200 × 3 final.
   - Priority order:
     1. privileged: intervention-frame weighting, plus more DAgger rounds;
     2. student: student-DAgger, plus the corner auxiliary head;
     3. image augmentation.
7. **Compute: 2 × 8 envs per process** (user approved, 2026-10-07). V measured about +30% throughput.
   - The thermal guard still runs, at 90 / 84 °C.
   - **VRAM:** 3 × 8 with rig cameras overflowed the 16 GB GPU. If 2 × 8 with cameras runs out of memory during
     collect or the vision finals, fall back to 2 × 6 for that stage only. Check `nvidia-smi` on the first stage.
   - Update the cpu-heat-budget memory and the status.md heat-cap note.
8. **F2 closed by the lead's judgement** (user: "if you think F2 is complete and good, move to F3"):
   - M's n = 40 holds 40/40 on both arms.
   - Its fix targets the measured right-arm mechanism.
   - F3's n = 300 full-task grasp and G2 rates are the confirmation. If F3 shows G2 slips above about 5%, return to
     F2.

### Honest timeline (local, 2 × 4 cap, after the cuts)

| step | wall time | done by (if started 21:15) |
|---|---|---|
| F3 expert eval (n = 100 × 3) | ~1.5 h | ~22:45 |
| F4 collect 400 eps | ~2.5 h | ~01:15 |
| BC seed 0 (+ check_resync / anchor in the Isaac lanes) | ~40 min | ~02:00 |
| DAgger r1 (128 rollouts + 15k steps + n = 100 × 2 sets) | ~2 h | ~04:00 |
| DAgger r2 (only if r1 gains) | ~2 h | ~06:00 |
| vision student (GPU) during the privileged finals | — | — |
| privileged finals n = 200 × 3 | ~3.5 h | ~09:30 |
| vision finals n = 200 × 3 (replan 2, cameras) | ~4 h | ~13:30 |
| detector, demos, F4 report rows | ~1 h | ~14:30 tomorrow |
| F5 A/Bs (3–4 screens + finals for the winners) | ~1–1.5 days | |
| F6 report | ~2 h | |

- **With 2 × 8 envs (approved)** the sim-bound rows shrink by about 25%:
  - F3 is about 1.1 h;
  - F4 is about 11 h, so the finals land around 08:30–10:00 tomorrow;
  - F5 follows.
- Modal was not chosen.

## Files to change

| file | change |
|---|---|
| `scripts/f4_retrain.ps1` | `-EnvsPerProc 8` default; one BC seed; DAgger `--eval-sets id_hard recovery`, `--rounds 1`, then round 2 conditional; start collect behind a flag that waits for F3 id_hard |
| `scripts/f3_gate.ps1` | `WORLDFOLD_ISAAC_ENVS_PER_PROC = 8` |
| `imitation/verify.py` | F2 reads M's run plus F3's grasp_success / G2; F3 makes check_resync informational |
| `docs/roadmap.md`, `docs/status.md`, `docs/results.md` | record the trims and the gate-2 stop |
| `isaac/fold_expert.py`, `isaac/isaac_env.py` | no change; M is the config |

## Verification

- `verify F1` (PASS) and `verify F2`/`F3` after the rewrite.
- Fast tests are run after each script change.
- Each F4 stage's STEP / DONE markers go to `f4.log` under a monitor.
- Finals are at n = 200 × 3 with Wilson intervals, recorded in results.md before any report.

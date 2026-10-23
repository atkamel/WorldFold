# Port Roy's grasp into our Isaac env (plan, 2026-10-19)

**Goal:** the perpendicular quarter fold on `IsaacClothFoldEnv`, using the mechanisms that make Roy's folds work
(`origin/ROY-vla-teacher`, inventoried 2026-10-18), then the fold-move RL for alignment. We port pieces; we don't merge
his branch (decided with the user, 2026-10-19).

**Rules:** every piece is a profile setting or expert attribute that is off by default, so the half-fold profiles
and frozen datasets stay reproducible. Each step is measured before the next is stacked on it. Numbers go into
results.md. Long runs on WATcloud (tmpdisk 200 GB, Kit hang detector off); Modal only for short checks.

## Steps

| # | what | sim-free test | Isaac check | pass bar |
|---|---|---|---|---|
| 1 | **Flat box pads**: `pads="box"` in `lab_scene` turns off the capsule colliders and adds Roy's pads (`pads.py`: 4 mm thick, 16 × 22 mm, at the jaw tips); closed target moves to −0.17 rad | the USD edit lists the right prims and box sizes (Isaac-venv test) | `grasp_bench` config M on the friction layout, n = 20 per arm | held ≥ 19/20 per arm (today 40/40 with the lining) |
| 2 | **Soft orientation + Roy's layout**: pinch orientation weight 0.005 (an `IsaacArmExpert` attribute); profile `roy` with his base poses (incl. the USD offset), bases ~1 cm above the table, cloth ~26 cm out, his drop | `layout_search.plan_move` on the `roy` profile: every corner grasp and place reachable within the collision bounds (tilt unbounded) | `grasp_bench` on `roy`, n = 20 per arm | held ≥ 18/20 per arm at the tilts Roy uses (30–46°) |
| 3 | **Press-and-slide grasp, slow release**: approach above grasp − 4 cm along the jaw axis, press 1.3 cm below contact, slide 4 cm, roll so the moving finger leads, close over ~0.1 s, hold; release = open ramp, hold, back off 2 cm, lift ~3 mm per step | waypoint generator: slide start, press height, roll choice, release path (unit tests on fixed inputs) | `grasp_bench` on single layer, and on the two-layer stack left by a half fold | single layer held ≥ 18/20; stack lifted with both layers (gap < 1 cm) ≥ 15/20 |
| 4 | **Drag + quarter-fold plan**: a drag move (slide grasp, flat pull at table + 1.2 cm); `layout_search` re-run with the new grasp model to choose fold 1's direction, the drag target for the half sheet and fold 2's grasps | search returns a plan whose every move passes the collision bounds | scripted quarter fold (`quarter_fold_run.py`), n = 10, with traces and video | both folds complete ≥ 5/10; alignment logged as the baseline |
| 5 | **RL for alignment**: `fold_rl.py` with the lessons of run strip_r2 (exploration std 0.08, a failed fold dominates the reward, 48 episodes per update), on WATcloud | existing `test_fold_rl.py` plus the reward change | learning curve; best version evaluated n = 20 with video | alignment better than step 4's baseline; one recorded episode meets the bar (mean ≤ 1.0 cm, edges ≤ 1.5 cm) |

**Stop rules:** if step 3 can't lift both layers of a stack (≥ 15/20), stop and report before building step 4, since the
second fold depends on it. If step 1 or 2 lowers holding, A/B the two remaining differences (CPU device vs `cuda:0`,
LeHome arm drives vs our MuJoCo-matched drives) with the same bench before going on.

**Timing:** our control step is 0.05 s and Roy's tick is 1/90 s, so his tick counts are scaled by time (his 8-tick
close ≈ 2 of our steps; his 0.7 mm/tick lift ≈ 3 mm per step).

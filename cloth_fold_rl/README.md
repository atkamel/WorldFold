# cloth_fold_rl

Fold-task wrappers around `mujoco_sim/sim_main.py`'s `ClothFoldEnv`, the scripted experts that drive them, and the
fold alignment score. The Isaac env (`isaac/sim/isaac_env.py`) has the same API, so the wrappers and experts here run
on both simulators.

| file | what it holds |
|---|---|
| `fold_env.py` | `SingleCornerFoldEnv`, the one-corner fold task (and the physical-grabber variant in `physical_env.py`) |
| `quarter_fold_env.py` | `QuarterFoldEnv` and `HalfFoldEnv`, the fold stages (`STAGES`, `Move`, `Stage`) and grid vertex names |
| `expert.py`, `quarter_fold_expert.py` | the scripted IK expert per arm and the two-arm fold expert built from it |
| `physical_expert.py` | the scoop expert for the physical grabber |
| `scripted_half_fold.py` | an open-loop joint-space half fold (hardcoded waypoints) |
| `fold_alignment.py` | how well the layers line up, per fold and over the outline (mean and edge, against the bar) |
| `train.py` | PPO on the one-corner task |

The ClothFoldEnv defaults are unchanged, so `cloth_angles/` collects data against the stock env. The quarter-fold
wrapper passes it two options, `grasp_corners` (several grasp vertices per gripper) and `grasp_radius`.

The PPO and behaviour-cloning results for the one-corner task, their checkpoints and the run, record and demo
scripts were removed in the October 2026 cleanup. Tag `pre-cleanup-2026-10-22` has them.

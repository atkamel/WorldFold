"""Observation / action sizes, importable without the simulator. See docs/imitation.md §2-3."""

OBS_DIM = 139
ACTION_DIM = 12

# Observation subsets for the privileged-features ablation (imitation.md 2.1, roadmap M2.3).
# Dims outside a subset are zeroed after normalization, so every policy keeps OBS_DIM inputs.
_SENSOR_PROPRIO = [i for i in range(50) if i not in (24, 49)]     # 24/49: sim grasp-weld flags
_CORNERS_AND_TASK = list(range(50, 62)) + list(range(120, 132)) + [137, 138]  # corner pos, goals, stage/settle
OBS_SUBSETS = {
    "full": list(range(OBS_DIM)),
    "proprio": _SENSOR_PROPRIO,
    "proprio_corners": _SENSOR_PROPRIO + _CORNERS_AND_TASK,
}

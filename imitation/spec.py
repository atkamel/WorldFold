"""Observation / action sizes, importable without the simulator. See docs/imitation.md §2-3."""

OBS_DIM = 139
ACTION_DIM = 12

# Obs v2 (Phase Q2): the quarter fold appends, per arm (left, right), the point its current move carries (the mean of
# the carried vertices), that move's goal, and goal - point -- 18 dims after the 139. The 139 only show the four
# corners, and the second fold's right arm carries vertex 5 (the crease end), which no corner slot holds.
CARRY_OBS_DIM = 18
OBS_DIM_QF = OBS_DIM + CARRY_OBS_DIM
CARRY_SLICE = slice(OBS_DIM, OBS_DIM_QF)


def obs_dim(task="half_fold"):
    """Observation width of a task's env and datasets."""
    return OBS_DIM_QF if task == "quarter_fold" else OBS_DIM

# proprio dims of the gripper actuator command (left, right); 1.0 = open, below 0.5 = closing or closed
GRIPPER_CTRL_DIMS = (10, 35)

# Observation subsets for the privileged-features ablation (imitation.md 2.1, roadmap M2.3).
# Dims outside a subset are zeroed after normalization, so every policy keeps OBS_DIM inputs.
_SENSOR_PROPRIO = [i for i in range(50) if i not in (24, 49)]     # 24/49: sim grasp-weld flags
_CORNERS_AND_TASK = list(range(50, 62)) + list(range(120, 132)) + [137, 138]  # corner pos, goals, stage/settle
OBS_SUBSETS = {
    "full": list(range(OBS_DIM)),
    "proprio": _SENSOR_PROPRIO,
    "proprio_corners": _SENSOR_PROPRIO + _CORNERS_AND_TASK,
}

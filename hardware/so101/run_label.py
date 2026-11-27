"""Whether a recorded run folded, as the control panel writes it and rig_dataset.py reads it: label.json in the run's
folder. Also the count of labelled runs of one kind under runs/.
"""

import json
import shutil
from pathlib import Path

LABEL = "label.json"        # rig_dataset.LABEL
WORDS = ["good", "bad", "discard"]


def label_of(folder):
    """"good", "bad" or None for a run folder with no label."""
    file = Path(folder) / LABEL
    if not file.exists():
        return None
    if json.loads(file.read_text())["success"]:
        return "good"
    return "bad"


def set_label(folder, word):
    """Label the run in `folder` "good" or "bad", or "discard" it, which deletes the folder. False for another word."""
    if word == "discard":
        shutil.rmtree(folder)
        return True
    if word == "good" or word == "bad":
        label = {"success": word == "good", "note": None}
        (Path(folder) / LABEL).write_text(json.dumps(label))
        return True
    return False


def is_teleop(meta):
    """A recording from the panel's record button."""
    return meta.get("task") == "teleop"


def is_panel_fold(meta):
    """A scripted fold played from the panel's Fold button."""
    return meta.get("source") == "control panel" and meta.get("mode") == "fold"


def count(runs_dir, wanted):
    """How many runs under `runs_dir` are labelled good, bad, or not at all. `wanted(meta)` says whether a run's
    meta.json is of the kind being counted."""
    counts = {"good": 0, "bad": 0, "unlabelled": 0}
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return counts
    for folder in runs_dir.iterdir():
        meta_file = folder / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
            label = label_of(folder)
        except (OSError, ValueError, KeyError):
            continue    # a folder being written or damaged is not counted
        if not wanted(meta):
            continue
        if label is None:
            counts["unlabelled"] += 1
        else:
            counts[label] += 1
    return counts

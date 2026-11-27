"""Record a teleop session from the control panel (ui.py) with recorder.Recorder.

Each tick saves one step (every arm's measured and target joints) and, when the phone has a
frame, one camera frame (color and LiDAR depth). Joints are raw encoder steps (4096 per turn)
in motor id order 1 to 6; each arm's twin.json turns them into angles.
"""

import json
import shutil
import threading
import time
from pathlib import Path

import recorder

RECORD_HZ = 10
RUNS_DIR = Path(__file__).parent / "runs"  # same folder as fold_live.py's runs; ignored by git
LABEL = "label.json"        # rig_dataset.LABEL


def label_of(folder):
    """"good", "bad" or None for a run folder with no label."""
    file = folder / LABEL
    if not file.exists():
        return None
    if json.loads(file.read_text())["success"]:
        return "good"
    return "bad"


def count_episodes(runs_dir):
    """How many teleop recordings under `runs_dir` are labelled good, bad, or not at all."""
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
            continue    # a folder being written or damaged is not an episode
        if meta.get("task") != "teleop":
            continue
        if label is None:
            counts["unlabelled"] += 1
        else:
            counts[label] += 1
    return counts


def record_tick(run, seconds, row, arms_by_name, phone):
    target = {}
    measured = {}
    for name in arms_by_name:
        present_joints, target_joints = arms_by_name[name].joints()
        measured[name] = present_joints
        target[name] = target_joints
    run.step(seconds, row, target, measured)

    if phone is None:
        return
    frame = phone.latest()
    if frame is not None:
        run.frame(seconds, frame.rgb, frame.depth, frame.K)


class TeleopRecording:
    """Start and stop one recording at a time; a thread samples at RECORD_HZ while it runs."""

    def __init__(self, arms_by_name, phone, runs_dir=RUNS_DIR):
        self.arms_by_name = arms_by_name
        self.phone = phone
        self.runs_dir = runs_dir
        self.lock = threading.Lock()
        self.run = None
        self.thread = None
        self.stop_requested = threading.Event()
        self.rows = 0
        self.last = None        # the folder of the recording stopped last, until it is discarded
        self.counts = count_episodes(self.runs_dir)

    def start(self, correction=False):
        """`correction`: the recording is a person taking over from a policy that went wrong."""
        with self.lock:
            if self.run is not None:
                return
            meta = {
                "task": "teleop",
                "arms": list(self.arms_by_name),
                "record_hz": RECORD_HZ,
                "joint_units": "encoder steps, 4096 per turn, motor ids 1 to 6",
            }
            if correction:
                meta["correction"] = True
            self.run = recorder.Recorder(self.runs_dir, meta)
            self.rows = 0
            self.stop_requested.clear()
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def loop(self):
        period = 1.0 / RECORD_HZ
        start = time.perf_counter()
        next_tick = start
        while not self.stop_requested.is_set():
            record_tick(self.run, time.perf_counter() - start, self.rows, self.arms_by_name, self.phone)
            self.rows += 1
            next_tick += period
            wait = next_tick - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                # A slow tick: carry on from now instead of bunching up the missed ones.
                next_tick = time.perf_counter()

    def stop(self):
        with self.lock:
            if self.run is None:
                return
            self.stop_requested.set()
            self.thread.join()
            self.run.close()
            self.last = self.run.path
            self.run = None
            self.counts = count_episodes(self.runs_dir)

    def label(self, word):
        """Say how the recording stopped last went: "good" or "bad" (what rig_dataset.py reads), or "discard",
        which deletes its folder. False when there is none to label, one is being recorded, or the word is unknown."""
        with self.lock:
            if self.run is not None or self.last is None:
                return False
            if word == "discard":
                shutil.rmtree(self.last)
                self.last = None
            elif word == "good" or word == "bad":
                label = {"success": word == "good", "note": None}
                (self.last / LABEL).write_text(json.dumps(label))
            else:
                return False
            self.counts = count_episodes(self.runs_dir)
            return True

    def status(self):
        with self.lock:
            last = None
            last_label = None
            if self.last is not None:
                last = self.last.name
                last_label = label_of(self.last)
            status = {"recording": False, "steps": 0, "folder": None, "last": last, "last_label": last_label,
                      "counts": dict(self.counts)}
            if self.run is not None:
                status["recording"] = True
                status["steps"] = self.rows
                status["folder"] = str(self.run.path)
            return status

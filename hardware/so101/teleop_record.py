"""Record a teleop session from the control panel (ui.py) with recorder.Recorder.

Each tick saves one step (every arm's measured and target joints) and, when the phone has a
frame, one camera frame (color and LiDAR depth). Joints are raw encoder steps (4096 per turn)
in motor id order 1 to 6; each arm's twin.json turns them into angles.
"""

import threading
import time
from pathlib import Path

import recorder

RECORD_HZ = 10
RUNS_DIR = Path(__file__).parent / "runs"  # same folder as fold_live.py's runs; ignored by git


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

    def start(self):
        with self.lock:
            if self.run is not None:
                return
            meta = {
                "task": "teleop",
                "arms": list(self.arms_by_name),
                "record_hz": RECORD_HZ,
                "joint_units": "encoder steps, 4096 per turn, motor ids 1 to 6",
            }
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
            self.run = None

    def status(self):
        with self.lock:
            if self.run is None:
                return {"recording": False, "steps": 0, "folder": None}
            return {"recording": True, "steps": self.rows, "folder": str(self.run.path)}

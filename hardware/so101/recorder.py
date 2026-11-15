"""Record teleop episodes: joint positions of every arm plus the camera's color and depth.

An episode is a folder:
    meta.json     arms, record rate, units
    steps.jsonl   one line per step: time, each arm's present and target joints, the frame files
    rgb/000000.jpg, depth/000000.npy   the camera frame of that step (depth: float32 meters)

Joints are raw encoder steps (4096 per turn) in motor id order 1 to 6.
"""

import json
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

RECORD_HZ = 10
JPEG_QUALITY = 90
# Ignored in .gitignore: recorded episodes stay out of git.
EPISODES_DIR = Path(__file__).parent.parent.parent / "outputs" / "so101_teleop"


class EpisodeWriter:
    def __init__(self, folder, arm_names, record_hz):
        self.folder = Path(folder)
        self.steps = 0
        (self.folder / "rgb").mkdir(parents=True)
        (self.folder / "depth").mkdir()
        meta = {
            "arms": list(arm_names),
            "record_hz": record_hz,
            "joint_units": "encoder steps, 4096 per turn, motor ids 1 to 6",
            "depth_units": "meters",
            "intrinsics": "fx, fy, cx, cy of the color image",
        }
        (self.folder / "meta.json").write_text(json.dumps(meta, indent=2))
        self.steps_file = open(self.folder / "steps.jsonl", "w")

    def write_step(self, seconds, joints, frame):
        """joints: {arm name: {"present": [6], "target": [6]}}.
        frame: None, or (rgb HxWx3 uint8, depth HxW float32, (fx, fy, cx, cy))."""
        step = {"t": seconds}
        for name in joints:
            step[name] = joints[name]

        step["rgb"] = None
        step["depth"] = None
        step["intrinsics"] = None
        if frame is not None:
            rgb, depth, intrinsics = frame
            rgb_name = "rgb/{:06d}.jpg".format(self.steps)
            depth_name = "depth/{:06d}.npy".format(self.steps)
            Image.fromarray(rgb).save(self.folder / rgb_name, quality=JPEG_QUALITY)
            np.save(self.folder / depth_name, depth)
            step["rgb"] = rgb_name
            step["depth"] = depth_name
            step["intrinsics"] = list(intrinsics)

        self.steps_file.write(json.dumps(step) + "\n")
        # Flushed every step so a crash keeps everything recorded so far.
        self.steps_file.flush()
        self.steps += 1

    def close(self):
        self.steps_file.close()


class Recorder:
    """Samples the arms and the camera at RECORD_HZ on its own thread while recording."""

    def __init__(self, arms, camera):
        self.arms = arms
        self.camera = camera
        self.lock = threading.Lock()
        self.writer = None
        self.thread = None
        self.stop_requested = threading.Event()

    def start(self):
        with self.lock:
            if self.writer is not None:
                return
            folder = EPISODES_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
            self.writer = EpisodeWriter(folder, list(self.arms), RECORD_HZ)
            self.stop_requested.clear()
            self.thread = threading.Thread(target=self.loop, args=(self.writer,), daemon=True)
            self.thread.start()

    def loop(self, writer):
        period = 1.0 / RECORD_HZ
        start = time.perf_counter()
        next_tick = start
        while not self.stop_requested.is_set():
            joints = {}
            for name in self.arms:
                joints[name] = self.arms[name].joints()
            frame = None
            if self.camera is not None:
                frame = self.camera.latest()
            writer.write_step(time.perf_counter() - start, joints, frame)

            next_tick += period
            wait = next_tick - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                # Saving took longer than a tick: skip the missed ticks instead of bunching up.
                next_tick = time.perf_counter()

    def stop(self):
        with self.lock:
            if self.writer is None:
                return
            self.stop_requested.set()
            self.thread.join()
            self.writer.close()
            self.writer = None

    def status(self):
        with self.lock:
            if self.writer is None:
                return {"recording": False, "steps": 0, "folder": None}
            return {"recording": True, "steps": self.writer.steps, "folder": str(self.writer.folder)}

"""Saves everything about one robot run to a folder: camera frames with depth, the joint readings and targets of every
control step, and events. Frames go through a writer thread so the control loop never waits on the disk (a full queue
drops the frame and counts it); steps and events are written at once and flushed, so a killed run keeps its data.

    with Recorder("outputs/hardware/runs", meta={"task": "half_fold"}) as rec:
        rec.frame(t, rgb, depth, K=[fx, fy, cx, cy])        # rgb uint8 [H, W, 3], depth float32 metres
        rec.step(t, row, target={"left": q6}, measured={"left": q6})
        rec.event(t, "grasp", arm="left", jaw=0.12)
        rec.save("plan.json", plan)
    run = load(rec.path)                                    # meta, summary, steps, events, frames (the jsonl rows)
    rgb, depth = load_frame(rec.path, 0)                    # depth in metres, NaN where the camera had no reading

Folder layout: meta.json, summary.json, steps.jsonl, events.jsonl, frames.jsonl, frames/000000.jpg, frames/000000.npz.
"""

import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def _plain(x):
    """json.dumps default: numpy arrays and scalars as lists and numbers."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON serialisable: {type(x)}")


def _floats(q):
    return [round(float(v), 5) for v in q]


class Recorder:
    """One run in root/<YYYYmmdd_HHMMSS>/ (a suffix _2, _3 ... if that folder exists)."""

    def __init__(self, root, meta=None, queue_size=64):
        stem = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        n = 1
        while True:
            path = root / (stem if n == 1 else f"{stem}_{n}")
            try:
                path.mkdir()
                break
            except FileExistsError:
                n += 1
        self.path = path
        (path / "frames").mkdir()
        self._dump("meta.json", {**(meta or {}), "started": datetime.now().isoformat()})
        self._files = {k: open(path / f"{k}.jsonl", "w") for k in ("frames", "steps", "events")}
        self._count = {"frames": 0, "dropped": 0, "steps": 0, "events": 0}
        self._queued = 0
        self._error = None
        self._closed = False
        self._queue = queue.Queue(queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def frame(self, t, rgb, depth=None, K=None, **extra):
        """Queue a frame for the writer thread; dropped (and counted) if the queue is full."""
        row = {"i": self._queued, "t": t, "K": None if K is None else [float(v) for v in K],
               "depth": depth is not None, **extra}
        try:
            self._queue.put_nowait((row, np.array(rgb), None if depth is None else np.array(depth)))
        except queue.Full:
            self._count["dropped"] += 1
            return
        self._queued += 1

    def step(self, t, row, target, measured, **extra):
        self._line("steps", {"t": t, "row": row, "target": {a: _floats(q) for a, q in target.items()},
                             "measured": {a: _floats(q) for a, q in measured.items()}, **extra})

    def event(self, t, name, **data):
        self._line("events", {"t": t, "name": name, **data})

    def save(self, name, obj):
        """A dict or list as JSON, a PIL image via .save, an ndarray via np.save."""
        if isinstance(obj, np.ndarray):
            with open(self.path / name, "wb") as f:
                np.save(f, obj)
        elif isinstance(obj, Image.Image):
            obj.save(self.path / name)
        else:
            self._dump(name, obj)

    def close(self):
        """Flush the queue, stop the writer, write summary.json; re-raises the writer's first exception."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
            for f in self._files.values():
                f.close()
            self._dump("summary.json", {**self._count, "ended": datetime.now().isoformat()})
        if self._error is not None:
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _dump(self, name, obj):
        (self.path / name).write_text(json.dumps(obj, indent=1, default=_plain))

    def _line(self, kind, row):
        f = self._files[kind]
        f.write(json.dumps(row, default=_plain) + "\n")
        f.flush()
        self._count[kind] += 1

    def _run(self):
        while (item := self._queue.get()) is not None:
            if self._error is not None:
                continue    # keep draining so close() does not wait on a full queue
            try:
                self._write(*item)
            except Exception as e:
                self._error = e

    def _write(self, row, rgb, depth):
        name = f"{row['i']:06d}"
        Image.fromarray(rgb).save(self.path / "frames" / f"{name}.jpg", quality=90)
        if depth is not None:
            mm = np.nan_to_num(depth.astype(np.float64) * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
            mm[mm <= 0] = 0
            mm = np.clip(np.round(mm), 0, 65535).astype(np.uint16)
            np.savez_compressed(self.path / "frames" / f"{name}.npz", depth_mm=mm)
        f = self._files["frames"]
        f.write(json.dumps(row, default=_plain) + "\n")
        f.flush()
        self._count["frames"] += 1


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(path):
    """{"meta", "summary" (None if the run was killed), "steps", "events", "frames"}."""
    path = Path(path)
    summary = path / "summary.json"
    return {"meta": json.loads((path / "meta.json").read_text()),
            "summary": json.loads(summary.read_text()) if summary.exists() else None,
            "steps": _read_jsonl(path / "steps.jsonl"), "events": _read_jsonl(path / "events.jsonl"),
            "frames": _read_jsonl(path / "frames.jsonl")}


def load_frame(path, i):
    """(rgb uint8 [H, W, 3], depth float32 metres [h, w] with NaN where there is no reading, or None)."""
    base = Path(path) / "frames" / f"{i:06d}"
    rgb = np.asarray(Image.open(base.with_suffix(".jpg")).convert("RGB"))
    npz = base.with_suffix(".npz")
    if not npz.exists():
        return rgb, None
    with np.load(npz) as z:
        mm = z["depth_mm"]
    return rgb, np.where(mm == 0, np.nan, mm / 1000.0).astype(np.float32)

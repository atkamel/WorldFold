import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
from PIL import Image

import recorder
from recorder import Recorder, load, load_frame


def gradient(h=48, w=64):
    x = np.linspace(0, 255, w)[None, :].repeat(h, 0)
    y = np.linspace(0, 255, h)[:, None].repeat(w, 1)
    return np.stack([x, y, (x + y) / 2], -1).astype(np.uint8)


def test_roundtrip(tmp_path):
    rgb = gradient()
    depths = []
    rec = Recorder(tmp_path, meta={"task": "x"})
    for i in range(5):
        d = None
        if i < 3:
            d = np.full((12, 16), 0.5 + 0.1 * i, np.float32)
            d[0, 0] = np.nan
            d[1, 1] = 0.0
            d[2, 2] = -1.0
        depths.append(d)
        rec.frame(0.1 * i, rgb, d, K=[1.0, 2.0, 3.0, 4.0] if d is not None else None, cam="top")
    for i in range(4):
        rec.step(0.05 * i, i, {"left": np.arange(6) / 7}, {"left": [0.123456789] * 6}, phase="a")
    rec.event(0.0, "start")
    rec.event(0.2, "grasp", q=np.array([1.0, 2.0]), n=np.int64(3))
    rec.save("plan.json", {"a": np.arange(3)})
    rec.save("snap.png", Image.fromarray(rgb))
    rec.close()
    rec.close()

    run = load(rec.path)
    assert run["meta"]["task"] == "x" and "started" in run["meta"]
    assert run["summary"]["frames"] == 5 and run["summary"]["dropped"] == 0
    assert run["summary"]["steps"] == 4 and run["summary"]["events"] == 2
    assert [f["i"] for f in run["frames"]] == list(range(5))
    assert run["frames"][0]["K"] == [1.0, 2.0, 3.0, 4.0] and run["frames"][4]["K"] is None
    assert run["frames"][0]["depth"] and not run["frames"][4]["depth"] and run["frames"][0]["cam"] == "top"
    assert len(run["steps"]) == 4 and run["steps"][1]["target"]["left"][1] == round(1 / 7, 5)
    assert run["steps"][0]["measured"]["left"][0] == 0.12346 and run["steps"][2]["phase"] == "a"
    assert run["events"][1]["q"] == [1.0, 2.0] and run["events"][1]["n"] == 3
    assert (rec.path / "plan.json").exists() and (rec.path / "snap.png").exists()

    for i in range(5):
        img, depth = load_frame(rec.path, i)
        assert img.shape == rgb.shape and img.dtype == np.uint8
        assert np.abs(img.astype(float) - rgb).mean() < 3
        if depths[i] is None:
            assert depth is None
            continue
        bad = np.isnan(depths[i]) | (depths[i] <= 0)
        assert depth.dtype == np.float32 and np.array_equal(np.isnan(depth), bad)
        assert np.abs(depth[~bad] - depths[i][~bad]).max() <= 0.001


def test_full_queue_drops_without_blocking(tmp_path, monkeypatch):
    gate = threading.Event()
    real = Recorder._write

    def slow(self, *a):
        gate.wait(5)
        real(self, *a)

    monkeypatch.setattr(Recorder, "_write", slow)
    rec = Recorder(tmp_path, queue_size=1)
    t0 = time.time()
    for i in range(5):
        rec.frame(i, gradient())
    assert time.time() - t0 < 0.5
    gate.set()
    rec.close()
    s = load(rec.path)["summary"]
    assert s["dropped"] >= 3 and s["frames"] + s["dropped"] == 5


def test_killed_run_keeps_lines(tmp_path):
    rec = Recorder(tmp_path)
    rec.step(0.0, 0, {"a": [0] * 6}, {"a": [0] * 6})
    rec.event(0.0, "go")
    run = load(rec.path)
    assert run["summary"] is None and len(run["steps"]) == 1 and run["events"][0]["name"] == "go"
    rec.close()


def test_same_second_gets_new_folder(tmp_path):
    a, b = Recorder(tmp_path), Recorder(tmp_path)
    assert a.path != b.path
    a.close()
    b.close()


def test_writer_exception_is_raised_on_close(tmp_path, monkeypatch):
    def boom(self, *a):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Recorder, "_write", boom)
    rec = Recorder(tmp_path)
    rec.frame(0.0, gradient())
    with pytest.raises(RuntimeError, match="disk full"):
        rec.close()


def test_module_exports():
    assert callable(recorder.load) and callable(recorder.load_frame)

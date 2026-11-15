import json

import numpy as np
from PIL import Image

from hardware.so101.recorder import EpisodeWriter


def make_joints():
    return {
        "arm1": {"present": [1, 2, 3, 4, 5, 6], "target": [1, 2, 3, 4, 5, 7]},
        "arm2": {"present": [11, 12, 13, 14, 15, 16], "target": [11, 12, 13, 14, 15, 16]},
    }


def read_steps(folder):
    lines = (folder / "steps.jsonl").read_text().splitlines()
    steps = []
    for line in lines:
        steps.append(json.loads(line))
    return steps


def test_step_with_a_camera_frame_saves_color_depth_and_joints(tmp_path):
    rgb = np.zeros((8, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    depth = np.linspace(0.2, 1.4, 12, dtype=np.float32).reshape(4, 3)
    writer = EpisodeWriter(tmp_path / "episode", ["arm1", "arm2"], record_hz=10)

    writer.write_step(0.5, make_joints(), (rgb, depth, (100.0, 101.0, 3.0, 4.0)))
    writer.close()

    steps = read_steps(tmp_path / "episode")
    assert len(steps) == 1
    step = steps[0]
    assert step["t"] == 0.5
    assert step["arm1"]["target"] == [1, 2, 3, 4, 5, 7]
    assert step["arm2"]["present"] == [11, 12, 13, 14, 15, 16]
    assert step["intrinsics"] == [100.0, 101.0, 3.0, 4.0]

    saved_depth = np.load(tmp_path / "episode" / step["depth"])
    np.testing.assert_array_equal(saved_depth, depth)
    saved_rgb = np.asarray(Image.open(tmp_path / "episode" / step["rgb"]))
    assert saved_rgb.shape == (8, 6, 3)


def test_step_without_a_camera_frame_still_records_joints(tmp_path):
    writer = EpisodeWriter(tmp_path / "episode", ["arm1", "arm2"], record_hz=10)

    writer.write_step(0.0, make_joints(), None)
    writer.close()

    step = read_steps(tmp_path / "episode")[0]
    assert step["rgb"] is None
    assert step["depth"] is None
    assert step["arm1"]["present"] == [1, 2, 3, 4, 5, 6]
    assert list((tmp_path / "episode" / "rgb").iterdir()) == []


def test_steps_are_numbered_in_order(tmp_path):
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    writer = EpisodeWriter(tmp_path / "episode", ["arm1"], record_hz=10)

    for index in range(3):
        writer.write_step(index * 0.1, {"arm1": {"present": [0] * 6, "target": [0] * 6}}, (rgb, depth, (1.0, 1.0, 0.0, 0.0)))
    writer.close()

    steps = read_steps(tmp_path / "episode")
    assert writer.steps == 3
    assert steps[2]["depth"] == "depth/000002.npy"
    meta = json.loads((tmp_path / "episode" / "meta.json").read_text())
    assert meta["arms"] == ["arm1"]
    assert meta["record_hz"] == 10

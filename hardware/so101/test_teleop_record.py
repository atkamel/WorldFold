"""teleop_record with stand-in arms and a FakePhone: what one tick saves, and a start/stop round trip."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import recorder
import rgbd
import teleop_record


class StubArm:
    def __init__(self, present, target):
        self.present = present
        self.target = target

    def joints(self):
        return self.present, self.target


def make_arms():
    return {
        "arm1": StubArm([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 9]),
        "arm2": StubArm([11, 12, 13, 14, 15, 16], [11, 12, 13, 14, 15, 16]),
    }


def make_phone():
    rgb = np.zeros((8, 6, 3), dtype=np.uint8)
    depth = np.full((4, 3), 0.5, dtype=np.float32)
    phone = rgbd.FakePhone([rgbd.Frame(0.0, rgb, depth, (100.0, 100.0, 3.0, 4.0))])
    phone.wait()
    return phone


def test_a_tick_saves_every_arms_joints_and_the_camera_frame(tmp_path):
    with recorder.Recorder(tmp_path) as run:
        teleop_record.record_tick(run, 0.25, 0, make_arms(), make_phone())

    saved = recorder.load(run.path)
    step = saved["steps"][0]
    assert step["measured"]["arm2"] == [11, 12, 13, 14, 15, 16]
    assert step["target"]["arm1"] == [1, 2, 3, 4, 5, 9]
    assert len(saved["frames"]) == 1
    rgb, depth = recorder.load_frame(run.path, 0)
    assert rgb.shape == (8, 6, 3)
    np.testing.assert_allclose(depth, 0.5, atol=1e-3)


def test_a_tick_without_a_camera_still_saves_joints(tmp_path):
    with recorder.Recorder(tmp_path) as run:
        teleop_record.record_tick(run, 0.0, 0, make_arms(), None)

    saved = recorder.load(run.path)
    assert len(saved["steps"]) == 1
    assert saved["frames"] == []


def take(recording, correction=False):
    """Record for a moment. Returns the run folder."""
    recording.start(correction)
    time.sleep(0.25)
    folder = Path(recording.status()["folder"])
    recording.stop()
    return folder


def test_the_last_recording_is_labelled_good_or_bad_and_counted(tmp_path):
    recording = teleop_record.TeleopRecording(make_arms(), make_phone(), tmp_path)
    assert recording.label("good") is False         # nothing recorded yet
    folder = take(recording)
    status = recording.status()
    assert status["last"] == folder.name
    assert status["counts"] == {"good": 0, "bad": 0, "unlabelled": 1}
    assert recording.label("good") is True
    assert json.loads((folder / "label.json").read_text())["success"] is True
    assert recording.status()["counts"] == {"good": 1, "bad": 0, "unlabelled": 0}
    assert recording.label("bad") is True
    assert json.loads((folder / "label.json").read_text())["success"] is False
    assert recording.status()["last_label"] == "bad"
    assert recording.label("perfect") is False


def test_discard_deletes_the_last_recording_and_no_other(tmp_path):
    recording = teleop_record.TeleopRecording(make_arms(), make_phone(), tmp_path)
    first = take(recording)
    recording.label("good")
    second = take(recording)
    assert recording.label("discard") is True
    assert not second.exists()
    assert first.exists()
    assert recording.status()["last"] is None
    assert recording.label("discard") is False
    assert first.exists()


def test_a_recording_in_progress_is_not_labelled(tmp_path):
    recording = teleop_record.TeleopRecording(make_arms(), make_phone(), tmp_path)
    first = take(recording)
    recording.start()
    assert recording.label("discard") is False
    recording.stop()
    assert first.exists()


def test_the_count_covers_teleop_recordings_already_in_the_folder(tmp_path):
    earlier = teleop_record.TeleopRecording(make_arms(), make_phone(), tmp_path)
    take(earlier)
    earlier.label("good")
    with recorder.Recorder(tmp_path, meta={"task": "half_fold"}):
        pass                                        # a scripted fold's run is not an episode of teleop
    recording = teleop_record.TeleopRecording(make_arms(), make_phone(), tmp_path)
    assert recording.status()["counts"] == {"good": 1, "bad": 0, "unlabelled": 0}


def test_a_takeover_is_marked_as_a_correction(tmp_path):
    recording = teleop_record.TeleopRecording(make_arms(), make_phone(), tmp_path)
    folder = take(recording, correction=True)
    assert json.loads((folder / "meta.json").read_text())["correction"] is True


def test_start_then_stop_leaves_a_finished_run(tmp_path):
    recording = teleop_record.TeleopRecording(make_arms(), make_phone(), tmp_path)
    assert recording.status()["recording"] is False

    recording.start()
    time.sleep(0.35)
    assert recording.status()["recording"] is True
    folder = recording.status()["folder"]
    recording.stop()

    assert recording.status()["recording"] is False
    saved = recorder.load(folder)
    assert saved["summary"] is not None
    assert len(saved["steps"]) >= 2

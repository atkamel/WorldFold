"""panel_camera with a stand-in phone: the status must say when frames are not arriving."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import panel_camera


class StubPhone:
    def __init__(self):
        self.count = 0

    def latest(self):
        return None


def connected_camera():
    camera = panel_camera.Camera()
    camera.phone = StubPhone()
    return camera


def test_before_the_phone_connects_the_status_is_not_streaming():
    camera = panel_camera.Camera()
    assert "streaming" not in camera.status(now=0.0)
    assert camera.latest() is None


def test_connected_without_any_frame_is_not_reported_as_streaming():
    camera = connected_camera()
    assert camera.status(now=0.0) != "streaming"
    assert "no frames" in camera.status(now=0.0)


def test_arriving_frames_are_reported_as_streaming():
    camera = connected_camera()
    camera.phone.count = 5
    assert camera.status(now=0.0) == "streaming"
    camera.phone.count = 9
    assert camera.status(now=10.0) == "streaming"


def test_frames_that_stop_arriving_are_reported():
    camera = connected_camera()
    camera.phone.count = 5
    assert camera.status(now=0.0) == "streaming"
    assert camera.status(now=panel_camera.STALE_SECONDS + 1.0) != "streaming"

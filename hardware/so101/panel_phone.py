"""The phone's pictures for a script that runs while the control panel (ui.py) holds the phone: the panel serves its
latest frame at /frame.npz and PanelPhone fetches it, with the count of frames the phone has sent so far.

PanelPhone has what camera_run.py asks of rgbd.Phone: count, latest() and wait().
"""

import io
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import rgbd

POLL_SECONDS = 0.05     # how often wait() asks the panel again


def pack(frame, count):
    """The bytes the panel sends for `frame` (rgbd.Frame), the phone having sent `count` frames so far."""
    buffer = io.BytesIO()
    np.savez(buffer, rgb=frame.rgb, depth=frame.depth, K=np.array(frame.K, float), count=np.array(count))
    return buffer.getvalue()


class PanelPhone:
    def __init__(self, url):
        self.url = url
        self.count = 0      # frames the phone had sent when the panel was asked last
        self.seen = 0       # the count of the frame last returned
        self.frame = None

    def start(self):
        pass

    def fetch(self):
        """Ask the panel for its latest frame. No picture yet (404) leaves the count where it was."""
        try:
            with urllib.request.urlopen(self.url, timeout=2) as answer:
                data = np.load(io.BytesIO(answer.read()))
        except urllib.error.HTTPError:
            return
        self.count = int(data["count"])
        self.frame = rgbd.Frame(time.monotonic(), data["rgb"], data["depth"], tuple(data["K"].tolist()))

    def latest(self):
        self.fetch()
        if self.count == 0:
            return None
        self.seen = self.count
        return self.frame

    def wait(self, timeout=5.0):
        """The first frame the phone sends after the one returned last. Exits if none comes in `timeout` s."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.fetch()
            if self.count > self.seen:
                self.seen = self.count
                return self.frame
            time.sleep(POLL_SECONDS)
        sys.exit("The panel sent no new picture. Unlock the phone and bring the Record3D app to the front.")

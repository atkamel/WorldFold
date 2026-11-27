"""The iPhone LiDAR for the control panel (ui.py). Optional: the arms run without it.

Camera keeps looking for the phone until Record3D accepts the connection, and status() says
what is missing: no phone, the app not serving, or connected but no frames arriving.
"""

import time

import rgbd

RETRY_SECONDS = 5   # how often to look for the iPhone until it connects
STALE_SECONDS = 2   # no new frame for this long means the stream has stopped

NO_FRAMES_HINT = "keep Record3D open on the phone with USB Streaming mode on and not recording"


class Camera:
    def __init__(self):
        self.phone = None
        self.message = "looking for the iPhone"
        self.last_count = 0
        self.last_change = None

    def connect_loop(self):
        while self.phone is None:
            phone = rgbd.Phone()
            try:
                phone.start()
            except ImportError:
                self.message = "record3d is not installed"
                return
            except SystemExit as reason:  # rgbd.Phone reports a missing or idle phone by exiting
                self.message = str(reason)
                time.sleep(RETRY_SECONDS)
                continue
            self.phone = phone

    def status(self, now=None):
        if now is None:
            now = time.monotonic()
        if self.phone is None:
            return self.message

        count = self.phone.count
        if count == 0:
            return "connected but no frames yet: " + NO_FRAMES_HINT
        if count != self.last_count or self.last_change is None:
            self.last_count = count
            self.last_change = now
        if now - self.last_change > STALE_SECONDS:
            # Record3D's library cannot reconnect inside one process, so this needs a panel restart.
            return "frames stopped arriving: " + NO_FRAMES_HINT + ", then restart the panel"
        return "streaming"

    def latest(self):
        if self.phone is None:
            return None
        return self.phone.latest()

    def count(self):
        """Frames the phone has sent so far: a script reading the panel's picture tells a new one by it."""
        if self.phone is None:
            return 0
        return self.phone.count

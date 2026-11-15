"""The iPhone LiDAR as a camera: the latest color and depth frame from Record3D over USB.

The phone needs Record3D open with USB Streaming mode on and NOT recording: the red record
button pauses the USB stream. Windows needs Apple Mobile Device Support.
"""

import threading

import numpy as np
from record3d import Record3DStream


class IphoneCamera:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.status = "not started"
        self.session = None

    def start(self, device_index=0):
        """Connect to the phone. Returns True when it is streaming; self.status says why not."""
        devices = Record3DStream.get_connected_devices()
        if len(devices) <= device_index:
            self.status = "no iPhone over USB: plug it in, unlock it and tap Trust"
            return False

        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        connected = self.session.connect(devices[device_index])
        if connected is False:
            self.status = "Record3D is not serving USB: open the app, turn on USB Streaming mode, and do not record"
            return False
        self.status = "connected, waiting for the first frame"
        return True

    def on_new_frame(self):
        # Called on the library's thread. Its buffers are reused, so copy.
        depth = np.array(self.session.get_depth_frame(), dtype=np.float32)
        rgb = np.array(self.session.get_rgb_frame())
        coeffs = self.session.get_intrinsic_mat()
        intrinsics = (coeffs.fx, coeffs.fy, coeffs.tx, coeffs.ty)
        with self.lock:
            self.frame = (rgb, depth, intrinsics)
            self.status = "streaming"

    def on_stream_stopped(self):
        with self.lock:
            self.frame = None
            self.status = "the phone stopped streaming"

    def latest(self):
        """None, or (rgb HxWx3 uint8, depth HxW float32 meters, (fx, fy, cx, cy) of the color image)."""
        with self.lock:
            return self.frame

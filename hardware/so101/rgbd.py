"""RGB plus LiDAR depth from an iPhone (Record3D over USB), and depth turned into height above the table.

    phone = rgbd.Phone(); phone.start()
    f = phone.wait()                                       # Frame: f.rgb, f.depth, f.K
    plane = rgbd.fit_plane(rgbd.points(f.depth, f.K, f.rgb.shape))        # once, with the table in view
    h = rgbd.height(f.depth, f.K, f.rgb.shape, plane)      # metres above the table, at the depth resolution
    rgbd.flat(h, f.rgb.shape)                              # True where the table is clear (an arm is not there)

Setup is as in fold_camera.record3d_frame: the app open in USB Streaming mode and not recording. FakePhone replays a
list of Frames through the same interface, for tests and recorded runs. Camera frame: x right, y down, z forward.
"""

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

MIN_DEPTH, MAX_DEPTH = 0.1, 3.0   # m: LiDAR readings outside this range are dropped (as hardware/iphone_lidar)


@dataclass
class Frame:
    t: float            # time.monotonic() when it arrived
    rgb: np.ndarray     # [H, W, 3] uint8
    depth: np.ndarray   # [h, w] float32 metres, same view at lower resolution; NaN or 0 where there is no reading
    K: tuple            # (fx, fy, cx, cy) for the RGB resolution


class Phone:
    """Record3D over USB: the latest frame.

    The stream's callback only counts the frame and wakes waiters; the arrays are read and copied on the caller's
    thread, as usb_viewer.py reads them from the main thread after its event fires.
    """

    def __init__(self, device=0):
        self.device = device
        self.session = None
        self.cond = threading.Condition()
        self.count = 0      # frames announced by the callback
        self.seen = 0       # the count of the frame last returned

    def start(self):
        from record3d import Record3DStream
        devices = Record3DStream.get_connected_devices()
        if len(devices) <= self.device:
            sys.exit(f"No iPhone at index {self.device}. Plug it in, unlock it, open Record3D.")
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_frame
        self.session.on_stream_stopped = lambda: None
        if self.session.connect(devices[self.device]) is False:
            sys.exit("Record3D is not serving USB. Turn on USB Streaming mode in its settings and stop any recording.")

    def _on_frame(self):
        with self.cond:
            self.count += 1
            self.cond.notify_all()

    def _read(self):
        s = self.session
        k = s.get_intrinsic_mat()
        return Frame(time.monotonic(), np.array(s.get_rgb_frame()), np.array(s.get_depth_frame(), dtype=np.float32),
                     (k.fx, k.fy, k.tx, k.ty))

    def latest(self):
        with self.cond:
            if self.count == 0:
                return None
            self.seen = self.count
        return self._read()

    def wait(self, timeout=5.0):
        with self.cond:
            if not self.cond.wait_for(lambda: self.count > self.seen, timeout):
                sys.exit("Record3D sent no new frame.")
            self.seen = self.count
        return self._read()

    def stop(self):
        self.session = None


class FakePhone:
    """The same interface over a list of Frames: each wait() returns the next one, the last one repeats."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0          # frames returned so far

    def start(self):
        pass

    def latest(self):
        return self.frames[self.i - 1] if self.i else None

    def wait(self, timeout=5.0):
        self.i = min(self.i + 1, len(self.frames))
        return self.frames[self.i - 1]

    def stop(self):
        pass


def points(depth, K, rgb_shape):
    """[h, w, 3] camera-frame points for the depth image; K is for the RGB resolution. Invalid depth gives NaN."""
    h, w = depth.shape
    s = w / rgb_shape[1]
    fx, fy, cx, cy = (v * s for v in K)
    d = depth.astype(np.float64)
    with np.errstate(invalid="ignore"):
        d = np.where((d > MIN_DEPTH) & (d <= MAX_DEPTH), d, np.nan)
    v, u = np.mgrid[:h, :w]
    return np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d], axis=2)


def fit_plane(pts, iters=300, tol=0.008, rng=None):
    """Plane (n, d) with n . p + d = 0 on the table: RANSAC over the valid points, refined on the inliers.
    n is a unit normal with the camera on its positive side (d > 0)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    p = pts.reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 100:
        raise ValueError(f"only {len(p)} valid points, need 100")
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        inliers = np.abs(p @ n - n @ a) < tol
        if inliers.sum() > best_n:
            best, best_n = inliers, inliers.sum()
    q = p[best]
    mean = q.mean(axis=0)
    n = np.linalg.svd(q - mean, full_matrices=False)[2][-1]      # smallest singular direction of the inliers
    d = -n @ mean
    return (n, d) if d > 0 else (-n, -d)


def height(depth, K, rgb_shape, plane):
    """[h, w] signed distance above the plane (metres, positive toward the camera), NaN where depth is invalid."""
    n, d = plane
    return points(depth, K, rgb_shape) @ n + d


def _upsample(m, rgb_shape):
    rows = np.arange(rgb_shape[0]) * m.shape[0] // rgb_shape[0]
    cols = np.arange(rgb_shape[1]) * m.shape[1] // rgb_shape[1]
    return m[rows][:, cols]


def flat(height_map, rgb_shape, max_height=0.02):
    """Mask at the RGB resolution: True where the surface is within max_height of the table or unknown (NaN)."""
    with np.errstate(invalid="ignore"):
        return _upsample(~(height_map > max_height), rgb_shape)


def lifted(height_map, rgb_shape, min_height=0.02):
    """Mask at the RGB resolution: True where the height is known and above min_height."""
    with np.errstate(invalid="ignore"):
        return _upsample(height_map > min_height, rgb_shape)

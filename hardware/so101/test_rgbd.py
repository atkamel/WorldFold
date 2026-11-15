import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rgbd  # noqa: E402

RGB, DEPTH = (720, 960, 3), (192, 256)
K = (700.0, 700.0, 480.0, 360.0)
TILT = np.radians(50)
N_TRUE = np.array([0.0, -np.sin(TILT), -np.cos(TILT)])
D_TRUE = 0.7 * np.cos(TILT)     # the table passes through (0, 0, 0.7)
BOX = (slice(60, 110), slice(90, 237))      # 50 x 147 of 192 x 256 = 15%


def scene(box=True, noise=0.0, seed=1):
    """Depth image of the tilted table, optionally with a 10 cm tall box, and Gaussian depth noise."""
    s = DEPTH[1] / RGB[1]
    v, u = np.mgrid[:DEPTH[0], :DEPTH[1]]
    rays = np.stack([(u - K[2] * s) / (K[0] * s), (v - K[3] * s) / (K[1] * s), np.ones(DEPTH)], axis=2)
    h = np.zeros(DEPTH)
    if box:
        h[BOX] = 0.10
    depth = (h - D_TRUE) / (rays @ N_TRUE)
    depth += np.random.default_rng(seed).normal(0, noise, DEPTH)
    return depth.astype(np.float32)


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


def test_points_rays_and_invalid():
    depth = np.full(DEPTH, 2.0, np.float32)
    depth[0, 1] = np.nan
    depth[0, 2] = 0.0
    depth[0, 3] = 5.0
    p = rgbd.points(depth, K, RGB)
    s = DEPTH[1] / RGB[1]
    assert np.allclose(p[100, 200], [(200 - K[2] * s) * 2 / (K[0] * s), (100 - K[3] * s) * 2 / (K[1] * s), 2.0])
    assert np.allclose(p[96, 128, :2], [0.0, 0.0], atol=1e-6)   # the principal point looks straight ahead
    assert np.isnan(p[0, 1:4]).all()


def test_fit_plane_with_noise_and_box():
    n, d = rgbd.fit_plane(rgbd.points(scene(noise=0.005), K, RGB))
    assert d > 0
    assert angle_deg(n, N_TRUE) < 1.0
    assert abs(d - D_TRUE) < 0.003


def test_fit_plane_too_few_points():
    with pytest.raises(ValueError):
        rgbd.fit_plane(np.full((5, 5, 3), np.nan))


def test_height_table_and_box():
    depth = scene(noise=0.005)
    plane = rgbd.fit_plane(rgbd.points(depth, K, RGB))
    h = rgbd.height(depth, K, RGB, plane)
    on_box = np.zeros(DEPTH, bool)
    on_box[BOX] = True
    assert abs(np.median(h[~on_box])) < 0.01
    assert abs(np.median(h[on_box]) - 0.10) < 0.015


def masks(depth):
    plane = rgbd.fit_plane(rgbd.points(depth, K, RGB))
    h = rgbd.height(depth, K, RGB, plane)
    return rgbd.flat(h, RGB), rgbd.lifted(h, RGB)


def box_region():
    """Box and table pixels at the RGB resolution, 2 pixels in from the box edge."""
    r = RGB[0] / DEPTH[0]
    box = np.zeros(RGB[:2], bool)
    box[int(60 * r) + 2:int(110 * r) - 2, int(90 * r) + 2:int(237 * r) - 2] = True
    table = np.ones(RGB[:2], bool)
    table[int(60 * r) - 2:int(110 * r) + 2, int(90 * r) - 2:int(237 * r) + 2] = False
    return box, table


def test_flat_and_lifted_masks():
    depth = scene(noise=0.002)
    depth[150:160, 10:20] = np.nan
    flat, lifted = masks(depth)
    box, table = box_region()
    assert flat.shape == lifted.shape == RGB[:2]
    assert not flat[box].any() and lifted[box].all()
    assert flat[table].mean() > 0.999 and lifted[table].mean() < 0.001
    hole = (slice(150 * 720 // 192 + 2, 160 * 720 // 192 - 2), slice(10 * 960 // 256 + 2, 20 * 960 // 256 - 2))
    assert flat[hole].all() and not lifted[hole].any()


def frames(n):
    return [rgbd.Frame(float(i), np.zeros(RGB, np.uint8), np.full(DEPTH, i + 1.0, np.float32), K) for i in range(n)]


def test_fakephone_order_and_repeat():
    fs = frames(3)
    phone = rgbd.FakePhone(fs)
    phone.start()
    assert phone.latest() is None
    assert [phone.wait() for _ in range(5)] == [fs[0], fs[1], fs[2], fs[2], fs[2]]
    assert phone.latest() is fs[2]
    phone.stop()


class FakeStream:
    devices = ["dev"]
    refuse = False
    last = None

    def __init__(self):
        FakeStream.last = self
        self.depth = np.full((192, 256), 1.5, np.float32)
        self.rgb = np.zeros((720, 960, 3), np.uint8)
        self.intr = types.SimpleNamespace(fx=700.0, fy=701.0, tx=480.0, ty=360.0)

    @staticmethod
    def get_connected_devices():
        return FakeStream.devices

    def connect(self, device):
        return not FakeStream.refuse

    def get_depth_frame(self):
        return self.depth

    def get_rgb_frame(self):
        return self.rgb

    def get_intrinsic_mat(self):
        return self.intr


@pytest.fixture
def fake_record3d(monkeypatch):
    FakeStream.devices, FakeStream.refuse = ["dev"], False
    monkeypatch.setitem(sys.modules, "record3d", types.SimpleNamespace(Record3DStream=FakeStream))


def test_phone_frame(fake_record3d):
    phone = rgbd.Phone()
    phone.start()
    assert phone.latest() is None
    FakeStream.last.on_new_frame()
    f = phone.wait(timeout=1.0)
    assert f.K == (700.0, 701.0, 480.0, 360.0)
    assert f.rgb.shape == (720, 960, 3) and f.depth.shape == (192, 256) and f.depth.dtype == np.float32
    assert f.depth is not FakeStream.last.depth
    assert phone.latest() is not None
    with pytest.raises(SystemExit):     # no frame newer than the one wait returned
        phone.wait(timeout=0.05)
    FakeStream.last.on_new_frame()
    phone.wait(timeout=1.0)
    phone.stop()


def test_phone_start_exits(fake_record3d):
    FakeStream.devices = []
    with pytest.raises(SystemExit):
        rgbd.Phone().start()
    FakeStream.devices, FakeStream.refuse = ["dev"], True
    with pytest.raises(SystemExit):
        rgbd.Phone().start()

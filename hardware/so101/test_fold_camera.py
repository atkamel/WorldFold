"""fold_camera without a phone or an arm: synthetic frames of a dark cloth on a light table through a known camera."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
try:
    import scservo_sdk  # noqa: F401
except ImportError:      # replay imports it; only validate() is used. An empty stand-in must not hide the real SDK
    sys.modules["scservo_sdk"] = types.ModuleType("scservo_sdk")

import fold_camera  # noqa: E402
import replay  # noqa: E402

SHAPE = (480, 640)
SIDE, NEAR_EDGE, SPACING = 0.30, 0.08, 0.30


def camera():
    """Pixel-to-table map of a camera behind the arms: the table's near part across the bottom, far part narrower."""
    table = [(-0.40, 0.0), (0.40, 0.0), (0.40, 0.60), (-0.40, 0.60)]
    px = [(40, 460), (600, 460), (500, 60), (140, 60)]
    a, b = [], []
    for (u, v), (x, y) in zip(px, table):
        a += [[u, v, 1, 0, 0, 0, -x * u, -x * v], [0, 0, 0, u, v, 1, -y * u, -y * v]]
        b += [x, y]
    return np.append(np.linalg.solve(a, b), 1).reshape(3, 3)


def cloth(shift=(0.0, 0.0), turn_deg=0.0):
    """Table corners [near-left, near-right, far-right, far-left] of the cloth, moved from the ruler position."""
    c, s = np.cos(np.radians(turn_deg)), np.sin(np.radians(turn_deg))
    square = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) * SIDE / 2
    return square @ np.array([[c, s], [-s, c]]) + (shift[0], NEAR_EDGE + SIDE / 2 + shift[1])


def render(corners, discs=()):
    xy, _ = fold_camera.table_grid(camera(), SHAPE)
    dark = np.ones(SHAPE, bool)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        dark &= (b[0] - a[0]) * (xy[..., 1] - a[1]) - (b[1] - a[1]) * (xy[..., 0] - a[0]) >= 0
    for (x, y), r in discs:
        dark |= (xy[..., 0] - x) ** 2 + (xy[..., 1] - y) ** 2 <= r ** 2
    return np.where(dark[..., None], 30, 200).astype(np.uint8).repeat(3, axis=2)


BASES = (((-0.15, 0.02), 0.05), ((0.15, 0.02), 0.05))        # the arms' dark bases, behind the search region


@pytest.fixture(scope="module")
def cal():
    return fold_camera.calibrate(render(cloth()), SIDE, NEAR_EDGE)


def test_look_finds_a_moved_and_turned_cloth(cal):
    truth = cloth(shift=(0.02, 0.015), turn_deg=8.0)
    corners, _ = fold_camera.look(render(truth, BASES), cal)
    for name, want in zip(("near_left", "near_right", "far_right", "far_left"), truth):
        assert np.linalg.norm(corners[name] - want) < 0.004


def test_look_refuses_an_outline_with_an_arm_on_it(cal):
    with pytest.raises(ValueError, match="not a 30 cm cloth"):
        fold_camera.look(render(cloth(), [((0.19, 0.36), 0.06)]), cal)
    with pytest.raises(ValueError, match="no dark cloth"):
        fold_camera.look(np.full((*SHAPE, 3), 200, np.uint8), cal)


def test_fold_points_are_in_each_arms_frame():
    corners = dict(zip(("near_left", "near_right", "far_right", "far_left"), cloth()))
    points = fold_camera.fold_points(corners, SPACING)
    d = fold_camera.INSET / np.sqrt(2)
    assert np.allclose(points["left"], ((0.38 - d, -d), (0.08 - d, -d)))       # forward, left of the left arm
    assert np.allclose(points["right"], ((0.38 - d, d), (0.08 - d, d)))


def test_plan_pinches_where_the_cloth_is(cal):
    truth = cloth(shift=(-0.02, -0.01), turn_deg=-6.0)
    traj, corners, _, points = fold_camera.plan(render(truth, BASES), cal, SPACING, -0.17)
    replay.validate(traj)
    for side in ("left", "right"):
        right, forward, up = traj["arms"][side]["tip"][traj["marks"]["pinch"] + 1]
        assert np.allclose((forward, -right), points[side][0], atol=0.004) and up < 0.008
        assert traj["source"][side]["pinch_error_mm"] < 6


def test_cli_calibrates_then_looks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fold_camera, "CAMERA", tmp_path / "camera.json")
    before, after = tmp_path / "ruler.png", tmp_path / "moved.png"
    Image.fromarray(render(cloth())).save(before)
    Image.fromarray(render(cloth(shift=(0.01, 0.0), turn_deg=5.0), BASES)).save(after)
    assert fold_camera.main(["calibrate", "--image", str(before), "--out", str(tmp_path / "a.png")]) == 0
    assert fold_camera.main(["look", "--image", str(after), "--out", str(tmp_path / "b.png")]) == 0
    assert (tmp_path / "b.png").exists() and "far_left" in capsys.readouterr().out


def test_depth_mask_takes_an_arm_off_the_outline(cal):
    arm = [((0.19, 0.36), 0.06)]                                   # a dark arm over the far right corner
    rgb = render(cloth(), arm)
    standing = (rgb != render(cloth())).any(axis=2)                # what rgbd.flat would mark as above the table
    with pytest.raises(ValueError):
        fold_camera.look(rgb, cal)
    corners, _ = fold_camera.look(rgb, cal, keep=~standing)
    assert np.linalg.norm(corners["far_right"] - cloth()[2]) < 0.004


def test_near_edge_follows_the_bottom_layer(cal):
    grid = fold_camera.search_region(cal)
    moved = cloth()
    moved[:2, 1] += 0.02
    slope0, y0 = fold_camera.near_edge(render(cloth(), BASES), cal, grid, (-0.12, 0.12))
    slope1, y1 = fold_camera.near_edge(render(moved, BASES), cal, grid, (-0.12, 0.12))
    assert abs(y0 - NEAR_EDGE) < 0.004 and abs((y1 - y0) - 0.02) < 0.004 and abs(slope0) < 0.02 and abs(slope1) < 0.02
    assert fold_camera.near_edge(np.full((*SHAPE, 3), 200, np.uint8), cal, grid, (-0.12, 0.12)) is None


# --- a light blue cloth among darker things ---

def render_blue(corners, discs=()):
    """A light blue cloth on a grey table, with dark discs (a laptop, a bag) that are darker than the cloth."""
    xy, _ = fold_camera.table_grid(camera(), SHAPE)
    cloth_here = np.ones(SHAPE, bool)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        cloth_here &= (b[0] - a[0]) * (xy[..., 1] - a[1]) - (b[1] - a[1]) * (xy[..., 0] - a[0]) >= 0
    image = np.full((*SHAPE, 3), 200, np.uint8)
    image[cloth_here] = (90, 150, 210)
    for (x, y), r in discs:
        disc_here = (xy[..., 0] - x) ** 2 + (xy[..., 1] - y) ** 2 <= r ** 2
        image[disc_here & ~cloth_here] = (30, 30, 30)
    return image


CLUTTER = (((-0.30, 0.50), 0.09), ((0.31, 0.47), 0.08))      # dark things on the table, inside the searched area


def test_a_blue_cloth_is_found_among_darker_things():
    blue_cal = fold_camera.calibrate(render_blue(cloth(), CLUTTER), SIDE, NEAR_EDGE, colour="blue")
    truth = cloth(shift=(0.02, 0.015), turn_deg=8.0)
    corners, _ = fold_camera.look(render_blue(truth, CLUTTER), blue_cal)
    for name, want in zip(("near_left", "near_right", "far_right", "far_left"), truth):
        assert np.linalg.norm(corners[name] - want) < 0.004


def test_a_calibration_saved_without_a_colour_still_looks_for_a_dark_cloth(cal):
    old_cal = dict(cal)
    old_cal.pop("colour", None)
    truth = cloth(shift=(0.02, 0.015), turn_deg=8.0)
    corners, _ = fold_camera.look(render(truth, BASES), old_cal)
    assert np.linalg.norm(corners["near_left"] - truth[0]) < 0.004


# --- one frame from the phone ---

class OneFrameStream:
    """A stand-in for record3d.Record3DStream that has a frame ready as soon as it connects."""

    made = []

    def __init__(self):
        OneFrameStream.made.append(self)
        self.on_new_frame = None

    @staticmethod
    def get_connected_devices():
        return ["phone"]

    def connect(self, device):
        self.on_new_frame()
        return True

    def get_rgb_frame(self):
        return np.full((960, 720, 3), 7, np.uint8)

    def get_depth_frame(self):
        return np.ones((256, 192), np.float32)

    def get_intrinsic_mat(self):
        return types.SimpleNamespace(fx=700.0, fy=700.0, tx=360.0, ty=480.0)


def test_one_frame_from_the_phone_keeps_its_stream_alive(monkeypatch):
    import gc
    import weakref

    OneFrameStream.made.clear()
    monkeypatch.setitem(sys.modules, "record3d", types.SimpleNamespace(Record3DStream=OneFrameStream))
    rgb = fold_camera.record3d_frame()
    assert rgb.shape == (960, 720, 3)
    stream = weakref.ref(OneFrameStream.made[0])
    OneFrameStream.made.clear()
    gc.collect()
    assert stream() is not None


def test_a_pinch_with_the_fingers_too_flat_is_not_reachable(capsys):
    def plan_with_tilt(left, right):
        source = {}
        for side, tilt in (("left", left), ("right", right)):
            source[side] = {"pinch_error_mm": 0.0, "place_error_mm": 0.0, "max_tilt_deg": tilt}
        return {"source": source}

    assert fold_camera.report(plan_with_tilt(24.0, 26.0), {})
    assert not fold_camera.report(plan_with_tilt(24.0, fold_camera.MAX_TILT + 5.0), {})
    assert "FLAT" in capsys.readouterr().out
    assert fold_camera.report(plan_with_tilt(24.0, fold_camera.MAX_TILT + 5.0), {}, max_tilt=50.0)


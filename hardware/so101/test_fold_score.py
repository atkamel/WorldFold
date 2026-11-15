"""fold_score on synthetic photos: convex table-frame polygons rendered through a known perspective homography."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import fold_score  # noqa: E402

SIDE = 0.30
SQUARE = [(0, 0), (SIDE, 0), (SIDE, SIDE), (0, SIDE)]
# pixels of the table corners (0,0), (side,0), (side,side), (0,side): near edge at the bottom of a 480x640 image
PIXELS = [(180, 400), (470, 410), (420, 140), (230, 130)]


def table_to_pixel(px):
    """Homography taking table metres to pixels, by the same direct solve as the tool but written out here."""
    a, b = [], []
    for (x, y), (u, v) in zip(SQUARE, px):
        a += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.append(np.linalg.solve(a, b), 1).reshape(3, 3)


def render(poly, px=PIXELS):
    """480x640 RGB image, dark (30) where the convex counter-clockwise table polygon lands, light (200) elsewhere."""
    inv = np.linalg.inv(table_to_pixel(px))
    v, u = np.mgrid[0:480, 0:640]
    q = inv @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    x, y = q[0] / q[2], q[1] / q[2]
    inside = np.ones(x.shape, bool)
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        inside &= (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) >= 0
    grey = np.where(inside, 30, 200).astype(np.uint8).reshape(480, 640)
    return np.stack([grey] * 3, axis=2)


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_perfect_fold():
    s = fold_score.score(render(SQUARE), render(rect(0, 0, SIDE, SIDE / 2)))
    assert s["far_edge_mean"] < 0.3
    assert s["far_edge_max"] < 0.5
    assert s["overhang_max"] < 0.3
    assert 0.95 < s["area_ratio"] < 1.05
    assert s["uncovered_near"] < 0.03


def test_fold_short_by_2cm():
    s = fold_score.score(render(SQUARE), render(rect(0, 0, SIDE, SIDE / 2 + 0.02)))
    assert s["far_edge_mean"] == pytest.approx(2.0, abs=0.4)


def test_skewed_fold():
    after = [(0, 0), (SIDE, 0), (SIDE, SIDE / 2 + 0.03), (0, SIDE / 2)]
    s = fold_score.score(render(SQUARE), render(after))
    assert s["far_edge_max"] == pytest.approx(3.0, abs=0.5)


def test_fold_shifted_past_near_edge():
    s = fold_score.score(render(SQUARE), render(rect(0, -0.02, SIDE, SIDE / 2 - 0.02)))
    assert s["overhang_max"] == pytest.approx(2.0, abs=0.4)


def test_near_top():
    flipped = [(640 - u, 480 - v) for u, v in PIXELS]   # the same scene seen with the near edge at the top
    before, after = render(SQUARE, flipped), render(rect(0, 0, SIDE, SIDE / 2), flipped)
    s = fold_score.score(before, after, near="top")
    assert s["far_edge_mean"] < 0.3 and s["overhang_max"] < 0.3 and s["uncovered_near"] < 0.03


def test_cli(tmp_path, capsys):
    Image.fromarray(render(SQUARE)).save(tmp_path / "before.png")
    Image.fromarray(render(rect(0, 0, SIDE, SIDE / 2))).save(tmp_path / "after.png")
    out = tmp_path / "overlay.png"
    fold_score.main([str(tmp_path / "before.png"), str(tmp_path / "after.png"), "--out", str(out)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert set(json.loads(lines[0])) == {"far_edge_mean", "far_edge_max", "overhang_max", "area_ratio", "uncovered_near"}
    assert out.exists()

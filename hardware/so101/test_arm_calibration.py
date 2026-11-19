"""arm_calibration: the touch sequence is a safe trajectory that rests on its points, the fit recovers a camera's map
from clicked touches, and fold_camera then follows the cloth through each arm's own map, wherever the bases stand."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import test_fold_camera as synthetic  # noqa: E402  (its camera, cloth and renderer; it also stands in for the servo SDK)

import arm_calibration  # noqa: E402
import fold_camera  # noqa: E402
import fold_score  # noqa: E402
import fold_traj  # noqa: E402
import replay  # noqa: E402
from kinematics import Arm  # noqa: E402

S = synthetic.SPACING
# each arm's frame (x forward, y left) from the table's (X right, Y forward), for two rigs
BEHIND = {"left": lambda p: (p[1], -(p[0] + S / 2)), "right": lambda p: (p[1], -(p[0] - S / 2))}
GAP = 0.16
SIDES = {"left": lambda p: (p[0] + 0.15 + GAP, p[1] - 0.17), "right": lambda p: (0.15 + GAP - p[0], 0.17 - p[1])}


def to_table(rig, side, point):
    """The table point an arm-frame point is, by inverting the rig's (linear) frame change numerically."""
    f = rig[side]
    o, ex, ey = np.array(f((0.0, 0.0))), np.array(f((1.0, 0.0))), np.array(f((0.0, 1.0)))
    return np.linalg.solve(np.column_stack([ex - o, ey - o]), np.asarray(point) - o)


def clicked_maps(rig, noise=0.0, seed=0):
    """The maps arm_calibration fits when every touch is clicked where the synthetic camera sees it (+ pixel noise)."""
    rng = np.random.default_rng(seed)
    to_px = np.linalg.inv(synthetic.camera())
    maps = {}
    for side in ("left", "right"):
        points = arm_calibration.touch_points()
        px = fold_score.apply(to_px, np.array([to_table(rig, side, p) for p in points])) + rng.normal(0, noise, (len(points), 2))
        maps[side] = {"H": arm_calibration.fit_map(px, points).tolist()}
    return maps


def test_the_touch_sequence_rests_on_every_point_and_is_a_safe_trajectory():
    collision = pytest.importorskip("collision")
    traj = arm_calibration.trajectory()
    replay.validate(traj)
    rows, arm = np.array(traj["arms"]["left"]["target"]), Arm()
    points, rests = traj["source"]["points"], traj["marks"]["rests"]
    assert len(points) == len(rests) == 9
    for (x, y), (a, b) in zip(points, rests):
        assert b - a == round(arm_calibration.HOLD / fold_traj.DT) and np.all(rows[a:b + 1] == rows[a])     # it rests
        assert np.allclose(arm.tip(rows[a, :5])[0], (x, y, fold_traj.PINCH_Z), atol=0.001)                  # on the point
    assert np.all(rows[:, 5] == arm_calibration.CLOSED)
    assert collision.closest(rows.tolist())[0] >= 2 * collision.CLEARANCE
    assert np.abs(np.diff(rows[:, :5], axis=0)).max() <= fold_traj.JOINT_RATE
    heights = np.array([arm.tip(r[:5])[0][2] for r in rows])
    assert heights.min() > 0.004 and np.isclose(heights[0], fold_traj.HOVER, atol=0.001)
    resting = np.zeros(len(rows), bool)
    for a, b in rests:
        resting[a:b + 1] = True
    travelling = np.array([arm.tip(r[:5])[0] for r in rows[~resting]])
    assert np.hypot(travelling[:, 0], travelling[:, 1]).min() > 0.15        # it goes round the base, never across it


def test_a_narrower_spread_keeps_the_touching_arm_near_its_own_centre_line():
    arm = Arm()
    assert arm_calibration.WIDEST == 25.0                                    # the default is the one for bases 30 cm apart
    for widest, limit in ((arm_calibration.WIDEST, 0.16), (55.0, 0.31)):
        rows = arm_calibration.trajectory(arm_calibration.touch_points(widest))["arms"]["left"]["target"]
        sideways = max(abs(p[1]) for row in rows for p in arm.joint_points(row[:5]))
        assert sideways < limit


def test_the_fit_recovers_the_cameras_map_and_says_how_well():
    points = np.array(arm_calibration.touch_points())
    to_px = np.linalg.inv(synthetic.camera())
    px = fold_score.apply(to_px, np.array([to_table(BEHIND, "left", p) for p in points]))
    H, inside, left_out, _ = arm_calibration.fit_report(px, points)
    assert inside.max() < 1e-6 and np.nanmax(left_out) < 1e-6
    fresh = np.array([(0.38, 0.0), (0.10, 0.05), (0.25, -0.12)])
    seen = fold_score.apply(to_px, np.array([to_table(BEHIND, "left", p) for p in fresh]))
    assert np.abs(arm_calibration.apply(H, seen) - fresh).max() < 1e-6
    noisy = px + np.random.default_rng(0).normal(0, 1.0, px.shape)          # clicks a pixel off
    _, inside, left_out, _ = arm_calibration.fit_report(noisy, points)
    assert inside.max() < 0.006 and np.nanmax(left_out) < arm_calibration.MAX_FIT_ERROR
    with pytest.raises(ValueError, match="at least 4"):
        arm_calibration.fit_map(px[:3], points[:3])


def clicked_touches(folder, side, rig=None, noise=0.0, seed=0):
    """touches.json and one picture for `side`, every touch clicked where the synthetic camera sees it."""
    from PIL import Image
    points = arm_calibration.touch_points()
    px = fold_score.apply(np.linalg.inv(synthetic.camera()), np.array([to_table(rig or BEHIND, side, p) for p in points]))
    px = px + np.random.default_rng(seed).normal(0, noise, px.shape)
    touches = [{"point": list(p), "frame": f"touch_{k}.jpg", "px": list(map(float, c))} for k, (p, c) in enumerate(zip(points, px))]
    (folder / side).mkdir(parents=True, exist_ok=True)
    for t in touches:
        Image.fromarray(np.zeros((*synthetic.SHAPE, 3), np.uint8)).save(folder / side / t["frame"])
    (folder / side / "touches.json").write_text(json.dumps({"arm": side, "touches": touches}))
    return px


def test_fit_wants_both_arms_names_a_wrong_click_and_keeps_the_rest_of_camera_json(tmp_path, monkeypatch):
    monkeypatch.setattr(arm_calibration, "CAMERA", tmp_path / "camera.json")
    monkeypatch.setattr(arm_calibration, "FOLDER", tmp_path / "touches")
    (tmp_path / "camera.json").write_text(json.dumps({"dark": 100.0, "colour": "blue"}))
    px = clicked_touches(tmp_path / "touches", "left", noise=1.0)
    with pytest.raises(ValueError, match="right arm has 0 clicked"):          # one arm alone is not a calibration
        arm_calibration.fit()
    assert "arms" not in json.loads((tmp_path / "camera.json").read_text())
    clicked_touches(tmp_path / "touches", "right", noise=1.0, seed=1)
    arm_calibration.record_click("left", "touch_0.jpg", None)                # one fingertip was out of the picture
    summary = arm_calibration.fit()
    assert summary["left"]["touches"] == 8 and summary["right"]["touches"] == 9 and summary["left"]["max_mm"] < 6
    saved = json.loads((tmp_path / "camera.json").read_text())
    assert saved["dark"] == 100.0 and saved["colour"] == "blue" and {"left", "right", "fitted", "shape"} == set(saved["arms"])
    assert saved["arms"]["shape"] == list(synthetic.SHAPE)
    # a click 12 px off (about 25 mm on this table): a fit of all nine touches bends to it and looks fine, the fit
    # of the other eight does not, and the others agree best without that picture
    arm_calibration.record_click("left", "touch_4.jpg", (px[4][0] + 12, px[4][1]))
    with pytest.raises(ValueError, match="fit best without touch_4.jpg"):
        arm_calibration.fit()
    assert json.loads((tmp_path / "camera.json").read_text()) == saved
    arm_calibration.record_click("left", "touch_4.jpg", px[4])               # clicked again, rightly
    for k in range(4):                                                        # too few clicks to tell a wrong one
        arm_calibration.record_click("right", f"touch_{k}.jpg", None)
    with pytest.raises(ValueError, match="right arm has 5 clicked"):
        arm_calibration.fit()


def test_each_touch_gets_a_picture_from_the_middle_of_its_rest_and_none_from_after_it(tmp_path):
    from PIL import Image
    traj = {"marks": {"rests": [[10, 50], [100, 140], [200, 240]]}, "source": {"points": [[0.3, 0.1], [0.3, -0.1], [0.2, 0.0]]}}
    lines = [{"phase": "glide to the first target", "row": None, "t": 99.0, "target": {"left": []}}]
    lines += [{"phase": "fold", "row": k, "t": 100.0 + 0.1 * k, "target": {"left": []}} for k in range(160)]    # stopped in touch 3
    (tmp_path / "steps.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
    (tmp_path / "frames").mkdir()
    # touch 1 rests 101.0 to 105.0 and, settled, 101.5 to 105.0; touch 2 rests 110.0 to 114.0 but the phone only
    # sent pictures before it arrived and after it lifted
    for t in [*np.arange(99.0, 109.0, 0.25), 109.75, 114.25, 114.5]:
        Image.fromarray(np.full((4, 4, 3), int(t * 4) % 255, np.uint8)).save(tmp_path / "frames" / f"{int(t * 1000)}.jpg")
    touches = arm_calibration.keep_touches(tmp_path, "left", traj, tmp_path / "steps.jsonl")
    assert [t["frame"] for t in touches] == ["touch_0.jpg"] and touches[0]["px"] is None and touches[0]["point"] == [0.3, 0.1]
    want = np.asarray(Image.open(tmp_path / "frames" / "103250.jpg"))        # the middle of 101.5 to 105.0
    assert np.array_equal(np.asarray(Image.open(tmp_path / "touch_0.jpg")), want)


def test_a_stalled_phone_adds_no_pictures(tmp_path):
    import time
    import rgbd
    frame = rgbd.Frame(0.0, np.zeros((4, 4, 3), np.uint8), None, (1, 1, 0, 0))
    phone = type("Phone", (), {"count": 3, "latest": lambda self: frame})()
    capture = arm_calibration.Capture(phone, tmp_path)
    capture.PERIOD = 0.01
    capture.start()
    time.sleep(0.15)
    phone.count = 4                                                           # one new frame arrives
    time.sleep(0.15)
    capture.stopped.set()
    capture.join()
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 2                # the first one seen, and the new one


def arm_cal(rig, noise=0.0):
    cal = fold_camera.calibrate(synthetic.render(synthetic.cloth()), synthetic.SIDE, synthetic.NEAR_EDGE)
    cal["arms"] = clicked_maps(rig, noise)
    return cal


def test_the_look_follows_a_moved_cloth_through_the_arms_maps():
    truth = synthetic.cloth(shift=(0.02, 0.015), turn_deg=6.0)
    want = fold_camera.fold_points(dict(zip(("near_left", "near_right", "far_right", "far_left"), truth)), S)
    for noise, tol in ((0.0, 0.004), (1.0, 0.008)):
        traj, _, _, points = fold_camera.plan(synthetic.render(truth, synthetic.BASES), arm_cal(BEHIND, noise), S, -0.17)
        for side in ("left", "right"):
            assert np.abs(np.array(points[side]) - np.array(want[side])).max() < tol
        assert traj["source"]["left"]["pinch_error_mm"] < 1.0


def test_calibrating_on_the_cloth_after_it_moved_no_longer_moves_the_arms(tmp_path, monkeypatch):
    """The trap the arm calibration removes: `calibrate` with the cloth away from the ruler position used to make the
    arms go to the ruler position."""
    from PIL import Image
    monkeypatch.setattr(fold_camera, "CAMERA", tmp_path / "camera.json")
    (tmp_path / "camera.json").write_text(json.dumps(arm_cal(BEHIND)))
    truth = synthetic.cloth(shift=(0.03, 0.015), turn_deg=0.0)
    moved = synthetic.render(truth, synthetic.BASES)
    Image.fromarray(moved).save(tmp_path / "moved.png")
    fold_camera.main(["calibrate", "--image", str(tmp_path / "moved.png"), "--out", str(tmp_path / "look.png")])
    cal = json.loads((tmp_path / "camera.json").read_text())
    _, _, _, points = fold_camera.plan(moved, cal, S, -0.17)
    want = fold_camera.fold_points(dict(zip(("near_left", "near_right", "far_right", "far_left"), truth)), S)
    ruler = fold_camera.fold_points(dict(zip(("near_left", "near_right", "far_right", "far_left"), synthetic.cloth())), S)
    assert np.abs(np.array(points["left"]) - np.array(want["left"])).max() < 0.004
    assert np.abs(np.array(points["left"]) - np.array(ruler["left"])).max() > 0.015


def test_with_the_arms_at_the_sides_each_arm_gets_its_own_corners_and_the_whole_fold():
    cal = arm_cal(SIDES)
    corners = np.array([(-0.15, 0.02), (0.15, 0.02), (0.15, 0.32), (-0.15, 0.32)]) + (0.01, 0.01)     # a little off centre
    traj, _, px, points = fold_camera.plan(synthetic.render(corners), cal, S, -0.17)
    d = fold_camera.INSET / np.sqrt(2)
    for side, far, near in (("left", corners[3] + (d, -d), corners[0] + (d, -d)),
                            ("right", corners[2] + (-d, -d), corners[1] + (-d, -d))):
        assert np.abs(np.array(points[side]) - np.array([SIDES[side](far), SIDES[side](near)])).max() < 0.004
        assert traj["source"][side]["release_short_mm"] == 0.0 and traj["source"][side]["tip_error_mm"] < 0.5
    drawn = np.asarray(fold_camera.overlay(synthetic.render(corners), cal, px, points, S))
    u, v = px["far_left"]
    assert drawn[int(v) - 12:int(v) + 12, int(u) - 12:int(u) + 12, 0].max() == 255            # the pinch circle, on the corner

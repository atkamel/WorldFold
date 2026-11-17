"""fold_live on a fake servo bus and a replayed camera: the look, the tracked near edge moving the place, the
record of the run."""

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import recorder  # noqa: E402
from test_fold_camera import BASES, NEAR_EDGE, SIDE, SPACING, cloth, render  # noqa: E402
from test_replay import FakePort, make_fake_sdk  # noqa: E402


class Replayed:
    """A phone that replays frames: wait() gives the next one, then repeats the last."""

    def __init__(self, images):
        self.frames = [types.SimpleNamespace(t=float(i), rgb=rgb, depth=None, K=None) for i, rgb in enumerate(images)]
        self.i = -1

    def wait(self, timeout=5.0):
        self.i = min(self.i + 1, len(self.frames) - 1)
        return self.frames[self.i]

    def latest(self):
        return self.frames[max(self.i, 0)]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    ports = {}
    monkeypatch.setitem(sys.modules, "scservo_sdk", make_fake_sdk(ports))
    for name in ("replay", "fold_live"):
        sys.modules.pop(name, None)
    import fold_camera
    import fold_live
    replay = fold_live.replay
    monkeypatch.setattr(replay.arms, "ARMS", {"L": "devL", "R": "devR"})
    monkeypatch.setattr(replay.arms, "folder", lambda name: tmp_path / name)
    monkeypatch.setattr(replay.arms, "port", lambda name: replay.arms.ARMS[name])
    for name, dev in (("L", "devL"), ("R", "devR")):
        (tmp_path / name).mkdir()
        limits = {str(m): {"true": {"min": 685, "max": 3410}} for m in range(1, 6)} | {"6": {"true": {"min": 1316, "max": 2779}}}
        (tmp_path / name / "limits.json").write_text(json.dumps(limits))
        ports[dev] = FakePort(dev)
        ports[dev].pos = {m: 2047 for m in range(1, 7)}
    monkeypatch.setattr(replay, "SETTLE_SECONDS", 0)
    monkeypatch.setattr(replay, "sleep_until", lambda t: time.sleep(0.002))   # fast, but the tracker thread keeps up
    cal = fold_camera.calibrate(render(cloth()), SIDE, NEAR_EDGE)
    args = types.SimpleNamespace(spacing=SPACING, closed=-0.17, speed=1.0, grasp_check=False, grasp_margin=0.5,
                                 retries=2, allow_clamp=False, no_depth=False, dry_run=False, yes=True, settle=0.3)
    followers = {"left": replay.Follower("L"), "right": replay.Follower("R")}
    return types.SimpleNamespace(fold_live=fold_live, cal=cal, args=args, followers=followers, ports=ports, tmp=tmp_path)


def slid(dy):
    """The cloth with its near edge moved dy forward: the bottom layer after sliding."""
    corners = cloth()
    corners[:2, 1] += dy
    return corners


def run_fold(rig, images):
    with recorder.Recorder(rig.tmp / "runs", {}) as run:
        code = rig.fold_live.fold(rig.followers, Replayed(images), rig.cal, run, rig.args)
    return code, recorder.load(run.path), run.path


def test_the_place_follows_the_near_edge(rig):
    code, data, path = run_fold(rig, [render(cloth(), BASES)] + [render(slid(0.02), BASES)] * 3)
    assert code == 0
    events = {e["name"]: e for e in data["events"]}
    shifts = events["place_correction"]["shifts"]
    assert set(shifts) == {"left", "right"} and all(abs(d - 0.02) < 0.004 for d in shifts.values())
    look = events["look"]["points"]
    for side in ("left", "right"):
        (_, before), (_, after) = look[side], events["place_correction"]["points"][side]
        assert abs((after[0] - before[0]) - shifts[side]) < 1e-9 and after[1] == before[1]      # forward only
    second = json.loads((path / "plan_2.json").read_text())
    first = json.loads((path / "plan_1.json").read_text())
    assert second["source"]["left"]["place_error_mm"] == first["source"]["left"]["place_error_mm"] == 0.0
    assert second["source"]["left"]["max_tilt_deg"] <= first["source"]["left"]["max_tilt_deg"]
    plans = [s["plan"] for s in data["steps"]]
    assert plans[0] == "plan_1" and plans[-1] == "plan_2" and (path / "look.png").exists()
    assert len(data["frames"]) >= 3 and "score" in events
    # the arms end standing, jaws open, on the second plan's last row
    last = json.loads((path / "plan_2.json").read_text())["arms"]["left"]["target"][-1]
    f = rig.followers["left"]
    assert abs(rig.ports["devL"].writes[-1][2] - f.to_ticks(2, last[1])) <= 1


def test_a_still_cloth_keeps_the_first_plan(rig):
    code, data, path = run_fold(rig, [render(cloth(), BASES)] * 3)
    assert code == 0 and not (path / "plan_2.json").exists()
    assert {s["plan"] for s in data["steps"]} == {"plan_1"}


def test_a_slide_past_the_cap_is_capped(rig):
    code, data, _ = run_fold(rig, [render(cloth(), BASES)] + [render(slid(0.06), BASES)] * 3)
    shifts = next(e for e in data["events"] if e["name"] == "place_correction")["shifts"]
    assert code == 0 and shifts and all(abs(d - rig.fold_live.MAX_SHIFT) < 1e-9 for d in shifts.values())


def test_no_cloth_means_no_motion(rig):
    code, data, _ = run_fold(rig, [np.full((480, 640, 3), 200, np.uint8)])
    assert code == 1 and data["events"][0]["name"] == "look_failed"
    assert all(p.writes == [] for p in rig.ports.values())

"""camera_run: on the (fake) arms a stand-in camera policy that gives the planner's rows is played to the end and
recorded; a row into the table, a phone that stops sending and an arm off its row stop it BEFORE the row is sent,
with torque on; a jump is cut to the rate cap; a bad dataset or start is refused before any port opens."""

import importlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fold_traj  # noqa: E402
import policy_data  # noqa: E402
import rgbd  # noqa: E402
import rig_dataset  # noqa: E402
from test_replay import env  # noqa: E402,F401  (env: replay.py on a fake servo bus)

pytest.importorskip("mujoco")
pytest.importorskip("so101_nexus")
SIZE, SCALE, SHAPE = 16, rig_dataset.ACTION_SCALE, (48, 64)


class Planned:
    """A stand-in camera policy: whatever it is shown, the planner's rows from the row it was last asked at.
    `change(k, chunk)` may alter the chunk given at row k."""

    obs_horizon, chunk, cameras = 2, 8, [("main", SIZE)]

    def __init__(self, traj, replan, change=None):
        self.rows = np.concatenate([np.array(traj["arms"][side]["target"]) for side in ("left", "right")], axis=1)
        self.replan, self.change, self.k, self.shown = replan, change, 0, []

    def predict(self, obs, images):
        self.shown.append((obs.copy(), images[0]["main"].copy()))
        rows = self.rows[np.clip(np.arange(self.k, self.k + self.chunk), 0, len(self.rows) - 1)] / SCALE
        chunk = rows[None].copy()
        if self.change:
            self.change(self.k, chunk[0])
        self.k += self.replan
        return chunk


class Phone:
    """A phone that counts the frames it is sent, as rgbd.Phone does. `sending`: how many more looks find a new
    frame (None: all of them); after that the stream has stalled and latest() repeats the last frame."""

    def __init__(self, sending=None, shape=SHAPE):
        self.count, self.sending, self.shape, self.waits = 0, sending, shape, 0

    def frame(self):
        return rgbd.Frame(0.0, np.full((*self.shape, 3), min(self.count, 250), np.uint8),
                          np.full((12, 16), 0.5, np.float32), (50, 50, 32, 24))

    def wait(self, timeout=5.0):
        self.count += 1
        self.waits += 1
        return self.frame()

    def latest(self):
        if self.sending is None or self.sending > 0:
            self.count += 1
            self.sending = None if self.sending is None else self.sending - 1
        return self.frame()


@pytest.fixture(scope="module")
def planned():
    return fold_traj.fold(policy_data.cloth_points("sides"))


@pytest.fixture
def rig(env, planned, monkeypatch):  # noqa: F811
    for name in ("policy_run", "camera_run"):
        sys.modules.pop(name, None)
    env.camera_run = importlib.import_module("camera_run")      # on the fake bus's replay
    monkeypatch.setattr(env.replay, "sleep_until", lambda t: None)
    env.traj = planned
    env.rows = len(planned["arms"]["left"]["target"])
    env.config = {"size": SIZE, "action_scale": SCALE, "rows": env.rows, "row_seconds": 0.05, "frame_shape": list(SHAPE),
                  "start": [float(v) for side in ("left", "right") for v in planned["arms"][side]["target"][0]]}
    env.followers = {"left": env.replay.Follower("L"), "right": env.replay.Follower("R")}
    return env


def options(**changed):
    return types.SimpleNamespace(**({"dry_run": False, "torque_limit": 100, "track_deg": 8.0, "stale": 5.0, "replan": 4,
                                     "log": None, "policy": "stand-in", "dataset": "stand-in"} | changed))


def play(rig, change=None, phone=None, **changed):
    opts = options(**changed)
    policy = Planned(rig.traj, opts.replan, change)
    return rig.camera_run.run(policy, rig.config, rig.followers, phone or Phone(), opts), policy


def ticks(rig, side, row):
    f = rig.followers[side]
    return {m: round(f.to_ticks(m, rig.traj["arms"][side]["target"][row][m - 1])) for m in rig.replay.MOTOR_IDS}


def writes_after_start(rig, dev="devL"):
    """What the port was sent once the glide had put the arm on the fold's first row."""
    writes = rig.ports[dev].writes
    start = max(i for i, w in enumerate(writes) if w == ticks(rig, "left", 0))
    return writes[start + 1:]


def test_the_planners_rows_given_by_a_camera_policy_are_played_and_recorded(rig, capsys):
    phone = Phone()
    code, policy = play(rig, phone=phone)
    assert code == 0 and phone.waits == 1                                         # a fresh picture before the first row
    for side, dev in (("left", "devL"), ("right", "devR")):
        port = rig.ports[dev]
        assert port.writes[-1] == ticks(rig, side, rig.rows - 1) and all(v == 1 for _, v in port.torque)
    assert len(policy.shown) == math.ceil(rig.rows / 4)                           # a look every 4 rows
    obs, picture = policy.shown[3]
    assert obs.shape == (1, 2, 139) and picture.shape == (3, SIZE, SIZE) and picture.dtype == np.uint8
    folder = next((rig.tmp / "runs").iterdir())                                   # recorded as rig_record.py records
    run = rig_dataset.load_run(folder)
    assert run["kind"] == "policy" and len(run["t"]) == rig.rows and (run["frame"] >= 0).all()
    assert json.loads((folder / "policy_meta.json").read_text())["camera_policy"] is True
    # what the policy was shown on the arms is what a dataset made from this run holds, rates included: the log
    # carries the time the joints were read, not the later time the line was written
    stored, seen = rig_dataset.proprio(run)[12], obs[0, -1]                       # the 4th look is at row 12
    rates = [i for b in (0, 25) for i in [*range(b + 5, b + 10), *range(b + 21, b + 24)]]
    still = [i for i in range(139) if i not in rates]
    assert np.allclose(seen[still], stored[still], atol=1e-5)
    # the fake arms play 200 rows in a second, so the rates are large and two clocks apart by microseconds differ a
    # little; had the log held the time its line was written, they would differ by the time the policy took
    assert np.abs(stored[rates]).max() > 0.05 and np.allclose(seen[rates], stored[rates], rtol=0.1, atol=0.02)
    assert np.allclose(run["target"], policy.rows, atol=0.002)


def test_a_dry_run_opens_no_port_and_a_policy_for_other_cameras_is_refused(rig, capsys):
    code, _ = play(rig, dry_run=True)
    assert code == 0 and rig.ports == {} and "No port opened" in capsys.readouterr().out
    policy = Planned(rig.traj, 4)
    policy.cameras = [("main", 128), ("left_wrist_cam", 64)]
    with pytest.raises(SystemExit, match="looks through"):
        rig.camera_run.run(policy, rig.config, rig.followers, Phone(), options())
    with pytest.raises(SystemExit, match="count the frames"):                     # a camera that cannot be told stalled
        rig.camera_run.run(Planned(rig.traj, 4), rig.config, rig.followers, types.SimpleNamespace(latest=lambda: None), options())
    assert rig.ports == {}


def test_a_start_the_arm_cannot_take_is_refused_before_any_port_opens(rig):
    rig.config["start"][1] = math.radians(170)                                    # outside the shoulder's limits
    with pytest.raises(SystemExit, match="outside its limits"):
        play(rig)
    rig.config["start"][1] = math.radians(60)                                     # leaning forward onto the table
    rig.config["start"][2] = math.radians(60)
    with pytest.raises(SystemExit, match="into the table|within .* of itself"):
        play(rig)
    assert rig.ports == {}


def test_a_dataset_that_cannot_be_played_from_is_refused(tmp_path, rig):
    good = dict(rig.config)

    def refused(match, **changed):
        (tmp_path / "manifest.json").write_text(json.dumps({"config": good | changed}))
        with pytest.raises(ValueError, match=match):
            rig.camera_run.load_config(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"size": 128}}))
    with pytest.raises(ValueError, match="does not say"):
        rig.camera_run.load_config(tmp_path)
    refused("not 12 joint angles", start=good["start"][:11])
    refused("not 12 joint angles", start=[float("nan")] + good["start"][1:])       # replay.convert would send a limit
    refused("rows", rows=0)
    refused("shorter than", row_seconds=0.01)
    refused("units of", action_scale=0.05)
    (tmp_path / "manifest.json").write_text(json.dumps({"config": good}))
    assert rig.camera_run.load_config(tmp_path)["rows"] == rig.rows


def test_a_row_into_the_table_is_stopped_before_it_is_sent(rig, capsys):
    def into_the_table(k, chunk):
        if k == 40:                         # the open jaw swung under the gripper: fingertip, wrist and elbow all high
            chunk[:, :6] = np.radians([-55.1, 25.7, 26.8, 86.8, 3.1, 57.3]) / SCALE
    code, _ = play(rig, change=into_the_table)
    out = capsys.readouterr().out
    assert code == 1 and "into the table" in out and "Stopped at row" in out
    stopped = int(out.split("Stopped at row ")[1].split(":")[0])
    assert 40 <= stopped < 60                                                     # on the way there, a row's worth at a time
    after = writes_after_start(rig)
    assert after[-1] == after[-2] or len(after) <= (stopped) * 3 + 3              # nothing sent for the refused row
    body = rig.camera_run.Body()
    f = rig.followers["left"]
    for w in after:                                                               # and no pose sent was itself in the table
        q = [f.to_rad(m, w[m]) for m in rig.replay.MOTOR_IDS]
        assert body.table_fault(np.array(q)) is None
    assert all(v == 1 for _, v in rig.ports["devL"].torque)


def test_the_table_is_checked_on_the_pose_the_limits_leave(rig, monkeypatch):
    """row_fault clamps a target up to 2 deg into the limits without refusing; the pose sent is the clamped one."""
    seen = []
    body_fault = rig.camera_run.Body.table_fault
    monkeypatch.setattr(rig.camera_run.Body, "table_fault", lambda self, q: seen.append(np.array(q)) or body_fault(self, q))
    low, high = rig.followers["left"].limits[4]
    limit = rig.followers["left"].to_rad(4, high)

    def past_the_limit(k, chunk):
        if k >= 40:
            chunk[:, 3] = (limit + math.radians(1.5)) / SCALE                      # the left wrist, 1.5 deg past its limit
    play(rig, change=past_the_limit)
    assert max(q[3] for q in seen[2::2]) <= limit + 1e-6                          # never the unclamped angle


def test_a_phone_that_stops_sending_stops_the_run(rig, capsys):
    code, _ = play(rig, phone=Phone(sending=2), stale=0.0)                        # new pictures at rows 0 and 4, then none
    out = capsys.readouterr().out
    assert code == 1 and "no new picture" in out and "Stopped at row 8" in out    # the third look finds nothing new
    assert len(writes_after_start(rig)) == 7 * 3                                  # rows 1 to 7 followed the start, no more


def test_a_picture_of_another_size_stops_the_run_before_the_first_row(rig, capsys):
    code, _ = play(rig, phone=Phone(shape=(64, 48)))                              # the phone was turned
    assert code == 1 and "the policy learned on" in capsys.readouterr().out
    assert writes_after_start(rig) == []


def test_a_policy_that_fails_stops_the_run(rig, capsys):
    def broken(k, chunk):
        if k == 8:
            raise RuntimeError("shape mismatch")
    code, _ = play(rig, change=broken)
    assert code == 1 and "the policy failed on this picture (RuntimeError: shape mismatch)" in capsys.readouterr().out


def test_a_jump_from_the_policy_is_cut_to_the_rate_cap(rig):
    def jump(k, chunk):
        if k == 40:
            chunk[:, 4] += math.radians(20) / SCALE                               # the left wrist roll, 20 deg at once
    code, _ = play(rig, change=jump)
    assert code == 0
    roll = np.array([w[5] for w in writes_after_start(rig)]) / rig.replay.STEPS_PER_DEGREE
    rows = roll[2::3]                                                             # the end of each row's three moves
    assert np.abs(np.diff(rows)).max() <= 3.0 + 1e-6 and np.ptp(roll) > 10        # it went, under the cap each row


def test_an_arm_far_from_its_last_row_stops_the_run(rig, monkeypatch, capsys):
    calls = {"n": 0}
    read = rig.replay.Follower.read_present

    def stuck(self):
        present = read(self)
        calls["n"] += 1
        if self.name == "L" and calls["n"] > 60:
            present = {m: v + (round(12 * rig.replay.STEPS_PER_DEGREE) if m == 2 else 0) for m, v in present.items()}
        return present
    monkeypatch.setattr(rig.replay.Follower, "read_present", stuck)
    code, _ = play(rig)
    assert code == 1 and "from the row sent before" in capsys.readouterr().out


def test_with_idle_poses_the_arms_glide_from_them_and_back(rig, capsys):
    for name in ("L", "R"):
        folder = rig.replay.arms.folder(name)
        (folder / "idle_pose.json").write_text(json.dumps({str(m): 2047 + 40 * m for m in rig.replay.MOTOR_IDS}))
    code, _ = play(rig)
    out = capsys.readouterr().out
    assert code == 0 and "back at its idle pose" in out
    assert rig.ports["devL"].writes[-1] == {m: 2047 + 40 * m for m in rig.replay.MOTOR_IDS}

"""camera_run: on the (fake) arms a stand-in camera policy that gives the planner's rows is played to the end and
recorded; a row into the table, a phone that stops sending and an arm off its row stop it with torque on; a jump is
cut to the rate cap; nothing opens a port in a dry run."""

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
SIZE, SCALE = 16, rig_dataset.ACTION_SCALE


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
    def __init__(self, counting=True):
        self.count, self.counting = 0, counting

    def latest(self):
        self.count += self.counting
        return rgbd.Frame(0.0, np.full((48, 64, 3), min(self.count, 250), np.uint8), np.full((12, 16), 0.5, np.float32),
                          (50, 50, 32, 24))


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
    env.config = {"size": SIZE, "action_scale": SCALE, "rows": env.rows, "row_seconds": 0.05,
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


def test_the_planners_rows_given_by_a_camera_policy_are_played_and_recorded(rig, capsys):
    code, policy = play(rig)
    assert code == 0
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
    # what the policy was shown on the arms is what a dataset made from this run holds: joints, jaw and fingertip
    stored = rig_dataset.proprio(run)
    slots = [i for b in (0, 25) for i in [*range(b, b + 5), *range(b + 10, b + 17)]]
    assert np.allclose(obs[0, -1, slots], stored[12, slots], atol=1e-5)           # the 4th look is at row 12
    assert np.allclose(run["target"], policy.rows, atol=0.002)


def test_a_dry_run_opens_no_port_and_a_policy_for_other_cameras_is_refused(rig, capsys):
    code, _ = play(rig, dry_run=True)
    assert code == 0 and rig.ports == {} and "No port opened" in capsys.readouterr().out
    policy = Planned(rig.traj, 4)
    policy.cameras = [("main", 128), ("left_wrist_cam", 64)]
    with pytest.raises(SystemExit, match="looks through"):
        rig.camera_run.run(policy, rig.config, rig.followers, Phone(), options())
    assert rig.ports == {}


def test_a_start_the_arm_cannot_take_is_refused_before_any_port_opens(rig):
    rig.config["start"][1] = math.radians(170)                                    # outside the shoulder's limits
    with pytest.raises(SystemExit, match="outside its limits"):
        play(rig)
    assert rig.ports == {}


def test_a_row_into_the_table_stops_the_run_with_torque_on(rig, monkeypatch, capsys):
    monkeypatch.setattr(rig.camera_run, "FINGERTIP_FLOOR", 0.006)                 # the fold's own pinch is at 5 mm
    code, _ = play(rig)
    out = capsys.readouterr().out
    assert code == 1 and "fingertip would be" in out and "above the table" in out
    pinch = rig.traj["marks"]["pinch"]
    assert len(rig.ports["devL"].writes) < (pinch + 1) * 5 + 400                  # stopped on the way down, not after
    assert all(v == 1 for _, v in rig.ports["devL"].torque)


def test_a_phone_that_sends_nothing_new_stops_the_run(rig, capsys):
    code, _ = play(rig, phone=Phone(counting=False), stale=0.0)                   # the same picture count every look
    assert code == 1 and "no new picture" in capsys.readouterr().out
    sent = len(rig.ports["devL"].writes)
    code, _ = play(rig, phone=types.SimpleNamespace(latest=lambda: None))         # no picture at all: stops at row 0
    assert code == 1 and "no new picture" in capsys.readouterr().out and len(rig.ports["devL"].writes) > 0 and sent > 0


def test_a_jump_from_the_policy_is_cut_to_the_rate_cap(rig):
    def jump(k, chunk):
        if k == 40:
            chunk[:, 4] += math.radians(20) / SCALE                               # the left wrist roll, 20 deg at once
    code, _ = play(rig, change=jump)
    assert code == 0
    writes = rig.ports["devL"].writes
    roll = np.array([w[5] for w in writes]) / rig.replay.STEPS_PER_DEGREE
    assert np.abs(np.diff(roll)).max() <= 3.0 / 1 + 1e-6 and np.ptp(roll) > 10    # it went, a row's worth at a time


def test_an_arm_far_from_its_last_row_stops_the_run(rig, monkeypatch, capsys):
    port_holder = {}
    read = rig.replay.Follower.read_present

    def stuck(self):
        present = read(self)
        port_holder.setdefault("calls", 0)
        port_holder["calls"] += 1
        if self.name == "L" and port_holder["calls"] > 60:
            present = {m: v + (round(12 * rig.replay.STEPS_PER_DEGREE) if m == 2 else 0) for m, v in present.items()}
        return present
    monkeypatch.setattr(rig.replay.Follower, "read_present", stuck)
    code, _ = play(rig)
    assert code == 1 and "from the row sent before" in capsys.readouterr().out


def test_a_dataset_that_does_not_say_how_it_was_recorded_is_refused(tmp_path):
    import camera_run
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"size": 128}}))
    with pytest.raises(ValueError, match="start"):
        camera_run.load_config(tmp_path)

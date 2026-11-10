"""Tests for replay.py against a fake scservo_sdk bus (no serial port, no arm).

    /Users/akamel/Code/WorldFold/.venv/bin/python -m pytest hardware/so101/test_replay.py
"""

import copy
import importlib
import json
import math
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
T = 10


class FakePort:
    def __init__(self, device):
        self.device = device
        self.pos = {m: 2047 for m in range(1, 7)}
        self.writes = []  # one {motor id: goal} per sync write
        self.torque = []  # (motor id, value) per torque-enable write
        self.on_write = None  # called with the write count; may raise
        self.ser = types.SimpleNamespace(reset_input_buffer=lambda: None)

    def openPort(self):
        return True

    def setBaudRate(self, rate):
        return True


def make_fake_sdk(ports):
    sdk = types.ModuleType("scservo_sdk")
    sdk.COMM_SUCCESS = 0
    sdk.SCS_LOBYTE = lambda x: x & 0xFF
    sdk.SCS_HIBYTE = lambda x: (x >> 8) & 0xFF
    sdk.PortHandler = lambda device: ports.setdefault(device, FakePort(device))

    class PacketHandler:
        def __init__(self, protocol):
            pass

        def write1ByteTxRx(self, port, motor, address, value):
            if address == 40:
                port.torque.append((motor, value))
            return 0, 0

        def write2ByteTxRx(self, port, motor, address, value):
            return 0, 0

    class GroupSyncRead:
        def __init__(self, port, packet, address, length):
            self.port = port

        def addParam(self, m):
            pass

        def txRxPacket(self):
            return 0

        def isAvailable(self, m, address, length):
            return True

        def getData(self, m, address, length):
            return self.port.pos[m]

    class GroupSyncWrite:
        def __init__(self, port, packet, address, length):
            self.port, self.goals = port, {}

        def addParam(self, m, data):
            self.goals[m] = data[0] | data[1] << 8

        def txPacket(self):
            self.port.writes.append(dict(self.goals))
            self.port.pos.update(self.goals)  # a perfect servo
            if self.port.on_write:
                self.port.on_write(len(self.port.writes))
            return 0

    sdk.PacketHandler, sdk.GroupSyncRead, sdk.GroupSyncWrite = PacketHandler, GroupSyncRead, GroupSyncWrite
    return sdk


@pytest.fixture
def env(tmp_path, monkeypatch):
    ports = {}
    monkeypatch.setitem(sys.modules, "scservo_sdk", make_fake_sdk(ports))
    sys.modules.pop("replay", None)
    replay = importlib.import_module("replay")
    monkeypatch.setattr(replay.arms, "ARMS", {"L": "devL", "R": "devR"})
    monkeypatch.setattr(replay.arms, "folder", lambda name: tmp_path / "arms" / name)
    monkeypatch.setattr(replay.arms, "port", lambda name: replay.arms.ARMS[name])
    for name in ("L", "R"):
        (tmp_path / "arms" / name).mkdir(parents=True)
        limits = {str(m): {"true": {"min": 685, "max": 3410}} for m in range(1, 7)}  # +-120 deg about 2047
        (tmp_path / "arms" / name / "limits.json").write_text(json.dumps(limits))
    monkeypatch.setattr(replay, "SETTLE_SECONDS", 0)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    return types.SimpleNamespace(replay=replay, ports=ports, tmp=tmp_path)


def make_traj(dt=0.02):
    target = [[0.3 * math.sin(k / 3 + j) for j in range(6)] for k in range(T)]
    for row in target:
        row[5] = 0.5
    for row in target[5:]:
        row[5] = 0.1  # grasp at step 5
    arm = {"target": target, "q": target + [target[-1]],
           "tip": [[0.1 * k, 0.2, 0.05] for k in range(T + 1)]}
    return {"dt": dt, "joints": JOINTS, "arms": {"left": arm, "right": copy.deepcopy(arm)},
            "layout": {"name": "test"}, "source": {"tool": "pytest"}}


def write_traj(env, traj):
    path = env.tmp / "traj.json"
    path.write_text(json.dumps(traj))
    return str(path)


def seed_at_first_target(env, traj):
    """Start both fake arms on the first target, so the ramp has nothing to do."""
    for side, dev in (("left", "devL"), ("right", "devR")):
        f = env.replay.Follower("L")
        ticks = env.replay.convert(f, traj["arms"][side]["target"][:1])[0][0]
        port = env.ports.setdefault(dev, FakePort(dev))
        port.pos = {m: round(t) for m, t in zip(range(1, 7), ticks)}


# --- validation ---

def test_validate_accepts_fixture(env):
    env.replay.validate(make_traj())


@pytest.mark.parametrize("break_it", [
    lambda t: t["arms"]["left"]["target"][0].pop(),
    lambda t: t["arms"]["left"]["q"].pop(),
    lambda t: t["arms"]["right"]["target"].pop(),
    lambda t: t["arms"]["left"]["target"][2].__setitem__(1, float("nan")),
    lambda t: t["arms"]["right"]["q"][0].__setitem__(0, float("inf")),
    lambda t: t["arms"]["left"]["tip"][0].pop(),
    lambda t: t.__setitem__("dt", 0),
    lambda t: t.__setitem__("joints", JOINTS[::-1]),
    lambda t: t["arms"].pop("right"),
])
def test_validate_rejects(env, break_it):
    traj = make_traj()
    break_it(traj)
    with pytest.raises(ValueError):
        env.replay.validate(traj)


# --- conversion ---

def test_round_trip_each_joint(env):
    f = env.replay.Follower("L")
    for m in range(1, 7):
        for rad in (-1.5, -0.7, 0.0, 0.123, 1.0, 1.5):
            back = f.to_rad(m, round(f.to_ticks(m, rad)))
            assert abs(math.degrees(back - rad)) < 0.2


def test_urdf_json_override(env):
    folder = env.tmp / "arms" / "L"
    (folder / "urdf.json").write_text(json.dumps({"zero": {"1": 1000}, "sign": {"2": -1}}))
    f = env.replay.Follower("L")
    assert f.to_ticks(1, 0) == 1000
    assert f.to_ticks(2, 0.1) < 2047 and f.to_ticks(3, 0.1) > 2047  # motor 3 keeps the default
    assert f.to_ticks(3, 0) == 2047
    assert "urdf.json" in f.describe()
    assert "unverified" in env.replay.Follower("R").describe()
    assert abs(f.to_rad(2, f.to_ticks(2, 0.4)) - 0.4) < 1e-9


def test_missing_limits_is_an_error(env):
    (env.tmp / "arms" / "L" / "limits.json").unlink()
    with pytest.raises(FileNotFoundError):
        env.replay.Follower("L")


def test_clamp_counts(env):
    f = env.replay.Follower("L")
    target = [[0.0] * 6 for _ in range(4)]
    target[1][0] = math.radians(121)  # 1 deg over (limit is 120 deg)
    target[2][0] = math.radians(125)  # 5 deg over
    ticks, stats = env.replay.convert(f, target)
    assert stats[1][0] == 2 and abs(stats[1][1] - 5) < 0.5
    assert stats[2] == [0, 0.0]
    assert ticks[2][0] == 3410


def test_refuses_over_two_degrees_unless_allowed(env, capsys):
    traj = make_traj()
    traj["arms"]["left"]["target"][3][0] = math.radians(125)
    path = write_traj(env, traj)
    with pytest.raises(SystemExit):
        env.replay.main([path, "--left", "L", "--right", "R", "--dry-run"])
    assert env.replay.main([path, "--left", "L", "--right", "R", "--dry-run", "--allow-clamp"]) == 0
    assert "1 steps clamped" in capsys.readouterr().out


def test_small_clamp_is_allowed(env):
    traj = make_traj()
    traj["arms"]["left"]["target"][3][0] = math.radians(121)
    assert env.replay.main([write_traj(env, traj), "--left", "L", "--right", "R", "--dry-run"]) == 0


# --- dry run and printout ---

def test_dry_run_opens_no_bus_and_prints_ranges_and_tips(env, capsys):
    assert env.replay.main([write_traj(env, make_traj()), "--left", "L", "--right", "R", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert env.ports == {}
    assert "max step change" in out and "shoulder_pan" in out
    assert "layout:" in out and "source:" in out
    assert "left tip at start: right 0.0, forward 20.0, up 5.0 cm" in out
    assert "left tip at the grasp (step 5): right 50.0, forward 20.0, up 5.0 cm" in out
    assert "unverified" in out


# --- motion ---

def test_ramp_never_exceeds_20_deg_s(env):
    r = env.replay
    start = {m: 2047 for m in range(1, 7)}
    end = {1: 3000, 2: 2047, 3: 1000, 4: 2100, 5: 2047, 6: 2047}
    pts = r.ramp_points(start, end)
    prev = start
    for p in pts:
        for m in p:
            assert abs(p[m] - prev[m]) / r.STEPS_PER_DEGREE <= r.RAMP_DEG_S * r.TICK_SECONDS + 1e-9
        prev = p
    assert prev == pytest.approx(end)


def test_play_sends_exactly_T_ticks_with_rate_cap(env):
    r = env.replay
    traj = make_traj()
    traj["arms"]["left"]["target"][4][0] = 1.0  # a ~57 deg jump in one 20 ms step
    ticks = {side: r.convert(r.Follower("L"), traj["arms"][side]["target"])[0] for side in ("left", "right")}
    followers = {"left": r.Follower("L"), "right": r.Follower("R")}
    for f in followers.values():
        f.connect()
    before = {s: len(f.port.writes) for s, f in followers.items()}
    engaged = r.play(followers, ticks, 0.02, 1.0, r.MOTOR_IDS, {})
    cap = r.MAX_DEG_S * 0.02 * r.STEPS_PER_DEGREE
    assert engaged > 0
    for s, f in followers.items():
        sent = f.port.writes[before[s]:]
        assert len(sent) == T
        for a, b in zip(sent, sent[1:]):
            assert all(abs(b[m] - a[m]) <= cap + 1 for m in a)  # +1: goals are rounded to whole ticks


def test_full_replay_prints_final_pose_and_leaves_torque_on(env, capsys):
    traj = make_traj()
    seed_at_first_target(env, traj)
    assert env.replay.main([write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1"]) == 0
    out = capsys.readouterr().out
    assert "Rate cap" in out and "measured vs trajectory q" in out
    assert all(v == 1 for p in env.ports.values() for _, v in p.torque)


def test_only_and_no_gripper(env):
    traj = make_traj()
    seed_at_first_target(env, traj)
    args = [write_traj(env, traj), "--left", "L", "--right", "R", "--only", "left", "--no-gripper", "--speed", "1"]
    assert env.replay.main(args) == 0
    assert env.ports["devR"].writes == [] and env.ports["devR"].torque == []
    sent = env.ports["devL"].writes[-T:]
    assert all(set(w) == {1, 2, 3, 4, 5} for w in sent)


def test_ctrl_c_stops_and_keeps_torque(env, capsys):
    traj = make_traj()
    seed_at_first_target(env, traj)

    def interrupt(n):
        if n >= 5:  # one torque-on hold write, then a few play ticks
            raise KeyboardInterrupt

    env.ports["devL"].on_write = interrupt
    code = env.replay.main([write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1"])
    assert code == 1
    assert "at step" in capsys.readouterr().out
    port = env.ports["devL"]
    assert len(port.writes) == 5  # nothing sent after the interrupt
    assert port.torque and all(v == 1 for _, v in port.torque)


def test_check_mode_moves_each_joint_in_urdf_positive_direction(env, monkeypatch):
    (env.tmp / "arms" / "L" / "urdf.json").write_text(json.dumps({"sign": {"2": -1}}))
    monkeypatch.setattr(env.replay, "sleep_until", lambda t: None)
    assert env.replay.main(["--check", "L"]) == 0
    writes = env.ports["devL"].writes
    for m, direction in ((1, 1), (2, -1), (3, 1), (4, 1), (5, 1)):
        peak = max(direction * (w[m] - 2047) for w in writes if m in w)
        assert abs(peak - 8 * 4096 / 360) < 1
    assert all(w[6] == 2047 for w in writes if 6 in w)  # check never moves the gripper
    assert all(v == 1 for _, v in env.ports["devL"].torque)

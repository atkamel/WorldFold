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
        self.words = []  # (motor id, address, value) per two-byte register write
        self.registers = {}  # (motor id, address) -> the value the motor holds
        self.deaf_to_words = False  # True: two-byte writes are acknowledged but not stored
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
            port.words.append((motor, address, value))
            if not port.deaf_to_words:
                port.registers[(motor, address)] = value
            return 0, 0

        def read2ByteTxRx(self, port, motor, address):
            return port.registers.get((motor, address), 0), 0, 0

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


def test_read_mode_prints_model_angles_and_sends_nothing(env, monkeypatch, capsys):
    def stop(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(env.replay.time, "sleep", stop)
    assert env.replay.main(["--read", "L"]) == 0
    assert env.ports["devL"].writes == []
    assert "shoulder_pan" in capsys.readouterr().out


def fold_like_traj():
    """Rows: 0-2 above, jaw open; 3 jaw shut in the air ("empty"); 4 open again ("hover"); 5 down; 6 down, jaw shut;
    7 lifted ("check"); 8-9 carried; 10 jaw open."""
    jaw = [0.5, 0.5, 0.5, -0.1, 0.5, 0.5, -0.1, -0.1, -0.1, -0.1, 0.5]
    down = [0, 0, 0, 0, 0, 0.2, 0.2, 0.1, 0, 0, 0]
    target = [[0.0, d, 0.0, 0.0, 0.0, j] for d, j in zip(down, jaw)]
    arm = {"target": target, "q": target + [target[-1]]}
    # dt 1 s: the rate cap (60 deg per step) never holds a jaw back, so each row is reached in its own step
    return {"dt": 1.0, "joints": JOINTS, "arms": {"left": arm, "right": copy.deepcopy(arm)},
            "marks": {"empty": 3, "hover": 4, "check": 7}}


def add_cloth(env, dev, catches):
    """The fake arm's jaw stops 40 ticks short of shut on cloth: on the pinch attempts listed in `catches` (0 = the
    first time it shuts while down), until it opens again."""
    f = env.replay.Follower("L")
    shut, down = f.to_ticks(6, -0.1), f.to_ticks(2, 0.2)
    port = env.ports.setdefault(dev, FakePort(dev))
    state = {"attempt": -1, "was_shut": False, "holding": False}

    def on_write(n):
        goal = port.writes[-1]
        is_shut = goal[6] < shut + 20
        if is_shut and not state["was_shut"] and abs(goal[2] - down) < 2:
            state["attempt"] += 1
            state["holding"] = state["attempt"] in catches
        state["holding"] &= is_shut
        state["was_shut"] = is_shut
        if state["holding"]:
            port.pos[6] = round(shut) + 40
    port.on_write = on_write
    return state


def run_fold(env, capsys, monkeypatch, *extra):
    monkeypatch.setattr(env.replay, "sleep_until", lambda t: None)
    traj = fold_like_traj()
    seed_at_first_target(env, traj)
    code = env.replay.main([write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1", "--grasp-check",
                            *extra])
    return code, capsys.readouterr().out


def test_grasp_check_passes_when_both_jaws_stop_on_cloth(env, capsys, monkeypatch):
    states = [add_cloth(env, dev, {0}) for dev in ("devL", "devR")]
    code, out = run_fold(env, capsys, monkeypatch)
    assert code == 0 and out.count("holding") == 2 and "MISSED" not in out
    assert [s["attempt"] for s in states] == [0, 0]


def test_grasp_check_pinches_again_after_a_miss(env, capsys, monkeypatch):
    left, right = add_cloth(env, "devL", {0, 1}), add_cloth(env, "devR", {1})      # the right arm misses once
    code, out = run_fold(env, capsys, monkeypatch)
    assert code == 0 and out.count("MISSED") == 1 and "retry 1 of 2" in out
    assert (left["attempt"], right["attempt"]) == (1, 1)                           # both went back down together
    jaw_open = env.replay.Follower("L").to_ticks(6, 0.5)
    assert env.ports["devR"].writes[-1][6] == round(jaw_open)                      # and the fold finished


def test_grasp_check_gives_up_after_its_retries_and_finishes(env, capsys, monkeypatch):
    state = add_cloth(env, "devL", {0, 1, 2})
    add_cloth(env, "devR", set())
    code, out = run_fold(env, capsys, monkeypatch, "--retries", "1")
    assert code == 0 and out.count("MISSED") == 2 and state["attempt"] == 1


def test_log_holds_the_measured_angles_per_step(env):
    traj = make_traj()
    seed_at_first_target(env, traj)
    log = env.tmp / "run.json"
    assert env.replay.main([write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1", "--log", str(log)]) == 0
    data = json.loads(log.read_text())
    assert [row[0] for row in data["measured"]["left"]] == list(range(T))
    assert len(data["measured"]["right"][0]) == 7
    # a perfect fake servo read before each send: step 0 reads the first target, where the arm was seeded
    assert abs(data["measured"]["left"][0][1] - traj["arms"]["left"]["target"][0][0]) < 2 * env.replay.RAD_PER_STEP


# --- idle pose: a replay starts from it and returns to it ---

def write_idle(env, name, pose):
    (env.tmp / "arms" / name / "idle_pose.json").write_text(json.dumps({str(m): p for m, p in pose.items()}))


def first_target(env, traj, side):
    f = env.replay.Follower("L")
    ticks = env.replay.convert(f, traj["arms"][side]["target"][:1])[0][0]
    return {m: round(t) for m, t in zip(range(1, 7), ticks)}


def test_replay_goes_to_the_idle_pose_first_and_ends_there(env):
    traj = make_traj()
    idle = {m: 1500 + 10 * m for m in range(1, 7)}
    write_idle(env, "L", idle)
    write_idle(env, "R", idle)
    assert env.replay.main([write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1"]) == 0
    for side, dev in (("left", "devL"), ("right", "devR")):
        writes = env.ports[dev].writes
        assert writes[-1] == idle
        assert writes.index(idle) < writes.index(first_target(env, traj, side))


def test_an_idle_pose_outside_the_limits_is_clamped(env):
    traj = make_traj()
    idle = {m: 2047 for m in range(1, 7)}
    idle[2] = 600  # below the fixture's lower limit of 685
    write_idle(env, "L", idle)
    write_idle(env, "R", idle)
    assert env.replay.main([write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1"]) == 0
    assert env.ports["devL"].writes[-1][2] == 685


def test_no_gripper_keeps_the_jaw_out_of_the_idle_moves(env):
    traj = make_traj()
    idle = {m: 1500 + 10 * m for m in range(1, 7)}
    write_idle(env, "L", idle)
    write_idle(env, "R", idle)
    args = [write_traj(env, traj), "--left", "L", "--right", "R", "--no-gripper", "--speed", "1"]
    assert env.replay.main(args) == 0
    assert set(env.ports["devL"].writes[-1]) == {1, 2, 3, 4, 5}


# --- torque cap and the lowest points of a trajectory ---

def torque_limit_writes(port):
    """{motor id: every value written to the torque limit register, in order}."""
    writes = {}
    for motor, address, value in port.words:
        if address == 48:
            writes.setdefault(motor, []).append(value)
    return writes


def test_torque_limit_caps_every_joint_and_full_torque_comes_back_at_the_idle_pose(env):
    traj = make_traj()
    idle = {m: 1500 + 10 * m for m in range(1, 7)}
    write_idle(env, "L", idle)
    write_idle(env, "R", idle)
    args = [write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1", "--torque-limit", "40"]
    assert env.replay.main(args) == 0
    writes = torque_limit_writes(env.ports["devL"])
    for motor in range(1, 7):
        assert writes[motor][0] == 400
    for motor in range(1, 6):
        assert writes[motor][-1] == 1000
    assert writes[6][-1] == 500


def test_without_a_torque_limit_the_arm_joints_get_full_torque(env):
    traj = make_traj()
    seed_at_first_target(env, traj)
    assert env.replay.main([write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1"]) == 0
    writes = torque_limit_writes(env.ports["devR"])
    for motor in range(1, 6):
        assert writes[motor][0] == 1000
    assert writes[6][0] == 500


def test_a_torque_limit_too_low_to_hold_the_arm_is_refused(env):
    with pytest.raises(SystemExit):
        env.replay.main([write_traj(env, make_traj()), "--left", "L", "--torque-limit", "5"])


def test_a_torque_limit_the_motor_did_not_take_stops_before_torque_goes_on(env):
    traj = make_traj()
    seed_at_first_target(env, traj)
    env.ports["devL"].deaf_to_words = True
    args = [write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1", "--torque-limit", "40"]
    assert env.replay.main(args) == 1
    assert env.ports["devL"].torque == []


def test_ctrl_c_leaves_the_torque_cap_on(env):
    traj = make_traj()
    seed_at_first_target(env, traj)

    def interrupt(n):
        if n >= 5:
            raise KeyboardInterrupt

    env.ports["devL"].on_write = interrupt
    args = [write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1", "--torque-limit", "40"]
    assert env.replay.main(args) == 1
    writes = torque_limit_writes(env.ports["devL"])
    for motor in range(1, 7):
        assert writes[motor] == [400]


def test_lowest_points_finds_the_step_that_leans_furthest(env):
    f = env.replay.Follower("L")
    rows = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]
    rows[3][1] = 0.9  # shoulder lift tips the whole arm forward and down
    ticks = env.replay.convert(f, rows)[0]
    lowest = env.replay.lowest_points(f, ticks)
    assert len(lowest) == 3
    for height, step in lowest:
        assert step == 3
    upright_tip = env.replay.lowest_points(f, ticks[:3])[2][0]
    assert lowest[2][0] < upright_tip


def test_lowest_points_follow_the_clamped_targets(env):
    f = env.replay.Follower("L")
    over = [[0.0, math.radians(140), 0.0, 0.0, 0.0, 0.0]]  # the fixture's limit is 120 deg
    at_limit = [[0.0, math.radians(120), 0.0, 0.0, 0.0, 0.0]]
    sent = env.replay.lowest_points(f, env.replay.convert(f, over)[0])
    expected = env.replay.lowest_points(f, env.replay.convert(f, at_limit)[0])
    for i in range(3):
        assert abs(sent[i][0] - expected[i][0]) < 0.002


def test_the_torque_cap_stays_on_an_arm_that_does_not_end_at_an_idle_pose(env):
    traj = make_traj()
    seed_at_first_target(env, traj)
    args = [write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1", "--torque-limit", "40"]
    assert env.replay.main(args) == 0
    writes = torque_limit_writes(env.ports["devL"])
    for motor in range(1, 7):
        assert writes[motor] == [400]


# --- smooth playback ---

def test_play_moves_in_small_steps_between_rows(env, monkeypatch):
    r = env.replay
    monkeypatch.setattr(r, "sleep_until", lambda t: None)
    traj = make_traj(dt=0.1)  # 0.1 s per row = 5 sends of 20 ms
    followers = {"left": r.Follower("L"), "right": r.Follower("R")}
    ticks = {}
    for side in followers:
        ticks[side] = r.convert(followers[side], traj["arms"][side]["target"])[0]
        followers[side].connect()
    r.play(followers, ticks, 0.1, 1.0, r.MOTOR_IDS, {})
    sent = followers["left"].port.writes
    assert len(sent) == T * 5
    biggest_row_jump = 0.0
    for k in range(1, T):
        for j in range(6):
            biggest_row_jump = max(biggest_row_jump, abs(ticks["left"][k][j] - ticks["left"][k - 1][j]))
    biggest_sent_jump = 0
    for a, b in zip(sent, sent[1:]):
        for m in a:
            biggest_sent_jump = max(biggest_sent_jump, abs(b[m] - a[m]))
    assert biggest_sent_jump <= biggest_row_jump / 5 + 1  # +1: goals are rounded to whole ticks
    for m in range(1, 7):
        assert sent[-1][m] == round(ticks["left"][T - 1][m - 1])


# --- self-collision ---

def through_the_base_traj():
    traj = make_traj()
    for side in ("left", "right"):
        row = traj["arms"][side]["target"][4]
        row[1] = math.radians(-81)
        row[2] = math.radians(80)
        row[3] = math.radians(95)
    return traj


def test_a_trajectory_through_the_arms_own_base_is_refused(env):
    pytest.importorskip("mujoco")
    pytest.importorskip("so101_nexus")
    path = write_traj(env, through_the_base_traj())
    with pytest.raises(SystemExit):
        env.replay.main([path, "--left", "L", "--right", "R", "--speed", "1"])
    assert env.ports == {}  # refused before any bus was opened


def test_no_collision_check_plays_it_anyway(env):
    path = write_traj(env, through_the_base_traj())
    assert env.replay.main([path, "--left", "L", "--right", "R", "--dry-run", "--no-collision-check"]) == 0


def fake_closest(band):
    """A stand-in for collision.closest: the arm overlaps itself wherever shoulder lift (rad) is inside `band`.
    Returns the stand-in and the list of row sets it was asked about."""
    asked = []

    def closest(rows):
        asked.append([list(row) for row in rows])
        for step, row in enumerate(rows):
            if band[0] <= row[1] <= band[1]:
                return -0.02, step, "stand-in parts"
        return 0.05, 0, "nothing"

    return closest, asked


def test_a_glide_to_the_idle_pose_through_the_arm_is_refused_before_torque_goes_on(env, monkeypatch):
    collision = pytest.importorskip("collision")
    closest, asked = fake_closest((-0.6, -0.4))
    monkeypatch.setattr(collision, "closest", closest)
    traj = make_traj()
    idle = {m: 2047 for m in range(1, 7)}
    idle[2] = 2047 - round(1.0 / env.replay.RAD_PER_STEP)  # shoulder lift -1 rad: the glide crosses the band
    write_idle(env, "L", idle)
    write_idle(env, "R", idle)
    with pytest.raises(SystemExit):
        env.replay.main([write_traj(env, traj), "--left", "L", "--right", "R", "--speed", "1"])
    assert env.ports["devL"].torque == []
    assert env.ports["devL"].writes == []


def test_without_the_gripper_the_check_takes_the_jaw_at_both_ends_of_its_travel(env, monkeypatch):
    collision = pytest.importorskip("collision")
    closest, asked = fake_closest((5.0, 6.0))  # never overlaps
    monkeypatch.setattr(collision, "closest", closest)
    traj = make_traj()
    args = [write_traj(env, traj), "--left", "L", "--right", "R", "--dry-run", "--no-gripper"]
    assert env.replay.main(args) == 0
    f = env.replay.Follower("L")
    ends = [f.to_rad(6, f.limits[6][0]), f.to_rad(6, f.limits[6][1])]
    jaws_seen = set()
    for rows in asked:
        for row in rows:
            jaws_seen.add(round(row[5], 6))
    assert jaws_seen == {round(ends[0], 6), round(ends[1], 6)}

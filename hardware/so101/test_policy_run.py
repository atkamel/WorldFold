"""policy_run: the dry run refuses a policy whose rollout goes through the arm's base, breaks the rate cap or misses
the planner's pinch; on the (fake) arms a row that fails a check, or an arm far from its last row, stops the run
with torque on and nothing more sent. A stand-in policy plays the planner's own rows, so each test breaks one thing.

    /Users/akamel/Code/WorldFold/.venv/bin/python -m pytest hardware/so101/test_policy_run.py
"""

import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fold_traj  # noqa: E402
import policy_data  # noqa: E402
from policy_net import DT  # noqa: E402
from test_replay import env, step_lines  # noqa: E402,F401  (env: the fake servo bus)

pytest.importorskip("mujoco")
pytest.importorskip("so101_nexus")


class Replayer:
    """A stand-in policy: the planner's row for the time asked, whatever it reads. `change(k, rows, live)` may alter
    the rows given for row k; `live` is True once the dry-run rollout is done (on the arms)."""

    def __init__(self, traj, change=None):
        self.plan = {side: np.array(traj["arms"][side]["target"]) for side in ("left", "right")}
        self.rows = len(self.plan["left"]) + 5
        self.change = change
        self.calls = 0

    def next_targets(self, points, measured, t, previous=None):
        k = min(int(round(t / DT)), len(self.plan["left"]) - 1)
        self.calls += 1
        out = {side: self.plan[side][k].copy() for side in self.plan}
        if self.change:
            self.change(k, out, self.calls > self.rows)
        return out


@pytest.fixture(scope="module")
def planned():
    points = policy_data.cloth_points("sides")
    return points, fold_traj.fold(points)


@pytest.fixture
def run_env(env, planned, monkeypatch):  # noqa: F811
    sys.modules.pop("policy_run", None)
    env.policy_run = importlib.import_module("policy_run")     # on the fake bus's replay
    env.points, env.traj = planned
    monkeypatch.setattr(fold_traj, "fold", lambda points: env.traj)
    monkeypatch.setattr(env.replay, "sleep_until", lambda t: None)
    env.followers = {"left": env.replay.Follower("L"), "right": env.replay.Follower("R")}
    return env


def options(**changed):
    return types.SimpleNamespace(**({"dry_run": False, "speed": 1.0, "torque_limit": 100, "max_error_mm": 5.0,
                                     "track_deg": 8.0, "stay_deg": 6.0, "log": None, "policy": "stand-in"} | changed))


def run(env, policy, **changed):
    return env.policy_run.run(env.points, env.followers, policy, options(**changed))


def refusal(env, policy):
    with pytest.raises(SystemExit) as stopped:
        run(env, policy, dry_run=True)
    assert env.ports == {}                                      # refused before any port opened
    return str(stopped.value)


# --- before any port opens ---

def test_the_planners_own_rows_pass_the_dry_run_without_opening_a_port(run_env, capsys):
    assert run(run_env, Replayer(run_env.traj), dry_run=True) == 0
    out = capsys.readouterr().out
    assert run_env.ports == {}
    assert "0.0 mm where the jaw shuts, 0.0 mm where it opens" in out and "Dry run" in out


def test_a_rollout_through_the_arms_own_base_is_refused(run_env):
    def through_the_base(k, rows, live):
        if 100 <= k <= 102:
            rows["left"][1:4] = np.radians([-81, 80, 95])      # test_replay's pose inside the base
    assert "pass through or too close to itself" in refusal(run_env, Replayer(run_env.traj, through_the_base))


def test_a_rollout_over_the_rate_cap_is_refused(run_env):
    def jump(k, rows, live):
        if k == 80:
            rows["right"][2] += math.radians(4)                # one row 4 deg out and back: 3 deg is the cap
    assert "over the 3.0 deg cap" in refusal(run_env, Replayer(run_env.traj, jump))


def test_a_rollout_that_misses_the_pinch_is_refused(run_env):
    def aside(k, rows, live):
        rows["left"][0] += math.radians(2)                     # pan 2 deg off throughout: ~8 mm at the corner
    assert "misses the planner's pinch or release" in refusal(run_env, Replayer(run_env.traj, aside))


def test_a_rollout_whose_jaw_never_opens_is_refused(run_env):
    def shut(k, rows, live):
        if k > 40:
            rows["left"][5] = -0.17
    assert "the jaw never opens again" in refusal(run_env, Replayer(run_env.traj, shut))


def test_a_rollout_that_is_not_a_number_is_refused(run_env):
    def nan(k, rows, live):
        if k == 60:
            rows["left"][3] = float("nan")
    assert "not a number" in refusal(run_env, Replayer(run_env.traj, nan))


def test_points_the_planner_refuses_are_refused(run_env, monkeypatch):
    def refuse(points):
        raise ValueError("no release keeps the arm clear of itself")
    monkeypatch.setattr(fold_traj, "fold", refuse)
    assert "planner refuses" in refusal(run_env, Replayer(run_env.traj))


# --- on the arms ---

def sent_to(env, side, row):
    """The goals the fake arm gets for the planner's row `row`."""
    f = env.followers[side]
    ticks = env.replay.convert(f, [env.traj["arms"][side]["target"][row]])[0][0]
    return {m: round(t) for m, t in zip(range(1, 7), ticks)}


def test_a_full_run_logs_every_row_sent_and_read(run_env, tmp_path):
    policy = Replayer(run_env.traj)
    assert run(run_env, policy, log=str(tmp_path / "steps.jsonl")) == 0
    rows = [line for line in step_lines(tmp_path / "steps.jsonl") if line["phase"] == "fold"]
    assert [line["row"] for line in rows] == list(range(1, policy.rows))
    for line in rows:
        for side in ("left", "right"):
            assert len(line["target"][side]) == 6 and len(line["measured"][side]) == 6
    last = len(run_env.traj["arms"]["left"]["target"]) - 1
    assert run_env.ports["devL"].writes[-1] == sent_to(run_env, "left", last)
    assert all(v == 1 for port in run_env.ports.values() for _, v in port.torque)


def test_a_row_that_fails_its_check_on_the_arms_stops_the_run_and_sends_nothing_more(run_env, capsys):
    def jump_on_the_arms(k, rows, live):
        if live and k == 50:
            rows["left"][2] += math.radians(10)                # the dry run never saw it
    assert run(run_env, Replayer(run_env.traj, jump_on_the_arms)) == 1
    assert "Stopped at row 50" in capsys.readouterr().out
    for side, dev in (("left", "devL"), ("right", "devR")):
        port = run_env.ports[dev]
        assert port.writes[-1] == sent_to(run_env, side, 49)   # the last row sent is the one before
        assert all(v == 1 for _, v in port.torque)             # torque on: the arms hold
    elbow = sent_to(run_env, "left", 50)[3] + round(10 * run_env.replay.STEPS_PER_DEGREE)
    assert all(abs(w[3] - elbow) > 50 for w in run_env.ports["devL"].writes)


def test_rows_that_drift_from_the_checked_rollout_on_the_arms_stop_the_run(run_env, capsys):
    def drift_on_the_arms(k, rows, live):
        if live and k >= 40:
            rows["left"][1] += math.radians(0.5 * (k - 39))     # half a degree a row: inside the rate cap, every row
    assert run(run_env, Replayer(run_env.traj, drift_on_the_arms)) == 1
    out = capsys.readouterr().out
    assert "the row the dry run checked" in out
    assert "Stopped at row 51" in out or "Stopped at row 52" in out               # 6 to 6.5 deg away, at the 6 deg limit
    assert all(v == 1 for _, v in run_env.ports["devL"].torque)


def hook_port(env, monkeypatch, name, on_write):
    """Give arm `name`'s fake bus the `on_write` hook once the run connects it."""
    connect = env.replay.Follower.connect

    def connected(follower):
        connect(follower)
        if follower.name == name:
            follower.port.on_write = on_write
    monkeypatch.setattr(env.replay.Follower, "connect", connected)


def test_an_arm_far_from_its_last_row_stops_the_run(run_env, monkeypatch, capsys):
    state = {"slipping": False}

    def from_row_30(k, rows, live):
        state["slipping"] |= live and k == 30

    def sag(n):
        if state["slipping"]:                                  # the right elbow stops 10 deg short of every goal
            port = run_env.ports["devR"]
            port.pos[3] = port.writes[-1][3] - round(10 * run_env.replay.STEPS_PER_DEGREE)
    hook_port(run_env, monkeypatch, "R", sag)
    assert run(run_env, Replayer(run_env.traj, from_row_30)) == 1
    out = capsys.readouterr().out
    assert "Stopped at row 31: the right arm's elbow_flex is 10." in out and "deg from the row sent before" in out
    for side, dev in (("left", "devL"), ("right", "devR")):
        assert run_env.ports[dev].writes[-1] == sent_to(run_env, side, 30)     # nothing after row 30
        assert all(v == 1 for _, v in run_env.ports[dev].torque)


def test_ctrl_c_on_the_arms_stops_with_torque_on(run_env, monkeypatch, capsys):
    state = {"row_20": False}

    def at_row_20(k, rows, live):
        state["row_20"] |= live and k == 20

    def interrupt(n):
        if state["row_20"]:
            raise KeyboardInterrupt
    hook_port(run_env, monkeypatch, "L", interrupt)
    assert run(run_env, Replayer(run_env.traj, at_row_20)) == 1
    assert "Stopped (Ctrl-C) at row 20" in capsys.readouterr().out
    assert run_env.ports["devL"].writes[-1] != sent_to(run_env, "left", 20)
    assert run_env.ports["devR"].writes[-1] == sent_to(run_env, "right", 19)    # the left arm is sent first
    assert all(v == 1 for port in run_env.ports.values() for _, v in port.torque)

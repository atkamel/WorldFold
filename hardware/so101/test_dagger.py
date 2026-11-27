"""dagger_run on the fake servo bus, through fold_camera.py's own fold on a synthetic picture: what reaches the arms
with the network trusted and not, which rows the scripted fold takes back, what is saved, and one round of training
on the saved rows. Plus the numpy network's gradients.

    /Users/akamel/Code/WorldFold/.venv/bin/python -m pytest hardware/so101/test_dagger.py
"""

import importlib
import json
import math
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

pytest.importorskip("mujoco")
pytest.importorskip("so101_nexus")

import dagger_data  # noqa: E402
import dagger_net  # noqa: E402
import dagger_train  # noqa: E402
import fold_camera  # noqa: E402
import fold_traj  # noqa: E402
from test_fold_camera import BASES, NEAR_EDGE, SIDE, cloth, render  # noqa: E402
from test_replay import env, step_lines  # noqa: E402,F401  (env: the fake servo bus)

PLANS = {}          # the planner's answers, kept across tests: a fold takes seconds to plan
REAL_FOLD = fold_traj.fold


def remembered_fold(points, closed=-0.17, empty_reading=False, pinch_z=fold_traj.PINCH_Z):
    key = json.dumps([np.round(np.array([points[s] for s in ("left", "right")]), 6).tolist(), closed, empty_reading,
                      pinch_z if not isinstance(pinch_z, dict) else sorted(pinch_z.items())])
    if key not in PLANS:
        PLANS[key] = REAL_FOLD(points, closed, empty_reading, pinch_z)
    return json.loads(json.dumps(PLANS[key]))


class OnPlan:
    """A stand-in network: the plan's own row for the time asked. `change(k, side, row)` may alter it."""

    def __init__(self, traj, change=None):
        self.plan = {side: np.array(traj["arms"][side]["target"]) for side in ("left", "right")}
        self.meta, self.path, self.change, self.asked = {"closed": -0.17}, Path("stand-in.npz"), change, 0

    def targets(self, inputs, measured, sent, t):
        k = int(round(t / dagger_net.DT))
        self.asked += 1
        out = {}
        for side in measured:
            out[side] = self.plan[side][k].copy()
            if self.change:
                out[side] = self.change(k, side, out[side])
        return out


@pytest.fixture
def rig(env, tmp_path, monkeypatch):  # noqa: F811
    """The fake bus, a calibrated camera, a picture of a moved cloth, and dagger_run on that bus's replay."""
    monkeypatch.setattr(fold_traj, "fold", remembered_fold)
    monkeypatch.setattr(fold_camera, "CAMERA", tmp_path / "camera.json")
    monkeypatch.setattr(fold_camera, "HERE", tmp_path)
    monkeypatch.setattr(env.replay, "sleep_until", lambda t: None)
    sys.modules.pop("dagger_run", None)
    env.dagger_run = importlib.import_module("dagger_run")
    ruler, moved = tmp_path / "ruler.png", tmp_path / "moved.png"
    Image.fromarray(render(cloth())).save(ruler)
    Image.fromarray(render(cloth(shift=(0.01, -0.01), turn_deg=4.0), BASES)).save(moved)
    assert fold_camera.main(["calibrate", "--image", str(ruler), "--out", str(tmp_path / "a.png")]) == 0
    cal = json.loads(fold_camera.CAMERA.read_text())
    env.traj = fold_camera.plan(np.asarray(Image.open(moved).convert("RGB")), cal, 0.30, -0.17)[0]
    env.store = tmp_path / "store"
    env.fold_args = ["--image", str(moved), "--out", str(tmp_path / "look.png"), "--left", "L", "--right", "R",
                     "--speed", "1"]
    return env


def run(rig, policy=None, *more, monkeypatch=None, trust="2"):
    """dagger_run on the saved picture, inside 2 deg unless `more` names a band or `trust` is None."""
    if policy is not None:
        monkeypatch.setattr(rig.dagger_run.dagger_net, "Policy", lambda path: policy)
    band = [] if trust is None or "--trust-deg" in more else ["--trust-deg", trust]
    return rig.dagger_run.main(["--store", str(rig.store), *band, *more, *rig.fold_args])


def played(rig, since):
    """{device: the goals written to it since `since` ({device: count}) that belong to the fold's rows}."""
    rows = len(rig.traj["arms"]["left"]["target"])
    sends = math.ceil(rig.traj["dt"] / rig.replay.TICK_SECONDS - 1e-9)
    return {dev: port.writes[since.get(dev, 0):][-rows * sends:] for dev, port in rig.ports.items()}


def saved_rows(rig):
    files = sorted(rig.store.glob("rows_*.npz"))
    assert len(files) == 1
    return np.load(files[0])


def plan_ticks(rig, side):
    return np.array(rig.replay.convert(rig.replay.Follower("L"), rig.traj["arms"][side]["target"])[0])


# --- what reaches the arms ---

def test_with_no_trust_the_arms_get_what_the_scripted_fold_sends(rig, monkeypatch):
    assert fold_camera.main(["fold", *rig.fold_args]) == 0
    scripted = played(rig, {})
    counts = {dev: len(port.writes) for dev, port in rig.ports.items()}
    wild = OnPlan(rig.traj, lambda k, side, row: row + 0.5)            # a network that is 29 deg off everywhere
    assert run(rig, wild, "--trust-deg", "0", "--no-train", monkeypatch=monkeypatch) == 0
    assert played(rig, counts) == scripted
    rows = saved_rows(rig)
    assert not rows["drove"].any() and wild.asked == len(rig.traj["arms"]["left"]["target"])


def test_a_network_on_the_plan_drives_every_row_but_the_last(rig, monkeypatch, capsys):
    assert run(rig, OnPlan(rig.traj), "--no-train", monkeypatch=monkeypatch) == 0
    rows = saved_rows(rig)
    T = len(rig.traj["arms"]["left"]["target"])
    assert len(rows["drove"]) == 2 * T and rows["obs"].shape == (2 * T, len(dagger_net.OBS))
    assert rows["drove"][rows["row"] < T - 1].all() and not rows["drove"][rows["row"] == T - 1].any()
    assert f"drove {T - 1} of {T} rows" in capsys.readouterr().out
    follower = rig.replay.Follower("L")
    for side, mirror in (("left", 1.0), ("right", dagger_net.MIRROR_JOINTS)):
        mine = rows["side"] == side                                  # each row's reading, time and answer line up
        plan = np.array([[follower.to_rad(m, t) for m, t in zip(range(1, 7), np.round(row))]
                         for row in plan_ticks(rig, side)]) * mirror
        assert np.array_equal(rows["row"][mine], np.arange(T))
        assert np.allclose(rows["teacher"][mine], plan, atol=2e-3)                      # the plan's row k
        assert np.allclose(rows["obs"][mine][1:, 10:16], plan[:-1], atol=2e-3)          # read: the row before
        assert np.allclose(rows["obs"][mine][1:, 16:22], plan[:-1], atol=2e-3)          # sent last: the same
        assert np.allclose(rows["obs"][mine][0, 16:22], plan[0], atol=2e-3)             # row 0: where it was glided
        assert np.allclose(rows["obs"][mine][:, 22], np.arange(T) * rig.traj["dt"])     # at time k * dt
        assert set(rows["obs"][mine][:, 23]) == {1.0 if side == "left" else -1.0}
    assert not np.allclose(rows["teacher"][rows["side"] == "left"], rows["teacher"][rows["side"] == "right"],
                           atol=0.02)                                # the two arms' plans do differ here
    for dev, side in (("devL", "left"), ("devR", "right")):         # and the arm ends on the plan's last row
        assert rig.ports[dev].writes[-1] == dict(zip(range(1, 7), np.round(plan_ticks(rig, side)[-1]).astype(int)))
    log = step_lines(next((rig.tmp / "runs").glob("*/steps.jsonl")))
    who = [line for line in log if line["phase"] == "dagger"]
    assert len(who) == T and who[0]["who"] == {"left": "policy", "right": "policy"}
    assert sum(line["phase"] == "fold" for line in log) == T


def test_a_row_that_strays_is_the_plans_and_the_arm_stays_in_the_band(rig, monkeypatch):
    def change(k, side, row):
        if side == "left" and 40 <= k < 60:
            row[2] += math.radians(9.0)                              # the elbow, far outside the band
        elif side == "right":
            row[0] += math.radians(1.9 * (-1) ** k)                  # the pan, jumping 3.8 deg a row inside it: over
        return row                                                   # the rate cap, so the cap is at work too

    assert run(rig, OnPlan(rig.traj, change), "--trust-deg", "2", "--no-train", monkeypatch=monkeypatch) == 0
    rows = saved_rows(rig)
    left, right = rows["side"] == "left", rows["side"] == "right"
    strayed = left & (rows["row"] >= 40) & (rows["row"] < 60)
    assert not rows["drove"][strayed].any() and set(rows["why"][strayed]) == {"strayed"}
    assert rows["drove"][right & (rows["row"] < rows["row"].max())].all()
    sends = math.ceil(rig.traj["dt"] / rig.replay.TICK_SECONDS - 1e-9)
    for dev, side in (("devL", "left"), ("devR", "right")):
        sent = np.array([[goal[m] for m in range(1, 7)] for goal in played(rig, {})[dev][sends - 1::sends]])
        off = np.abs(sent - plan_ticks(rig, side)) / rig.replay.STEPS_PER_DEGREE
        assert off.max() <= 2.0 + 0.1                                # never outside the band around the plan
        last = dict(zip(range(1, 7), np.round(plan_ticks(rig, side)[-1]).astype(int)))
        assert rig.ports[dev].writes[-1] == last                     # and it ends on the plan's last row
    assert off.max() > 1.0                                           # and the right arm did follow the network


def test_an_arm_the_rate_cap_held_short_of_the_last_row_is_stepped_onto_it(rig, monkeypatch):
    far = OnPlan(rig.traj, lambda k, side, row: row + np.array([math.radians(4.5), 0, 0, 0, 0, 0]))
    assert run(rig, far, "--trust-deg", "5", "--no-train", monkeypatch=monkeypatch) == 0
    assert saved_rows(rig)["drove"].sum() > 300                      # 4.5 deg off, inside 5: the network drove
    sends = math.ceil(rig.traj["dt"] / rig.replay.TICK_SECONDS - 1e-9)
    for dev, side in (("devL", "left"), ("devR", "right")):         # 4.5 deg is more than one row's 3 deg
        last = dict(zip(range(1, 7), np.round(plan_ticks(rig, side)[-1]).astype(int)))
        assert rig.ports[dev].writes[-1] == last
        steps = np.abs(np.diff([[g[m] for m in range(1, 7)] for g in rig.ports[dev].writes[-3 * sends:]], axis=0))
        assert steps.max() <= rig.replay.MAX_DEG_S * rig.traj["dt"] * rig.replay.STEPS_PER_DEGREE / sends + 1


def roughest(writes, sends):
    """The most any motor's row-to-row step changes from one row to the next (ticks), over `writes`."""
    rows = np.array([[g[m] for m in range(1, 7)] for g in writes[sends - 1::sends]], float)
    return np.abs(np.diff(rows, n=2, axis=0)).max()


def test_smoothing_rounds_the_corners_keeps_the_rate_cap_and_still_ends_on_the_last_row(rig, monkeypatch):
    sends = math.ceil(rig.traj["dt"] / rig.replay.TICK_SECONDS - 1e-9)
    cap = rig.replay.MAX_DEG_S * rig.traj["dt"] * rig.replay.STEPS_PER_DEGREE / sends
    assert run(rig, OnPlan(rig.traj), "--trust-deg", "0", "--no-train", monkeypatch=monkeypatch) == 0
    counts = {dev: len(port.writes) for dev, port in rig.ports.items()}
    plain = {dev: roughest(played(rig, {})[dev], sends) for dev in rig.ports}
    for f in rig.store.glob("rows_*.npz"):
        f.unlink()
    assert run(rig, OnPlan(rig.traj), "--trust-deg", "0", "--no-train", "--smooth", "0.5",
               monkeypatch=monkeypatch) == 0
    rows = len(rig.traj["arms"]["left"]["target"])
    for dev, side in (("devL", "left"), ("devR", "right")):
        fold = rig.ports[dev].writes[counts[dev]:][-(rows + 1) * sends:]
        assert roughest(fold[:rows * sends], sends) < 0.7 * plain[dev]
        steps = np.abs(np.diff([[g[m] for m in range(1, 7)] for g in fold], axis=0))
        assert steps.max() <= cap + 1
        last = dict(zip(range(1, 7), np.round(plan_ticks(rig, side)[-1]).astype(int)))
        assert rig.ports[dev].writes[-1] == last


def test_smoothing_outside_a_tenth_to_one_is_refused(rig, monkeypatch):
    with pytest.raises(SystemExit):
        run(rig, OnPlan(rig.traj), "--no-train", "--smooth", "0", monkeypatch=monkeypatch)


def test_a_pose_too_close_to_itself_is_the_plans(rig, monkeypatch):
    import collision
    real, live = collision.closest, {"on": False}

    def closest(rows):
        if live["on"] and len(rows) == 1:
            return 0.001, 0, ("gripper", "base")
        return real(rows)

    def change(k, side, row):
        live["on"] = True                                            # the network is only asked once the fold plays
        return row

    monkeypatch.setattr(collision, "closest", closest)
    assert run(rig, OnPlan(rig.traj, change), "--no-train", monkeypatch=monkeypatch) == 0
    rows = saved_rows(rig)
    assert not rows["drove"].any() and "too close to itself" in set(rows["why"])


def test_a_network_that_gives_no_number_is_never_sent(rig, monkeypatch):
    nan = OnPlan(rig.traj, lambda k, side, row: row * np.nan)
    assert run(rig, nan, "--no-train", monkeypatch=monkeypatch) == 0
    assert set(saved_rows(rig)["why"]) == {"not a number"}
    sends = math.ceil(rig.traj["dt"] / rig.replay.TICK_SECONDS - 1e-9)
    for dev, side in (("devL", "left"), ("devR", "right")):         # what was sent is the plan, row for row
        sent = np.array([[goal[m] for m in range(1, 7)] for goal in played(rig, {})[dev][sends - 1::sends]])
        assert np.array_equal(sent, np.round(plan_ticks(rig, side)))


def test_a_row_that_puts_the_fingertip_under_the_plans_is_the_plans(rig, monkeypatch):
    from kinematics import Arm
    arm, pinch = Arm(), rig.traj["marks"]["pinch"]
    plan = np.array(rig.traj["arms"]["left"]["target"])
    drop = {}
    for sign in (1.0, -1.0):                                         # which way the shoulder lowers the fingertip
        bent = plan[pinch].copy()
        bent[1] += sign * math.radians(1.5)
        drop[sign] = arm.tip(plan[pinch][:5])[0][2] - arm.tip(bent[:5])[0][2]
    down = max(drop, key=drop.get)
    assert drop[down] > 0.004                                        # 1.5 deg, inside the band, is 4 mm or more

    def change(k, side, row):
        if side == "left" and k == pinch:
            row[1] += down * math.radians(1.5)
        return row

    assert run(rig, OnPlan(rig.traj, change), "--no-train", monkeypatch=monkeypatch) == 0
    rows = saved_rows(rig)
    at = (rows["side"] == "left") & (rows["row"] == pinch)
    assert list(rows["why"][at]) == ["fingertip under the plan's"] and not rows["drove"][at].any()


def test_ctrl_c_in_the_fold_keeps_the_rows_played_and_puts_the_functions_back(rig, monkeypatch, capsys):
    fold, play = fold_traj.fold, rig.replay.play
    policy = OnPlan(rig.traj)

    def change(k, side, row):
        if k == 30:
            raise KeyboardInterrupt
        return row

    policy.change = change
    assert run(rig, policy, "--no-train", monkeypatch=monkeypatch) == 1
    assert "Stopped (Ctrl-C)" in capsys.readouterr().out
    rows = saved_rows(rig)
    assert rows["row"].max() == 29 and len(rows["row"]) == 60
    assert fold_traj.fold is fold and rig.replay.play is play


def test_the_next_round_never_takes_a_number_in_use(rig, tmp_path):
    store = tmp_path / "rounds"
    store.mkdir()
    for name in ("round_002.npz", "round_002.json", "round_003.npz", "round_003.json", "round_009.npz"):
        (store / name).write_text("")
    assert rig.dagger_run.next_round(store).name == "round_010.npz"
    assert rig.dagger_run.latest(store).name == "round_003.npz"      # round_009 has no JSON: not a finished round


def test_a_plan_the_network_was_not_trained_for_is_left_to_the_scripted_fold(rig, monkeypatch, capsys):
    policy = OnPlan(rig.traj)
    assert run(rig, policy, "--closed", "-0.15", monkeypatch=monkeypatch) == 0
    assert policy.asked == 0 and not list(rig.store.glob("rows_*.npz"))
    assert "nothing was learned: the jaw shuts to -0.15" in capsys.readouterr().out
    assert rig.ports["devL"].writes                                  # the fold itself was played


def test_a_dry_run_opens_no_port_and_says_what_the_network_would_drive(rig, monkeypatch, capsys):
    assert run(rig, OnPlan(rig.traj), "--dry-run", monkeypatch=monkeypatch) == 0
    assert rig.ports == {} and not rig.store.exists()
    assert "the network drove" in capsys.readouterr().out


def test_the_scripted_folds_functions_are_put_back(rig, monkeypatch):
    fold, play = fold_traj.fold, rig.replay.play
    assert run(rig, OnPlan(rig.traj), "--dry-run", monkeypatch=monkeypatch) == 0
    assert fold_traj.fold is fold and rig.replay.play is play


# --- starting off with the scripted fold, and the pictures ---

def test_the_first_fold_is_the_scripted_fold_and_the_network_drives_once_it_has_earned_it(rig, monkeypatch, capsys):
    assert fold_camera.main(["fold", *rig.fold_args]) == 0
    scripted = played(rig, {})
    counts = {dev: len(port.writes) for dev, port in rig.ports.items()}
    assert run(rig, OnPlan(rig.traj), "--no-train", monkeypatch=monkeypatch, trust=None) == 0
    assert "No fold saved yet: the scripted fold drives this one" in capsys.readouterr().out
    assert played(rig, counts) == scripted                           # nothing saved yet: the scripted fold, exactly
    first = np.load(sorted(rig.store.glob("rows_*.npz"))[0])
    assert not first["drove"].any()
    assert rig.dagger_run.earned(rig.store) == 1.0                   # the network was on the plan throughout
    (rig.store / "rows_00000000_000000.npz").unlink(missing_ok=True)
    next(rig.store.glob("rows_*.npz")).rename(rig.store / "rows_20000101_000000.npz")
    assert run(rig, OnPlan(rig.traj), "--no-train", monkeypatch=monkeypatch, trust=None) == 0
    assert "it drives this one inside 2 deg" in capsys.readouterr().out
    newest = np.load(sorted(rig.store.glob("rows_*.npz"))[-1])
    assert newest["drove"].sum() == len(newest["drove"]) - 2         # every row but each arm's last


def test_a_network_that_was_off_the_plan_on_the_last_fold_hands_the_next_one_back(rig, monkeypatch, capsys):
    off = OnPlan(rig.traj, lambda k, side, row: row + (0.1 if k % 5 == 0 else 0.0))    # 5.7 deg off on a fifth of rows
    assert run(rig, off, "--no-train", monkeypatch=monkeypatch) == 0                    # a fold inside 2 deg
    assert rig.dagger_run.earned(rig.store) == pytest.approx(0.8, abs=0.01)
    capsys.readouterr()
    next(rig.store.glob("rows_*.npz")).rename(rig.store / "rows_20000101_000000.npz")
    assert run(rig, OnPlan(rig.traj), "--no-train", monkeypatch=monkeypatch, trust=None) == 0
    assert "under 90%: the scripted fold drives this one" in capsys.readouterr().out
    assert not np.load(sorted(rig.store.glob("rows_*.npz"))[-1])["drove"].any()


def test_a_fold_with_the_phone_saves_its_pictures_beside_the_log(rig, monkeypatch):
    import rgbd
    import rig_record
    picture = np.asarray(Image.open(rig.fold_args[1]).convert("RGB"))

    class Phone:
        def __init__(self, device=0):
            self.device, self.started = device, False

        def start(self):
            self.started = True

        def latest(self):
            return types.SimpleNamespace(rgb=picture, depth=np.full(picture.shape[:2], 0.5, np.float32),
                                         K=(500.0, 500.0, 320.0, 240.0))

        def wait(self, timeout=5.0):
            assert self.started
            return self.latest()

    monkeypatch.setattr(rgbd, "Phone", Phone)
    monkeypatch.setattr(rig_record, "FRAME_SECONDS", 0.01)
    live = [a for a in rig.fold_args if a not in ("--image", rig.fold_args[1])]        # no --image: the phone is used
    monkeypatch.setattr(rig.dagger_run.dagger_net, "Policy", lambda path: OnPlan(rig.traj))
    assert rig.dagger_run.main(["--store", str(rig.store), "--trust-deg", "2", "--no-train", *live]) == 0
    folder = next(p for p in (rig.tmp / "runs").iterdir() if (p / "steps.jsonl").exists())
    frames = step_lines(folder / "frames.jsonl")
    assert len(frames) >= 2 and len(list((folder / "frames").iterdir())) >= 2
    note = json.loads((folder / "dagger.json").read_text())
    assert Path(note["rows"]).exists() and note["trust_deg"] == 2.0
    assert any(line["phase"] == "dagger" for line in step_lines(folder / "steps.jsonl"))
    assert not list(Path(tempfile.gettempdir()).glob("dagger_look_*.png"))


# --- the intervention button ---

class Hands:
    """Stand-ins for the Xbox button and the gamepad teleop: the fold is told to wait at the button read numbered
    `at` (one read a row until then) for `ticks` steps of the teleop, in each of which the fingertip of each arm in
    `moves` ({side: (x, y, z) m in all}) goes a `ticks`-th of its way."""

    def __init__(self, followers, at=None, ticks=0, moves=None):
        from kinematics import Arm
        self.followers, self.at, self.ticks, self.moves = followers, at, ticks, moves or {}
        self.reads, self.arm, self.waiting = 0, Arm(), False

    def read(self):
        self.reads += 1
        return self.at is not None and self.at < self.reads <= self.at + self.ticks + 1

    def take(self, command):
        self.commands = {side: dict(c) for side, c in command.items()}
        self.start = {side: np.array([self.followers[side].to_rad(m, c[m]) for m in range(1, 7)])
                      for side, c in command.items()}
        self.done = 0

    def tick(self):
        self.done += 1
        for side, move in self.moves.items():
            f, q = self.followers[side], self.start[side]
            tip = self.arm.tip(q[:5])[0] + np.array(move) * self.done / self.ticks
            solved = self.arm.solve(tip, q[:5])[0]
            self.commands[side] = {m: f.to_ticks(m, a) for m, a in zip(range(1, 7), [*solved, q[5]])}
        return "Driving."

    def command(self, side):
        return dict(self.commands[side])

    def close(self):
        pass


def nudged(rig, monkeypatch, *more, **hands):
    import dagger_nudge
    made = {}

    def controls(followers):
        made["hands"] = Hands(followers, **hands)
        return made["hands"], made["hands"]

    monkeypatch.setattr(dagger_nudge, "controls", controls)
    return run(rig, OnPlan(rig.traj), "--trust-deg", "0", "--no-train", *more, monkeypatch=monkeypatch)


def tips_sent(rig, side, run_folder):
    """(row numbers, the fingertip of each row sent to `side`, the fingertip of the plan's row) from a run's log."""
    from kinematics import Arm
    arm, plan = Arm(), np.array(rig.traj["arms"][side]["target"])
    lines = [line for line in step_lines(run_folder / "steps.jsonl") if line["phase"] == "fold"]
    rows = np.array([line["row"] for line in lines])
    sent = np.array([arm.tip(line["target"][side][:5])[0] for line in lines])
    return rows, sent, np.array([arm.tip(plan[k][:5])[0] for k in rows])


def newest_run(rig):
    return sorted(p for p in (rig.tmp / "runs").iterdir() if (p / "steps.jsonl").exists())[-1]


def test_where_the_teleop_leaves_a_fingertip_moves_the_rest_of_the_fold_and_is_kept(rig, monkeypatch, capsys):
    rig.store.mkdir()
    (rig.store / "rows_19990101_000000.npz").write_bytes(b"an earlier fold")
    T = len(rig.traj["arms"]["left"]["target"])
    assert nudged(rig, monkeypatch, at=20, ticks=30, moves={"left": (0.01, 0.0, -0.002)}) == 0
    out = capsys.readouterr().out
    assert "Waiting on row 20" in out and "Driving." in out and "Going on" in out
    kept = json.loads((rig.store / "nudges.json").read_text())
    assert np.allclose(kept["left"]["pinch"], [0.01, 0.0, -0.002], atol=2e-4)
    assert kept["left"]["carry"] == kept["left"]["pinch"]            # a corrected pinch carries to the place
    assert kept["right"] == {"pinch": [0.0, 0.0, 0.0], "carry": [0.0, 0.0, 0.0]}
    rows, sent, plan = tips_sent(rig, "left", newest_run(rig))
    assert list(rows) == list(range(T))                              # no row lost or played twice by the wait
    after = (rows > 22) & (rows < T - 1)
    assert np.allclose(sent[after] - plan[after], [0.01, 0.0, -0.002], atol=1.5e-3)
    assert np.allclose(sent[rows < 20], plan[rows < 20], atol=1e-4)  # before it: the plan
    rows, sent, plan = tips_sent(rig, "right", newest_run(rig))
    assert np.allclose(sent, plan, atol=1e-4)                        # the arm left alone: the plan throughout
    for dev, side in (("devL", "left"), ("devR", "right")):         # and both end on the plan's last row
        assert rig.ports[dev].writes[-1] == dict(zip(range(1, 7), np.round(plan_ticks(rig, side)[-1]).astype(int)))
    saved = saved_rows(rig)                                          # the earlier fold's rows were put aside
    assert len(list(rig.store.glob("before_nudge_*/rows_19990101_000000.npz"))) == 1
    from kinematics import Arm
    arm, left = Arm(), saved["side"] == "left"
    taught = np.array([arm.tip(q[:5])[0] for q in saved["teacher"][left]])
    planned = np.array([arm.tip(q[:5])[0] for q in np.array(rig.traj["arms"]["left"]["target"])])
    assert np.allclose((taught - planned)[:-1], [0.01, 0.0, -0.002], atol=1.5e-3)    # every row, the early ones too
    assert np.allclose(taught[-1], planned[-1], atol=1e-4)                           # but the last


def test_the_next_fold_starts_with_the_stored_offsets_and_they_can_be_forgotten(rig, monkeypatch, capsys):
    rig.store.mkdir()
    (rig.store / "nudges.json").write_text(json.dumps({"right": {"pinch": [0.0, -0.008, 0.0],
                                                                 "carry": [0.0, -0.008, 0.004]}}))
    T, check = len(rig.traj["arms"]["left"]["target"]), rig.traj["marks"]["check"]
    assert nudged(rig, monkeypatch) == 0
    assert "right pinch (+0.0, -0.8, +0.0) cm" in capsys.readouterr().out
    rows, sent, plan = tips_sent(rig, "right", newest_run(rig))
    early, late = (rows > 3) & (rows <= check), (rows > check + 3) & (rows < T - 1)
    assert np.allclose(sent[early] - plan[early], [0.0, -0.008, 0.0], atol=1.5e-3)
    assert np.allclose(sent[late] - plan[late], [0.0, -0.008, 0.004], atol=1.5e-3)
    rows, sent, plan = tips_sent(rig, "left", newest_run(rig))
    assert np.allclose(sent, plan, atol=1e-4)
    assert not list(rig.store.glob("before_nudge_*"))                # nothing changed in this fold
    assert nudged(rig, monkeypatch, "--forget-nudges") == 0
    assert "Stored offsets: none" in capsys.readouterr().out
    rows, sent, plan = tips_sent(rig, "right", newest_run(rig))
    assert np.allclose(sent, plan, atol=1e-4)
    assert len(list(rig.store.glob("nudges_until_*.json"))) == 1


def test_a_fingertip_left_far_off_is_kept_at_the_limit_and_one_too_close_to_the_arm_is_not_kept(rig, monkeypatch,
                                                                                               capsys):
    import collision
    import dagger_nudge
    assert nudged(rig, monkeypatch, at=20, ticks=30, moves={"left": (0.0, 0.048, 0.0)}) == 0    # 4.8 cm to the left
    kept = json.loads((rig.store / "nudges.json").read_text())
    assert kept["left"]["pinch"][1] == pytest.approx(dagger_nudge.MAX_NUDGE)
    rows, sent, plan = tips_sent(rig, "left", newest_run(rig))
    assert np.abs(sent[rows > 30][:, 1] - plan[rows > 30][:, 1]).max() <= dagger_nudge.MAX_NUDGE + 1.5e-3
    real = collision.closest
    monkeypatch.setattr(collision, "closest", lambda rows: (0.001, 0, ("gripper", "base")) if len(rows) == 1
                        else real(rows))
    (rig.store / "nudges.json").unlink()
    capsys.readouterr()
    assert nudged(rig, monkeypatch, at=20, ticks=30, moves={"left": (0.01, 0.0, 0.0)}) == 0
    assert "left arm's fingertip is not kept" in capsys.readouterr().out
    assert not (rig.store / "nudges.json").exists()


def test_the_xbox_button_turns_waiting_over_on_each_press_and_a_lost_controller_lets_the_fold_go_on():
    import dagger_nudge
    now = [None]
    button = dagger_nudge.Button(lambda: now[0], period=0.001)
    button.close()                                                   # its thread: the test does the looking
    time.sleep(0.02)
    waited = []
    for down in (False, True, True, False, None, True, True, False):   # two presses, held; a gap between them
        now[0] = down
        button.look()
        waited.append(button.read())
    assert waited == [False, True, True, True, True, False, False, False]
    button.waiting, button.seen = True, time.monotonic() - dagger_nudge.LOST - 0.1
    assert button.read() is False                                    # the controller gone: the fold goes on


def test_the_gamepad_teleop_drives_the_folds_own_arms_through_its_own_loop(env, monkeypatch):  # noqa: F811
    """The real gamepad.Gamepad on two fake-bus followers of the repo's own arms: after its stick rest, a left
    stick pushed forward moves the left arm and leaves the right where the fold had it."""
    import dagger_nudge
    from hardware.controller.gip import ControllerState
    monkeypatch.undo()                                               # the repo's arms.json, limits and idle poses
    monkeypatch.setitem(sys.modules, "scservo_sdk", sys.modules["scservo_sdk"])
    sys.modules.pop("gamepad", None)
    followers = {"left": env.replay.Follower("arm2"), "right": env.replay.Follower("arm1")}
    for f in followers.values():
        f.connect()
    teleop = dagger_nudge.Teleop(followers)
    teleop.close()                                                   # its controller thread: the test feeds samples
    time.sleep(0.05)
    teleop.pad.shutdown.clear()
    from kinematics import Arm
    arm = Arm()
    pose = [*arm.solve(np.array([0.28, 0.0, 0.08]), np.radians([0, 40, -30, 80, 0]))[0], 0.4]
    command = {side: {m: f.to_ticks(m, a) for m, a in zip(range(1, 7), pose)} for side, f in followers.items()}
    for side, f in followers.items():
        f.port.pos = {m: round(t) for m, t in command[side].items()}
    teleop.take(command)

    def feed(state, ticks):
        status = None
        for _ in range(ticks):
            teleop.pad.sample = (state, time.monotonic(), "")
            status = teleop.tick()
            time.sleep(0.02)
        return status

    assert feed(ControllerState(), 25).startswith("Driving")         # the sticks rest, then Drive is pressed for you
    status = feed(ControllerState(left_y=30000), 15)
    assert status.startswith("Driving")
    moved = arm.tip([followers["left"].to_rad(m, teleop.command("left")[m]) for m in range(1, 6)])[0]
    assert moved[0] > 0.28 + 0.01                                    # the left fingertip went forward
    assert teleop.command("right") == command["right"]
    assert followers["left"].port.writes and not followers["right"].port.writes


# --- the network and a round of training ---

def test_the_gradients_are_the_slopes_of_the_error():
    rng = np.random.default_rng(0)
    layers = [(w.astype(np.float64), b.astype(np.float64) + 0.1) for w, b in dagger_net.initial(rng, [5, 7, 4, 3])]
    x, y = rng.normal(size=(6, 5)), rng.normal(size=(6, 3))
    output, seen = dagger_net.forward(layers, x)
    loss, grads = dagger_net.gradients(layers, seen, output, y)

    def error(changed):
        return float(np.mean((dagger_net.forward(changed, x)[0] - y) ** 2))

    assert loss == pytest.approx(error(layers))
    for i, j, index in ((0, 0, (2, 3)), (1, 0, (0, 1)), (2, 0, (3, 2)), (0, 1, (4,)), (2, 1, (1,))):
        bumped = [[w.copy(), b.copy()] for w, b in layers]
        bumped[i][j][index] += 1e-6
        slope = (error([tuple(p) for p in bumped]) - loss) / 1e-6
        assert grads[i][j][index] == pytest.approx(slope, rel=1e-3, abs=1e-7)


def test_the_right_arm_is_the_left_arm_in_a_mirror():
    inputs = dagger_net.fold_inputs({"left": ((0.3, -0.02), (0.1, -0.02)), "right": ((0.3, 0.02), (0.1, 0.02))},
                                    {"left": -0.004, "right": -0.004})
    q = np.array([0.2, 0.5, -0.4, 0.9, -0.3, 0.1])
    left = dagger_net.observation(inputs, "left", q, q + 0.01, 1.0)
    right = dagger_net.observation(inputs, "right", q * dagger_net.MIRROR_JOINTS,
                                   (q + 0.01) * dagger_net.MIRROR_JOINTS, 1.0)
    assert np.allclose(left[:-1], right[:-1]) and left[-1] == 1.0 and right[-1] == -1.0


def test_a_fold_on_the_arms_then_a_round_of_training_brings_the_network_nearer(rig, monkeypatch, capsys):
    monkeypatch.setattr(dagger_net, "HIDDEN", (48, 48))
    planned = []
    for i, (shift, turn) in enumerate((((0.01, -0.01), 4.0), ((0.0, 0.0), 0.0), ((-0.01, -0.02), -5.0))):
        corners = dict(zip(("near_left", "near_right", "far_right", "far_left"), cloth(shift=shift, turn_deg=turn)))
        points = fold_camera.fold_points(corners, 0.30)
        planned.append((i, points, fold_traj.PINCH_Z, fold_traj.fold(points, -0.17, False, fold_traj.PINCH_Z)))
    base, first = rig.tmp / "base.npz", rig.tmp / "first.npz"
    dagger_data.save(base, [(i, p, {"left": z, "right": z}, traj) for i, p, z, traj in planned] * 4, [], 0)
    dagger_train.train(base, first, steps=400, quiet=True)
    args = ["--policy", str(first), "--base", str(base), "--steps", "300", "--trust-deg", "2"]
    assert run(rig, None, *args) == 0
    out = capsys.readouterr().out
    assert "Trained on" in out and "round_001.npz" in out
    rows = saved_rows(rig)
    before = dagger_train.errors(dagger_net.Policy(first), rows["obs"], rows["teacher"])["rms_deg"]
    after = dagger_train.errors(dagger_net.Policy(rig.store / "round_001.npz"), rows["obs"], rows["teacher"])["rms_deg"]
    assert max(after.values()) < max(before.values())
    assert rig.dagger_run.latest(rig.store) == rig.store / "round_001.npz"
    history = dagger_net.Policy(rig.store / "round_001.npz").meta["history"]
    assert len(history) == 2 and history[1]["rows_from_arms"] == len(rows["obs"])


def test_without_the_planned_folds_the_rows_are_kept_and_the_command_is_given(rig, monkeypatch, capsys):
    assert run(rig, OnPlan(rig.traj), "--base", str(rig.tmp / "none.npz"), monkeypatch=monkeypatch) == 0
    assert "dagger_data.py --episodes" in capsys.readouterr().out
    assert len(list(rig.store.glob("rows_*.npz"))) == 1 and not list(rig.store.glob("round_*.npz"))

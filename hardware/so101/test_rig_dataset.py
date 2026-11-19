"""rig_record + rig_dataset: a fold played on the (fake) arms with a (fake) phone is recorded, labelled, converted
into the store the simulator's students train from, and a camera policy trains on it and gives an action."""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import recorder  # noqa: E402
import rgbd  # noqa: E402
import rig_dataset  # noqa: E402
import rig_record  # noqa: E402
from kinematics import Arm  # noqa: E402
from test_replay import JOINTS, env  # noqa: E402,F401  (env: replay.py on a fake servo bus)

ROWS = 40


def traj_file(folder):
    """A two-arm trajectory of ROWS rows the fake arms can play: slow sines (under replay.py's rate cap), the jaw
    shutting as it goes."""
    target = [[0.3 * math.sin(k / 40 + j) for j in range(5)] + [0.5 - 0.01 * min(k, 30)] for k in range(ROWS)]
    arm = {"target": target, "q": target[:1] + target}
    path = Path(folder) / "traj.json"
    path.write_text(json.dumps({"dt": 0.02, "joints": JOINTS, "arms": {"left": arm, "right": json.loads(json.dumps(arm))}}))
    return path, np.array(target)


class Phone:
    """A phone whose every picture is one grey level higher than the last, so pictures can be told apart. It counts
    the frames it is sent, as rgbd.Phone does."""

    def __init__(self):
        self.count = 0

    def latest(self):
        self.count += 1
        level = min(self.count, 250)
        return rgbd.Frame(0.0, np.full((48, 64, 3), level, np.uint8), np.full((12, 16), 0.5, np.float32), (50, 50, 32, 24))

    def wait(self, timeout=5.0):
        return self.latest()


def record(env, monkeypatch, answer, argv=("--no-collision-check",)):  # noqa: F811
    monkeypatch.setattr(rig_record, "FRAME_SECONDS", 0.02)
    path, target = traj_file(env.tmp)
    code, folder = rig_record.record(env.replay.main, [str(path), "--left", "L", "--right", "R", "--speed", "1", *argv],
                                     Phone(), runs=env.tmp / "recorded", ask=lambda question: answer)
    return code, folder, target


def test_a_fold_is_recorded_with_its_pictures_steps_and_label(env, monkeypatch):  # noqa: F811
    code, folder, target = record(env, monkeypatch, "y")
    assert code == 0 and json.loads((folder / "label.json").read_text())["success"] is True
    saved = recorder.load(folder)
    assert len(saved["frames"]) >= 5 and (folder / "plan.json").exists()         # the phone was read while it played
    run = rig_dataset.load_run(folder)
    assert run["kind"] == "script" and run["label"]["success"]
    assert run["target"].shape == run["measured"].shape == (ROWS, 12)
    assert np.abs(run["target"][:, :6] - target).max() < 0.002 and np.abs(run["target"][:, 6:] - target).max() < 0.002
    assert np.abs(run["measured"][1:] - run["target"][:-1]).max() < 0.002        # read before a row: the row before it
    assert (run["frame"] >= 0).all() and (np.diff(run["frame"]) >= 0).all() and run["frame"][-1] > run["frame"][0]
    times = np.array([f["t"] for f in saved["frames"]])
    for t, i in zip(run["t"], run["frame"]):
        assert times[i] <= t and (i + 1 == len(times) or times[i + 1] > t)       # the last picture before the row


def test_a_run_that_never_moves_leaves_no_folder(env, monkeypatch):  # noqa: F811
    with pytest.raises(SystemExit):
        rig_record.record(env.replay.main, [str(env.tmp / "missing.json"), "--left", "L", "--right", "R"], Phone(),
                          runs=env.tmp / "recorded", ask=lambda question: "y")
    assert list((env.tmp / "recorded").iterdir()) == []


def test_an_unanswered_run_stays_unlabelled_and_out_of_a_dataset(env, monkeypatch):  # noqa: F811
    _, folder, _ = record(env, monkeypatch, "")
    assert not (folder / "label.json").exists()
    line, usable = rig_dataset.describe(folder)
    assert "unlabelled" in line and not usable
    with pytest.raises(ValueError, match="no good fold to learn from"):
        rig_dataset.build([folder], env.tmp / "store", "rig_t")
    assert rig_dataset.main(["label", str(folder), "bad", "--note", "left corner slipped"]) == 0
    assert json.loads((folder / "label.json").read_text()) == {"success": False, "note": "left corner slipped"}


def test_a_one_arm_run_and_a_run_without_pictures_cannot_be_used(env, monkeypatch):  # noqa: F811
    _, folder, _ = record(env, monkeypatch, "y", argv=("--no-collision-check", "--only", "left"))
    assert "unusable" in rig_dataset.describe(folder)[0]
    _, folder, _ = record(env, monkeypatch, "y", argv=("--no-collision-check", "--no-gripper"))
    assert "never sent" in rig_dataset.describe(folder)[0]                       # the jaw rows logged were not played
    _, folder, _ = record(env, monkeypatch, "y")
    shutil.rmtree(folder / "frames")
    (folder / "frames.jsonl").unlink()
    line, usable = rig_dataset.describe(folder)
    assert "   0 pictures" in line and not usable


def test_a_teleop_recording_becomes_radians_through_each_arms_tick_map(env):  # noqa: F811
    run = recorder.Recorder(env.tmp / "teleop", meta={"task": "teleop", "arms": ["R", "L", "lead"]})
    for k in range(30):
        ticks = [2047 + 10 * k + m for m in range(6)]
        run.step(0.1 * k, k, target={"L": [v + 5 for v in ticks], "R": ticks, "lead": ticks},
                 measured={"L": ticks, "R": [v - 3 for v in ticks], "lead": ticks})
        run.frame(0.1 * k, np.full((48, 64, 3), k, np.uint8))
    run.close()
    got = rig_dataset.load_run(run.path, left="L", right="R")
    step = env.replay.RAD_PER_STEP
    assert got["kind"] == "teleop" and got["measured"].shape == (30, 12)
    assert np.allclose(got["measured"][4, :5], [(10 * 4 + m) * step for m in range(5)])              # left = L, about 2047
    assert np.allclose(got["target"][4, :5] - got["measured"][4, :5], 5 * step)
    assert np.allclose(got["measured"][4, 6:11], [(10 * 4 + m - 3) * step for m in range(5)])        # right = R
    assert list(got["frame"]) == list(range(30))
    with pytest.raises(ValueError, match="not in this recording"):
        rig_dataset.load_run(run.path, left="L", right="arm9")


def test_recorded_folds_become_a_dataset_a_camera_policy_trains_on(env, monkeypatch):  # noqa: F811
    torch = pytest.importorskip("torch")
    from imitation.data.schema import load_dataset
    from imitation.policies.common import load_policy
    from imitation.train import train
    _, good, target = record(env, monkeypatch, "y")
    shutil.copytree(good, good.with_name("second_good"))
    _, bad, _ = record(env, monkeypatch, "n")
    folders = rig_dataset.run_folders(env.tmp / "recorded")
    assert len(folders) == 3
    store = env.tmp / "store"
    assert rig_dataset.main(["build", str(env.tmp / "recorded"), "--out", str(store), "--version", "rig_t", "--size", "64"]) == 0
    manifest, episodes = load_dataset(store, "rig_t", images=True)
    assert manifest["n_episodes"] == 2 and manifest["n_success"] == 2 and {e.meta["run"] for e in episodes} == {good.name, "second_good"}
    ep = next(e for e in episodes if e.meta["run"] == good.name)
    run = rig_dataset.load_run(good)
    T = ROWS + rig_dataset.HOLD                                 # the fold, then the arms at rest where it ended
    assert ep.obs.shape == (T, 139) and ep.actions.shape == (T, 12) and ep.images["main"].shape == (T, 3, 64, 64)
    assert ep.meta["rows_recorded"] == ROWS and ep.meta["success"] is True
    assert np.allclose(ep.actions[:ROWS] * rig_dataset.ACTION_SCALE, run["target"], atol=1e-6)        # what was sent
    assert np.allclose(ep.obs[:ROWS, 0:5], run["measured"][:, 0:5], atol=1e-6) and np.allclose(ep.obs[:ROWS, 25:30], run["measured"][:, 6:11], atol=1e-6)
    assert np.allclose(ep.obs[:ROWS, 10], run["measured"][:, 5], atol=1e-6) and np.allclose(ep.obs[:ROWS, 35], run["measured"][:, 11], atol=1e-6)
    assert np.allclose(ep.obs[7, 11:14], Arm().tip(run["measured"][7, :5])[0], atol=1e-6)             # the fingertip
    assert np.all(ep.obs[:, 50:] == 0) and np.all(ep.obs[:, [24, 49]] == 0)                           # nothing privileged
    assert ep.images["main"][0].max() < ep.images["main"][ROWS - 1].min()                             # each row its own picture
    # rates look back only: row 7's is what rows 6 and 7 read, whatever row 8 read
    dt = run["t"][7] - run["t"][6]
    assert np.allclose(ep.obs[7, 5:10], (run["measured"][7, :5] - run["measured"][6, :5]) / dt, rtol=1e-4, atol=1e-5) and np.all(ep.obs[0, 5:10] == 0)
    # the rest after the fold: the last joints, target and picture, no rates; and the fold is never "terminated",
    # which the trainer would follow with zeros, "go to the zero pose" for these absolute targets
    assert np.all(ep.actions[ROWS:] == ep.actions[ROWS - 1]) and np.all(ep.obs[ROWS:, 0:5] == ep.obs[ROWS - 1, 0:5])
    assert np.all(ep.obs[ROWS:, 5:10] == 0) and np.all(ep.images["main"][ROWS:] == ep.images["main"][ROWS - 1])
    assert not ep.terminated.any() and ep.truncated[-1] and np.all(ep.discount == 1)
    from imitation.data.dataset import build_samples
    X, Y, M, _ = build_samples([ep], 2, 16)
    for t in range(len(Y)):                                    # every row a policy is taught is a row that was sent
        for k in np.flatnonzero(M[t]):
            assert t + k < T and np.array_equal(Y[t, k], ep.actions[t + k])
    assert M[ROWS - 1].all() and not M[-1].all()               # the fold's last row is taught its whole look-ahead
    with pytest.raises(FileExistsError):                                                              # write-once
        rig_dataset.build(folders, store, "rig_t", size=64)
    kept, left_out = rig_dataset.build(folders, store, "rig_all", size=64, failures=True)
    assert kept["n_episodes"] == 3 and kept["n_success"] == 2 and left_out == []
    (bad / "policy_meta.json").write_text(json.dumps({"policy": "half_fold_v0.npz", "speed": 1.0}))    # as policy_run.py leaves
    by_policy = rig_dataset.episode(rig_dataset.load_run(bad), 64)
    assert by_policy.meta["kind"] == "policy" and by_policy.meta["source"] == "student" and np.all(by_policy.actor == 1)

    out = env.tmp / "vision_run"
    train("vision", "rig_t", str(out), root=str(store), steps=6, batch=8, warmup=2, eval_every=3, log=lambda *a: None)
    policy = load_policy(str(out / "final.pt"), "cpu")
    assert [name for name, _ in policy.cameras] == ["main"]
    history = np.repeat(ep.obs[:1], policy.obs_horizon, axis=0)[None]
    chunk = policy.predict(history, [{"main": ep.images["main"][0]}])
    assert chunk.shape == (1, policy.chunk, 12) and np.isfinite(chunk).all() and torch.is_tensor(policy.obs_mean)

    # and the trained policy goes back onto the (fake) arms through camera_run.py, with what the dataset says of
    # how it was recorded. Six updates teach it nothing: the point is that it is played, checked and recorded
    import importlib
    import types
    pytest.importorskip("mujoco")
    pytest.importorskip("so101_nexus")
    for name in ("policy_run", "camera_run"):
        sys.modules.pop(name, None)
    camera_run = importlib.import_module("camera_run")
    manifest = json.loads((store / "rig_t" / "manifest.json").read_text())
    config = manifest["config"]
    assert config["size"] == 64 and config["rows"] == ROWS and config["frame_shape"] == [48, 64] and config["hold"] == rig_dataset.HOLD
    assert np.allclose(config["start"], np.concatenate([target[0], target[0]]), atol=0.002)           # a start a fold had
    # the fake arms played these folds at 0.02 s a row; the real ones never go under 0.05, which camera_run insists on
    with pytest.raises(ValueError, match="shorter than"):
        camera_run.load_config(store / "rig_t")
    config = config | {"row_seconds": 0.05}
    monkeypatch.setattr(env.replay, "sleep_until", lambda t: None)
    monkeypatch.setattr(env.replay, "RUNS_DIR", env.tmp / "played")
    options = types.SimpleNamespace(dry_run=False, torque_limit=50, track_deg=8.0, stale=5.0, replan=4, log=None,
                                    policy=str(out / "final.pt"), dataset=str(store / "rig_t"))
    followers = {"left": env.replay.Follower("L"), "right": env.replay.Follower("R")}
    code = camera_run.run(policy, config, followers, Phone(), options)
    lines = [json.loads(l) for l in (next((env.tmp / "played").iterdir()) / "steps.jsonl").read_text().splitlines()]
    folds = [l for l in lines if l["phase"] == "fold"]
    assert any(l["phase"] == "glide to the start" for l in lines)                 # the arms were taken to the folds' start
    if code == 0:                                                                 # played to the end...
        assert len(folds) == ROWS and all("read_t" in l for l in folds)
    else:                                                                         # ...or a check stopped it, and said why
        assert code == 1 and lines[-1]["phase"] == "stopped" and lines[-1]["reason"] and len(folds) == lines[-1]["row"]

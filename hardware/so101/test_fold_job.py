"""fold_job.FoldJob with a stand-in script in place of fold_camera.py, stand-in arms and a stand-in camera."""

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fold_job  # noqa: E402

STAND_IN = '''
import sys
import time
mode = sys.argv[1]
print("mode " + mode)
print("args " + " ".join(sys.argv[2:]))
if "--log" in sys.argv:
    with open(sys.argv[sys.argv.index("--log") + 1], "w") as steps:
        steps.write('{"t": 1.0, "phase": "fold", "row": 0, "target": {}, "measured": {}}\\n')
if mode == "fold":
    input("Check the outline. Enter to fold, Ctrl-C to stop ")
    print("folding")
    input("At the first target. Enter to play ")
    print("played")
if mode == "look" and "--speed" in sys.argv:
    sys.exit("Look failed: stand-in")
if mode == "calibrate":
    time.sleep(30)
'''


class StandInArm:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def lend(self):
        self.calls.append("lend " + self.name)

    def take_back(self):
        self.calls.append("take back " + self.name)


class StandInCamera:
    def __init__(self, has_frame=True):
        self.has_frame = has_frame

    def latest(self):
        if not self.has_frame:
            return None
        return types.SimpleNamespace(rgb=np.zeros((6, 4, 3), np.uint8), depth=np.ones((3, 2), np.float32),
                                     K=(100.0, 100.0, 2.0, 3.0))


@pytest.fixture
def rig(tmp_path):
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN)
    calls = []
    arms = {"arm1": StandInArm("arm1", calls), "arm2": StandInArm("arm2", calls)}
    runs = tmp_path / "runs"
    job = fold_job.FoldJob(arms, StandInCamera(), command=[sys.executable, "-u", str(script)], runs_dir=runs)
    return types.SimpleNamespace(job=job, calls=calls, arms=arms, runs=runs)


def wait_for(job, states, seconds=15):
    deadline = time.time() + seconds
    while time.time() < deadline:
        status = job.status()
        if status["state"] in states:
            return status
        time.sleep(0.05)
    raise AssertionError(f"still {job.status()['state']} after {seconds} s: {job.status()['lines']}")


def image_path(status):
    for line in status["lines"]:
        if line.startswith("args "):
            words = line.split()
            return Path(words[words.index("--image") + 1])
    raise AssertionError("the stand-in never printed its arguments")


def test_a_look_runs_to_the_end_without_touching_the_arms(rig):
    assert rig.job.start("look", {}) is None
    status = wait_for(rig.job, ["done"])
    assert status["lines"][0] == "mode look"
    assert status["exit_code"] == 0
    assert rig.calls == []
    assert not image_path(status).exists()      # the frame handed to the script is gone afterwards


def test_a_fold_waits_at_each_prompt_and_borrows_the_arms(rig):
    assert rig.job.start("fold", {"speed": 0.3, "torque": 50, "grasp_check": True}) is None
    status = wait_for(rig.job, ["waiting"])
    assert "Enter to fold" in status["prompt"]
    assert rig.calls == ["lend arm2", "lend arm1"]
    rig.job.go_on()
    deadline = time.time() + 15
    while "folding" not in rig.job.status()["lines"] and time.time() < deadline:
        time.sleep(0.05)
    status = wait_for(rig.job, ["waiting"])
    assert "Enter to play" in status["prompt"]
    rig.job.go_on()
    status = wait_for(rig.job, ["done"])
    assert "played" in status["lines"]
    assert rig.calls == ["lend arm2", "lend arm1", "take back arm2", "take back arm1"]
    args_line = status["lines"][1]
    for flag in ("--left arm2", "--right arm1", "--speed 0.3", "--torque-limit 50", "--grasp-check"):
        assert flag in args_line


def test_stop_ends_a_running_job_and_the_arms_come_back(rig):
    assert rig.job.start("fold", {}) is None
    wait_for(rig.job, ["waiting"])
    rig.job.stop()
    status = wait_for(rig.job, ["stopped"])
    assert rig.calls == ["lend arm2", "lend arm1", "take back arm2", "take back arm1"]
    assert not image_path(status).exists()


def test_a_second_job_is_refused_while_one_runs(rig):
    assert rig.job.start("calibrate", {"colour": "blue"}) is None
    wait_for(rig.job, ["running"])
    assert rig.job.start("look", {}) is not None
    rig.job.stop()
    wait_for(rig.job, ["stopped"])


def test_a_script_that_fails_is_reported(rig):
    assert rig.job.start("look", {"speed": 0.5}) is None      # the stand-in fails a look given a speed
    status = wait_for(rig.job, ["failed"])
    assert status["exit_code"] != 0
    assert any("failed" in line for line in status["lines"])


def test_start_is_refused_without_a_frame_or_with_a_bad_request(rig):
    assert rig.job.start("dance", {}) is not None
    assert rig.job.start("fold", {"speed": "fast"}) is not None
    rig.job.camera = StandInCamera(has_frame=False)
    assert rig.job.start("look", {}) is not None
    assert rig.calls == []
    assert rig.job.status()["state"] == "idle"


def test_a_fold_is_refused_when_an_arm_is_missing(rig):
    del rig.arms["arm1"]
    assert rig.job.start("fold", {}) is not None
    assert rig.calls == []


# --- every fold is recorded ---

def test_a_fold_is_recorded_with_frames_steps_and_its_output(rig):
    import recorder

    assert rig.job.start("fold", {"speed": 0.3}) is None
    wait_for(rig.job, ["waiting"])
    rig.job.go_on()
    deadline = time.time() + 15
    while "folding" not in rig.job.status()["lines"] and time.time() < deadline:
        time.sleep(0.05)
    wait_for(rig.job, ["waiting"])
    rig.job.go_on()
    wait_for(rig.job, ["done"])
    folders = sorted(rig.runs.iterdir())
    assert len(folders) == 1
    saved = recorder.load(folders[0])
    assert saved["meta"]["mode"] == "fold"
    assert saved["meta"]["options"] == {"speed": 0.3}
    assert len(saved["frames"]) >= 1
    assert saved["frames"][0]["t"] > 1_000_000_000          # wall-clock seconds, the clock the step log uses
    assert len(saved["steps"]) == 1                         # what the script wrote to the --log it was given
    names = [event["name"] for event in saved["events"]]
    assert "waiting" in names and "continue" in names and names[-1] == "ended"
    assert "played" in (folders[0] / "output.txt").read_text()
    assert saved["summary"] is not None


def test_a_look_is_not_recorded(rig):
    assert rig.job.start("look", {}) is None
    wait_for(rig.job, ["done"])
    assert not rig.runs.exists() or list(rig.runs.iterdir()) == []


def test_a_fold_that_cannot_start_gives_the_arms_back(rig):
    rig.job.command = [str(rig.runs / "no_such_program.exe")]
    refusal = rig.job.start("fold", {})
    assert refusal is not None
    assert rig.calls == ["lend arm2", "lend arm1", "take back arm2", "take back arm1"]
    assert rig.job.status()["state"] == "failed"

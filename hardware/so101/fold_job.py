"""One cloth-fold job for the control panel (ui.py): run fold_camera.py as a child process, keep its printed output,
answer its two "Enter to ..." pauses, and stop it. Two more modes run other scripts the same way: "learn" is
dagger_run.py (the camera fold with DAgger's network in the loop) and "policy" is camera_run.py (a camera policy
trained on recorded episodes, which reads its pictures from the panel and records its own run).

The panel keeps the phone, so the script gets the camera's latest frame as a file (--image) that is deleted when
the job ends. For a fold the panel's arms are lent to the script (their serial ports closed) and taken back when
it ends, however it ends.

Every fold is recorded to a new folder under runs/ (recorder.py's layout): the camera picture and depth a few
times a second, the script's step log (steps.jsonl, written by replay.py: targets and encoder readings per control
step), the look, the calibration, the options, the printed output and what happened when. Frame and step times are
both time.time(), so a picture can be matched to the joints at that moment.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

import recorder
import run_label

HERE = Path(__file__).parent
SCRIPT = HERE / "fold_camera.py"
CAMERA_FILE = HERE / "camera.json"
LOOK_FILE = HERE / "look.png"
RUNS_DIR = HERE / "runs"    # ignored by git
MODES = ["calibrate", "look", "fold", "learn", "policy"]
# The script of each mode that is not fold_camera.py's own. "learn" is the camera fold with DAgger's network in the
# loop, "policy" a camera policy trained on recorded episodes.
OTHER_SCRIPTS = {"learn": HERE / "dagger_run.py", "policy": HERE / "camera_run.py"}
ARM_MODES = ["fold", "learn", "policy"]     # these move the arms, so the panel lends them
RECORDED_MODES = ["fold", "learn"]          # camera_run.py records its own run
MAX_TRUST_DEG = 5.0                         # dagger_run.py's --trust-deg goes from 0 to this
MAX_LINES = 200
PROMPT_MARKS = ["Enter to", "press Enter"]  # every pause of the scripts prints one of these and waits for Enter
BUSY_STATES = ["running", "waiting"]
FRAME_SECONDS = 0.2         # a recorded fold saves the camera's picture this often


def saved_colour():
    """The cloth colour of the saved calibration, or None when there is none."""
    if not CAMERA_FILE.exists():
        return None
    return json.loads(CAMERA_FILE.read_text()).get("colour", "dark")


class FoldJob:
    def __init__(self, arms_by_name, camera, followers, command=None, runs_dir=RUNS_DIR):
        """`followers`: the follower arms' names, left then right as seen from behind the arms."""
        self.arms_by_name = arms_by_name
        self.camera = camera
        self.followers = list(followers)
        self.command = command      # tests give a stand-in script, run for every mode
        self.runs_dir = runs_dir
        self.lock = threading.Lock()
        self.state = "idle"
        self.mode = None
        self.lines = []
        self.partial = b""          # output since the last line end: a prompt sits here while the script waits
        self.exit_code = None
        self.process = None
        self.stop_asked = False
        self.run = None             # the recorder.Recorder of the fold in progress
        self.last_run = None        # the folder of the recorded fold that ended last, until it is discarded
        self.counts = run_label.count(self.runs_dir, run_label.is_panel_fold)

    def status(self):
        with self.lock:
            prompt = ""
            if self.state == "waiting":
                prompt = self.partial.decode("utf-8", "replace")
            last_run = None
            last_label = None
            if self.last_run is not None:
                last_run = self.last_run.name
                last_label = run_label.label_of(self.last_run)
            return {"state": self.state, "mode": self.mode, "lines": list(self.lines), "prompt": prompt,
                    "exit_code": self.exit_code, "colour": saved_colour(), "last_run": last_run,
                    "last_label": last_label, "counts": dict(self.counts)}

    def label(self, word):
        """Say how the recorded fold that ended last went: "good" or "bad" (what rig_dataset.py reads), or
        "discard", which deletes its folder. False when there is none, a job is running, or the word is unknown."""
        with self.lock:
            if self.state in BUSY_STATES or self.last_run is None:
                return False
            if not run_label.set_label(self.last_run, word):
                return False
            if word == "discard":
                self.last_run = None
            self.counts = run_label.count(self.runs_dir, run_label.is_panel_fold)
            return True

    def command_for(self, mode):
        """The program and script that run `mode`."""
        if self.command is not None:
            return list(self.command)
        script = OTHER_SCRIPTS.get(mode, SCRIPT)
        return [sys.executable, "-u", str(script)]

    def policy_arguments(self, options):
        """camera_run.py's arguments: it takes its pictures from the panel (`frames`, the panel's /frame.npz)."""
        # whole paths: the script runs from this folder, not from where the panel was started
        args = ["--policy", str(Path(str(options["policy"])).resolve()),
                "--dataset", str(Path(str(options["dataset"])).resolve()),
                "--left", self.followers[0], "--right", self.followers[1]]
        if "torque" in options:
            args = args + ["--torque-limit", str(int(options["torque"]))]
        if options.get("frames"):
            args = args + ["--frames", str(options["frames"])]
        return args

    def arguments(self, mode, image, options, steps_file=None):
        """The script's arguments for `mode`. ValueError for an option that is not a number."""
        if mode == "policy":
            return self.policy_arguments(options)
        if mode == "learn":
            # dagger_run.py takes its own flag and hands the rest to `fold_camera.py fold`
            args = ["--trust-deg", str(float(options.get("trust", 0))), "--image", image]
        else:
            args = [mode, "--image", image]
        if mode == "calibrate" and options.get("colour"):
            args = args + ["--cloth-colour", str(options["colour"])]
        if mode in ("fold", "learn"):
            args = args + ["--left", self.followers[0], "--right", self.followers[1]]
        if "speed" in options:
            args = args + ["--speed", str(float(options["speed"]))]
        if "torque" in options:
            args = args + ["--torque-limit", str(int(options["torque"]))]
        if options.get("grasp_check"):
            args = args + ["--grasp-check"]
        if steps_file is not None:
            args = args + ["--log", str(steps_file)]
        return args

    def start(self, mode, options):
        """Start a job. Returns None, or the reason it was not started."""
        if mode not in MODES:
            return f"unknown mode {mode!r}"
        if mode in ARM_MODES and len(self.followers) != 2:
            return "a fold needs two follower arms in arms.json, listed left then right"
        if mode == "policy":
            for what in ("policy", "dataset"):
                if not options.get(what) or not Path(str(options[what])).exists():
                    return f"no {what} at {options.get(what)!r}: train one from recorded episodes first"
        try:
            self.arguments(mode, "", options)
            trust = float(options.get("trust", 0))
        except (TypeError, ValueError):
            return "speed, torque cap and trust must be numbers"
        if mode == "learn" and not 0.0 <= trust <= MAX_TRUST_DEG:
            return f"trust is 0 to {MAX_TRUST_DEG:g} degrees"
        lent = []
        if mode in ARM_MODES:
            for name in self.followers:
                if name not in self.arms_by_name:
                    return f"the panel is not running the {name} arm"
                lent.append(self.arms_by_name[name])
        with self.lock:
            if self.state in BUSY_STATES:
                return "a job is already running"
            frame = self.camera.latest()
            if frame is None:
                return "no picture from the camera yet"
            self.mode = mode
            self.lines = []
            self.partial = b""
            self.exit_code = None
            self.stop_asked = False
            image = None
            run = None
            borrowed = []
            try:
                handle, image = tempfile.mkstemp(suffix=".png", prefix="panel_frame_")
                os.close(handle)
                Image.fromarray(frame.rgb).save(image)
                steps_file = None
                if mode in RECORDED_MODES:
                    run = recorder.Recorder(self.runs_dir, meta={"source": "control panel", "mode": mode,
                                                                 "options": options, "left": self.followers[0],
                                                                 "right": self.followers[1]})
                    steps_file = run.path / "steps.jsonl"
                for arm in lent:
                    arm.lend()
                    borrowed.append(arm)
                environment = dict(os.environ)
                environment["PYTHONIOENCODING"] = "utf-8"
                self.process = subprocess.Popen(self.command_for(mode) + self.arguments(mode, image, options, steps_file),
                                                cwd=str(HERE), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT, bufsize=0, env=environment)
            except OSError as err:
                reason = f"could not start the script: {err}"
                self.add_line(reason)
                self.finish(None, image, borrowed, run)
                return reason
            self.state = "running"
            self.run = run
        threading.Thread(target=self.follow, args=(self.process, image, lent, run), daemon=True).start()
        if run is not None:
            threading.Thread(target=self.record, args=(self.process, run), daemon=True).start()
        return None

    def add_line(self, text):
        self.lines.append(text)
        if len(self.lines) > MAX_LINES:
            del self.lines[0]

    def note(self, name, **data):
        """An event in the fold's recording, if there is one. Called with the lock held."""
        if self.run is not None:
            self.run.event(time.time(), name, **data)

    def record(self, process, run):
        """Save the camera's picture and depth while the script runs."""
        while process.poll() is None:
            frame = self.camera.latest()
            if frame is not None:
                run.frame(time.time(), frame.rgb, frame.depth, frame.K)
            time.sleep(FRAME_SECONDS)

    def at_a_prompt(self):
        """Whether the output since the last line end holds one of the scripts' pauses."""
        for mark in PROMPT_MARKS:
            if mark.encode() in self.partial:
                return True
        return False

    def follow(self, process, image, lent, run):
        """Read the child's output a byte at a time until it ends, then clean up."""
        while True:
            byte = process.stdout.read(1)
            if byte == b"":
                break
            with self.lock:
                if byte == b"\n":
                    self.add_line(self.partial.decode("utf-8", "replace").rstrip("\r"))
                    self.partial = b""
                    if self.state == "waiting":     # the line ended, so it was not a prompt
                        self.state = "running"
                else:
                    self.partial = self.partial + byte
                    if self.at_a_prompt() and self.state != "waiting":
                        self.state = "waiting"
                        self.note("waiting", prompt=self.partial.decode("utf-8", "replace"))
        code = process.wait()
        with self.lock:
            if self.partial:
                self.add_line(self.partial.decode("utf-8", "replace"))
                self.partial = b""
            self.finish(code, image, lent, run)

    def finish(self, code, image, lent, run):
        """The end of a job, however it ended: delete the frame file, take the arms back, close the recording,
        set the state. `code` is the script's exit code, or None if it never started. Called with the lock held."""
        problems = []
        if image is not None:
            try:
                os.remove(image)
            except OSError as err:
                problems.append(f"could not delete the frame file {image}: {err}")
        for arm in lent:
            try:
                arm.take_back()
            except OSError as err:
                problems.append(f"could not take the {arm.name} arm back: {err}. Restart the panel.")
        for problem in problems:
            self.add_line(problem)
        self.exit_code = code
        if self.stop_asked:
            self.state = "stopped"
            self.add_line("Stopped. The arms hold where they are, torque on. A torque cap set for the fold "
                          "stays on the motors until the next fold or a power cycle.")
        elif code == 0 and not problems:
            self.state = "done"
        else:
            self.state = "failed"
            if code is not None:
                self.add_line(f"The script failed (exit code {code}).")
        if run is not None:
            self.add_line(f"Recorded to {run.path}")
            self.close_recording(run, code)
            self.last_run = run.path
            self.counts = run_label.count(self.runs_dir, run_label.is_panel_fold)
        self.run = None

    def close_recording(self, run, code):
        """Put the look, the calibration and the output beside the frames and steps, and close the run."""
        for source in (LOOK_FILE, CAMERA_FILE):
            if source.exists():
                shutil.copy(source, run.path / source.name)
        (run.path / "output.txt").write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        run.event(time.time(), "ended", state=self.state, exit_code=code)
        try:
            run.close()
        except Exception as err:    # a frame that failed to save must not hide how the fold ended
            self.add_line(f"The recording did not close cleanly: {err}")

    def go_on(self):
        """Answer the prompt the script is waiting at."""
        with self.lock:
            if self.state != "waiting":
                return
            self.add_line(self.partial.decode("utf-8", "replace"))
            self.partial = b""
            self.state = "running"
            self.note("continue")
            try:
                self.process.stdin.write(b"\n")
                self.process.stdin.flush()
            except OSError:
                pass    # the script ended on its own; follow() reports how

    def stop(self):
        with self.lock:
            if self.state not in BUSY_STATES:
                return
            self.stop_asked = True
            self.note("stop")
            self.process.terminate()

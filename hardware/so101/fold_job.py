"""One cloth-fold job for the control panel (ui.py): run fold_camera.py as a child process, keep its printed output,
answer its two "Enter to ..." pauses, and stop it.

The panel keeps the phone, so the script gets the camera's latest frame as a file (--image) that is deleted when
the job ends. For a fold the panel's arms are lent to the script (their serial ports closed) and taken back when
it ends, however it ends.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
SCRIPT = HERE / "fold_camera.py"
CAMERA_FILE = HERE / "camera.json"
LEFT_ARM = "arm2"       # seen from behind the arms
RIGHT_ARM = "arm1"
MODES = ["calibrate", "look", "fold"]
MAX_LINES = 200
PROMPT_MARK = "Enter to"    # both of the fold's pauses print this and then wait for Enter
BUSY_STATES = ["running", "waiting"]


def saved_colour():
    """The cloth colour of the saved calibration, or None when there is none."""
    if not CAMERA_FILE.exists():
        return None
    return json.loads(CAMERA_FILE.read_text()).get("colour", "dark")


class FoldJob:
    def __init__(self, arms_by_name, camera, command=None):
        self.arms_by_name = arms_by_name
        self.camera = camera
        self.command = command
        if self.command is None:
            self.command = [sys.executable, "-u", str(SCRIPT)]
        self.lock = threading.Lock()
        self.state = "idle"
        self.mode = None
        self.lines = []
        self.partial = b""          # output since the last line end: a prompt sits here while the script waits
        self.exit_code = None
        self.process = None
        self.stop_asked = False

    def status(self):
        with self.lock:
            prompt = ""
            if self.state == "waiting":
                prompt = self.partial.decode("utf-8", "replace")
            return {"state": self.state, "mode": self.mode, "lines": list(self.lines), "prompt": prompt,
                    "exit_code": self.exit_code, "colour": saved_colour()}

    def arguments(self, mode, image, options):
        """The script's arguments for `mode`. ValueError for an option that is not a number."""
        args = [mode, "--image", image]
        if mode == "calibrate" and options.get("colour"):
            args = args + ["--cloth-colour", str(options["colour"])]
        if mode == "fold":
            args = args + ["--left", LEFT_ARM, "--right", RIGHT_ARM]
        if "speed" in options:
            args = args + ["--speed", str(float(options["speed"]))]
        if "torque" in options:
            args = args + ["--torque-limit", str(int(options["torque"]))]
        if options.get("grasp_check"):
            args = args + ["--grasp-check"]
        return args

    def start(self, mode, options):
        """Start a job. Returns None, or the reason it was not started."""
        if mode not in MODES:
            return f"unknown mode {mode!r}"
        try:
            self.arguments(mode, "", options)
        except (TypeError, ValueError):
            return "speed and torque cap must be numbers"
        lent = []
        if mode == "fold":
            for name in (LEFT_ARM, RIGHT_ARM):
                if name not in self.arms_by_name:
                    return f"the panel is not running the {name} arm"
                lent.append(self.arms_by_name[name])
        with self.lock:
            if self.state in BUSY_STATES:
                return "a job is already running"
            frame = self.camera.latest()
            if frame is None:
                return "no picture from the camera yet"
            handle, image = tempfile.mkstemp(suffix=".png", prefix="panel_frame_")
            os.close(handle)
            Image.fromarray(frame.rgb).save(image)
            for arm in lent:
                arm.lend()
            self.state = "running"
            self.mode = mode
            self.lines = []
            self.partial = b""
            self.exit_code = None
            self.stop_asked = False
            environment = dict(os.environ)
            environment["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(self.command + self.arguments(mode, image, options), cwd=str(HERE),
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            bufsize=0, env=environment)
        threading.Thread(target=self.follow, args=(self.process, image, lent), daemon=True).start()
        return None

    def add_line(self, text):
        self.lines.append(text)
        if len(self.lines) > MAX_LINES:
            del self.lines[0]

    def follow(self, process, image, lent):
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
                    if PROMPT_MARK.encode() in self.partial:
                        self.state = "waiting"
        code = process.wait()
        problems = []
        try:
            os.remove(image)
        except OSError as err:
            problems.append(f"could not delete the frame file {image}: {err}")
        for arm in lent:
            try:
                arm.take_back()
            except OSError as err:
                problems.append(f"could not take the {arm.name} arm back: {err}. Restart the panel.")
        with self.lock:
            if self.partial:
                self.add_line(self.partial.decode("utf-8", "replace"))
                self.partial = b""
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
                self.add_line(f"The script failed (exit code {code}).")

    def go_on(self):
        """Answer the prompt the script is waiting at."""
        with self.lock:
            if self.state != "waiting":
                return
            self.add_line(self.partial.decode("utf-8", "replace"))
            self.partial = b""
            self.state = "running"
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
            self.process.terminate()

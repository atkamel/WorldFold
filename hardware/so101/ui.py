"""Real-time control panel + live 3D twin for one or two SO-101 followers.

    .venv/bin/python ui.py             # every arm in arms.json; open http://localhost:8101
    .venv/bin/python ui.py arm2        # just one of them
                                       # phone: open the https link it prints
    .venv/bin/python ui.py --gamepad    # both followers, left/right sticks and triggers
    On Windows the interpreter is .venv/Scripts/python.exe; set PLATFORM in arms.py.

The page also shows the iPhone LiDAR point cloud and has a record button: a recording saves
every arm's joints plus the phone's color and depth to runs/ (teleop_record.py). The phone
is optional; without it the arms run and a recording holds joints only.

One background thread per arm owns that arm's serial port: 50 times a second
it reads every motor's position and, with torque on, nudges each goal toward
its target. The browser only talks to this server, never to the motors.

Mirror mode: one arm leads and the other copies its target as a mirror image
(pan and wrist roll flipped), so any input that moves the leader moves both.
With the leader limp, moving it by hand drives the other arm.

Leader arm: arms.json "leaders" pairs a follower with a leader arm (a limp
SO-101 moved by hand), e.g. {"arm1": "leader"}. With it on, the follower
copies the leader's joint angles.
"""

import argparse
import importlib.util
import io
import json
import math
import os
import secrets
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import scservo_sdk as scs

HERE = Path(__file__).parent
# This file runs as a script, so the repo root is not on the path; hardware.iphone_lidar needs it.
sys.path.insert(0, str(HERE.parent.parent))

import arms  # noqa: E402
import fold_job  # noqa: E402
import panel_camera  # noqa: E402
import replay  # noqa: E402
import teleop_record  # noqa: E402
from hardware.iphone_lidar.pointcloud import pack_frame, pack_no_points  # noqa: E402

BAUDRATE = 1_000_000
HTTP_PORT = 8101  # laptop only (127.0.0.1)
PHONE_PORT = 8102  # HTTPS on the Wi-Fi: phones only give motion sensors to secure pages
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_ID = 6
TICK_SECONDS = 0.02  # 50 Hz control loop
STEPS_PER_DEGREE = 4096 / 360
DEFAULT_SPEED = 50  # deg/s; adjustable live from the page
MAX_SPEED = 180  # deg/s ceiling for the speed slider
MAX_BURST_SPEED = 270  # deg/s ceiling for one-off fast moves (punches); near the motor's top speed

PAGE_FILE = HERE / "ui.html"
URDF_FILE = HERE / "so101.urdf"
PHONE_PAGE_FILE = HERE / "phone.html"
CERT_FILE = HERE / "certs" / "cert.pem"
KEY_FILE = HERE / "certs" / "key.pem"
LIDAR_DIR = HERE.parent / "iphone_lidar"
CLOUD_PAGE_FILE = LIDAR_DIR / "viewer.html"
# The only files the cloud page loads; anything else under /vendor/ is refused.
VENDOR_FILES = {
    "three.module.js": LIDAR_DIR / "vendor" / "three.module.js",
    "OrbitControls.js": LIDAR_DIR / "vendor" / "OrbitControls.js",
}

# STS3215 control table addresses
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_TORQUE_LIMIT = 48
ADDR_PRESENT_POSITION = 56

GRIPPER_TORQUE_LIMIT = 500  # 50% of max, as lerobot does, so it can't burn out squeezing
MIRROR_SIGN = {1: -1, 5: -1}  # a mirror image swaps left and right: pan and wrist roll change sign


def copy_pose(pose, twin_from, twin_to, mirror):
    """One arm's encoder pose -> the same joint angles on another arm (or their
    mirror image), through each arm's straight-up reference (same convention as
    the pages: angle = sign * (encoder - zero))."""
    def zero(twin, m):
        return twin.get("zero", {}).get(str(m), 2047)

    def sign(twin, m):
        return -1 if twin.get("flip", {}).get(str(m)) else 1

    out = {}
    for m, pos in pose.items():
        center = twin_from.get("center", {}).get(str(m))
        if center is not None:
            # A leader arm's encoder can roll over 4095 -> 0 (its wrist roll turns all the way
            # round): read it within half a turn of the pose it was matched to its follower in.
            pos = center + (pos - center + 2048) % 4096 - 2048
        angle = sign(twin_from, m) * (pos - zero(twin_from, m)) * (MIRROR_SIGN.get(m, 1) if mirror else 1)
        out[m] = zero(twin_to, m) + sign(twin_to, m) * angle
    return out


class Arm:
    def __init__(self, name):
        self.name = name
        folder = arms.folder(name)
        self.pose_file = folder / "idle_pose.json"
        self.twin_file = folder / "twin.json"
        self.presets_file = folder / "presets.json"
        self.port = scs.PortHandler(arms.port(name))
        self.packet = scs.PacketHandler(0)  # protocol 0 = Feetech STS series
        if not self.port.openPort() or not self.port.setBaudRate(BAUDRATE):
            sys.exit(f"Could not open {arms.port(name)} ({name} arm)")

        limits = json.loads((folder / "limits.json").read_text())
        self.limits = {m: (limits[str(m)]["true"]["min"], limits[str(m)]["true"]["max"]) for m in MOTOR_IDS}

        # Sync read: one request, every motor answers in turn. Much faster
        # than six separate request/reply round trips.
        self.reader = scs.GroupSyncRead(self.port, self.packet, ADDR_PRESENT_POSITION, 2)
        for m in MOTOR_IDS:
            self.reader.addParam(m)

        self.lock = threading.Lock()
        self.present = {m: None for m in MOTOR_IDS}
        for _ in range(10):
            self.read_present()
            if None not in self.present.values():
                break
        else:
            sys.exit(f"Could not read all motors of the {name} arm: {self.present}")
        self.target = dict(self.present)  # where the sliders want each joint
        self.command = dict(self.present)  # goal actually sent, eased toward target
        # Keep whatever torque state the arm is in (idle.py may have left it holding).
        self.torque = bool(self.read1(GRIPPER_ID, ADDR_TORQUE_ENABLE))
        self.pending_torque = None
        self.pending_pose = None
        self.speed = DEFAULT_SPEED
        # The map the collision check and the fold use, so the page draws the arm they see.
        follower = replay.Follower(name)
        self.model = {"zero": follower.zero, "sign": follower.sign}
        self.burst_speed = None  # temporary faster speed until the current pose is reached
        self.pending_burst = None
        self.twin = json.loads(self.twin_file.read_text()) if self.twin_file.exists() else None
        self.follows = None  # (leading Arm, mirror?) while this arm copies another
        self.lent = False  # True while another program holds this arm's serial port

    # --- bus access: only ever called from the control thread (or __init__) ---

    def read1(self, motor_id, address):
        for _ in range(5):
            self.port.ser.reset_input_buffer()
            try:
                value, comm, error = self.packet.read1ByteTxRx(self.port, motor_id, address)
            except IndexError:  # SDK crashes on a truncated reply
                comm = None
            if comm == scs.COMM_SUCCESS and not error:
                return value
            time.sleep(0.02)
        return 0

    def read_present(self):
        self.port.ser.reset_input_buffer()
        try:
            ok = self.reader.txRxPacket() == scs.COMM_SUCCESS
        except IndexError:
            ok = False
        if not ok:
            return False  # keep the last good readings; the next tick retries
        available = [m for m in MOTOR_IDS if self.reader.isAvailable(m, ADDR_PRESENT_POSITION, 2)]
        for m in available:
            self.present[m] = self.reader.getData(m, ADDR_PRESENT_POSITION, 2)
        return len(available) == len(MOTOR_IDS)

    def send_goals(self, goals):
        """Send every motor its goal in ONE packet, so all joints move in step."""
        sync = scs.GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, 2)
        for m, goal in goals.items():
            goal = round(goal)  # command is eased in fractional steps
            sync.addParam(m, [scs.SCS_LOBYTE(goal), scs.SCS_HIBYTE(goal)])
        return sync.txPacket() == scs.COMM_SUCCESS

    def apply_torque(self, on):
        if on:
            # Goal = where each joint is NOW, so switching torque on holds
            # still instead of snapping to a stale goal.
            self.command = dict(self.present)
            self.target = dict(self.present)
            self.send_goals(self.command)
            self.packet.write2ByteTxRx(self.port, GRIPPER_ID, ADDR_TORQUE_LIMIT, GRIPPER_TORQUE_LIMIT)
        for m in MOTOR_IDS:
            self.packet.write1ByteTxRx(self.port, m, ADDR_TORQUE_ENABLE, int(on))
        self.torque = on

    def control_loop(self):
        while True:
            self.tick()
            time.sleep(TICK_SECONDS)

    def tick(self):
        """One pass of the control loop. A lent arm's bus belongs to another program, so it is left alone."""
        follows = self.follows
        if follows:
            with follows[0].lock:
                lead_target = dict(follows[0].target)
        with self.lock:
            if self.lent:
                return
            pending, self.pending_torque = self.pending_torque, None
            if pending is not None:
                self.apply_torque(pending)
            pose, self.pending_pose = self.pending_pose, None
            if pose is not None:
                if not self.torque:
                    self.apply_torque(True)
                self.target.update(pose)
                self.burst_speed, self.pending_burst = self.pending_burst, None

            self.read_present()
            if self.torque:
                if follows:
                    leader, mirror = follows
                    self.target = self.clamp(copy_pose(lead_target, leader.twin or {}, self.twin or {}, mirror))
                # Speed = how far a goal may move per tick. 50 deg/s is
                # ~11.4 steps per 20 ms tick; fractions carry over.
                max_step = (self.burst_speed or self.speed) * STEPS_PER_DEGREE * TICK_SECONDS
                # Ease each goal toward its slider by at most max_step per
                # tick, so a fast slider flick can't slam the joint.
                for m in MOTOR_IDS:
                    delta = self.target[m] - self.command[m]
                    self.command[m] += max(-max_step, min(max_step, delta))
                self.send_goals(self.command)
                if self.burst_speed and all(abs(self.target[m] - self.command[m]) < 1 for m in MOTOR_IDS):
                    self.burst_speed = None  # arrived: back to the normal speed
            else:
                # Limp: the arm is moved by hand, so sliders follow it.
                self.target = dict(self.present)
                self.command = dict(self.present)

    def lend(self):
        """Give the serial port up so another program (a cloth fold) can drive the arm."""
        with self.lock:
            self.lent = True
            self.follows = None
            self.port.closePort()

    def take_back(self):
        """Open the serial port again and hold the arm where the other program left it. Whatever was asked of the
        arm while it was lent is dropped. OSError if the port or the motors do not answer."""
        with self.lock:
            if not self.port.openPort() or not self.port.setBaudRate(BAUDRATE):
                raise OSError(f"could not reopen {arms.port(self.name)}")
            self.present = {m: None for m in MOTOR_IDS}
            for _ in range(10):
                self.read_present()
                if None not in self.present.values():
                    break
            if None in self.present.values():
                raise OSError(f"could not read all motors of the {self.name} arm")
            self.torque = bool(self.read1(GRIPPER_ID, ADDR_TORQUE_ENABLE))
            self.target = dict(self.present)
            self.command = dict(self.present)
            self.pending_torque = None
            self.pending_pose = None
            self.pending_burst = None
            self.burst_speed = None
            self.follows = None
            self.lent = False

    # --- called from HTTP threads: only touch shared state, never the bus ---

    def clamp(self, pose):
        return {m: max(self.limits[m][0], min(self.limits[m][1], int(p))) for m, p in pose.items()}

    def set_target(self, pose):
        """Aim the listed joints ({motor id: encoder value}) at new targets."""
        pose = self.clamp({int(m): p for m, p in pose.items()})
        with self.lock:
            self.target.update(pose)

    def request_torque(self, on):
        with self.lock:
            self.pending_torque = on

    def go_pose(self, pose, speed=None):
        """Queue a pose ({motor id: encoder value}, any subset of joints); the
        control thread switches torque on (holding still) first, then eases
        each listed joint toward it. speed (deg/s) overrides the normal speed
        just until this pose is reached."""
        clamped = {int(m): max(self.limits[int(m)][0], min(self.limits[int(m)][1], int(p))) for m, p in pose.items()}
        with self.lock:
            self.pending_pose = {**(self.pending_pose or {}), **clamped}
            self.pending_burst = min(MAX_BURST_SPEED, float(speed)) if speed else None

    def go_idle(self):
        self.go_pose(json.loads(self.pose_file.read_text()))

    def hold(self):
        """Stop: aim every joint at the goal it's already been sent, so the
        easing has nowhere left to go and the arm freezes where it is."""
        with self.lock:
            self.target = dict(self.command)

    def save_twin(self, twin):
        self.twin_file.write_text(json.dumps(twin, indent=2))
        self.twin = twin

    def load_presets(self):
        """Saved poses, by name: {motor id: encoder value}."""
        return json.loads(self.presets_file.read_text()) if self.presets_file.exists() else {}

    def save_presets(self, presets):
        self.presets_file.write_text(json.dumps(presets, indent=2))

    def state(self):
        with self.lock:
            return {
                "present": self.present,
                "target": self.target,
                "torque": self.torque,
                "lent": self.lent,
                "limits": self.limits,
                "twin": self.twin,
                "model": self.model,
                "presets": self.load_presets(),
            }

    def joints(self):
        """(measured, target) encoder positions in motor id order, for teleop_record."""
        with self.lock:
            measured = []
            target = []
            for m in MOTOR_IDS:
                measured.append(self.present[m])
                target.append(self.target[m])
            return measured, target


def mesh_files():
    """{file name: path} of the arm's mesh files, which ship with the so101_nexus package. Empty without it."""
    package = importlib.util.find_spec("so101_nexus")
    if package is None:
        return {}
    folder = Path(package.submodule_search_locations[0]) / "assets" / "SO101" / "assets"
    found = {}
    for file in folder.glob("*.stl"):
        found[file.name] = file
    return found


def cloud_blob(camera):
    """The camera's latest frame as packed points for the cloud page."""
    frame = camera.latest()
    if frame is None:
        return pack_no_points()
    return pack_frame(frame.rgb, frame.depth, frame.K)


def camera_jpeg(camera):
    """The camera's latest picture as JPEG bytes, or None with no frame. Encoded in memory: nothing is written."""
    frame = camera.latest()
    if frame is None:
        return None
    from PIL import Image  # only needed with a phone; the arms run without it

    buffer = io.BytesIO()
    Image.fromarray(frame.rgb).save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()


def run(arm):
    """The arm's control loop. A lost USB link ends the whole server: a page
    that keeps showing frozen readings as live is worse than no page."""
    try:
        arm.control_loop()
    except OSError as err:  # pyserial's SerialException is an OSError
        print(f"\nLost the {arm.name} arm's USB connection ({err}). Stopping; torque is left as is.", flush=True)
        os._exit(1)


def ensure_certificate():
    """Self-signed HTTPS certificate for the phone link (the phone will warn
    once, since no authority vouches for it; the traffic is still encrypted)."""
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    CERT_FILE.parent.mkdir(exist_ok=True)
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
         "-keyout", str(KEY_FILE), "-out", str(CERT_FILE), "-subj", "/CN=so101-arm"],
        check=True, capture_output=True,
    )


def lan_ip():
    """This laptop's address on the Wi-Fi (no packet is actually sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


LOOK_FILE = HERE / "look.png"
# Requests that move one arm: refused while that arm is lent to a cloth fold.
ARM_MOVES = ["/goal", "/torque", "/pose", "/idle"]


def make_handler(arms_by_name, camera, recording, token=None, fold=None, stop_fold=None, gamepad=None):
    # follower -> leader arm, for the pairs whose arms are both running
    leaders = {f: arms_by_name[l] for f, l in arms.CONFIG.get("leaders", {}).items()
               if f in arms_by_name and l in arms_by_name}

    """token=None: trusted (localhost). Otherwise every request must carry ?t=token,
    so only someone with the printed link can drive the arm over Wi-Fi."""

    meshes = mesh_files()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, content_type="application/json", status=200):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(data)

        def route(self):
            """Return the request path, or None (after replying 403) if the token is wrong."""
            url = urlparse(self.path)
            if token and not secrets.compare_digest(parse_qs(url.query).get("t", [""])[0], token):
                self.reply({"error": "bad token"}, status=403)
                return None
            if not token:
                # The laptop page has no token, so it only answers requests addressed to this machine by name.
                # A web page elsewhere that points its own name at 127.0.0.1 arrives with that other name.
                port = self.server.server_address[1]
                if self.headers.get("Host", "") not in (f"localhost:{port}", f"127.0.0.1:{port}"):
                    self.reply({"error": "this panel only answers on localhost"}, status=403)
                    return None
            return url.path

        def do_GET(self):
            path = self.route()
            if path is None:
                return
            if path == "/state":
                first = next(iter(arms_by_name.values()))
                self.reply({
                    "arms": {name: arm.state() for name, arm in arms_by_name.items()},
                    "layout": arms.CONFIG.get("layout", "side_by_side"),
                    "spacing_cm": arms.CONFIG.get("spacing_cm", 40),
                    "leaders": {f: l.name for f, l in leaders.items()},
                    "follow": {a.name: {"leader": a.follows[0].name, "mirror": a.follows[1]}
                               for a in arms_by_name.values() if a.follows},
                    "speed": first.speed,
                    "gamepad": gamepad.status if gamepad is not None else None,
                    "gamepad_tips": gamepad.tips_state() if gamepad is not None else None,
                    "gamepad_sticks": gamepad.sticks_state() if gamepad is not None else None,
                })
            elif path == "/phone":
                self.reply(PHONE_PAGE_FILE.read_bytes(), "text/html")
            elif path == "/cloud":
                self.reply(CLOUD_PAGE_FILE.read_bytes(), "text/html")
            elif path == "/frame":
                self.reply(cloud_blob(camera), "application/octet-stream")
            elif path.startswith("/vendor/"):
                file_name = path[len("/vendor/"):]
                if file_name in VENDOR_FILES:
                    self.reply(VENDOR_FILES[file_name].read_bytes(), "text/javascript")
                else:
                    self.reply({"error": "not found"}, status=404)
            elif path == "/so101.urdf":
                self.reply(URDF_FILE.read_bytes(), "application/xml")
            elif path.startswith("/assets/"):
                # Only a name from the listing is served, so the request cannot reach another file.
                file = meshes.get(path[len("/assets/"):])
                if file is None:
                    self.reply({"error": "not found"}, status=404)
                else:
                    self.reply(file.read_bytes(), "model/stl")
            elif path == "/record":
                status = recording.status()
                status["camera"] = camera.status()
                self.reply(status)
            elif path == "/camera.jpg":
                picture = camera_jpeg(camera)
                if picture is None:
                    self.reply({"error": "no picture yet"}, status=404)
                else:
                    self.reply(picture, "image/jpeg")
            elif path == "/look.png":
                if LOOK_FILE.exists():
                    self.reply(LOOK_FILE.read_bytes(), "image/png")
                else:
                    self.reply({"error": "no look yet"}, status=404)
            elif path == "/fold":
                if fold is None:
                    self.reply({"error": "not available here"}, status=404)
                else:
                    self.reply(fold.status())
            else:
                self.reply(PAGE_FILE.read_bytes(), "text/html")

        def do_POST(self):
            path = self.route()
            if path is None:
                return
            self.path = path
            # A browser lets any web page send a plain form to this port without asking; JSON it must ask about
            # first, and this server never says yes.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self.reply({"error": "send JSON"}, status=415)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or "{}")
            arm = arms_by_name.get(body.get("arm"))
            if gamepad is not None and self.path in ("/goal", "/pose", "/idle", "/mirror", "/leader", "/fold/start"):
                self.reply({"error": "The controller has exclusive control in gamepad mode"}, status=409)
                return
            if self.path == "/gamepad/claw":
                # A claw key pressed on the page: that arm's claw goes to its other end.
                if gamepad is None:
                    self.reply({"error": "not in gamepad mode"}, status=404)
                    return
                if arm is None:
                    self.reply({"error": "send an arm"}, status=400)
                    return
                if not gamepad.toggle_claw(arm.name):
                    self.reply({"error": "Press A on the controller to drive, then use the claw keys"}, status=409)
                    return
                self.reply({"ok": True})
                return
            if self.path == "/gamepad/target":
                # A fingertip point dragged in the 3D view: the arm's base frame, metres.
                if gamepad is None:
                    self.reply({"error": "not in gamepad mode"}, status=404)
                    return
                point = body.get("tip")
                plausible = arm is not None and isinstance(point, list) and len(point) == 3
                if plausible:
                    for value in point:
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            plausible = False
                        elif not math.isfinite(value) or abs(value) > 1.0:
                            plausible = False
                if not plausible:
                    self.reply({"error": "send an arm and a point of three numbers within a metre"}, status=400)
                    return
                if not gamepad.set_tip(arm.name, point):
                    self.reply({"error": "Press A on the controller to drive, then drag"}, status=409)
                    return
                self.reply({"ok": True})
                return
            if self.path.startswith("/fold/"):
                if fold is None:
                    self.reply({"error": "not available here"}, status=404)
                    return
                if self.path == "/fold/start":
                    refusal = fold.start(body.get("mode"), body.get("options", {}))
                    if refusal is not None:
                        self.reply({"error": refusal}, status=409)
                        return
                elif self.path == "/fold/continue":
                    fold.go_on()
                elif self.path == "/fold/stop":
                    fold.stop()
                self.reply({"ok": True})
                return
            if self.path in ARM_MOVES and arm is not None and arm.lent:
                self.reply({"error": f"{arm.name} is busy with a cloth fold"}, status=409)
                return
            if self.path in ("/mirror", "/leader"):
                for a in arms_by_name.values():
                    if a.lent:
                        self.reply({"error": "the arms are busy with a cloth fold"}, status=409)
                        return
            if self.path == "/goal":
                arm.set_target(body["pose"])
            elif self.path == "/torque":
                arm.request_torque(bool(body["on"]))
            elif self.path == "/speed":
                for a in arms_by_name.values():
                    a.speed = max(1, min(MAX_SPEED, float(body["speed"])))
            elif self.path == "/hold":
                # Stop everything, mirror and leader arm included, so no arm keeps chasing another. A cloth fold
                # holds the arms itself, so it is stopped too: `stop_fold` is given to the phone's page as well,
                # which has no fold buttons.
                if gamepad is not None:
                    gamepad.stop()
                if stop_fold is not None:
                    stop_fold()
                for a in arms_by_name.values():
                    a.follows = None
                    a.hold()
            elif self.path == "/mirror":
                # Every other arm that isn't a leader arm mirrors this one.
                leader = arms_by_name.get(body.get("leader"))
                lead_arms = set(leaders.values())
                for a in arms_by_name.values():
                    a.follows = (leader, True) if leader and a is not leader and a not in lead_arms else None
            elif self.path == "/leader":
                for follower, leader in leaders.items():
                    arms_by_name[follower].follows = (leader, False) if body["on"] else None
            elif self.path == "/pose":
                arm.go_pose(body["pose"], body.get("speed"))
            elif self.path == "/preset/save":
                presets = arm.load_presets()
                presets[body["name"]] = body["pose"]
                arm.save_presets(presets)
            elif self.path == "/preset/delete":
                presets = arm.load_presets()
                presets.pop(body["name"], None)
                arm.save_presets(presets)
            elif self.path == "/idle":
                arm.go_idle()
            elif self.path == "/twin":
                arm.save_twin(body["twin"])
            elif self.path == "/record/start":
                recording.start()
            elif self.path == "/record/stop":
                recording.stop()
            self.reply({"ok": True})

        def log_message(self, *args):
            pass  # keep the terminal quiet

    return Handler


def main():
    parser = argparse.ArgumentParser(description="SO-101 teleop panel")
    parser.add_argument("arms", nargs="*", metavar="ARM", help="which serial arms to open")
    parser.add_argument("--gamepad", action="store_true", help="drive the two side-by-side followers")
    args = parser.parse_args()
    names = args.arms or list(arms.ARMS)
    gamepad = None
    if args.gamepad:
        from gamepad import Gamepad
        from gamepad import MAX_JOINT_SPEED as gamepad_top_speed

        leaders = set(arms.CONFIG.get("leaders", {}).values())
        followers = [name for name in arms.CONFIG["arms"] if name not in leaders]
        if arms.CONFIG.get("layout") != "side_by_side" or len(followers) != 2:
            parser.error("--gamepad requires two side-by-side followers in arms.json, listed left then right")
        if args.arms and set(args.arms) != set(followers):
            parser.error("--gamepad requires both followers")
        names = followers
    unknown = [n for n in names if n not in arms.ARMS]
    if unknown:
        parser.error(f"No serial port for {', '.join(unknown)} on platform {arms.PLATFORM}")
    arms_by_name = {name: Arm(name) for name in names}
    if args.gamepad:
        spacing = arms.CONFIG.get("spacing_cm")
        if spacing is None:
            parser.error("--gamepad requires spacing_cm in arms.json")
        gamepad = Gamepad(*arms_by_name.values(), float(spacing) / 100)
        # The panel's speed slider caps the controller too: start it at the controller's own top speed.
        for arm in arms_by_name.values():
            arm.speed = gamepad_top_speed
        log_folder = HERE / "runs"
        log_folder.mkdir(exist_ok=True)
        log_file = log_folder / f"gamepad_{time.strftime('%Y%m%d_%H%M%S')}.log"
        gamepad.log = open(log_file, "w", buffering=1, encoding="utf-8")
        print(f"Gamepad log: {log_file}")
        threading.Thread(target=gamepad.run, daemon=True).start()
    else:
        for arm in arms_by_name.values():
            threading.Thread(target=run, args=(arm,), daemon=True).start()
    camera = panel_camera.Camera()
    # On its own thread: looking for the phone can block, and the arms must not wait for it.
    threading.Thread(target=camera.connect_loop, daemon=True).start()
    recording = teleop_record.TeleopRecording(arms_by_name, camera)
    fold = fold_job.FoldJob(arms_by_name, camera)
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), make_handler(arms_by_name, camera, recording, fold=fold,
                                                                           stop_fold=fold.stop, gamepad=gamepad))

    print(f"Arms: {', '.join(f'{n} ({arms.port(n)})' for n in names)}")
    print(f"Laptop: http://localhost:{HTTP_PORT}")
    start_phone_server(arms_by_name, camera, recording, fold.stop, gamepad)
    print("Ctrl+C to stop; torque state is left as is.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    if gamepad is not None:
        gamepad.shutdown.set()
        gamepad.stop()
    recording.stop()


def start_phone_server(arms_by_name, camera, recording, stop_fold=None, gamepad=None):
    """The phone teleop page over HTTPS. Skipped when openssl is missing (a stock Windows install)."""
    try:
        ensure_certificate()
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        print(f"Phone teleop is off: could not make its HTTPS certificate with openssl ({err})")
        return
    token = secrets.token_urlsafe(8)
    phone_server = ThreadingHTTPServer(("0.0.0.0", PHONE_PORT), make_handler(arms_by_name, camera, recording, token,
                                                                           stop_fold=stop_fold, gamepad=gamepad))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)
    phone_server.socket = context.wrap_socket(phone_server.socket, server_side=True)
    threading.Thread(target=phone_server.serve_forever, daemon=True).start()

    phone_url = f"https://{lan_ip()}:{PHONE_PORT}/phone?t={token}"
    print(f"Phone (same Wi-Fi): {phone_url}")
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(phone_url)
        qr.print_ascii(invert=True)
    except ImportError:
        pass


if __name__ == "__main__":
    main()

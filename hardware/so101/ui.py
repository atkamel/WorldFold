"""Real-time control panel + live 3D twin for the SO-101 follower.

    .venv/bin/python ui.py        # then open http://localhost:8101
                                  # phone: open the https link it prints

One background thread owns the serial port: 50 times a second it reads every
motor's position and, with torque on, nudges each goal toward its slider. The
browser only talks to this server, never to the motors directly.
"""

import json
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

PORT = "/dev/tty.usbmodem5B790153891"
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

HERE = Path(__file__).parent
LIMITS_FILE = HERE / "limits.json"
POSE_FILE = HERE / "idle_pose.json"
TWIN_FILE = HERE / "twin.json"
PRESETS_FILE = HERE / "presets.json"
PAGE_FILE = HERE / "ui.html"
PHONE_PAGE_FILE = HERE / "phone.html"
CERT_FILE = HERE / "certs" / "cert.pem"
KEY_FILE = HERE / "certs" / "key.pem"

# STS3215 control table addresses
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_TORQUE_LIMIT = 48
ADDR_PRESENT_POSITION = 56

GRIPPER_TORQUE_LIMIT = 500  # 50% of max, as lerobot does, so it can't burn out squeezing


class Arm:
    def __init__(self):
        self.port = scs.PortHandler(PORT)
        self.packet = scs.PacketHandler(0)  # protocol 0 = Feetech STS series
        if not self.port.openPort() or not self.port.setBaudRate(BAUDRATE):
            sys.exit(f"Could not open {PORT}")

        limits = json.loads(LIMITS_FILE.read_text())
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
            sys.exit(f"Could not read all motors: {self.present}")
        self.target = dict(self.present)  # where the sliders want each joint
        self.command = dict(self.present)  # goal actually sent, eased toward target
        # Keep whatever torque state the arm is in (idle.py may have left it holding).
        self.torque = bool(self.read1(GRIPPER_ID, ADDR_TORQUE_ENABLE))
        self.pending_torque = None
        self.pending_pose = None
        self.speed = DEFAULT_SPEED
        self.burst_speed = None  # temporary faster speed until the current pose is reached
        self.pending_burst = None

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
            return  # keep the last good readings; the next tick retries
        for m in MOTOR_IDS:
            if self.reader.isAvailable(m, ADDR_PRESENT_POSITION, 2):
                self.present[m] = self.reader.getData(m, ADDR_PRESENT_POSITION, 2)

    def send_goals(self, goals):
        """Send every motor its goal in ONE packet, so all joints move in step."""
        sync = scs.GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, 2)
        for m, goal in goals.items():
            goal = round(goal)  # command is eased in fractional steps
            sync.addParam(m, [scs.SCS_LOBYTE(goal), scs.SCS_HIBYTE(goal)])
        sync.txPacket()

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
            with self.lock:
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
            time.sleep(TICK_SECONDS)

    # --- called from HTTP threads: only touch shared state, never the bus ---

    def set_target(self, motor_id, pos):
        lo, hi = self.limits[motor_id]
        with self.lock:
            self.target[motor_id] = max(lo, min(hi, int(pos)))

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
        self.go_pose(json.loads(POSE_FILE.read_text()))

    def hold(self):
        """Stop: aim every joint at the goal it's already been sent, so the
        easing has nowhere left to go and the arm freezes where it is."""
        with self.lock:
            self.target = dict(self.command)

    def state(self):
        with self.lock:
            return {
                "present": self.present,
                "target": self.target,
                "torque": self.torque,
                "limits": self.limits,
                "twin": json.loads(TWIN_FILE.read_text()) if TWIN_FILE.exists() else None,
                "presets": load_presets(),
                "speed": self.speed,
            }


def load_presets():
    """Saved poses, by name: {motor id: encoder value}."""
    return json.loads(PRESETS_FILE.read_text()) if PRESETS_FILE.exists() else {}


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


def make_handler(arm, token=None):
    """token=None: trusted (localhost). Otherwise every request must carry ?t=token,
    so only someone with the printed link can drive the arm over Wi-Fi."""

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
            return url.path

        def do_GET(self):
            path = self.route()
            if path is None:
                return
            if path == "/state":
                self.reply(arm.state())
            elif path == "/phone":
                self.reply(PHONE_PAGE_FILE.read_bytes(), "text/html")
            else:
                self.reply(PAGE_FILE.read_bytes(), "text/html")

        def do_POST(self):
            path = self.route()
            if path is None:
                return
            self.path = path
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or "{}")
            if self.path == "/goal":
                arm.set_target(int(body["id"]), body["pos"])
            elif self.path == "/torque":
                arm.request_torque(bool(body["on"]))
            elif self.path == "/speed":
                arm.speed = max(1, min(MAX_SPEED, float(body["speed"])))
            elif self.path == "/hold":
                arm.hold()
            elif self.path == "/pose":
                arm.go_pose(body["pose"], body.get("speed"))
            elif self.path == "/preset/save":
                presets = load_presets()
                presets[body["name"]] = body["pose"]
                PRESETS_FILE.write_text(json.dumps(presets, indent=2))
            elif self.path == "/preset/delete":
                presets = load_presets()
                presets.pop(body["name"], None)
                PRESETS_FILE.write_text(json.dumps(presets, indent=2))
            elif self.path == "/idle":
                arm.go_idle()
            elif self.path == "/twin":
                TWIN_FILE.write_text(json.dumps(body, indent=2))
            self.reply({"ok": True})

        def log_message(self, *args):
            pass  # keep the terminal quiet

    return Handler


def main():
    arm = Arm()
    threading.Thread(target=arm.control_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), make_handler(arm))

    ensure_certificate()
    token = secrets.token_urlsafe(8)
    phone_server = ThreadingHTTPServer(("0.0.0.0", PHONE_PORT), make_handler(arm, token))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)
    phone_server.socket = context.wrap_socket(phone_server.socket, server_side=True)
    threading.Thread(target=phone_server.serve_forever, daemon=True).start()

    phone_url = f"https://{lan_ip()}:{PHONE_PORT}/phone?t={token}"
    print(f"Laptop: http://localhost:{HTTP_PORT}")
    print(f"Phone (same Wi-Fi): {phone_url}")
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(phone_url)
        qr.print_ascii(invert=True)
    except ImportError:
        pass
    print("Ctrl+C to stop; torque state is left as is.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

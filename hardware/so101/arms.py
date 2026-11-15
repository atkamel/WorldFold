"""Which serial port and config folder belong to each SO-101 follower.

Each arm has its own USB serial adapter, so each is its own bus and both arms
use motor IDs 1-6 without clashing. An arm's calibration lives in its own
folder (arm1/, arm2/): limits.json, idle_pose.json, twin.json, presets.json.

Set PLATFORM to the computer the arms are plugged into.

To find an arm's port, plug in only that arm, then:
  mac:      ls /dev/tty.usbmodem*   (the name carries the adapter's serial number,
            so it stays the same across replugs and reboots)
  windows:  Device Manager > Ports (COM & LPT). Windows gives a COM number per
            adapter and USB socket, so it can change when an arm moves to another socket.
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent

PLATFORM = "windows"  # "windows" or "mac"

PORTS = {
    "mac": {
        "arm1": "/dev/tty.usbmodem5B790153891",
        "arm2": "/dev/tty.usbmodem5B8E1131611",
    },
    "windows": {
        "arm1": "COM3",
        "arm2": "COM5",
    },
}

ARM_NAMES = ["arm1", "arm2"]


def add_arm_argument(parser):
    # Required on purpose: a default arm would let a forgotten flag write one
    # arm's calibration into the other arm's motors and files.
    parser.add_argument("--arm", required=True, choices=ARM_NAMES, help="which follower to talk to")


def resolve(name):
    """Return (serial port, config folder) for an arm."""
    if PLATFORM not in PORTS:
        sys.exit(f"PLATFORM in arms.py is {PLATFORM!r}; it must be one of {list(PORTS)}")
    port = PORTS[PLATFORM][name]
    if port is None:
        sys.exit(f"No {PLATFORM} port set for {name}. Plug in only {name}, find its port, add it to arms.py")
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    return port, folder

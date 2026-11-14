"""Which serial port and config folder belong to each SO-101 follower.

Each arm has its own USB serial adapter, so each is its own bus and both arms
use motor IDs 1-6 without clashing. An arm's calibration lives in its own
folder (arm1/, arm2/): limits.json, idle_pose.json, twin.json, presets.json.

To find an arm's port, plug in only that arm and run: ls /dev/tty.usbmodem*
The name carries the adapter's serial number, so it stays the same across
replugs and reboots.
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent

PORTS = {
    "arm1": "/dev/tty.usbmodem5B790153891",
    "arm2": None,  # not known yet: plug in arm 2 alone, run ls /dev/tty.usbmodem*, paste it here
}


def add_arm_argument(parser):
    # Required on purpose: a default arm would let a forgotten flag write one
    # arm's calibration into the other arm's motors and files.
    parser.add_argument("--arm", required=True, choices=PORTS, help="which follower to talk to")


def resolve(name):
    """Return (serial port, config folder) for an arm."""
    port = PORTS[name]
    if port is None:
        sys.exit(f"No port set for {name}. Plug in only {name}, run ls /dev/tty.usbmodem*, add it to arms.py")
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    return port, folder

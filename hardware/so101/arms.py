"""Which SO-101 arms exist, and where each one's files live.

arms.json names every arm and its serial port, plus how the arms stand:
    "layout": "side_by_side"  both face the same way, the first arm listed on the left
    "layout": "facing"        they face each other across the table
    "spacing_cm"              distance between the two bases
    "leaders"                 optional {follower: leader arm}: a limp SO-101 moved by
                              hand whose joint angles the follower copies (ui.py)
Left is seen from behind the arms (side_by_side) or from the side of the
table between them (facing). Find a port with `ls /dev/tty.usbmodem*`: plug in
only that arm; the name carries the adapter's serial number, so it stays the
same across replugs and reboots.

Each arm has its own USB serial adapter, so each is its own bus and all arms
use motor IDs 1-6 without clashing. An arm's calibration lives in its own
folder (arm1/, arm2/, leader/): limits.json (set_limits.py), idle_pose.json
(idle.py record), twin.json and presets.json (ui.py).
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "arms.json").read_text())
ARMS = CONFIG["arms"]


def port(name):
    return ARMS[name]


def folder(name):
    return HERE / name


def add_arm_argument(parser):
    # Required on purpose: a default arm would let a forgotten flag write one
    # arm's calibration into the other arm's motors and files.
    parser.add_argument("--arm", required=True, choices=ARMS, help="which arm to talk to")


def resolve(name):
    """Return (serial port, config folder) for an arm."""
    folder(name).mkdir(exist_ok=True)
    return port(name), folder(name)

"""Which SO-101 arms exist, and where each one's files live.

arms.json names every arm and its serial port, plus how the arms stand:
    "layout": "side_by_side"  both face the same way, the first arm listed on the left
    "layout": "facing"        they face each other across the table
    "spacing_cm"              distance between the two bases
    "leaders"                 optional {follower: leader arm}: a limp SO-101 moved by
                              hand whose joint angles the follower copies (ui.py)
Left is seen from behind the arms (side_by_side) or from the side of the
table between them (facing).

Each arm lists one port per platform; PLATFORM below picks which set is used.
An arm with no port on that platform (null) is left out. To find a port, plug
in only that arm, then:
  mac:      ls /dev/tty.usbmodem*   (the name carries the adapter's serial
            number, so it stays the same across replugs and reboots)
  windows:  Device Manager > Ports (COM & LPT). The COM number can change when
            the arm moves to another USB socket.

Each arm has its own USB serial adapter, so each is its own bus and all arms
use motor IDs 1-6 without clashing. An arm's calibration lives in its own
folder (arm1/, arm2/, leader/): limits.json (set_limits.py), idle_pose.json
(idle.py record), twin.json and presets.json (ui.py).
"""

import json
import sys
from pathlib import Path

PLATFORM = "windows"  # "windows" or "mac": the computer the arms are plugged into
PLATFORMS = ["windows", "mac"]

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "arms.json").read_text())


def ports_for(config_arms, platform):
    """{arm name: serial port} for the arms that have a port on this platform."""
    if platform not in PLATFORMS:
        sys.exit(f"PLATFORM in arms.py is {platform!r}; it must be one of {PLATFORMS}")
    ports = {}
    for name in config_arms:
        port_name = config_arms[name].get(platform)
        if port_name is not None:
            ports[name] = port_name
    return ports


ARMS = ports_for(CONFIG["arms"], PLATFORM)


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

"""Which SO-101 arms exist, and where each one's files live.

arms.json names every arm and its serial port, plus how the arms stand:
    "layout": "side_by_side"  both face the same way, "left" on the left
    "layout": "facing"        they face each other across the table
    "spacing_cm"              distance between the two bases
    "leaders"                 optional {follower: leader arm}: a limp SO-101 moved by
                              hand whose joint angles the follower copies (ui.py)
Names are seen from behind the arms (side_by_side) or from the side of the
table between them (facing). Find a port with `ls /dev/tty.usbmodem*`.

Each arm keeps its own calibration in arms/<name>/: limits.json (set_limits.py),
idle_pose.json (idle.py record), twin.json and presets.json (ui.py).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "arms.json").read_text())
ARMS = CONFIG["arms"]


def port(name):
    return ARMS[name]


def folder(name):
    return HERE / "arms" / name


def choose(args):
    """Pop an arm name off the front of args. With none given, use the only
    arm in arms.json, so single-arm commands stay as they were."""
    if args and args[0] in ARMS:
        return args.pop(0)
    if len(ARMS) == 1:
        return next(iter(ARMS))
    sys.exit(f"Name an arm first: one of {', '.join(ARMS)}")

"""arms: which serial port each arm gets on the chosen platform."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

import arms

CONFIG_ARMS = {
    "arm1": {"mac": "/dev/tty.one", "windows": "COM3"},
    "leader": {"mac": "/dev/tty.leader", "windows": None},
}


def test_each_platform_gets_its_own_port_names():
    assert arms.ports_for(CONFIG_ARMS, "mac") == {"arm1": "/dev/tty.one", "leader": "/dev/tty.leader"}
    assert arms.ports_for(CONFIG_ARMS, "windows")["arm1"] == "COM3"


def test_an_arm_with_no_port_on_this_platform_is_left_out():
    assert "leader" not in arms.ports_for(CONFIG_ARMS, "windows")


def test_an_unknown_platform_is_refused():
    with pytest.raises(SystemExit):
        arms.ports_for(CONFIG_ARMS, "linux")

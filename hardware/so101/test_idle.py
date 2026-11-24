"""Tests for idle.py against a fake packet handler (no serial port, no arm)."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import arms

OVERLOAD_ERROR = 32


class FakePacket:
    """Motor 3 answers every torque write with an error byte; the others accept it."""

    def __init__(self):
        self.torque = []  # (motor id, value) per torque-enable write

    def write1ByteTxRx(self, port, motor, address, value):
        self.torque.append((motor, value))
        if motor == 3:
            return 0, OVERLOAD_ERROR
        return 0, 0


@pytest.fixture
def idle(monkeypatch):
    arm_name = list(arms.ARMS)[0]
    monkeypatch.setattr(sys, "argv", ["idle.py", "--arm", arm_name, "release"])
    sys.modules.pop("idle", None)
    module = importlib.import_module("idle")
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


def test_release_reaches_every_motor_when_one_fails(idle):
    packet = FakePacket()
    failures = idle.torque_off_all(None, packet)

    motors_written = set()
    for motor, value in packet.torque:
        assert value == 0
        motors_written.add(motor)
    assert motors_written == {1, 2, 3, 4, 5, 6}

    assert len(failures) == 1
    assert "3" in failures[0]
    assert str(OVERLOAD_ERROR) in failures[0]

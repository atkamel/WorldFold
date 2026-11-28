"""ui.Arm lending its serial port to another program and taking it back, against a fake servo SDK."""

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


class FakePort:
    def __init__(self, device):
        self.device = device
        self.open = False
        self.pos = {m: 2000 + m for m in range(1, 7)}
        self.torque_on = 1
        self.traffic = 0        # reads and writes that reached the bus
        self.goals = []         # one {motor id: goal} per sync write
        self.ser = types.SimpleNamespace(reset_input_buffer=lambda: None)

    def openPort(self):
        self.open = True
        return True

    def closePort(self):
        self.open = False

    def setBaudRate(self, rate):
        return True


def make_fake_sdk(ports):
    sdk = types.ModuleType("scservo_sdk")
    sdk.COMM_SUCCESS = 0
    sdk.SCS_LOBYTE = lambda x: x & 0xFF
    sdk.SCS_HIBYTE = lambda x: (x >> 8) & 0xFF
    sdk.PortHandler = lambda device: ports.setdefault(device, FakePort(device))

    def use(port):
        assert port.open, "the bus was used while the port was closed"
        port.traffic += 1

    class PacketHandler:
        def __init__(self, protocol):
            pass

        def read1ByteTxRx(self, port, motor, address):
            use(port)
            return port.torque_on, 0, 0

        def write1ByteTxRx(self, port, motor, address, value):
            use(port)
            return 0, 0

        def write2ByteTxRx(self, port, motor, address, value):
            use(port)
            return 0, 0

    class GroupSyncRead:
        def __init__(self, port, packet, address, length):
            self.port = port

        def addParam(self, m):
            pass

        def txRxPacket(self):
            use(self.port)
            return 0

        def isAvailable(self, m, address, length):
            return True

        def getData(self, m, address, length):
            return self.port.pos[m]

    class GroupSyncWrite:
        def __init__(self, port, packet, address, length):
            self.port = port
            self.goals = {}

        def addParam(self, m, data):
            self.goals[m] = data[0] | data[1] << 8

        def txPacket(self):
            use(self.port)
            self.port.goals.append(dict(self.goals))
            return 0

    sdk.PacketHandler = PacketHandler
    sdk.GroupSyncRead = GroupSyncRead
    sdk.GroupSyncWrite = GroupSyncWrite
    return sdk


@pytest.fixture
def arm(tmp_path, monkeypatch):
    ports = {}
    monkeypatch.setitem(sys.modules, "scservo_sdk", make_fake_sdk(ports))
    sys.modules.pop("ui", None)
    ui = importlib.import_module("ui")
    monkeypatch.setattr(ui.arms, "folder", lambda name: tmp_path)
    monkeypatch.setattr(ui.arms, "port", lambda name: "dev")
    limits = {str(m): {"true": {"min": 685, "max": 3410}} for m in range(1, 7)}
    (tmp_path / "limits.json").write_text(json.dumps(limits))
    made = ui.Arm("L")
    yield made
    sys.modules.pop("ui", None)


def test_a_lent_arm_leaves_the_bus_alone(arm):
    arm.lend()
    assert not arm.port.open
    assert arm.state()["lent"]
    before = arm.port.traffic
    arm.tick()
    arm.tick()
    assert arm.port.traffic == before


def test_taking_an_arm_back_holds_it_where_the_other_program_left_it(arm):
    arm.lend()
    arm.request_torque(False)                # asked for while lent: must not fire later
    arm.go_pose({1: 3000})
    arm.port.pos = {m: 2500 + m for m in range(1, 7)}       # the other program moved the arm
    arm.take_back()
    assert arm.port.open
    assert not arm.state()["lent"]
    assert arm.state()["torque"]
    assert arm.state()["target"] == arm.port.pos
    arm.tick()
    assert arm.port.goals[-1] == arm.port.pos       # the goal sent is where the arm already is

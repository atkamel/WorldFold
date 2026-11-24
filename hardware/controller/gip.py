"""Decode input from the PowerA EnWired controller's USB GIP reports."""

from dataclasses import dataclass, replace
from struct import Struct


MAIN_REPORT = 0x20
GUIDE_REPORT = 0x07
_HEADER = Struct("<BBBB")
_INPUT = Struct("<HHHhhhh")
_BUTTON_BITS = (
    (2, "menu"),
    (3, "view"),
    (4, "a"),
    (5, "b"),
    (6, "x"),
    (7, "y"),
    (8, "dpad_up"),
    (9, "dpad_down"),
    (10, "dpad_left"),
    (11, "dpad_right"),
    (12, "lb"),
    (13, "rb"),
    (14, "left_stick"),
    (15, "right_stick"),
)


@dataclass(frozen=True)
class ControllerState:
    buttons: tuple[str, ...] = ()
    guide: bool = False
    left_trigger: int = 0
    right_trigger: int = 0
    left_x: int = 0
    left_y: int = 0
    right_x: int = 0
    right_y: int = 0


@dataclass(frozen=True)
class InputReport:
    command: int
    options: int
    sequence: int
    payload: bytes

    def acknowledge(self) -> bytes | None:
        if self.command == GUIDE_REPORT and self.options & 0x10:
            return bytes((0x01, 0x20, self.sequence, 9, 0, GUIDE_REPORT, 0x20, 2, 0, 0, 0, 0, 0))
        return None

    def state(self, previous: ControllerState = ControllerState()) -> ControllerState:
        if self.command == GUIDE_REPORT:
            return replace(previous, guide=bool(self.payload[0] & 0x03))

        mask, lt, rt, lx, ly, rx, ry = _INPUT.unpack_from(self.payload)
        buttons = tuple(name for bit, name in _BUTTON_BITS if mask & (1 << bit))
        if len(self.payload) > _INPUT.size and self.payload[_INPUT.size] & 0x01:
            buttons += ("share",)
        return ControllerState(buttons, previous.guide, lt, rt, lx, ly, rx, ry)


def parse_input_report(packet: bytes) -> InputReport | None:
    """Return a complete main or Guide report, or None for unrelated GIP traffic."""
    if not packet or packet[0] not in (GUIDE_REPORT, MAIN_REPORT):
        return None
    if len(packet) < _HEADER.size:
        raise ValueError("truncated GIP input header")
    command, options, sequence, length = _HEADER.unpack_from(packet)
    if options & 0xC0 or length & 0x80:
        raise ValueError("unsupported GIP input header")
    minimum = 2 if command == GUIDE_REPORT else _INPUT.size
    end = _HEADER.size + length
    if length < minimum or len(packet) < end:
        raise ValueError("truncated GIP input payload")
    return InputReport(command, options, sequence, packet[_HEADER.size:end])

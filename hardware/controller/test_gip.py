from struct import pack

import pytest

from hardware.controller.gip import ControllerState, parse_input_report


def test_main_report_decodes_buttons_triggers_and_signed_axes():
    payload = pack("<HHHhhhh", 0xFFFC, 1023, 499, -32768, 32767, -7, 256)
    packet = b"\x20\x00\xff\x20" + payload + b"\x01" + bytes(17)
    report = parse_input_report(packet)

    assert report.sequence == 255
    assert report.state() == ControllerState(
        buttons=("menu", "view", "a", "b", "x", "y", "dpad_up", "dpad_down",
                 "dpad_left", "dpad_right", "lb", "rb", "left_stick", "right_stick", "share"),
        left_trigger=1023, right_trigger=499, left_x=-32768, left_y=32767, right_x=-7, right_y=256,
    )
    assert report.acknowledge() is None


def test_guide_updates_only_guide_and_acknowledges_the_same_sequence():
    previous = ControllerState(buttons=("a",), left_x=123)
    press = parse_input_report(bytes.fromhex("07 30 51 02 01 5b"))
    assert press.acknowledge() == bytes.fromhex("01 20 51 09 00 07 20 02 00 00 00 00 00")
    assert press.state(previous) == ControllerState(buttons=("a",), guide=True, left_x=123)

    main = parse_input_report(b"\x20\x00\x52\x0e" + bytes(14))
    assert main.state(press.state(previous)) == ControllerState(guide=True)
    release = parse_input_report(bytes.fromhex("07 20 52 02 00 5b"))
    assert release.acknowledge() is None
    assert release.state(main.state(press.state(previous))) == ControllerState()


def test_ignores_status_and_rejects_malformed_input():
    assert parse_input_report(bytes.fromhex("03 20 01 04 80 00 00 00")) is None
    with pytest.raises(ValueError, match="truncated GIP input header"):
        parse_input_report(b"\x20")
    with pytest.raises(ValueError, match="truncated GIP input payload"):
        parse_input_report(b"\x20\x00\x00\x20" + bytes(14))
    with pytest.raises(ValueError, match="unsupported GIP input header"):
        parse_input_report(b"\x20\x40\x00\x0e" + bytes(14))

"""Controller samples for teleop, using raw USB on macOS and XInput on Windows."""

import ctypes
import sys

from .controller import ControllerError, read_controller
from .gip import ControllerState


class _XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("buttons", ctypes.c_uint16), ("lt", ctypes.c_uint8), ("rt", ctypes.c_uint8),
        ("lx", ctypes.c_int16), ("ly", ctypes.c_int16), ("rx", ctypes.c_int16), ("ry", ctypes.c_int16),
    ]


class _XInputState(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_uint32), ("gamepad", _XInputGamepad)]


def xinput_state(pad):
    buttons = tuple(name for bit, name in ((0x0100, "lb"), (0x0200, "rb"), (0x2000, "b")) if pad.buttons & bit)
    return ControllerState(buttons=buttons, left_trigger=round(pad.lt * 1023 / 255),
                           right_trigger=round(pad.rt * 1023 / 255), left_x=pad.lx, left_y=pad.ly,
                           right_x=pad.rx, right_y=pad.ry)


def controller_samples(stopped, period=0.02):
    """Yield fresh controller state, or None until the first input or when unplugged."""
    if sys.platform == "win32":
        get_state = ctypes.WinDLL("xinput1_4.dll").XInputGetState
        get_state.argtypes = (ctypes.c_uint32, ctypes.POINTER(_XInputState))
        get_state.restype = ctypes.c_uint32
        state = _XInputState()
        while not stopped.is_set():
            result = get_state(0, ctypes.byref(state))
            if result not in (0, 1167):  # ERROR_DEVICE_NOT_CONNECTED
                raise ControllerError(f"XInputGetState failed ({result})")
            yield xinput_state(state.gamepad) if result == 0 else None
            stopped.wait(period)
    else:
        with read_controller(timeout_ms=75, heartbeat=True) as readings:
            for reading in readings:
                if stopped.is_set():
                    break
                if reading.state is not None:
                    yield reading.state

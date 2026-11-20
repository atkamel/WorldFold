"""Read the wired PowerA 20d6:2026 from Python via PyUSB and libusb."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from .gip import MAIN_REPORT, ControllerState, parse_input_report


VENDOR_ID = 0x20D6
PRODUCT_ID = 0x2026
_STARTUP = tuple(map(bytes.fromhex, (
    "05 20 00 01 00",        # Power on
    "0a 20 01 03 00 01 14",  # Enable the controller LED
    "06 20 02 02 01 00",     # Complete startup
)))
_STARTUP_PLACEHOLDER = bytes.fromhex("00 00 00 00 00 00 00 80 00 80 00 80 00 80")


class ControllerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reading:
    raw: bytes
    state: ControllerState | None


@contextmanager
def read_controller(timeout_ms: int = 250) -> Iterator[Iterator[Reading]]:
    """Claim the USB interface for the lifetime of a stream of raw and decoded reports."""
    if timeout_ms < 1:
        raise ValueError("timeout_ms must be positive")
    try:
        import usb.core
        import usb.util
    except ImportError as exc:
        raise ControllerError("Install PyUSB with python -m pip install -r hardware/controller/requirements.txt") from exc

    try:
        device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    except usb.core.NoBackendError as exc:
        raise ControllerError("PyUSB could not load libusb. On macOS, install it with brew install libusb.") from exc
    if device is None:
        raise ControllerError("PowerA 20d6:2026 is not accessible. Check the USB cable and run from Terminal.")

    try:
        try:
            config = device.get_active_configuration()
        except usb.core.USBError as exc:
            # An unconfigured device returns configuration 0 and PyUSB's error has no backend code.
            if exc.backend_error_code is not None or exc.errno is not None:
                raise
            device.set_configuration()
            config = device.get_active_configuration()

        interface = usb.util.find_descriptor(config, bInterfaceNumber=0, bAlternateSetting=0)
        if interface is None:
            raise ControllerError("PowerA USB input interface was not found")
        endpoint_in = usb.util.find_descriptor(interface, bEndpointAddress=0x81)
        endpoint_out = usb.util.find_descriptor(interface, bEndpointAddress=0x01)
        if endpoint_in is None or endpoint_out is None:
            raise ControllerError("PowerA USB input and output endpoints were not found")
        usb.util.claim_interface(device, interface.bInterfaceNumber)

        def write(packet: bytes) -> None:
            if endpoint_out.write(packet, timeout=1000) != len(packet):
                raise ControllerError("Short USB write to the controller")

        for packet in _STARTUP:
            write(packet)

        def readings() -> Iterator[Reading]:
            state = ControllerState()
            saw_main_report = False
            while True:
                try:
                    raw = bytes(endpoint_in.read(endpoint_in.wMaxPacketSize, timeout=timeout_ms))
                except usb.core.USBTimeoutError:
                    continue
                try:
                    report = parse_input_report(raw)
                except ValueError:
                    report = None  # Keep malformed frames available for raw inspection.
                if report is None:
                    yield Reading(raw, None)
                    continue
                if report.command == MAIN_REPORT and not saw_main_report:
                    saw_main_report = True
                    # A cold start can report four minimum axes before real input arrives.
                    if report.payload == _STARTUP_PLACEHOLDER.ljust(len(report.payload), b"\x00"):
                        yield Reading(raw, None)
                        continue
                ack = report.acknowledge()
                if ack is not None:
                    write(ack)
                state = report.state(state)
                yield Reading(raw, state)

        stream = readings()
        try:
            yield stream
        finally:
            stream.close()
    except usb.core.USBError as exc:
        raise ControllerError(f"USB communication with PowerA 20d6:2026 failed ({exc})") from exc
    finally:
        usb.util.dispose_resources(device)

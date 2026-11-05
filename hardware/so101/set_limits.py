"""Set the min/max position limits for ONE Feetech STS3215 servo by hand.

Plug in only the motor you want, set MOTOR_ID, then run (whatever ID the
motor currently has is overwritten with MOTOR_ID):
    .venv/bin/python set_limits.py

Move the joint to one boundary and press SPACE, then to the other and press
SPACE. The homing offset is set so the middle of that range reads 2047,
keeping it clear of the 4095 -> 0 rollover. The offset and limits are written
to the motor and saved to limits.json. Press q to quit without saving.
"""

import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

import scservo_sdk as scs

MOTOR_ID = 6
PORT = "/dev/tty.usbmodem5B790153891"
BAUDRATE = 1_000_000
LIMITS_FILE = Path(__file__).parent / "limits.json"

# STS3215 control table addresses
ADDR_ID = 5
ADDR_MIN_POSITION_LIMIT = 9
ADDR_MAX_POSITION_LIMIT = 11
ADDR_HOMING_OFFSET = 31
ADDR_TORQUE_ENABLE = 40
ADDR_LOCK = 55
ADDR_PRESENT_POSITION = 56

ENCODER_RESOLUTION = 4096  # one full turn = 0..4095
CENTER = ENCODER_RESOLUTION // 2 - 1  # 2047, same as lerobot-calibrate


def read_register(port, packet, address, size, tries=5):
    """Read a 1- or 2-byte register. A late reply from an earlier EEPROM
    write can still be in the receive buffer and garble this read, so drop
    leftover bytes and retry."""
    read = packet.read1ByteTxRx if size == 1 else packet.read2ByteTxRx
    for _ in range(tries):
        port.ser.reset_input_buffer()
        try:
            value, comm, error = read(port, MOTOR_ID, address)
        except IndexError:  # SDK crashes on a truncated reply
            comm = None
        if comm == scs.COMM_SUCCESS and not error:
            return value
        time.sleep(0.05)
    sys.exit(f"\nReading address {address} failed after {tries} tries")


def read_position(port, packet):
    return read_register(port, packet, ADDR_PRESENT_POSITION, 2)


def key_pressed():
    """Return the key pressed since the last call, or None (non-blocking)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def encode_sign_magnitude(value, sign_bit=11):
    """Homing_Offset stores the sign in bit 11 and the size in bits 0-10."""
    return (int(value < 0) << sign_bit) | abs(value)


def capture_range(port, packet):
    """Stream the live position and record two boundaries on SPACE.

    The raw reading jumps 4095 -> 0 when the joint passes that point, so we
    "unwrap" it: add up each small step between reads into a running total
    that keeps counting past 4095 (or below 0). Boundaries are recorded on
    that unwrapped scale, so a range crossing the rollover is still one
    continuous arc."""
    recorded = []
    last = read_position(port, packet)
    unwrapped = last

    print("Move to one boundary, SPACE; then the other, SPACE. q = quit.\n")
    while len(recorded) < 2:
        pos = read_position(port, packet)
        step = (pos - last + ENCODER_RESOLUTION // 2) % ENCODER_RESOLUTION - ENCODER_RESOLUTION // 2
        unwrapped += step
        last = pos

        marks = "  ".join(f"boundary {i + 1}: {p % ENCODER_RESOLUTION}" for i, p in enumerate(recorded))
        print(f"\r[motor {MOTOR_ID}] position: {pos:4d}   {marks}   ", end="", flush=True)

        key = key_pressed()
        if key == " ":
            recorded.append(unwrapped)
        elif key == "q":
            sys.exit("\nQuit, nothing saved.")
        time.sleep(0.02)

    print()
    lo, hi = min(recorded), max(recorded)
    if hi - lo >= ENCODER_RESOLUTION:
        sys.exit("Range is a full turn or more, so it can't be stored as limits. Nothing saved.")
    return lo, hi


def centre_range(lo, hi):
    """Pick the homing offset that puts the middle of [lo, hi] at 2047.

    Feetech: Present_Position = Actual_Position - Homing_Offset, so shifting
    by (middle - 2047) makes the middle read 2047 and the boundaries land at
    2047 -/+ half the span: never near 0 or 4095."""
    middle = (lo + hi) // 2 % ENCODER_RESOLUTION
    # The offset is stored in 11 bits (+/-2047), so take the shorter way round.
    offset = (middle - CENTER + ENCODER_RESOLUTION // 2) % ENCODER_RESOLUTION - ENCODER_RESOLUTION // 2
    offset = max(-2047, offset)
    new_lo = (lo - offset) % ENCODER_RESOLUTION
    return offset, new_lo, new_lo + (hi - lo)


def write_eeprom(port, packet, address, value, what):
    """EEPROM survives power-off but is locked; unlock, write 2 bytes, relock."""
    write_verified(port, packet, ADDR_LOCK, 1, 0, "Unlocking EEPROM")
    write_verified(port, packet, address, 2, value, what)
    write_verified(port, packet, ADDR_LOCK, 1, 1, "Locking EEPROM")


def write_verified(port, packet, address, size, value, what, tries=5):
    """Write, then read the register back to confirm it stuck.

    While saving to EEPROM the motor often answers late, so the write's own
    reply is unreliable: it can report failure even though the value landed.
    Reading the register back is the real check."""
    write = packet.write1ByteTxRx if size == 1 else packet.write2ByteTxRx
    for _ in range(tries):
        write(port, MOTOR_ID, address, value)
        time.sleep(0.05)  # let any late reply arrive before we flush it
        if read_register(port, packet, address, size) == value:
            return
    sys.exit(f"\n{what} failed: value {value} didn't stick after {tries} tries")


def assign_id(port, packet):
    """Find the single motor on the bus and give it MOTOR_ID.

    Every packet is addressed to one ID, so two motors with the same ID
    answer at once and garble each other. That's why only one motor may be
    plugged in while its ID is changed."""
    found = []
    for motor_id in range(254):  # 254 is broadcast, not a real ID
        port.ser.reset_input_buffer()
        _, comm, _ = packet.ping(port, motor_id)
        if comm == scs.COMM_SUCCESS:
            found.append(motor_id)
    if len(found) != 1:
        sys.exit(f"Expected exactly one motor on the bus, found IDs {found}. Plug in only one.")

    old_id = found[0]
    if old_id == MOTOR_ID:
        return
    # Unlock and write as the old ID; from then on the motor answers to the new one.
    packet.write1ByteTxRx(port, old_id, ADDR_LOCK, 0)
    packet.write1ByteTxRx(port, old_id, ADDR_ID, MOTOR_ID)
    time.sleep(0.05)
    if read_register(port, packet, ADDR_ID, 1) != MOTOR_ID:
        sys.exit(f"Changing ID {old_id} -> {MOTOR_ID} failed")
    write_verified(port, packet, ADDR_LOCK, 1, 1, "Locking EEPROM")
    print(f"Changed motor ID {old_id} -> {MOTOR_ID}\n")


def limits_entry(offset, lo, hi):
    """'true' = what the motor reports after the offset (no rollover in range).
    'raw' = what the bare encoder reads at those same spots: true + offset,
    wrapped to 0..4095. Raw min > raw max means the range crosses 4095 -> 0."""
    raw = lambda value: (value + offset) % ENCODER_RESOLUTION
    return {
        "homing_offset": offset,
        "raw": {"min": raw(lo), "max": raw(hi)},
        "true": {"min": lo, "max": hi},
    }


def save_limits(offset, lo, hi):
    limits = json.loads(LIMITS_FILE.read_text()) if LIMITS_FILE.exists() else {}
    limits[str(MOTOR_ID)] = limits_entry(offset, lo, hi)
    LIMITS_FILE.write_text(json.dumps(limits, indent=2))


def main():
    port = scs.PortHandler(PORT)
    packet = scs.PacketHandler(0)  # protocol 0 = Feetech STS series
    if not port.openPort() or not port.setBaudRate(BAUDRATE):
        sys.exit(f"Could not open {PORT}")

    print(f"\n========== MOTOR ID: {MOTOR_ID} ==========\n")
    assign_id(port, packet)

    # Torque off so the joint moves freely by hand.
    write_verified(port, packet, ADDR_TORQUE_ENABLE, 1, 0, "Disabling torque")
    # Clear any old offset so we record the true encoder angle.
    write_eeprom(port, packet, ADDR_HOMING_OFFSET, 0, "Clearing homing offset")

    old_terminal = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)  # read single keys without waiting for Enter
    try:
        raw_lo, raw_hi = capture_range(port, packet)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)

    offset, lo, hi = centre_range(raw_lo, raw_hi)
    write_eeprom(port, packet, ADDR_HOMING_OFFSET, encode_sign_magnitude(offset), "Writing homing offset")
    write_eeprom(port, packet, ADDR_MIN_POSITION_LIMIT, lo, "Writing min limit")
    write_eeprom(port, packet, ADDR_MAX_POSITION_LIMIT, hi, "Writing max limit")
    save_limits(offset, lo, hi)
    port.closePort()

    entry = limits_entry(offset, lo, hi)
    print(f"Motor {MOTOR_ID}: homing offset = {offset}")
    print(f"  raw:  min = {entry['raw']['min']}, max = {entry['raw']['max']}")
    print(f"  true: min = {lo}, max = {hi}")
    print(f"Written to the motor and saved to {LIMITS_FILE.name}")


if __name__ == "__main__":
    main()

"""Check that an arm's motors hold the limits and homing offsets saved in <arm>/limits.json.

    python check_limits.py --arm arm2

Read only: nothing is written to the motors and torque is left as it is. Stop ui.py first,
since only one program can use an arm's serial port.
"""

import argparse
import json
import sys

import scservo_sdk as scs

import arms

BAUDRATE = 1_000_000
MOTOR_IDS = [1, 2, 3, 4, 5, 6]

# STS3215 control table addresses (same as set_limits.py)
ADDR_MIN_POSITION_LIMIT = 9
ADDR_MAX_POSITION_LIMIT = 11
ADDR_HOMING_OFFSET = 31
ADDR_PRESENT_POSITION = 56


def read2(port, packet, motor_id, address):
    for attempt in range(5):
        value, comm, error = packet.read2ByteTxRx(port, motor_id, address)
        if comm == scs.COMM_SUCCESS and not error:
            return value
    return None


def decode_sign_magnitude(value, sign_bit=11):
    """The homing offset register: bit 11 is the sign, the lower bits the size."""
    size = value & ((1 << sign_bit) - 1)
    if value & (1 << sign_bit):
        return -size
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arms.add_arm_argument(parser)
    args = parser.parse_args()
    port_name, folder = arms.resolve(args.arm)
    saved = json.loads((folder / "limits.json").read_text())

    port = scs.PortHandler(port_name)
    packet = scs.PacketHandler(0)  # protocol 0 = Feetech STS series
    try:
        opened = port.openPort() and port.setBaudRate(BAUDRATE)
    except OSError:  # pyserial's SerialException is an OSError
        opened = False
    if not opened:
        sys.exit(f"Could not open {port_name} ({args.arm}). Is ui.py still running?")

    all_match = True
    for motor_id in MOTOR_IDS:
        low = read2(port, packet, motor_id, ADDR_MIN_POSITION_LIMIT)
        high = read2(port, packet, motor_id, ADDR_MAX_POSITION_LIMIT)
        raw_offset = read2(port, packet, motor_id, ADDR_HOMING_OFFSET)
        position = read2(port, packet, motor_id, ADDR_PRESENT_POSITION)
        if low is None or high is None or raw_offset is None:
            print(f"motor {motor_id}: no answer")
            all_match = False
            continue
        offset = decode_sign_magnitude(raw_offset)
        entry = saved[str(motor_id)]
        matches = (low == entry["true"]["min"] and high == entry["true"]["max"]
                   and offset == entry["homing_offset"])
        if not matches:
            all_match = False
        verdict = "match" if matches else "DIFFERENT"
        print(f"motor {motor_id}: motor min {low} max {high} offset {offset} | "
              f"file min {entry['true']['min']} max {entry['true']['max']} offset {entry['homing_offset']} | "
              f"now at {position} | {verdict}")
    port.closePort()

    if all_match:
        print(f"{args.arm}: every motor matches limits.json")
    else:
        sys.exit(f"{args.arm}: at least one motor differs from limits.json")


if __name__ == "__main__":
    main()

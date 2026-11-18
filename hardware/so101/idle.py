"""Move the SO-101 follower to an idle pose and hold it with torque on.

    .venv/bin/python idle.py --arm arm1 record    # torque off, pose the arm by hand, Enter to save
    .venv/bin/python idle.py --arm arm1           # glide to the saved pose and hold it
    .venv/bin/python idle.py --arm arm1 release   # torque off, arm goes limp (support it!)

The pose is saved to <arm>/idle_pose.json. Targets are clamped to <arm>/limits.json.
"""

import argparse
import json
import sys
import threading
import time

import scservo_sdk as scs

import arms

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
arms.add_arm_argument(parser)
parser.add_argument("mode", nargs="?", default="go", choices=["go", "record", "release"])
args = parser.parse_args()

PORT, ARM_DIR = arms.resolve(args.arm)
BAUDRATE = 1_000_000
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_ID = 6
MOVE_SECONDS = 3.0  # how long the glide to the idle pose takes
STEP_SECONDS = 0.02  # 50 goal updates per second
DISPLAY_SECONDS = 0.1  # how often record shows the encoder positions

LIMITS_FILE = ARM_DIR / "limits.json"
POSE_FILE = ARM_DIR / "idle_pose.json"

# STS3215 control table addresses
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_TORQUE_LIMIT = 48
ADDR_PRESENT_POSITION = 56

GRIPPER_TORQUE_LIMIT = 500  # 50% of max, as lerobot does, so it can't burn out squeezing


def read(port, packet, motor_id, address, tries=5):
    """Read a 2-byte register, dropping stale bytes and retrying on a bad reply."""
    for _ in range(tries):
        port.ser.reset_input_buffer()
        try:
            value, comm, error = packet.read2ByteTxRx(port, motor_id, address)
        except IndexError:  # SDK crashes on a truncated reply
            comm = None
        if comm == scs.COMM_SUCCESS and not error:
            return value
        time.sleep(0.05)
    sys.exit(f"Motor {motor_id}: reading address {address} failed")


def try_write(port, packet, motor_id, address, value, size=2):
    """Write a register. Returns None on success, or a message saying why it failed."""
    write_fn = packet.write1ByteTxRx if size == 1 else packet.write2ByteTxRx
    comm = None
    error = None
    for _ in range(5):
        comm, error = write_fn(port, motor_id, address, value)
        if comm == scs.COMM_SUCCESS and not error:
            return None
        time.sleep(0.05)
    return f"Motor {motor_id}: writing address {address} failed (comm result {comm}, servo error byte {error})"


def write(port, packet, motor_id, address, value, size=2):
    failure = try_write(port, packet, motor_id, address, value, size)
    if failure is not None:
        sys.exit(failure)


def set_torque(port, packet, on):
    for motor_id in MOTOR_IDS:
        write(port, packet, motor_id, ADDR_TORQUE_ENABLE, int(on), size=1)


def torque_off_all(port, packet):
    """Turn torque off on every motor, carrying on past a motor that fails. Returns the failure messages."""
    failures = []
    for motor_id in MOTOR_IDS:
        failure = try_write(port, packet, motor_id, ADDR_TORQUE_ENABLE, 0, size=1)
        if failure is not None:
            failures.append(failure)
    return failures


def release(port, packet):
    failures = torque_off_all(port, packet)
    if len(failures) > 0:
        port.closePort()
        sys.exit("\n".join(failures) + "\nTorque is off on the other motors. Power-cycle the arm to free the ones listed.")


def present_positions(port, packet):
    return {m: read(port, packet, m, ADDR_PRESENT_POSITION) for m in MOTOR_IDS}


def send_goals(port, packet, goals):
    """Send every motor its goal in ONE packet, so all joints move in step."""
    sync = scs.GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, 2)
    for motor_id, goal in goals.items():
        sync.addParam(motor_id, [scs.SCS_LOBYTE(goal), scs.SCS_HIBYTE(goal)])
    sync.txPacket()


def clamp_to_limits(pose):
    limits = json.loads(LIMITS_FILE.read_text())
    return {
        m: max(limits[str(m)]["true"]["min"], min(limits[str(m)]["true"]["max"], pos))
        for m, pos in pose.items()
    }


def record(port, packet):
    release(port, packet)
    print("Torque off. Pose the arm by hand in its idle position, then press Enter.")

    # Enter is waited for on another thread so this one can keep showing the encoders.
    enter_pressed = threading.Event()

    def wait_for_enter():
        input()
        enter_pressed.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    while not enter_pressed.is_set():
        pose = present_positions(port, packet)
        parts = []
        for motor_id in MOTOR_IDS:
            parts.append("{}:{:4d}".format(motor_id, pose[motor_id]))
        print("\r" + "  ".join(parts), end="", flush=True)
        time.sleep(DISPLAY_SECONDS)
    print()

    pose = present_positions(port, packet)
    POSE_FILE.write_text(json.dumps({str(m): p for m, p in pose.items()}, indent=2))
    print(f"Saved idle pose to {POSE_FILE.name}: {pose}")


def go_idle(port, packet):
    if not POSE_FILE.exists():
        sys.exit(f"No idle pose yet. Run: .venv/bin/python idle.py --arm {args.arm} record")
    target = clamp_to_limits({int(m): p for m, p in json.loads(POSE_FILE.read_text()).items()})
    start = present_positions(port, packet)

    # The goal register may hold an old value (e.g. 0). Turning torque on
    # would make the joint snap there at full speed, so first make the goal
    # "stay where you are", THEN enable torque.
    send_goals(port, packet, start)
    write(port, packet, GRIPPER_ID, ADDR_TORQUE_LIMIT, GRIPPER_TORQUE_LIMIT)
    set_torque(port, packet, True)

    # Glide: step the goal from start to target in small increments, so the
    # motor never sees a big jump and moves gently instead of slamming.
    steps = int(MOVE_SECONDS / STEP_SECONDS)
    for i in range(1, steps + 1):
        fraction = i / steps
        goals = {m: round(start[m] + (target[m] - start[m]) * fraction) for m in MOTOR_IDS}
        send_goals(port, packet, goals)
        print(f"\rmoving... {fraction:4.0%}", end="", flush=True)
        time.sleep(STEP_SECONDS)

    time.sleep(0.5)  # let the joints settle
    print("\nHolding idle pose (torque stays on after this script exits).")
    for m, pos in present_positions(port, packet).items():
        print(f"  motor {m}: target {target[m]:4d}  now {pos:4d}  error {pos - target[m]:+d}")


def main():
    port = scs.PortHandler(PORT)
    packet = scs.PacketHandler(0)  # protocol 0 = Feetech STS series
    if not port.openPort() or not port.setBaudRate(BAUDRATE):
        sys.exit(f"Could not open {PORT}")

    if args.mode == "record":
        record(port, packet)
    elif args.mode == "release":
        release(port, packet)
        print("Torque off: the arm is limp.")
    else:
        go_idle(port, packet)
    port.closePort()


if __name__ == "__main__":
    main()

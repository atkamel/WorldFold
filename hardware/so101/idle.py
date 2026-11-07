"""Move the SO-101 follower to an idle pose and hold it with torque on.

    .venv/bin/python idle.py record    # torque off, pose the arm by hand, Enter to save
    .venv/bin/python idle.py           # glide to the saved pose and hold it
    .venv/bin/python idle.py release   # torque off, arm goes limp (support it!)

The pose is saved to idle_pose.json. Targets are clamped to limits.json.
"""

import json
import sys
import time
from pathlib import Path

import scservo_sdk as scs

PORT = "/dev/tty.usbmodem5B790153891"
BAUDRATE = 1_000_000
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_ID = 6
MOVE_SECONDS = 3.0  # how long the glide to the idle pose takes
STEP_SECONDS = 0.02  # 50 goal updates per second

HERE = Path(__file__).parent
LIMITS_FILE = HERE / "limits.json"
POSE_FILE = HERE / "idle_pose.json"

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


def write(port, packet, motor_id, address, value, size=2):
    write_fn = packet.write1ByteTxRx if size == 1 else packet.write2ByteTxRx
    for _ in range(5):
        comm, error = write_fn(port, motor_id, address, value)
        if comm == scs.COMM_SUCCESS and not error:
            return
        time.sleep(0.05)
    sys.exit(f"Motor {motor_id}: writing address {address} failed")


def set_torque(port, packet, on):
    for motor_id in MOTOR_IDS:
        write(port, packet, motor_id, ADDR_TORQUE_ENABLE, int(on), size=1)


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
    set_torque(port, packet, False)
    input("Torque off. Pose the arm by hand in its idle position, then press Enter.")
    pose = present_positions(port, packet)
    POSE_FILE.write_text(json.dumps({str(m): p for m, p in pose.items()}, indent=2))
    print(f"Saved idle pose to {POSE_FILE.name}: {pose}")


def go_idle(port, packet):
    if not POSE_FILE.exists():
        sys.exit("No idle pose yet. Run: .venv/bin/python idle.py record")
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

    mode = sys.argv[1] if len(sys.argv) > 1 else "go"
    if mode == "record":
        record(port, packet)
    elif mode == "release":
        set_torque(port, packet, False)
        print("Torque off: the arm is limp.")
    else:
        go_idle(port, packet)
    port.closePort()


if __name__ == "__main__":
    main()

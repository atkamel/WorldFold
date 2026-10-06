"""GPU thermal guard (2026-10-06): hold the pause flag that rollouts and training honour (imitation/thermal.py).

Polls nvidia-smi every --every seconds. Writes the flag at >= --hot °C, removes it at <= --cool °C (hysteresis),
prints one line per state change and a heartbeat with the max temperature since the last one. The ceiling the user
set is 94 °C; the default trip point leaves a 4 °C margin.
    .venv/Scripts/python.exe scripts/thermal_guard.py [--hot 90 --cool 84]
CPU temperature needs admin on Windows, so the CPU is guarded by a load cap instead (Isaac <= 2 processes x 4 envs).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imitation.thermal import flag_path  # noqa: E402


def read_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    temp, util, power = (x.strip() for x in out.splitlines()[0].split(","))
    return int(temp), int(util), float(power) if power not in ("[N/A]", "") else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", type=int, default=90)
    ap.add_argument("--cool", type=int, default=84)
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--heartbeat", type=float, default=600.0)
    args = ap.parse_args()
    flag = flag_path()
    flag.parent.mkdir(parents=True, exist_ok=True)
    hot = flag.exists()
    peak, last_beat = 0, time.monotonic()

    def say(msg):
        print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)

    say(f"guard on: pause at >= {args.hot} C, resume at <= {args.cool} C, flag {flag} (now {'set' if hot else 'clear'})")
    while True:
        try:
            temp, util, power = read_gpu()
        except Exception as e:           # a failed read must not leave a stale pause behind or kill the guard
            say(f"nvidia-smi read failed: {e}")
            time.sleep(args.every)
            continue
        peak = max(peak, temp)
        if not hot and temp >= args.hot:
            flag.write_text(f"{datetime.now().isoformat()} {temp} C\n")
            hot = True
            say(f"PAUSE gpu {temp} C ({util}%, {power:.0f} W)")
        elif hot and temp <= args.cool:
            flag.unlink(missing_ok=True)
            hot = False
            say(f"RESUME gpu {temp} C")
        if time.monotonic() - last_beat >= args.heartbeat:
            say(f"heartbeat gpu {temp} C, max {peak} C since last, util {util}%, {power:.0f} W, "
                f"{'paused' if hot else 'running'}")
            peak, last_beat = 0, time.monotonic()
        time.sleep(args.every)


if __name__ == "__main__":
    main()

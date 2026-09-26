"""Live view of the background pipeline runs (refreshes every few seconds).

    python -m imitation.viz.watch            # Ctrl+C to exit; the runs keep going

Long runs are started detached (they survive closing the editor or the chat) and write
only log files, so this is the window onto them: which queue step is running, the
current job's latest log lines, the processes involved, and GPU / CPU load.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

RUNS = Path("outputs/imitation/runs")
QUEUE_LOG = RUNS / "phase5b_seq.out"
JOB_LOGS = ["distill_v2.out", "harvest_v2.log", "iql_v3.log", "iql_v3.eval.log", "success_v2.log", "figures.log",
            "demo_priv.log", "demo_sensor.log"]


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, shell=isinstance(cmd, str)).stdout.strip()
    except Exception:
        return ""


def tail(path, n):
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def processes():
    out = sh(["powershell", "-NoProfile", "-c",
              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "ForEach-Object { $_.CommandLine }"])
    jobs = [" ".join(l.split(" -m ", 1)[1].split()[:3]) for l in out.splitlines()
            if " -m imitation." in l and "imitation.viz.watch" not in l]
    workers = sum(1 for l in out.splitlines() if "multiprocessing.spawn" in l)
    return list(dict.fromkeys(jobs)), workers      # the venv launcher and its child share a command line


def frame(n_lines):
    gpu = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
              "--format=csv,noheader"])
    cpu = sh(["powershell", "-NoProfile", "-c",
              "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"])
    jobs, workers = processes()
    live = max((RUNS / j for j in JOB_LOGS if (RUNS / j).exists()), key=lambda p: p.stat().st_mtime, default=None)
    lines = [f"WorldFold pipeline  |  {time.strftime('%H:%M:%S')}  |  Ctrl+C exits (runs keep going)", "",
             f"GPU  {gpu}  (util %, used, total, C)", f"CPU  {cpu}% average load  |  sim workers alive: {workers}",
             "", "running: " + (" | ".join(jobs) if jobs else "nothing (queue idle or finished)"), "",
             f"queue ({QUEUE_LOG}):"] + [f"  {l}" for l in tail(QUEUE_LOG, 6) or ["  (no output yet)"]]
    if live is not None:
        lines += ["", f"current job log ({live}):"] + [f"  {l[:150]}" for l in tail(live, n_lines)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--lines", type=int, default=14)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        text = frame(args.lines)
        if args.once:
            print(text)
            return
        os.system("cls" if os.name == "nt" else "clear")
        print(text, flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()

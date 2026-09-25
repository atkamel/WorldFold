"""One CPU slot shared across processes (roadmap M5c.3).

Simulation (rollouts, evaluation, harvest) is CPU-bound; training is GPU-bound. With
`IMITATION_CPU_SLOT=<lock file>` set, every `rollout()` call holds this slot for its
duration. Several queues can then run at once: one trains on the GPU while another
simulates, but their CPU phases never overlap, so the sim-worker cap (10 of 16 cores) still
holds. Without the variable it's a no-op.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

ENV = "IMITATION_CPU_SLOT"


@contextlib.contextmanager
def cpu_slot(poll=2.0):
    path = os.environ.get(ENV)
    if not path:
        yield 0.0
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    t0 = time.perf_counter()
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(poll)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield time.perf_counter() - t0          # seconds spent waiting for the slot
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()

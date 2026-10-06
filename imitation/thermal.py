"""Cooperative thermal pause (2026-10-06): keep the laptop under 94 °C during long runs.

`scripts/thermal_guard.py` watches the GPU temperature and holds a flag file while it is too hot. Rollouts and
training call `wait_while_hot()` at points where pausing is safe: nothing is suspended, the parent just stops
sending work, and idle Isaac workers keep ticking Kit (its hang detector aborts after 120 s without a tick).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

DEFAULT_FLAG = Path(__file__).resolve().parents[1] / "outputs" / ".thermal_pause"
_CHECK_EVERY_S = 1.0
_last_check, _last_hot = 0.0, False


def flag_path() -> Path:
    return Path(os.environ.get("WORLDFOLD_THERMAL_FLAG", DEFAULT_FLAG))


def hot(now: float | None = None) -> bool:
    """Whether the guard currently asks for a pause (the flag is re-read at most once a second)."""
    global _last_check, _last_hot
    now = time.monotonic() if now is None else now
    if now - _last_check >= _CHECK_EVERY_S:
        _last_check, _last_hot = now, flag_path().exists()
    return _last_hot


def wait_while_hot(log=print, poll_s: float = 2.0) -> float:
    """Block while the flag exists; returns the seconds paused (0 when not hot)."""
    global _last_check, _last_hot
    if not hot():
        return 0.0
    t0 = time.monotonic()
    log(f"thermal pause: {flag_path()} present, waiting")
    while flag_path().exists():
        time.sleep(poll_s)
    _last_check, _last_hot = time.monotonic(), False
    paused = time.monotonic() - t0
    log(f"thermal resume after {paused:.0f}s")
    return paused

"""imitation/thermal.py: the pause flag the GPU thermal guard holds (scripts/thermal_guard.py)."""

from __future__ import annotations

import threading

import imitation.thermal as thermal


def _fresh(monkeypatch, path):
    monkeypatch.setenv("WORLDFOLD_THERMAL_FLAG", str(path))
    monkeypatch.setattr(thermal, "_last_check", -1e9)
    monkeypatch.setattr(thermal, "_last_hot", False)


def test_flag_path_follows_env(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path / "f")
    assert thermal.flag_path() == tmp_path / "f"


def test_hot_reads_flag_at_most_once_a_second(monkeypatch, tmp_path):
    flag = tmp_path / "f"
    _fresh(monkeypatch, flag)
    assert not thermal.hot(now=100.0)
    flag.write_text("x")
    assert not thermal.hot(now=100.5)      # cached
    assert thermal.hot(now=101.1)


def test_wait_while_hot(monkeypatch, tmp_path):
    flag = tmp_path / "f"
    _fresh(monkeypatch, flag)
    assert thermal.wait_while_hot(log=lambda m: None) == 0.0
    flag.write_text("x")
    monkeypatch.setattr(thermal, "_last_check", -1e9)
    msgs = []
    threading.Timer(0.3, flag.unlink).start()
    paused = thermal.wait_while_hot(log=msgs.append, poll_s=0.05)
    assert paused > 0.2 and len(msgs) == 2 and not thermal.hot()

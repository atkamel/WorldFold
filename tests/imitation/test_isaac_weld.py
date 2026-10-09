"""Weld grasp on Isaac (Phase W, W1), via isaac/weld_check.py in its own process. Isaac venv only.

The weld needs the GPU pipeline (device cuda:0): on LeHome's CPU device the particle positions can only be written
through USD points, which PhysX ignores mid-simulation (pinned particles drifted 14-36 mm per substep). Since W3 the
weld is soft, calibrated to MuJoCo's: the held corner may lag its target as much as MuJoCo's does (TRACK_PARITY)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.isaac, pytest.mark.slow,
              pytest.mark.skipif(importlib.util.find_spec("isaacsim") is None, reason="needs the Isaac venv")]

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("weld")
    r = subprocess.run([sys.executable, "-u", "isaac/weld_check.py", "--device", "cuda:0", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads((out / "weld.json").read_text())


@pytest.mark.parametrize("name", ["engaged", "tracking_like_mujoco", "corner_rose_8cm", "held_on_neutral",
                                  "released_and_fell", "mask_respected", "cloth_rests_on_table"])
def test_weld_check(report, name):
    assert report["checks"][name], {k: v for k, v in report.items() if k != "phases"}

"""Isaac camera rig (Phase I, I1.3), via isaac/rig_check.py in its own process. Isaac venv only."""

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
    out = tmp_path_factory.mktemp("rig")
    r = subprocess.run([sys.executable, "-u", "isaac/rig_check.py", "--out", str(out)], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads((out / "rig.json").read_text())


def test_frames_have_the_rig_shapes(report):
    for name, f in report["frames"].items():
        assert f["shape"] == f["expected"] and f["dtype"] == "uint8", name
        assert f["std"] > 1.0, f"{name} looks blank"


def test_wrist_cameras_sit_at_the_mjcf_mount(report):
    for prefix, p in report["wrist_pose"].items():
        assert p["pos_err_mm"] < 1.0, (prefix, p)
        assert p["rot_err_deg"] < 0.5, (prefix, p)


def test_state_stays_finite_with_cameras(report):
    assert report["state_finite"]

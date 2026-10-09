"""Vectorised Isaac env (milestone V), via isaac/vec_check.py in its own process: B = 3 copies of the weld-profile
scene in one Isaac process. Isaac venv only."""

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
    out = tmp_path_factory.mktemp("vec")
    r = subprocess.run([sys.executable, "-u", "isaac/vec_check.py", "--n", "3", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads((out / "vec.json").read_text())


@pytest.mark.parametrize("name", ["copies_separate", "local_frame_matches", "dr_per_copy", "dr_deterministic",
                                  "reset_isolated_dynamics", "reset_isolated", "weld_per_copy"])
def test_vec_check(report, name):
    assert report["checks"][name], {k: v for k, v in report.items() if k != "dr"}

"""Isaac scripted expert as the imitation teacher (Phase I, I2.1), via isaac/expert_smoke.py in its own process
(one Isaac env per process). Isaac venv only. Success itself isn't asserted: the friction grasp is used as built
and tuned in a later milestone (roadmap IG)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.isaac, pytest.mark.slow,
              pytest.mark.skipif(importlib.util.find_spec("isaacsim") is None, reason="needs the Isaac venv")]

ROOT = Path(__file__).resolve().parents[2]


def test_teacher_drives_a_full_fold_attempt():
    r = subprocess.run([sys.executable, "-u", "isaac/expert_smoke.py", "--seeds", "100002"], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    trace = r.stdout
    for phase in ("'close'", "'carry'", "'release'"):        # both arms got through pinch, carry and release
        assert trace.count(phase) >= 1, f"no {phase} in the phase trace"
    ep = json.loads(next(l for l in trace.splitlines() if l.startswith("EPISODE"))[len("EPISODE "):])
    assert ep["steps"] > 100 and len(ep["move_distance"]) == 2

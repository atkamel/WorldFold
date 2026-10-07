"""check_resync's backend-independent episode loop and the Isaac runner's resume (IG.3), on a fake env: no sim."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from imitation import check_resync as cr


class FakeExpert:
    def __init__(self, log):
        self.log = log

    def resync(self):
        self.log.append("resync")
        return {"phase": "carry"}


class FakeTeacher:
    def __init__(self):
        self.log = []
        self.expert = FakeExpert(self.log)

    def reset(self):
        self.log.append("reset")

    def act(self):
        self.log.append("act")
        return np.zeros(12)


class FakeEnv:
    """Succeeds at step `end`; records every action."""
    def __init__(self, end=30):
        self.unwrapped = SimpleNamespace(max_episode_steps=60)
        self.end, self.t, self.actions, self.seeds = end, 0, [], []

    def reset(self, seed=None):
        self.t = 0
        self.seeds.append(seed)

    def step(self, a):
        self.t += 1
        self.actions.append(np.asarray(a))
        done = self.t >= self.end
        return None, 0.0, done, False, {"success": done, "termination_reason": "success" if done else None}


def test_episode_resyncs_once_after_the_handover():
    for mode, resyncs_at_k in (("resync_only", False), ("noisy", True), ("random", True)):
        env, teacher = FakeEnv(), FakeTeacher()
        row = cr.episode(env, teacher, (3, mode, 4), switch_t=(5, 10))
        assert teacher.log.count("resync") == 1 and row["phases"] == {"phase": "carry"}
        assert row["success"] and row["reason"] == "success" and 5 <= row["t_switch"] < 10 and env.seeds == [3]
        # the expert acts on every step except the k handed-over ones (resync_only hands over none)
        assert teacher.log.count("act") == len(env.actions) - (4 if resyncs_at_k else 0) + (4 if mode == "noisy" else 0)


def test_same_seed_draws_the_same_handover_step():
    rows = [cr.episode(FakeEnv(), FakeTeacher(), (7, "noisy", 4), switch_t=(5, 100)) for _ in range(2)]
    assert rows[0]["t_switch"] == rows[1]["t_switch"]


def test_isaac_run_resumes_from_its_rows_file(tmp_path):
    path = tmp_path / "cr.json.rows.jsonl"
    made = []

    def make():
        made.append(1)
        return FakeEnv(), FakeTeacher()

    # a first, cut-off run: only resync_only seeds 0..1 finished
    rows = cr.run_isaac("isaac", 2, 3, path, make=make, switch_t=(5, 10), log=lambda *_: None)
    assert len(rows) == 6 and len(made) == 1
    keep = [json.loads(line) for line in path.read_text().splitlines()][:2]
    path.write_text("".join(json.dumps(r) + "\n" for r in keep))
    rows = cr.run_isaac("isaac", 2, 3, path, make=make, switch_t=(5, 10), log=lambda *_: None)
    assert len(rows) == 6 and len(made) == 2
    assert {(r["seed"], r["mode"]) for r in rows} == {(s, m) for s in (0, 1) for m in cr.MODES}
    # fully done: nothing is built or run
    cr.run_isaac("isaac", 2, 3, path, make=make, switch_t=(5, 10), log=lambda *_: None)
    assert len(made) == 2
    summary = cr.summarize_rows(rows, 2)
    assert all(v == {"n": 2, "success": 2, "reasons": {"success": 2}} for v in summary.values())


def test_isaac_cli_needs_out():
    code = "import sys\nsys.argv = ['x', '--backend', 'isaac']\nfrom imitation.check_resync import main\nmain()"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode != 0 and "--out" in out.stderr

"""Offline tests for the board: python teacher/harness/test_wf.py"""
import contextlib
import glob
import importlib
import io
import os
import sys
import tempfile
import time

with tempfile.TemporaryDirectory() as root:
    os.environ["WF_HARNESS_ROOT"] = root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    wf = importlib.import_module("wf")

    def run(*argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wf.main(list(argv))
        return buf.getvalue()

    def fails(*argv):
        try:
            run(*argv)
        except SystemExit as e:
            return bool(e.code)
        return False

    run("team", "Verifier A", "--owner", "roy", "--approach", "calibrate neatness")
    assert os.path.exists(f"{root}/teams/verifier-a.md")
    assert fails("team", "verifier a", "--owner", "x", "--approach", "y"), "duplicate team must fail"
    assert fails("post", "nobody", "question", "hi"), "unknown team must fail"
    assert fails("post", "verifier-a", "rumor", "hi"), "unknown kind must fail"
    assert fails("post", "verifier-a", "finding", "no proof"), "a finding without evidence must fail"
    run("post", "verifier-a", "finding", "pass line at 15", "--evidence", "results/x.json", "--to", "plumbing")
    time.sleep(1.1)
    run("post", "verifier-a", "question", "same title twice")
    notes = sorted(glob.glob(f"{root}/posts/*.md"))
    assert len(notes) == 2
    head = wf.read_fields(notes[0])
    assert head["kind"] == "finding" and head["verified"] == "no" and head["to"] == "plumbing"
    out = run("status")
    assert "verifier-a" in out and "calibrate neatness" in out and "-> plumbing" in out and "same title twice" in out
    assert "pass line" not in run("status", "-n", "1")
print("ok")

import json
import subprocess

import pytest

from imitation import verify

PHASE_I = ["I0.1", "I0.2", "I0.3", "I0.4", "I1.1", "I1.2", "I1.3", "I2.1", "I2.2", "I3.1", "I3.2", "I3.3"]


def test_wilson_matches_results_md():
    lo, hi = verify.wilson(97, 100)
    assert round(lo, 1) == 91.5 and round(hi, 1) == 99.0


def test_wilson_zero_successes():
    lo, hi = verify.wilson(0, 10)
    assert lo == 0 and 0 < hi < 50


def test_eval_counts(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"results": {"id": {"n": 50, "success_rate": 0.96},
                                         "ood": {"n": 20, "success_rate": 0.5}}}))
    assert verify.eval_counts(p) == {"id": (48, 50), "ood": (10, 20)}


def test_loss_dropped(tmp_path):
    good, bad = tmp_path / "g.json", tmp_path / "b.json"
    good.write_text(json.dumps([{"step": 1, "val_loss": 1.0}, {"step": 2, "val_loss": 0.4}]))
    bad.write_text(json.dumps({"history": [{"step": 1, "val_loss": 1.0}, {"step": 2, "val_loss": 0.9}]}))
    assert verify.loss_dropped(good) is True
    assert verify.loss_dropped(bad) is False
    assert verify.loss_dropped(tmp_path / "missing.json") is False


def test_results_md_has_and_isaac_available(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "results.md").write_text("row I0.1 here")
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    assert verify.results_md_has("I0.1") and not verify.results_md_has("I9.9")
    assert isinstance(verify.isaac_available(), bool)


def _git_repo(tmp_path):
    def run(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    run("config", "commit.gpgsign", "false")
    return run


def test_git_tracked(tmp_path, monkeypatch):
    run = _git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    run("add", "a.txt")
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    assert verify.git_tracked("a.txt") and not verify.git_tracked("b.txt")


def _i02_tree(tmp_path, hashed=True):
    run = _git_repo(tmp_path)
    (tmp_path / "isaac").mkdir()
    (tmp_path / "isaac" / "INSTALL_REVIEW.md").write_text("verdict: SAFE")
    line = "foo==1.0 \\\n    --hash=sha256:abc\n    # via x\n" if hashed else "foo==1.0\n"
    for n in ("lehome", "isaaclab", "torch-cu128"):
        (tmp_path / "isaac" / f"requirements-{n}-windows.lock").write_text(line)
    run("add", "-A")
    return run


def test_i02_pass_and_fail(tmp_path, monkeypatch):
    _i02_tree(tmp_path)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    r = verify.CHECKS["I0.2"][1]()
    assert r.status == "PASS", r.evidence
    (tmp_path / "isaac" / "requirements-torch-cu128-windows.lock").write_text("foo==1.0\n")
    r = verify.CHECKS["I0.2"][1]()
    assert r.status == "FAIL" and any("torch-cu128" in e for e in r.evidence)


def test_missing_artifact_names_path(tmp_path, monkeypatch):
    _git_repo(tmp_path)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    r = verify.CHECKS["I0.2"][1]()
    assert r.status == "FAIL" and any("INSTALL_REVIEW.md" in e for e in r.evidence)
    r = verify.CHECKS["I0.1"][1]()
    assert r.status == "FAIL" and any("expert_benchmark_i0_1.json" in e for e in r.evidence)


def test_i01_benchmark_equal(tmp_path, monkeypatch):
    d = tmp_path / "outputs" / "imitation"
    d.mkdir(parents=True)
    (d / "expert_benchmark_i0_1.json").write_text('{"a": 1,  "b": [1,2]}')
    (d / "expert_benchmark_m1_5.json").write_text('{"b": [1, 2], "a": 1}')
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "results.md").write_text("I0.1")
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "_is_ancestor", lambda c: True)
    assert verify.CHECKS["I0.1"][1]().status == "PASS"
    (d / "expert_benchmark_m1_5.json").write_text('{"a": 2}')
    assert verify.CHECKS["I0.1"][1]().status == "FAIL"


def test_list_has_all_ids(capsys):
    assert verify.main(["--list"]) == 0
    out = capsys.readouterr().out
    for i in PHASE_I:
        assert i in out
        assert i in verify.CHECKS


def test_stubs_skip_and_main_writes_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    assert verify.main(["I1.1"]) == 0
    assert "SKIP I1.1" in capsys.readouterr().out
    rec = json.loads((tmp_path / "outputs/imitation/isaac/verify/I1.1.json").read_text())
    assert rec["milestone"] == "I1.1" and rec["status"] == "SKIP" and "evidence" in rec and "time" in rec


def test_main_exit_code_on_fail(tmp_path, monkeypatch):
    _git_repo(tmp_path)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    assert verify.main(["I0.2"]) == 1
    assert verify.main(["nope"]) == 1

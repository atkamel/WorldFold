"""Phase I, I3.1: an end-to-end micro chain of the imitation pipeline on the Isaac backend (the Isaac counterpart
of roadmap M1.6), kept as a regression guard. Isaac venv only.

The real CLIs run as subprocesses into a pytest tmp dir, once (module-scoped fixture); the tests assert on the
artifacts each stage left behind:

    collect (mixed, 3 cams) -> train chunk_mlp / diffusion -> evaluate both -> DAgger round (takeover labels)
        -> vision BC -> success detector train + agree

It is tiny on purpose (3 demos, 2-episode evals, a few hundred train steps): it checks that the stages hand
each other valid artifacts, not that the policies are any good. Nothing about success rates is asserted.
"""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.isaac, pytest.mark.slow,
              pytest.mark.skipif(importlib.util.find_spec("isaacsim") is None, reason="needs the Isaac venv")]

ROOT = Path(__file__).resolve().parents[2]
VERSION = "chain_v1"
CAMERAS = {"main": 128, "left_wrist_cam": 64, "right_wrist_cam": 64}
N_DEMOS = 3
DAGGER_OUT = "dagger_chain"                       # dagger names its version <dataset>_<out name>_r<round>
DAGGER_VERSION = f"{VERSION}_{DAGGER_OUT}_r1"


class Chain:
    """Stage results of the one chain run. A stage that failed (or whose predecessor failed) is reported by
    `ok(stage)` with the tail of its output."""

    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.datasets = self.tmp / "datasets"
        self.runs = self.tmp / "runs"
        self.results = {}
        self.skipped = {}                         # stage -> reason it was not run (not a failure)
        self.failed = None

    def run(self, stage, module, args, timeout):
        if self.failed:
            self.results[stage] = None
            return
        cmd = [sys.executable, "-u", "-m", module, *map(str, args)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        self.results[stage] = r
        if r.returncode != 0:
            self.failed = stage

    def ok(self, stage):
        if stage in self.skipped:
            pytest.skip(self.skipped[stage])
        r = self.results.get(stage)
        if r is None:
            pytest.fail(f"stage '{stage}' did not run: upstream stage '{self.failed}' failed")
        assert r.returncode == 0, (f"stage '{stage}' exited {r.returncode}\n--- stdout ---\n{r.stdout[-3000:]}"
                                   f"\n--- stderr ---\n{r.stderr[-3000:]}")
        return r


def _val_seed_base(n=N_DEMOS):
    """A seed base whose n seeds include a validation seed. The detector validates on val-seed frames and
    cannot run with none; with 3 seeds that happens by chance only ~1 time in 4, so pick them deliberately."""
    from imitation.data.dataset import is_val_seed
    first = next(s for s in range(1000) if is_val_seed(s))
    return max(0, first - 1) if first + 1 >= n else 0


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    c = Chain(tmp_path_factory.mktemp("isaac_chain"))
    root = c.datasets
    cams = ",".join(f"{k}={v}" for k, v in CAMERAS.items())

    c.run("collect", "imitation.data.collect",
          ["--backend", "isaac", "--episodes", N_DEMOS, "--workers", 2, "--recovery-fraction", 0.0, "--render",
           "--cameras", cams, "--version", VERSION, "--root", root, "--mixed", "--seed-base", _val_seed_base()],
          timeout=3600)

    for kind in ("chunk_mlp", "diffusion"):
        c.run(f"train_{kind}", "imitation.train",
              ["--policy", kind, "--dataset", VERSION, "--root", root, "--run", c.runs / kind, "--steps", 200,
               "--batch", 32, "--allow-failures"], timeout=1800)
        c.run(f"eval_{kind}", "imitation.evaluate",
              ["--ckpt", c.runs / kind / "final.pt", "--backend", "isaac", "--sets", "id_easy", "--n", 2,
               "--workers", 2, "--out", c.runs / kind / "eval.json"], timeout=3600)

    c.run("dagger", "imitation.dagger",
          ["--backend", "isaac", "--init", c.runs / "diffusion" / "final.pt", "--dataset", VERSION, "--root", root,
           "--out", c.runs / DAGGER_OUT, "--rounds", 1, "--episodes", 2, "--train-steps", 200, "--eval-n", 2,
           "--eval-sets", "id_easy", "--score-sets", "id_easy", "--workers", 2, "--labels", "takeover",
           "--takeover-p", 0.5, "--allow-failures"], timeout=5400)

    c.run("train_vision", "imitation.train",
          ["--policy", "vision", "--dataset", VERSION, "--root", root, "--run", c.runs / "vision", "--steps", 100,
           "--batch", 16, "--allow-failures"], timeout=1800)

    # the detector needs positive ("folded") training frames for its class-balanced sampler; the micro dataset
    # may hold none (the Isaac expert succeeds ~1 in 3), which is a property of the data, not a pipeline fault
    if not c.failed and c.results.get("collect") is not None:
        c.skipped.update(_detector_skips(c))
    if not c.skipped.get("detector_train"):
        det = c.runs / "detector"
        c.run("detector_train", "imitation.vision.success",
              ["train", "--versions", VERSION, "--root", root, "--out", det, "--steps", 100], timeout=1800)
        c.run("detector_agree", "imitation.vision.success",
              ["agree", "--ckpt", det / "detector.pt", "--versions", VERSION, "--root", root], timeout=1800)
    return c


def _detector_skips(c):
    from imitation.data.dataset import is_val_seed
    from imitation.data.schema import load_dataset
    from imitation.vision.success import folded_labels
    _, eps = load_dataset(c.datasets, VERSION)
    train = [e for e in eps if not is_val_seed(e.meta["seed"])]
    if not any(is_val_seed(e.meta["seed"]) for e in eps):
        reason = "no validation-seed episode in the micro dataset"
    elif not sum(int(folded_labels(e).sum()) for e in train):
        reason = "no folded frames among the micro dataset's training episodes (all demos failed)"
    else:
        return {}
    return {"detector_train": reason, "detector_agree": reason}


def _check_dataset(root, version, expect_images):
    from imitation.data.schema import load_dataset, load_manifest
    from imitation.spec import ACTION_DIM, OBS_DIM
    assert (Path(root) / version / "manifest.json").exists()
    manifest = load_manifest(root, version)
    _, eps = load_dataset(root, version, verify=True, images=expect_images)   # hashes re-verify
    assert len(eps) == manifest["n_episodes"] == len(manifest["episodes"])
    for ep in eps:
        assert ep.obs.shape[1] == OBS_DIM and ep.actions.shape[1] == ACTION_DIM
        assert ep.steps > 0 and np.isfinite(ep.obs).all() and np.isfinite(ep.actions).all()
        assert np.abs(ep.actions).max() <= 1.0 + 1e-6
        if expect_images:
            for cam, size in CAMERAS.items():
                img = ep.images[cam]
                assert img.dtype == np.uint8 and img.shape == (ep.steps, 3, size, size), (cam, img.shape)
    return manifest, eps


def _check_run(run_dir, steps):
    assert (run_dir / "final.pt").exists()
    info = json.loads((run_dir / "run.json").read_text())
    assert info["train"]["steps"] == steps and info["checkpoint"]["sha256"]
    assert info["history"], "no loss history"
    for h in info["history"]:
        assert math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]), h
    return info


def test_collect_wrote_a_valid_three_camera_dataset(chain):
    chain.ok("collect")
    manifest, eps = _check_dataset(chain.datasets, VERSION, expect_images=True)
    assert manifest["n_episodes"] == N_DEMOS and manifest["config"]["backend"] == "isaac"
    assert {e["source"] for e in manifest["episodes"]} == {"expert"}
    assert not (chain.datasets / f"{VERSION}_failures" / "manifest.json").exists()    # --mixed: one version


@pytest.mark.parametrize("kind", ["chunk_mlp", "diffusion"])
def test_state_policy_trains_and_evaluates(chain, kind):
    chain.ok(f"train_{kind}")
    info = _check_run(chain.runs / kind, 200)
    assert info["policy"]["kind"] == kind and info["dataset"]["version"] == VERSION
    chain.ok(f"eval_{kind}")
    report = json.loads((chain.runs / kind / "eval.json").read_text())
    assert report["results"]["id_easy"]["n"] == 2


def test_dagger_round_wrote_a_new_version_with_takeover_labels(chain):
    chain.ok("dagger")
    out = chain.runs / DAGGER_OUT
    history = json.loads((out / "history.json").read_text())
    assert [h["round"] for h in history] == [0, 1]
    assert history[0]["eval"]["id_easy"]["n"] == 2 and history[1]["eval"]["id_easy"]["n"] == 2
    assert history[1]["dataset"] == DAGGER_VERSION
    # the round's checkpoint is written whether or not it was kept (it only beats round 0 sometimes)
    assert (out / "round_1" / "final.pt").exists() and (out / "round_1" / "run.json").exists()
    _check_run(out / "round_1", 200)

    manifest, eps = _check_dataset(chain.datasets, DAGGER_VERSION, expect_images=False)
    assert manifest["parent"] == VERSION
    new = [(e, ep) for e, ep in zip(manifest["episodes"], eps) if e["file"].startswith(f"{DAGGER_VERSION}/")]
    assert len(new) == 2 and len(manifest["episodes"]) == N_DEMOS + 2        # parent's demos + this round's
    assert all(e["source"] == "dagger" for e, _ in new)
    n_labels = 0
    for e, ep in new:
        assert e["n_labels"] == len(ep.label_steps)
        if len(ep.labels):
            from imitation.spec import ACTION_DIM
            assert ep.labels.ndim == 3 and ep.labels.shape[2] == ACTION_DIM, ep.labels.shape
            assert ep.labels.shape[0] == len(ep.label_steps) >= 1
            assert np.isfinite(ep.labels).all() and np.abs(ep.labels).max() <= 1.0 + 1e-6
        n_labels += len(ep.label_steps)
    assert history[1]["n_labels"] == n_labels
    # n_labels == 0 is tolerated: with 2 rollouts the expert may never take over by chance; the checkpoint and
    # the frozen version above are then the whole evidence that the round ran.


def test_vision_bc_trains_on_the_recorded_images(chain):
    chain.ok("train_vision")
    info = _check_run(chain.runs / "vision", 100)
    assert info["policy"]["kind"] == "vision"
    cams = {name: size for name, size in info["policy"]["config"]["cameras"]}
    assert cams == CAMERAS


def test_success_detector_trains_and_reports_agreement(chain):
    chain.ok("detector_train")
    det = chain.runs / "detector" / "detector.pt"
    assert det.exists()
    chain.ok("detector_agree")
    res = json.loads((chain.runs / "detector" / "agreement.json").read_text())
    assert res["n"] == N_DEMOS and 0 <= res["agree"] <= res["n"] and 0.0 <= res["rate"] <= 1.0
    assert len(res["wilson"]) == 2 and all(math.isfinite(x) for x in res["wilson"])

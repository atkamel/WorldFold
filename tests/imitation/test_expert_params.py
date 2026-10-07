"""WORLDFOLD_EXPERT_PARAMS sets IsaacArmExpert class attributes in every process that imports isaac.fold_expert."""

import os
import subprocess
import sys

import pytest

pytest.importorskip("isaac.pinch")


def test_env_params_override_class_attributes():
    code = "from isaac.fold_expert import IsaacArmExpert as E; print(E.SETTLE_LIFT, E.REGRASP, E.CARRY_SPEED)"
    env = dict(os.environ, WORLDFOLD_EXPERT_PARAMS="SETTLE_LIFT=1,REGRASP=2,CARRY_SPEED=0.5")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.stdout.split() == ["1", "2", "0.5"], out.stderr
    env["WORLDFOLD_EXPERT_PARAMS"] = "NOPE=1"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).returncode != 0

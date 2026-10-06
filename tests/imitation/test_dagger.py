import json

import numpy as np
import torch

from imitation.dagger import clamp_fraction, gain_in_se, resume_state


def _res(p, n):
    return {"id_easy": {"success_rate": p, "n": n}, "recovery": {"success_rate": p, "n": n}}


def test_gain_is_measured_in_standard_errors():
    assert gain_in_se(_res(0.80, 200), _res(0.70, 200)) > 2.0
    assert abs(gain_in_se(_res(0.72, 48), _res(0.70, 48))) < 1.0     # n=48 noise, not a gain


def test_resume_restores_best_checkpoint_dataset_and_next_round(tmp_path):
    hist = [{"round": 0, "checkpoint": "a.pt", "dataset": "v1", "eval": _res(0.5, 200), "kept": True},
            {"round": 1, "checkpoint": "b.pt", "dataset": "v1_d_r1", "eval": _res(0.4, 200), "kept": False}]
    (tmp_path / "history.json").write_text(json.dumps(hist))
    best_ckpt, best, dataset, next_round, history = resume_state(tmp_path)
    assert (str(best_ckpt), dataset, next_round, len(history)) == ("a.pt", "v1_d_r1", 2, 2)
    assert best == hist[0]["eval"]


def test_clamp_fraction_counts_normalized_obs_at_the_clip():
    mean, std = torch.zeros(3), torch.ones(3)
    X = torch.tensor([[[0.0, 20.0, 0.0]], [[0.0, 0.0, 0.0]]])
    assert clamp_fraction(X, mean, std) == np.float32(1 / 6)

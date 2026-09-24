import numpy as np

from imitation.evaluate import failure_code, save_version_name, summarize
from tests.imitation.test_data_and_policies import make_episode


def _failed(perturb, grasped_ever=(True, True), reason="truncated"):
    ep = make_episode(0)
    ep.meta.update(success=False, termination_reason=reason, perturb=perturb, final_fold_score=0.3,
                   final_move_distance=[0.2, 0.2])
    ep.grasped[:, 0], ep.grasped[:, 1] = grasped_ever
    return ep


def test_perturbed_failures_get_a_mechanistic_code():
    assert failure_code(_failed([10, 5], grasped_ever=(False, True))) == "G1"
    assert failure_code(_failed([10, 5], reason="cloth_dragged")) == "M1"
    assert failure_code(_failed(None, grasped_ever=(False, True))) == "G1"


def test_summary_counts_perturbed_failures_separately():
    ok = make_episode(1)
    ok.meta.update(final_fold_score=1.0, perturb=[3, 3])
    s = summarize([ok, _failed([10, 5]), _failed(None)])
    assert s["perturbed_failures"] == 1 and "R1" not in s["failure_codes"]


def test_saved_eval_versions_do_not_collide():
    a = save_version_name("id_easy", "runs/a/final.pt", 200)
    assert a != save_version_name("id_easy", "runs/b/final.pt", 200)
    assert a != save_version_name("id_easy", "runs/a/final.pt", 100)

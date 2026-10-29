import numpy as np

from imitation.data.dataset import Normalizer
from imitation.viz.report_figures import cams_panel, heatmap, layout_json, trace
from tests.imitation.test_data_and_policies import make_episode


def test_layout_covers_all_139_dims_and_matches_the_sensor_subset():
    lay = layout_json()
    assert sum(b["dims"] for b in lay["blocks"]) == 139
    starts = [b["start"] for b in lay["blocks"]]
    assert starts == sorted(starts) and lay["blocks"][-1]["end"] == 138
    sensor = sum(b["dims"] for b in lay["blocks"] if b["policy_sensor"])
    assert sensor == 48


def test_figures_render_from_a_tiny_episode(tmp_path):
    ep = make_episode(0, T=12)
    ep.grasped[4:9, 0] = True
    ep.images = {"main": np.zeros((12, 3, 16, 16), np.uint8), "left_wrist_cam": np.zeros((12, 3, 8, 8), np.uint8)}
    t = trace(ep, "x")
    assert len(t["fold_score"]) == 12 and len(t["actions"]) == 12
    heatmap(ep, Normalizer.fit([ep]), tmp_path / "h.png")
    info = cams_panel(ep, tmp_path / "c.png", size=32)
    assert (tmp_path / "h.png").exists() and info["moments"][1]["step"] == 4


def test_gpu_busy_counts_only_samples_inside_the_window(tmp_path):
    from imitation.viz.gpu_busy import busy, read_samples
    p = tmp_path / "gpu.csv"
    p.write_text("epoch,util_pct,mem_mib\n0,90,1\n5,0,1\n10,80,1\n15,,\n20,0,1\n25,99,1\n")
    s = read_samples(p)
    assert s.shape == (5, 2)                                  # header and the failed read are skipped
    r = busy(s, 5, 20)
    assert r["n_samples"] == 3 and r["mean_util_pct"] == round(80 / 3, 1) and r["busy_share"] == round(1 / 3, 3)

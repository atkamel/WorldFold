"""The held-corner summary, the hardware trajectory export and the cloth-pose reset knobs, without Isaac."""

import numpy as np

from isaac.tools.export_replay import trajectory
from isaac.tools.held_summary import carry_drops, fisher, held, summarise, wilson


def _row(**kw):
    return {"reason": "settled", "mean": 0.01, "edge_max": 0.02, "gate": {"valid": True}, "drop_at": {}, **kw}


def test_held_means_no_corner_left_the_jaws_in_the_carry():
    assert held(_row())
    assert not held(_row(drop_at={"right_": {"phase": "carry"}}))
    assert held(_row(drop_at={"left_": {"phase": "place"}}))          # a place-phase flag is a drag at the goal
    assert not held({"seed": 1, "no_window": True})
    assert not held(_row(reason="unstable"))
    assert carry_drops(_row(drop_at={"right_": {"phase": "carry"}, "left_": {"phase": "place"}})) == ["right_"]
    old = {"reason": "settled", "dropped": ["left_"]}                  # no drop_at: logged before phases were
    assert carry_drops(old) == ["left_"] and not held(old)


def test_a_late_flag_with_the_corner_above_the_site_is_held_only_in_the_lenient_count():
    up = _row(drop_at={"right_": {"phase": "carry", "wp": [12, 13], "dz_cm": 4.1}})
    fell = _row(drop_at={"right_": {"phase": "carry", "wp": [12, 13], "dz_cm": -9.0}})
    early = _row(drop_at={"right_": {"phase": "carry", "wp": [5, 13], "dz_cm": 4.1}})
    assert not held(up) and held(up, lenient=True)
    assert not held(fell, lenient=True) and not held(early, lenient=True)


def test_summary_counts_held_and_the_bars_separately():
    rows = [_row(), _row(drop_at={"right_": {"phase": "carry"}}), _row(mean=0.03), {"no_window": True},
            _row(gate={"valid": False})]
    s = summarise(rows)
    assert (s["n"], s["held"], s["drop_right_"], s["drop_left_"], s["no_window"]) == (5, 3, 1, 0, 1)
    assert s["working"] == 2 and s["held_working"] == 1 and s["strict"] == 0


def test_wilson_and_fisher_match_known_values():
    lo, hi = wilson(21, 64)
    assert abs(lo - 22.6) < 0.05 and abs(hi - 45.0) < 0.05
    assert abs(fisher(30, 64, 21, 64) - 0.1483) < 1e-3
    assert fisher(16, 32, 32, 64) == 1.0


def test_trajectory_targets_are_measured_joints_plus_the_clipped_delta():
    T = 3
    obs = np.zeros((T, 139), np.float32)
    obs[:, 0:5] = np.arange(T)[:, None] * 0.1           # left joints
    obs[:, 25:30] = 1.0                                 # right joints
    obs[:, 10], obs[:, 35] = [1.0, 1.0, 0.05], 0.5      # jaw targets
    final = obs[-1].copy()
    final[10] = -0.1
    final[11:14] = [-0.22, 0.0, 0.52]                   # left site at the end
    act = np.zeros((T, 12), np.float32)
    act[:, 0], act[:, 6] = 2.0, -0.5                    # left pan over the clip, right pan half speed
    tr = trajectory(obs, act, final, 0.05, {"left": (-0.22, -0.3, 0.48), "right": (0.22, -0.3, 0.48)}, 0.42)
    assert tr["left"]["target"].shape == (T, 6) and tr["left"]["q"].shape == (T + 1, 6)
    assert np.allclose(tr["left"]["target"][:, 0], obs[:, 0] + 0.05)
    assert np.allclose(tr["right"]["target"][:, 0], 1.0 - 0.025)
    assert np.allclose(tr["left"]["target"][:, 5], [1.0, 0.05, -0.1])          # the jaw target each step produced
    assert np.allclose(tr["left"]["tip"][-1], [0.0, 0.3, 0.1])


def test_cloth_yaw_adds_a_third_pose_entry_only_when_set():
    from cloth_fold_rl.quarter_fold_env import QuarterFoldEnv

    def pose(yaw, seed=5):
        env = object.__new__(QuarterFoldEnv)
        env.cloth_jitter, env.cloth_yaw, env._rng = 0.01, yaw, None

        class Stop(Exception):
            pass

        class Env:
            def reset(self, seed=None, options=None):
                raise Stop(options)
        env.env = Env()
        try:
            QuarterFoldEnv.reset(env, seed=seed)
        except Stop as e:
            return np.asarray(e.args[0]["cloth_pose"])

    plain, yawed = pose(0.0), pose(15.0)
    assert plain.shape == (2,) and yawed.shape == (3,)
    assert np.allclose(plain, yawed[:2])                 # the xy draw is unchanged by the yaw draw after it
    assert abs(yawed[2]) <= 15.0 and yawed[2] != 0.0

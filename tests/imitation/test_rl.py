import numpy as np

from imitation.rl.transitions import SUCCESS_REWARD, build_transitions, step_rewards
from tests.imitation.test_data_and_policies import A, D, make_episode


def test_macro_transitions_cover_the_episode_and_only_terminals_stop_bootstrap():
    ep = make_episode(0, T=20)                       # terminated success at step 19
    tr = build_transitions([ep], obs_horizon=2, macro=8)
    assert tr["obs"].shape == (3, 2, D) and tr["act"].shape == (3, 8, A)
    assert list(tr["valid"].sum(1)) == [8, 8, 4]
    assert list(tr["done"]) == [0, 0, 1]
    np.testing.assert_array_equal(tr["act"][1], ep.actions[8:16])
    np.testing.assert_array_equal(tr["next_obs"][-1, -1], ep.final_obs)
    assert tr["rew"][-1] > SUCCESS_REWARD * 0.9      # the sparse success label lands in the last interval
    trunc = make_episode(1, T=20)
    trunc.terminated[-1], trunc.truncated[-1], trunc.discount[-1] = False, True, 1.0
    trunc.meta["success"] = False
    assert build_transitions([trunc])["done"].sum() == 0


def test_reward_clips_spikes_and_unstable_episodes_are_dropped():
    ep = make_episode(0, T=5)
    ep.fold_score[:] = [0.0, 0.5, 0.5, 0.0, 0.1]      # toggle spikes
    r = step_rewards(ep)
    assert np.abs(r[:-1]).max() <= 0.05 + 1e-6
    bad = make_episode(1, T=10)
    bad.meta["termination_reason"] = "unstable"
    assert len(build_transitions([ep, bad], macro=8)["obs"]) == 1


def test_unstable_episodes_are_tagged_and_long_episodes_subsampled():
    from imitation.rl.transitions import CODES, stratified_weights
    bad = make_episode(1, T=40)
    bad.meta.update(termination_reason="unstable", success=False)
    bad.terminated[-1], bad.truncated[-1], bad.discount[-1] = False, True, 1.0
    ok = make_episode(0, T=40)
    tr = build_transitions([ok, bad], include_unstable=True)
    assert tr["unstable"].sum() == 5 and (~tr["unstable"]).sum() == 5
    sub = build_transitions([ok], max_per_episode=2)
    assert len(sub["obs"]) == 2 and sub["done"][-1] == 1          # the terminal interval is kept
    w = stratified_weights(np.array([0, 0, 0, 3]))
    assert abs(w[:3].sum() - w[3]) < 1e-5                         # each outcome code: equal total weight
    assert CODES[tr["code"][0]] == "success"

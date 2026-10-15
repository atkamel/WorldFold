"""isaac.fold_rl without Isaac: version 0 is the scripted fold, parameters land on the experts, PPO moves the policy."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaac import fold_rl as R   # noqa: E402


def test_unit_map_round_trips_and_clips():
    v = R.defaults(0)
    assert np.allclose(R.from_unit(R.to_unit(v)), v)
    lo, hi = R._bounds()
    assert np.allclose(R.from_unit(np.full(R.ACT_DIM, 5.0)), hi) and np.allclose(R.from_unit(-np.ones(R.ACT_DIM)), lo)


def test_version0_mean_is_scripted():
    pol = R.make_policy()
    for s in (0, 1):
        u, _ = R.decide(pol, np.random.default_rng(s).normal(size=R.OBS_DIM).astype(np.float32), s, explore=False)
        assert np.allclose(R.from_unit(u), R.defaults(s), atol=1e-6)


class _Arm:
    pass


class _Expert:
    def __init__(self):
        self.experts = {(s, p): _Arm() for s in (0, 1) for p in R.ARMS}
        self.base_overshoot = {(s, p): np.full(3, 0.01 * s) for s in (0, 1) for p in R.ARMS}
        self.overshoot = {k: v.copy() for k, v in self.base_overshoot.items()}


def test_apply_sets_each_arm():
    e = _Expert()
    v = R.defaults(1).copy()
    v[R.N_ARM + [n for n, _, _ in R.ARM_PARAMS].index("SQUEEZE")] = -0.07       # right arm squeezes harder
    v[[n for n, _, _ in R.ARM_PARAMS].index("place_dx")] = 0.01                   # left arm overshoots in x
    R.apply(e, 1, v)
    assert e.experts[(1, "right_")].SQUEEZE == pytest.approx(-0.07)
    assert e.experts[(1, "left_")].SQUEEZE == pytest.approx(0.05)
    assert np.allclose(e.overshoot[(1, "left_")], [0.02, 0.01, 0.01])


def test_ppo_moves_toward_rewarded_actions():
    pol = R.make_policy()
    opt = torch.optim.Adam(pol.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    obs = np.zeros(R.OBS_DIM, dtype=np.float32)
    u0 = pol.u0[0].numpy()
    batch = []
    for _ in range(64):
        u = u0 + rng.normal(0, R.INIT_STD, R.ACT_DIM)
        lp = float(pol.dist(torch.as_tensor(obs)[None], 0).log_prob(torch.as_tensor(u, dtype=torch.float32)).sum())
        batch.append({"decisions": [{"stage": 0, "obs": obs.tolist(), "act": u.tolist(), "logp": lp,
                                     "reward": float(u[0] - u0[0])}]})         # reward: push parameter 0 up
    for _ in range(5):
        R.ppo_update(pol, opt, batch)
    mean = pol.dist(torch.as_tensor(obs)[None], 0).mean[0].detach().numpy()
    assert mean[0] > u0[0] + 0.05

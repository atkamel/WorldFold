"""Milestone V: several envs per worker in lockstep (imitation/lockstep.py, the multi-slot worker in
imitation/rollout.py), on a pure-numpy fake batch -- no Isaac. The fake's advance() steps every copy, idle or not, so
an episode's trajectory depends only on its seed exactly when the barrier is right."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import multiprocessing.connection  # noqa: F401 (mp.connection.wait)
import threading
from pathlib import Path

import numpy as np
import pytest

from imitation.lockstep import Lockstep
from imitation.rollout import (ExpertController, Perturbation, PolicyController, _rollout, _serve_slots,
                               envs_per_proc)

# loaded by path: `tests` can resolve to mujuco/tests during a full collection (status.md defect 13)
_spec = importlib.util.spec_from_file_location("worldfold_fake_batch", Path(__file__).with_name("fake_batch.py"))
_fake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fake)
FakeBatch, FakePolicy, FakeTeacher = _fake.FakeBatch, _fake.FakePolicy, _fake.FakeTeacher

TIMEOUT = 60


class ThreadPool:
    """EnvPool's driver-facing surface (`pipes`, len) over in-process multi-slot workers: n_procs batches of B."""

    def __init__(self, n_procs, B):
        self.pipes, self.batches, self.threads = [], [], []
        for _ in range(n_procs):
            ends = [mp.Pipe() for _ in range(B)]
            batch = FakeBatch(B)
            t = threading.Thread(target=_serve_slots, daemon=True,
                                 args=([b for _, b in ends], batch, batch.envs),
                                 kwargs={"teacher_factory": FakeTeacher, "tick_s": 0.05})
            t.start()
            for a, _ in ends:
                assert a.recv() == ("ready", None)
                self.pipes.append(a)
            self.batches.append(batch)
            self.threads.append(t)

    def __len__(self):
        return len(self.pipes)

    def close(self):
        for p in self.pipes:
            p.send(("close", None))
        for p in self.pipes:
            while p.recv() != ("ok", None):
                pass
        for t in self.threads:
            t.join(timeout=10)
            assert not t.is_alive()


def _run(n_procs, B, seeds, controller, **kw):
    pool = ThreadPool(n_procs, B)
    out = {}

    def go():
        out["eps"] = _rollout(pool, seeds, controller, **kw)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(TIMEOUT)
    assert not t.is_alive(), "rollout deadlocked"
    pool.close()
    return out["eps"], pool


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.meta["seed"] == y.meta["seed"]
        for k in ("obs", "actions", "actor", "rewards", "terminated", "truncated", "label_steps", "labels"):
            np.testing.assert_array_equal(getattr(x, k), getattr(y, k), err_msg=f"seed {x.meta['seed']}: {k}")
        assert x.meta == y.meta


def _perturb(seed, rng):
    return Perturbation(t=2, k=2) if seed % 2 else None


SEEDS = list(range(100, 111))     # 11 seeds: slots finish at different times and some end idle


@pytest.mark.parametrize("make", [
    lambda: ExpertController(),
    lambda: PolicyController(FakePolicy(), replan_every=2, takeover=0.4),
    lambda: PolicyController(FakePolicy(), replan_every=3, label=True, beta=0.5),
], ids=["expert", "takeover", "label_beta"])
def test_episodes_depend_only_on_seed(make):
    ref, _ = _run(3, 1, SEEDS, make(), perturb_fn=_perturb)
    for n_procs, B in ((1, 3), (2, 3), (1, 4)):
        eps, pool = _run(n_procs, B, SEEDS, make(), perturb_fn=_perturb)
        _same(ref, eps)
    assert any(len(e.label_steps) for e in ref) or isinstance(make(), ExpertController)


def test_physics_steps_are_shared():
    """B envs in one batch take ~1/B as many global steps as B separate batches (the point of V)."""
    _, single = _run(3, 1, SEEDS[:6], ExpertController())
    _, shared = _run(1, 3, SEEDS[:6], ExpertController())
    separate = sum(b.steps for b in single.batches)
    together = shared.batches[0].steps
    assert together < 0.6 * separate, (together, separate)


def test_main_thread_calls_run_on_the_scheduler_thread():
    _, pool = _run(1, 3, SEEDS[:4], ExpertController())
    names = {name for _, name in pool.batches[0].main_calls}
    assert names == {pool.threads[0].name}      # the thread running the scheduler, not a slot thread


def test_idle_slots_never_block():
    """More slots than seeds: the unused slots stay inactive and the others still finish."""
    eps, _ = _run(1, 4, [7, 8], ExpertController())
    assert [e.meta["seed"] for e in eps] == [7, 8]


def _script(seed, rng):
    """One slot's command sequence: an episode with steps, resyncs and a label, then maybe a second one."""
    cmds = [("reset", (seed, None, True))]
    for _ in range(int(rng.integers(2, 6))):
        kind = rng.choice(["step_teacher", "step_action", "resync", "label"])
        if kind == "step_teacher":
            cmds.append(("step", None))
        elif kind == "step_action":
            cmds.append(("step", rng.uniform(-1, 1, 12).astype(np.float32)))
        elif kind == "resync":
            cmds.append(("resync", None))
        else:
            cmds.append(("label", int(rng.integers(1, 4))))
    return cmds


def _drive(pool, scripts, rng):
    """A random but protocol-respecting driver: each idle slot gets its next command in a random order, any reply
    is taken as it comes (never waits on one slot before sending to another), and an episode still in flight when
    its script runs out is stepped by the teacher to its end, as _rollout always finishes an episode."""
    scripts = {i: list(s) for i, s in scripts.items()}
    nxt = {i: 0 for i in scripts}
    inflight, live, replies = set(), set(), {i: [] for i in scripts}
    while True:
        for i in scripts:
            if i not in inflight and nxt[i] == len(scripts[i]) and i in live:
                scripts[i].append(("step", None))
        idle = [i for i in scripts if i not in inflight and nxt[i] < len(scripts[i])]
        rng.shuffle(idle)
        for i in idle[:int(rng.integers(1, len(idle) + 1))] if idle else []:
            pool.pipes[i].send(scripts[i][nxt[i]])
            nxt[i] += 1
            inflight.add(i)
        if not inflight and not idle:
            return replies
        for conn in mp.connection.wait([pool.pipes[i] for i in inflight], timeout=0.01):
            i = pool.pipes.index(conn)
            status, payload = conn.recv()
            assert status == "ok", payload
            cmd = scripts[i][nxt[i] - 1][0]
            if cmd == "label":
                payload = payload[0]                  # (labels, seconds): drop the wall time
            elif cmd == "reset":
                live.add(i)
            elif cmd == "step" and (payload[2] or payload[3]):
                live.discard(i)
            replies[i].append(payload)
            inflight.discard(i)


def _key(payload):
    """Comparable fingerprint of a reply (obs arrays, labels, scalars)."""
    if isinstance(payload, tuple):
        return tuple(_key(p) for p in payload)
    if isinstance(payload, np.ndarray):
        return payload.round(12).tobytes()
    if isinstance(payload, dict):
        return tuple(sorted((k, _key(v)) for k, v in payload.items() if k != "timing"))
    if isinstance(payload, list):
        return tuple(_key(p) for p in payload)
    if isinstance(payload, float):
        return round(payload, 12)
    return payload


@pytest.mark.parametrize("trial", range(5))
def test_random_command_orders_match_solo_runs(trial):
    rng = np.random.default_rng(trial)
    B = 3
    scripts = {i: _script(1000 * trial + i, rng) for i in range(B)}
    # solo reference: each script alone in its own batch of one
    ref = {}
    for i, s in scripts.items():
        pool = ThreadPool(1, 1)
        ref[i] = _drive(pool, {0: s}, np.random.default_rng(0))[0]
        pool.close()
    pool = ThreadPool(1, B)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=_drive(pool, scripts, rng)), daemon=True)
    t.start()
    t.join(TIMEOUT)
    assert not t.is_alive(), "deadlock under a random command order"
    pool.close()
    for i in scripts:
        # replies without timing noise: step payload's info carries per-call timing, dropped by _key
        assert [_key(p) for p in out["r"][i]] == [_key(p) for p in ref[i]], f"slot {i}"


def test_unfinished_episode_blocks_until_its_next_command():
    """Strict barrier: a slot with an episode in flight holds the global step until its next command arrives."""
    pool = ThreadPool(1, 2)
    a, b = pool.pipes
    zero = np.zeros(12, np.float32)
    a.send(("reset", (1, None, False)))
    assert a.poll(5)                       # b has no episode: a resets alone
    a.recv()
    b.send(("reset", (2, None, False)))
    assert not b.poll(0.3), "b's settle steps ran while a's episode waited for its next command"
    for k in range(3):                     # each of a's steps is one of b's SETTLE = 3 settle steps
        a.send(("step", zero))
        assert a.poll(5)
        a.recv()
        assert b.poll(5 if k == 2 else 0.3) == (k == 2)
    b.recv()
    b.send(("step", zero))
    assert not b.poll(0.3), "b stepped while a's episode waited"
    a.send(("step", zero))
    assert a.poll(5) and b.poll(5)
    a.recv(), b.recv()
    pool.close()


def test_pool_call_refuses_physics_with_several_envs_per_process():
    """EnvPool.call sends one command per slot and then waits in slot order; with lockstep slots out of phase that
    can wait forever (a slot done first sits idle mid-episode), so it refuses physics commands there."""
    from imitation.rollout import EnvPool
    pool = EnvPool.__new__(EnvPool)
    pool.envs_per_proc, pool.pipes = 2, []
    with pytest.raises(NotImplementedError):
        pool.call([0, 1], "reset", [None, None])


def test_advance_failure_reaches_every_parked_slot():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("solver blew up")

    ls = Lockstep(boom, tick_s=0.05)
    errors = []

    def slot(i):
        with ls.baton:
            ls.set_active(i, True)
            try:
                ls.sync(i)
            except RuntimeError as e:
                errors.append(str(e))
            ls.set_active(i, False)

    runner = threading.Thread(target=ls.run, daemon=True)
    runner.start()
    ts = [threading.Thread(target=slot, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    ls.close()
    runner.join(10)
    assert calls["n"] >= 1 and len(errors) == 3 and all("solver blew up" in e for e in errors)


def test_keep_alive_ticks_while_idle():
    ticks = []
    ls = Lockstep(lambda: None, keep_alive=lambda: ticks.append(1), tick_s=0.02)
    runner = threading.Thread(target=ls.run, daemon=True)
    runner.start()
    threading.Event().wait(0.2)
    ls.close()
    runner.join(5)
    assert len(ticks) >= 3


def test_envs_per_proc(monkeypatch):
    monkeypatch.delenv("WORLDFOLD_ISAAC_ENVS_PER_PROC", raising=False)
    assert envs_per_proc("isaac_weld") == 1
    monkeypatch.setenv("WORLDFOLD_ISAAC_ENVS_PER_PROC", "4")
    assert envs_per_proc("isaac_weld") == 4
    assert envs_per_proc("mujoco") == 1 and envs_per_proc(None) == 1
    with pytest.raises(ValueError):
        envs_per_proc("isaac")             # the CPU friction profile is not vectorised
    monkeypatch.setenv("WORLDFOLD_ISAAC_ENVS_PER_PROC", "0")
    with pytest.raises(ValueError):
        envs_per_proc("isaac_weld")


def test_thermal_pause_holds_work_and_keeps_episodes(monkeypatch, tmp_path):
    """The thermal guard's flag (imitation/thermal.py) pauses the driver before it sends work; the slots idle and
    keep ticking, and once the flag clears the episodes are the same as an unpaused run."""
    import imitation.thermal as thermal
    flag = tmp_path / "pause"
    monkeypatch.setenv("WORLDFOLD_THERMAL_FLAG", str(flag))
    monkeypatch.setattr(thermal, "_last_check", 0.0)
    monkeypatch.setattr(thermal, "_last_hot", False)
    ref, _ = _run(1, 3, SEEDS[:5], PolicyController(FakePolicy(), replan_every=2, takeover=0.4), perturb_fn=_perturb)
    flag.write_text("hot")
    monkeypatch.setattr(thermal, "_last_check", 0.0)
    threading.Timer(0.6, flag.unlink).start()
    eps, pool = _run(1, 3, SEEDS[:5], PolicyController(FakePolicy(), replan_every=2, takeover=0.4),
                     perturb_fn=_perturb)
    _same(ref, eps)
    assert not flag.exists()

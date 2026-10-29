import multiprocessing as mp
import time

from imitation.cpu_slot import ENV, cpu_slot


def _hold(path, q, hold):
    import os
    os.environ[ENV] = path
    with cpu_slot(poll=0.05):
        q.put(("in", time.perf_counter()))
        time.sleep(hold)
        q.put(("out", time.perf_counter()))


def test_cpu_slot_serializes_processes(tmp_path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    path = str(tmp_path / "cpu.lock")
    procs = [ctx.Process(target=_hold, args=(path, q, 0.6)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    events = sorted([q.get() for _ in range(4)], key=lambda e: e[1])
    assert [e[0] for e in events] == ["in", "out", "in", "out"]      # never both inside


def test_cpu_slot_is_a_noop_without_the_variable(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    with cpu_slot() as waited:
        assert waited == 0.0

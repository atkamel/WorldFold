"""Run his scripts/serve.py under yappi (profiles EVERY thread, incl. the 'inference-batcher' thread).

    YAPPI_OUT=/vol/profile/x.txt python /opt/teacher/yappi_serve.py --port 8000 policy:checkpoint ...
Send SIGUSR1 to dump per-thread stats to $YAPPI_OUT and exit.
"""
import os
import runpy
import signal
import sys
import threading

import yappi

OUT = os.environ["YAPPI_OUT"]
yappi.set_clock_type(os.environ.get("YAPPI_CLOCK", "wall"))
yappi.start(builtins=True, profile_threads=True)


def _rows(stats, key, n=35):
    stats.sort(key, "desc")
    out = []
    for i, s in enumerate(stats):
        if i >= n:
            break
        out.append(f"{s.tsub:9.3f}s self {s.ttot:9.3f}s total {s.ncall:9d} calls  {s.module.split('site-packages/')[-1]}:{s.lineno} {s.name}")
    return out


def _dump(signum, frame):
    yappi.stop()
    names = {t.ident: t.name for t in threading.enumerate()}
    lines = []
    threads = yappi.get_thread_stats()
    for t in threads:
        lines.append(f"THREAD ctx={t.id} name={names.get(t.tid, t.name)} ttot={t.ttot:.2f}s")
    for t in threads:
        name = names.get(t.tid, t.name)
        try:
            fs = yappi.get_func_stats(ctx_id=t.id)
        except TypeError:
            fs = yappi.get_func_stats(filter={"ctx_id": t.id})
        lines.append(f"\n===== THREAD {name} (ctx {t.id}) — top by SELF time =====")
        lines += _rows(fs, "tsub")
        lines.append(f"----- THREAD {name} — top by TOTAL time -----")
        lines += _rows(fs, "ttot")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, OUT)
    os._exit(0)


signal.signal(signal.SIGUSR1, _dump)
sys.path.insert(0, os.path.abspath("scripts"))
sys.argv = ["scripts/serve.py"] + sys.argv[1:]
runpy.run_path("scripts/serve.py", run_name="__main__")

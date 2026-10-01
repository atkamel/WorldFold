"""Runtime timing patch for his lehome-challenge sim loop (scripts/utils/remote_loop.py).

Per step it records: serialise observations, POST /infer (transport + worker proxy + teacher when called),
env.step (physics + render). After each episode it times physics-only and render-only steps (the floor
components) and writes everything to /vol/isaac/timing/<worker>_<time>.json.

    python isaac_timing_patch.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-timing-patch]"
p = ROOT / "lehome-challenge/scripts/utils/remote_loop.py"
s = p.read_text()
if MARK in s:
    print("already patched: timing")
    sys.exit(0)

edits = [
    ("    length = 0\n    success = False\n",
     "    length = 0\n    success = False\n"
     "    import time as _tm\n"
     "    _T = {\"ser\": [], \"post\": [], \"step\": []}\n"),
    ("        resp = policy.post(\"/infer\", policy.serialize_observation(obs))\n",
     "        _t0 = _tm.perf_counter()\n"
     "        _payload = policy.serialize_observation(obs)\n"
     "        _t1 = _tm.perf_counter()\n"
     "        resp = policy.post(\"/infer\", _payload)\n"
     "        _T[\"ser\"].append(_t1 - _t0); _T[\"post\"].append(_tm.perf_counter() - _t1)\n"),
    ("        for _ in range(int(resp.get(\"n_steps\", 1))):\n            env.step(action)\n",
     "        _t3 = _tm.perf_counter()\n"
     "        for _ in range(int(resp.get(\"n_steps\", 1))):\n            env.step(action)\n"
     "        _T[\"step\"].append(_tm.perf_counter() - _t3)\n"),
    ("    return {\"return\": episode_return, \"length\": length, \"success\": success}\n",
     "    try:\n"
     "        import json as _json, os as _os\n"
     "        _phys, _rend = [], []\n"
     "        for _ in range(20):\n"
     "            _a = _tm.perf_counter(); env.sim.step(render=False); _phys.append(_tm.perf_counter() - _a)\n"
     "        for _ in range(20):\n"
     "            _a = _tm.perf_counter(); env.sim.render(); _rend.append(_tm.perf_counter() - _a)\n"
     "        _os.makedirs(\"/vol/isaac/timing\", exist_ok=True)\n"
     "        _fn = f\"/vol/isaac/timing/{_os.environ.get('TEACHER_TIMING_TAG', 'run')}_{_os.environ.get('LEHOME_WORKER_LABEL', 'W')}_{int(_tm.time() * 1000)}.json\"\n"
     "        with open(_fn, \"w\") as _f:\n"
     "            _json.dump({\"ser\": _T[\"ser\"], \"post\": _T[\"post\"], \"step\": _T[\"step\"], \"phys_only\": _phys,\n"
     "                        \"render_only\": _rend, \"length\": length, \"success\": bool(success)}, _f)\n"
     "        print(f\"[timing] wrote {_fn}\", flush=True)\n"
     "    except Exception as _e:\n"
     "        print(f\"[timing] failed: {_e!r}\", flush=True)\n"
     "    return {\"return\": episode_return, \"length\": length, \"success\": success}\n"),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected 1 match, found {n} for:\n{old[:120]}"
    s = s.replace(old, new)
p.write_text(MARK + "\n" + s)
print("TIMING_PATCH_OK")

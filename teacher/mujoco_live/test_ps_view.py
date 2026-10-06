"""End-to-end check of the Polyscope (state) view against the fake Isaac server, no GPU:
    python test_ps_view.py        (starts `python -m teleop.fake_isaac PORT --state`, connects, draws 300 frames)
Reports client frame time, state frames received per second, payload size, and that the drawn robot matches."""
import os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "..", "oracle")
PORT = 7796
srv = subprocess.Popen([sys.executable, "-m", "teleop.fake_isaac", str(PORT), "--state"], cwd=ORACLE,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    from teleop_client import Client
    from teleop_client_ps import PolyscopeClient
    conn = Client._connect("127.0.0.1", PORT)
    assert conn[1].get("view") == "state", "server did not offer the state view"
    c = PolyscopeClient("127.0.0.1", PORT, restore_tuning=False, conn=conn)
    n0 = 0; frames = []; got = 0; size = 0
    t_start = time.perf_counter()
    for i in range(300):
        t = time.perf_counter()
        with c.lock:
            new, fr = c.new, c.frame
        if new:
            got += 1; size = len(fr)
        c.send_input({}, [])
        c.draw()
        frames.append((time.perf_counter() - t) * 1000)
        time.sleep(max(0, 1 / 60 - (time.perf_counter() - t)))
    wall = time.perf_counter() - t_start
    f = np.array(frames[30:])
    print(f"client draw+present: median {np.median(f):.2f} ms, p95 {np.percentile(f, 95):.2f} ms over {len(f)} frames")
    print(f"state frames received: {got / wall:.1f}/s, payload {size} bytes ({size * got / wall * 8 / 1e6:.1f} Mbit/s)")
    tip = c.parts["L"][0].get_transform()[:3, 3]
    print("left base part placed at (table frame, m):", np.round(tip, 3), "| hud msg:", c.hud.get("msg"))
    c.close()
finally:
    srv.terminate()

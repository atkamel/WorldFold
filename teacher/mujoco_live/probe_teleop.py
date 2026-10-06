"""Headless teleop probe over ONE connection (the server accepts only one client): assigns two fake mice to the
arms, then plays for `dur` s (circles + a grab/release every 8 s);
counts state updates/s and round-trip lag (input sent -> state carrying its echo received)."""
import math, os, socket, sys, threading, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
from teleop import protocol
host, port, dur = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
t = time.perf_counter()
s = socket.create_connection((host, port), timeout=30)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
print(f"TCP connect: {(time.perf_counter() - t) * 1000:.0f} ms", flush=True)
h, _ = protocol.recv(s); QUANT = float(h.get("quant", 1e-4))
print("hello:", h.get("t"), h.get("view"), h.get("backend"), flush=True)
times, lags, srv_fps, end, huds, cloth = [], [], [], [None], [], []
dec, nbytes, wire = protocol.StateDecoder(), [], [0]
def reader():
    while True:
        try:
            m = protocol.recv(s)
            if m: wire[0] = len(m[1])
        except OSError:
            m = None
        if m is None:
            end[0] = end[0] or "closed"; return
        hd = m[0]
        if hd["t"] == "state":
            m = (hd, dec.decode(hd, m[1])); nbytes.append(len(m[1]) if "fid" not in hd else wire[0])
        if hd["t"] in ("state", "frame"):
            now = time.time(); times.append(now)
            if hd.get("echo"): lags.append((now - hd["echo"]) * 1000)
            srv_fps.append(hd.get("fps", 0)); huds.append(hd.get("hud") or {})
            if hd["t"] == "state" and len(m[1]) > 48:
                cloth.append(np.frombuffer(m[1][48:], "<i2").reshape(-1, 3).astype(float) * QUANT)
        elif hd["t"] == "end":
            end[0] = hd.get("why"); return
threading.Thread(target=reader, daemon=True).start()
def send(mice, i):
    protocol.send(s, dict(t="input", mice=mice, keys=[], ext={"111": True, "222": True} if i == 0 else {}, ct=time.time(),
                          got=dec.last))

# assign the two fake mice to the arms the way a player does: click LEFT on the left mouse, then on the right one
for i, (h, ev) in enumerate([("111", ["left_down"]), ("111", ["left_up"]), ("222", ["left_down"]), ("222", ["left_up"])]):
    send({h: [0, 0, 0, 1 if ev == ["left_down"] else 0, 0, ev]}, i); time.sleep(0.3)
# then play: both arms trace circles at a hand-like speed; every 8 s both grab (hold LEFT 2.5 s) and let go
t0 = time.time(); i = 0; held = False
while time.time() - t0 < dur and not end[0]:
    a, t = i * 0.04, time.time() - t0
    want = (t % 8.0) > 5.5
    ev = (["left_down"] if want and not held else ["left_up"] if held and not want else [])
    held = want
    mice = {"111": [int(14 * math.cos(a)), int(14 * math.sin(a)), 0, int(held), 0, ev],
            "222": [int(-14 * math.cos(a)), int(14 * math.sin(a)), 0, int(held), 0, list(ev)]}
    send(mice, 4 + i)
    i += 1
    if i % 600 == 0:
        w = [x for x in times if x > time.time() - 10]
        print(f"  {time.time() - t0:4.0f}s: last 10 s {len(w) / 10:.1f} updates/s, lag median {np.median(lags[-100:]) if lags else 0:.0f} ms", flush=True)
    time.sleep(1 / 60)
try: protocol.send(s, dict(t="bye"))
except OSError: pass
time.sleep(1)
tt = np.array(times); tt = tt[tt > t0 + 5]
print(f"RESULT: {(len(tt) - 1) / (tt[-1] - tt[0]):.1f} updates/s (after 5 s warm-up), "
      f"lag median {np.median(lags):.0f} ms p95 {np.percentile(lags, 95):.0f} ms, end: {end[0]}")
aims = [tuple(round(x, 3) for x in (h.get("arms", {}).get("L", {}).get("aim") or [0, 0])[:2]) for h in huds[::30]]
print("arms assigned:", (huds[-1].get("mice") if huds else None), "| left-arm aim samples:", aims[:6],
      "| phases seen:", sorted({h.get("arms", {}).get("L", {}).get("phase", "?") for h in huds}))
if len(cloth) > 2:
    c0 = cloth[0]
    moved = [float(np.abs(c - c0).max()) * 1000 for c in cloth[1::max(1, len(cloth) // 8)]]
    print("cloth points:", len(c0), "| z range (table frame) mm:", round(c0[:, 2].min() * 1000, 1), "..", round(c0[:, 2].max() * 1000, 1),
          "| max displacement vs first frame, mm, over the session:", [round(x, 1) for x in moved])
if nbytes:
    print(f"state frame size on the wire: median {np.median(nbytes) / 1000:.1f} kB, max {max(nbytes) / 1000:.1f} kB")

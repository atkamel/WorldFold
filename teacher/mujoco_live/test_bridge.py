"""End-to-end bridge test (no window): connect to a running teleop server, play a sleeve fold with two
virtual mice over the network, measure frame rate, lag and what comes back.
    python teleop_server_mujoco.py 7790 &   python test_bridge.py 127.0.0.1 7790"""
import os, sys, time, socket, threading, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
from teleop import protocol  # noqa: E402

host, port = sys.argv[1], int(sys.argv[2])
s = socket.create_connection((host, port), timeout=60)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
hello = protocol.recv(s)[0]
print("hello:", {k: (f"{len(v)} items" if isinstance(v, list) and len(v) > 4 else v) for k, v in hello.items()})
state = dict(frames=0, bytes=0, lags=[], hud={}, recs=[], ended=None, t0=time.time())


def reader():
    while True:
        m = protocol.recv(s)
        if m is None:
            state["ended"] = state["ended"] or "closed"; return
        h, p = m
        if h["t"] == "frame":
            state["frames"] += 1; state["bytes"] += len(p); state["hud"] = h["hud"]
            if h.get("echo"):
                state["lags"].append(time.time() - h["echo"])
        elif h["t"] == "log":
            state["recs"] += h["recs"]
        elif h["t"] == "end":
            state["ended"] = h["why"]


threading.Thread(target=reader, daemon=True).start()
M1, M2 = "101", "202"
GAIN = 8e-5


def send(mice=None, keys=()):
    protocol.send(s, dict(t="input", mice=mice or {}, keys=list(keys), ext={M1: True, M2: True}, ct=time.time()))
    time.sleep(1 / 60)


def aim(arm):
    return np.array(state["hud"]["arms"][arm]["aim"][:2]) * 100


def goto(h, arm, xy, steps=240):
    for _ in range(steps):
        d = np.array(xy) - aim(arm)
        if np.linalg.norm(d) < 0.3:
            break
        st = d / max(1.0, np.linalg.norm(d) / 0.8)
        send({h: [st[0] / 100 / GAIN, -st[1] / 100 / GAIN, 0, False, False, []]})
    for _ in range(30):
        send()


def wait_phase(arm, phases, limit=240):
    for _ in range(limit):
        if state["hud"].get("arms", {}).get(arm, {}).get("phase") in phases:
            return True
        send()
    return False


while not state["hud"]:
    time.sleep(0.05)
send({M1: [0, 0, 0, True, False, ["left_down"]]}); send({M2: [0, 0, 0, True, False, ["left_down"]]})
for _ in range(20):
    send()
print("assigned:", not state["hud"]["assigning"], "|", state["hud"]["msg"])
t_play = time.time()
goto(M1, "L", [-19, 26])
send({M1: [0, 0, 0, True, False, ["left_down"]]})
ok_grab = wait_phase("L", ("carry",))
goto(M1, "L", [-2, 26])
send({M1: [0, 0, 0, False, False, ["left_up"]]})
ok_drop = wait_phase("L", ("hover",))
for _ in range(120):
    send()
dur = time.time() - t_play
protocol.send(s, dict(t="bye"))
time.sleep(0.5)
ev = [r for r in state["recs"] if r.get("event")]
lags = np.array(state["lags"]) * 1000
fps = state["frames"] / (time.time() - state["t0"])
print(f"grab ok {ok_grab}, drop ok {ok_drop} | events back: {[(r['event'], r.get('arm'), r.get('n')) for r in ev]}")
print(f"frames {state['frames']} ({fps:.0f}/s), avg {state['bytes'] / max(1, state['frames']) / 1024:.0f} KB/frame, "
      f"{state['bytes'] / (time.time() - state['t0']) / 1024 / 1024 * 8:.1f} Mbit/s")
print(f"input->frame lag: median {np.median(lags):.0f} ms, p95 {np.percentile(lags, 95):.0f} ms")
print(f"demo records received on the laptop side: {len(state['recs'])} (frames {sum(1 for r in state['recs'] if r.get('frame'))}) "
      f"| server says: {state['hud'].get('msg')} | ended: {state['ended']}")

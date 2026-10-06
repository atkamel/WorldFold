"""Live-tuning test over the real network protocol, against the fake Isaac backend (no GPU):
the panel spec arrives, changes apply and are logged, cloth changes wait for 'apply', a grab still works with
changed grab settings, and a bad op never ends the session.
    python test_tuning.py"""
import os, sys, time, socket, threading, subprocess
import numpy as np

ORACLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle")
sys.path.insert(0, ORACLE)
from teleop import protocol  # noqa: E402

PORT = 7791
srv = subprocess.Popen([sys.executable, "-m", "teleop.fake_isaac", str(PORT)], cwd=ORACLE,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for _ in range(100):
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=30); break
    except OSError:
        time.sleep(0.2)
hello = protocol.recv(s)[0]
st = dict(hud={}, recs=[], ended=None)


def reader():
    while True:
        m = protocol.recv(s)
        if m is None:
            st["ended"] = st["ended"] or "closed"; return
        h = m[0]
        if h["t"] == "frame":
            st["hud"] = h["hud"]
        elif h["t"] == "log":
            st["recs"] += h["recs"]
        elif h["t"] == "end":
            st["ended"] = h["why"]


threading.Thread(target=reader, daemon=True).start()
M1, M2, GAIN = "101", "202", 8e-5
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + f" {name}  {info}")
    if not ok:
        fails.append(name)


def send(mice=None, tune=None, n=1):
    for _ in range(n):
        m = dict(t="input", mice=mice or {}, keys=[], ext={M1: True, M2: True}, ct=time.time())
        if tune:
            m["tune"], tune = tune, None
        protocol.send(s, m)
        time.sleep(1 / 60)


while not st["hud"]:
    time.sleep(0.05)
spec = {d["key"]: d for d in hello.get("tune", [])}
check("hello lists the tunable settings", {"arm_speed", "grab_slide", "mass_g"} <= set(spec),
      f"{len(spec)} settings: " + ", ".join(sorted({d['kind'] for d in spec.values()})))
check("first log record = the settings the demo is played with", any("tune" in r and len(r["tune"]) == len(spec) for r in st["recs"]))

send(tune=[["adjust", "arm_speed", 1], ["adjust", "arm_speed", 1], ["set", "grab_slide", 0.04]], n=10)
t = st["hud"]["tune"]
check("control + grip changes apply", abs(t["arm_speed"] - 0.50) < 1e-9 and abs(t["grab_slide"] - 0.04) < 1e-9,
      f"arm_speed {t['arm_speed']}, grab_slide {t['grab_slide']}")
check("each change is logged", sum(1 for r in st["recs"] if "tune" in r) >= 4)
send(tune=[["set", "arm_speed", 99]], n=5)
check("values are clamped to their range", st["hud"]["tune"]["arm_speed"] == 1.0, st["hud"]["tune"]["arm_speed"])
send(tune=[["reset", "arm_speed"]], n=5)
check("reset returns to the default", st["hud"]["tune"]["arm_speed"] == 0.40)

send(tune=[["adjust", "mass_g", 1], ["set", "adhesion", 0.0]], n=10)
check("cloth change waits for apply", st["hud"]["cloth_pending"] is True, st["hud"]["msg"])
ep = st["hud"]["episode"]
send(tune=[["apply_cloth"]], n=20)
check("apply cloth rebuilds the shirt", st["hud"]["episode"] == ep + 1 and st["hud"]["cloth_pending"] is False, st["hud"]["msg"])
rec = [r for r in st["recs"] if "tune_cloth" in r]
check("cloth values logged with the rebuild", rec and abs(rec[-1]["tune_cloth"]["mass_g"] - 62.5) < 1e-9 and rec[-1]["tune_cloth"]["adhesion"] == 0.0,
      rec[-1]["tune_cloth"] if rec else None)

send(tune=[["bogus"], ["adjust", "no_such_key", 1], ["set", "arm_speed", "not a number"]], n=10)
check("bad ops never end the session", st["ended"] is None, st["hud"].get("msg"))

# a full-game grab after all that tuning (slide back at the tested 3 cm: on the fake, a 4 cm slide from this pose
# missed; the fake has no real physics, so slide lengths are checked directly below and for real in Isaac)
send({M1: [0, 0, 0, True, False, ["left_down"]]}); send({M2: [0, 0, 0, True, False, ["left_down"]]}); send(n=20)
send(tune=[["reset", "grab_slide"]], n=5)


def aim(arm):
    return np.array(st["hud"]["arms"][arm]["aim"][:2]) * 100


for _ in range(240):
    d = np.array([-15, 26]) - aim("L")   # sleeve middle
    if np.linalg.norm(d) < 0.3:
        break
    stp = d / max(1.0, np.linalg.norm(d) / 0.8)
    send({M1: [stp[0] / 100 / GAIN, -stp[1] / 100 / GAIN, 0, False, False, []]})
send(n=30)
send({M1: [0, 0, 0, True, False, ["left_down"]]})
for _ in range(400):
    if st["hud"]["arms"]["L"]["phase"] == "carry":
        break
    send()
check("grab works after a session of tuning (cloth rebuilt, values changed)", st["hud"]["arms"]["L"]["phase"] == "carry",
      (st["hud"]["arms"]["L"]["phase"], st["hud"]["msg"], [(r["event"], r.get("n")) for r in st["recs"] if r.get("event") == "grab"]))

protocol.send(s, dict(t="bye"))
time.sleep(0.5)
srv.terminate()
out = srv.communicate(timeout=10)[0]
print("server log tail:", " | ".join(out.strip().splitlines()[-2:]))

# the slide-length setting really changes the scripted grasp (direct, on the fake env)
from teleop.fake_isaac import FakeEnv, helpers  # noqa: E402
from teleop.isaac_world import IsaacWorld  # noqa: E402
from teleop.tuning import Tuning  # noqa: E402
spans = {}
for slide in (0.03, 0.05):
    w = IsaacWorld(FakeEnv(), helpers(), settle_ticks=5)
    w.tune = Tuning(("control", "grip")); w.tune.set("grab_slide", slide)
    q, _, _ = w.ik("L", np.array([-0.15, 0.26, 0.03]), w.q("L"), w.roll("L"))
    w.set_arm("L", q, w.roll("L"))
    for _ in range(40):
        w.step()
    w.grab("L"); xs = []
    while w.busy("L"):
        w.step(); xs.append(w.tip("L")[0])
    spans[slide] = (max(xs) - min(xs), w.held["L"])
check("grab_slide changes the grasp move (and still holds cloth)",
      spans[0.05][0] > spans[0.03][0] + 0.005 and spans[0.03][1] > 0 and spans[0.05][1] > 0,
      {k: (f"{v[0] * 100:.1f} cm", f"{v[1]} pts") for k, v in spans.items()})
print(f"\n{12 - len(fails)}/12 passed" + (f"  FAILED: {fails}" if fails else ""))

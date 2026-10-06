"""Control tests for game.Game with fake mice (no window). Each check prints PASS/FAIL with numbers."""
import time, json
import numpy as np
from game import Game, ARMS

g = Game(log=True)
M1, M2 = 101, 202            # two fake USB mice
TOUCHPAD = 999
results = []


def check(name, ok, info=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)


def tick(n=1, mice=None):
    for _ in range(n):
        g.tick(mice or {}, 1 / 60, is_external=lambda h: h != TOUCHPAD)
        mice = None


def ev(h, dx=0, dy=0, wh=0, events=()):
    return {h: (dx, dy, wh, False, False, list(events))}


def goto(h, arm, xy, steps=90):
    """Move a mouse so its arm's aim point reaches xy (cm), like a player would."""
    for _ in range(steps):
        c = g.cursor[arm][:2] * 100
        d = (np.array(xy) - c)
        if np.linalg.norm(d) < 0.1 and np.linalg.norm(g.cmd[arm][:2] * 100 - xy) < 0.3:
            break
        step = d / max(1.0, np.linalg.norm(d) / 1.2)  # at most 1.2 cm of cursor per tick
        gn = g.gain[h]
        tick(1, ev(h, step[0] / 100 / gn, -step[1] / 100 / gn))
    tick(30)


# 1. assignment: the touchpad is ignored, then two USB mice
tick(1, ev(TOUCHPAD, events=["left_down"]))
check("touchpad ignored at assignment", g.devmap == {} and "touchpad" in g.msg, g.msg)
tick(1, ev(M1, events=["left_down"])); tick(1, ev(M2, events=["left_down"]))
check("two mice assigned", g.devmap == {M1: "L", M2: "R"} and not g.assigning, str(g.devmap))

# 2. each mouse moves only its own arm
r0 = g.cmd["R"].copy()
goto(M1, "L", [-12, 20])
check("mouse 1 moves the LEFT arm only", np.linalg.norm(g.cmd["L"][:2] * 100 - [-12, 20]) < 0.5 and np.linalg.norm(g.cmd["R"][:2] - r0[:2]) < 1e-3,
      f"L aim {np.round(g.cmd['L'][:2] * 100, 1)}, R moved {np.linalg.norm(g.cmd['R'][:2] - r0[:2]) * 100:.2f} cm sideways")
l0 = g.cmd["L"].copy()
goto(M2, "R", [12, 20])
check("mouse 2 moves the RIGHT arm only", np.linalg.norm(g.cmd["R"][:2] * 100 - [12, 20]) < 0.5 and np.linalg.norm(g.cmd["L"][:2] - l0[:2]) < 1e-3)
tip_err = max(np.linalg.norm(g.w.tip(a) - g.cmd[a]) for a in ARMS)
check("arms follow their aim points", tip_err < 0.006, f"max tip error {tip_err * 100:.2f} cm")

# 3. wheel turns only its own arm's jaws; [ ] change carry height
yL, yR = g.w.jaw_yaw("L"), g.w.jaw_yaw("R")
tick(1, ev(M1, wh=3)); tick(40)
dL = np.degrees(((g.w.jaw_yaw("L") - yL + np.pi) % (2 * np.pi)) - np.pi)
dR = np.degrees(((g.w.jaw_yaw("R") - yR + np.pi) % (2 * np.pi)) - np.pi)
check("wheel turns only its own arm's jaws", abs(dL - 45) < 4 and abs(dR) < 2, f"L jaws turned {dL:+.1f} deg (3 notches), R {dR:+.1f} deg")
tick(1, ev(M1, wh=-3)); tick(40)
cL, cR = g.carry["L"], g.carry["R"]
g.last_moved = M1; g.key("]"); g.key("]")
tick(60)
zL = g.w.tip("L")[2] - g.w.surface_under(g.cmd["L"][:2])
check("] raises only that arm's carry height", abs(g.carry["L"] - cL - 0.01) < 1e-9 and g.carry["R"] == cR and abs(zL - g.carry["L"]) < 0.004,
      f"L {cL * 100:.1f}->{g.carry['L'] * 100:.1f} cm (hovers at {zL * 100:.1f}), R stays {g.carry['R'] * 100:.1f}")
g.key("["); g.key("[")

# 4. left sleeve: grab, carry, drop (one arm)
goto(M1, "L", [-19, 26])
tick(1, {M1: (0, 0, 0, True, False, ["left_down"])})
tick(60)
check("hold left = grab", g.phase["L"] == "carry" and g.w.held["L"] > 0, f"phase {g.phase['L']}, holding {g.w.held['L']} pts")
goto(M1, "L", [-1, 26])
tick(1, ev(M1, events=["left_up"])); tick(150)
check("release = drop", g.phase["L"] == "hover" and g.w.held["L"] == 0, f"area {100 * g.score['area_vs_flat']:.0f}% of flat")
a1 = g.score["area_vs_flat"]
check("flowback measured and small after a drop", g.last_flowback is not None and g.last_flowback[1] < 15,
      f"flowback avg {g.last_flowback[0]:.1f} mm, worst 5% {g.last_flowback[1]:.1f} mm" if g.last_flowback else "none")

# 4b. turning the wrist while holding turns the held cloth
g.key("Z"); tick(30)
goto(M1, "L", [-8, 15])
tick(1, {M1: (0, 0, 0, True, False, ["left_down"])}); tick(60)
held = g.w.cloth.att[0][0]
v0 = g.w.cloth.x[held, :2] - g.w.tip("L")[:2]
tick(1, {M1: (0, 0, 6, True, False, [])}); tick(90)    # 6 notches = 90 degrees
v1 = g.w.cloth.x[held, :2] - g.w.tip("L")[:2]
ang = np.degrees(np.arctan2(v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0], (v0 * v1).sum(1)))
big = np.linalg.norm(v0, axis=1) > 0.004
check("wheel while holding turns the held cloth", big.any() and abs(np.median(ang[big]) - 90) < 15,
      f"held points turned {np.median(ang[big]):+.0f} deg around the gripper")
tick(1, ev(M1, events=["left_up"])); tick(90)
g.key("Z"); tick(30)

# 4c. raised arm still reaches far (it lowers / tilts instead of locking)
g.last_moved = M1
for _ in range(10): g.key("]")   # carry 8 cm
goto(M1, "L", [-4, 31], steps=300)
y_reached = g.cmd["L"][1] * 100
check("raised arm reaches the far side of the shirt (no lock)", y_reached > 29.5,
      f"aimed at y=31 with carry {g.carry['L'] * 100:.0f} cm -> reached y={y_reached:.1f} at z={g.cmd['L'][2] * 100:.1f} cm")
for _ in range(10): g.key("[")

# 5. undo restores the shirt exactly
x_before = g.w.cloth.x.copy()
goto(M2, "R", [19, 26]); tick(1, {M2: (0, 0, 0, True, False, ["left_down"])}); tick(60)
goto(M2, "R", [1, 26]); tick(1, ev(M2, events=["left_up"])); tick(90)
g.key("Z"); tick(1)
check("Z undo restores the cloth", np.abs(g.w.cloth.x - x_before).max() < 0.003, f"max diff {np.abs(g.w.cloth.x - x_before).max() * 1000:.2f} mm")

# 6. tandem mirror: one mouse folds the right sleeve while... (here: both sleeves pattern on fresh shirt)
g.key("R"); tick(30)
g.key("T"); g.key("T")  # mirror
check("tandem mirror on", g.tandem == 2)
goto(M1, "L", [-19, 26])
check("mirror: right arm mirrors left", np.linalg.norm(g.cmd["R"][:2] * 100 - [19, 26]) < 3.0, f"R aim {np.round(g.cmd['R'][:2] * 100, 1)}")
tick(1, {M1: (0, 0, 0, True, False, ["left_down"])}); tick(60)
check("tandem: one button grabs with both arms", g.w.held["L"] > 0 and g.w.held["R"] > 0, f"held L {g.w.held['L']} R {g.w.held['R']}")
goto(M1, "L", [-1.5, 26])
tick(1, ev(M1, events=["left_up"])); tick(90)
check("tandem drop: both sleeves folded", g.w.held["L"] == 0 and g.w.held["R"] == 0, f"area {100 * g.score['area_vs_flat']:.0f}% of flat")
g.key("T")  # mirror -> off
check("tandem off again", g.tandem == 0)

# 7. bottom-up fold with both arms in parallel tandem
goto(M1, "L", [-6, 8.5]); goto(M2, "R", [6, 8.5])
g.key("T")  # off -> parallel
check("tandem parallel on", g.tandem == 1)
tick(1, {M2: (0, 0, 0, True, False, ["left_down"])}); tick(60)
check("parallel: both arms grabbed the hem", g.w.held["L"] > 0 and g.w.held["R"] > 0, f"held L {g.w.held['L']} R {g.w.held['R']}")
goto(M1, "L", [-6, 28], steps=150)
check("parallel: right arm moved with the left", abs(g.cmd["R"][1] - g.cmd["L"][1]) < 0.01, f"L {np.round(g.cmd['L'][:2]*100,1)} R {np.round(g.cmd['R'][:2]*100,1)}")
tick(1, ev(M1, events=["left_up"])); tick(120)
s = g.score
check("sleeves + bottom-up made the shirt compact", s["area_vs_flat"] < 0.55, f"area {100 * s['area_vs_flat']:.0f}% of flat, box {s['rect_cm']}, rect {s['rectangularity']}")
g.key("T"); g.key("T")

# 8. no jam: drop target unreachable -> it still lets go
g.key("R"); tick(30)
goto(M1, "L", [-19, 26]); tick(1, {M1: (0, 0, 0, True, False, ["left_down"])}); tick(60)
goto(M1, "L", [20, 38], steps=200)  # push far out of reach
tick(1, ev(M1, events=["left_up"])); tick(70)
check("never stuck: drop completes even at the reach limit", g.w.held["L"] == 0 and g.phase["L"] in ("rising", "hover"), f"phase {g.phase['L']}")

# 9. arms collide instead of passing through each other
g.key("R"); tick(30)
goto(M1, "L", [0, 18]); goto(M2, "R", [0, 18])
gap = np.linalg.norm(g.w.tip("L") - g.w.tip("R"))
check("arms block each other (no phasing)", gap > 0.02, f"tip distance {gap * 100:.1f} cm with both aimed at the same point")

# 10. slide (right button): drag along the table, stays low
g.key("R"); tick(30)
goto(M1, "L", [-8, 10])
tick(1, {M1: (0, 0, 0, False, True, ["right_down"])}); tick(60)
goto(M1, "L", [-12, 10])
hmax = g.w.tip("L")[2]
tick(1, ev(M1, events=["right_up"])); tick(60)
check("hold right = drag low along the table", hmax < 0.02 and g.w.held["L"] == 0, f"carry height while dragging {hmax * 100:.1f} cm")

# 11. logging: frames keep coming after reset and undo
recs = [json.loads(l) for l in open(g.log_path)]
eps = {}
for r in recs:
    if r.get("frame"):
        eps[r["episode"]] = eps.get(r["episode"], 0) + 1
check("frames logged in every episode (after resets too)", all(v > 20 for v in eps.values()) and len(eps) >= 4, str(eps))

# 12. speed
t0 = time.perf_counter(); tick(120); ms = (time.perf_counter() - t0) / 120 * 1000
check("game logic + physics fast enough for 60 Hz", ms < 10, f"{ms:.1f} ms per tick")
print(f"\n{sum(results)}/{len(results)} passed")

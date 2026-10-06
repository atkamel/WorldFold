"""Two-mouse cloth-folding controller: all game logic, no window, no physics engine.

The same code drives the local MuJoCo game, the local bridge test and the Isaac (LeHome) teleop server.
It needs a `world` backend with this interface (positions in metres, table surface at z = 0):

    tip(a) -> (3,)            gripper point of arm a ("L"/"R")       q(a) -> (4,) arm joints (no wrist roll)
    roll(a) -> float          current wrist roll                      jaw_yaw(a) -> float  jaw line angle (rad)
    ik(a, goal, q4, roll, seeds=True) -> (q4, pos_err_m, tilt_deg)
    roll_for_yaw(a, q4, yaw) -> (roll, achieved_yaw)                  set_arm(a, q4, roll)
    grab(a, all_layers) -> n  start a grab (n = cloth points caught if known now)
    release(a)                start letting go                        busy(a) -> bool  a grab/release move still running
    held -> {a: n}            step(dt)                                surface_under(xy) -> top of resting cloth (z)
    score() -> dict           cloth.x -> (N,3) cloth points           get_state() / set_state(s) / reset()
    xlim, ylim                table area the aim points may cover

Mouse input per tick: {handle: (dx, dy, wheel, left_held, right_held, events)} (rawmouse.MiceReader.take()).
"""
import time
import numpy as np

from .tuning import Tuning

ARMS = ("L", "R")
CARRY_DEFAULT = 0.03
TANDEM = ["off", "parallel", "mirror"]


class Game:
    def __init__(self, world, sink=None, tune=None):
        self.w = world
        self.sink = sink                # callable(record dict) for the demo log (file, network, ...)
        # live-tunable settings (tuning.py): grip settings if the world runs scripted grasps, cloth if it can rebuild
        kinds = ("control",) + (("grip",) if getattr(world, "TUNE_GRIP", False) else ()) + \
                (("cloth",) if hasattr(world, "apply_cloth") else ())
        self.tune = Tuning(kinds, tune)
        world.tune = self.tune
        self._tune_logged = False
        self.xlim = getattr(world, "xlim", (-0.24, 0.24))
        self.ylim = getattr(world, "ylim", (-0.03, 0.40))
        self.devmap = {}                # mouse handle -> arm
        self.gain = {}                  # mouse handle -> metres per count
        self.last_input = {}            # mouse handle -> time
        self.last_moved = None
        self.assigning = True
        self.tandem = 0
        self.undo = []
        self.n_grabs = 0
        self.msg = "click the LEFT button on the mouse for the LEFT (orange) arm"
        self.score = self.w.score() or {}
        self.best = self.score
        self.tick_n = 0
        self.time = 0.0                 # simulated seconds
        self.episode = 0
        self._new_episode()

    # ---------------------------------------------------------------- bookkeeping
    def write(self, rec):
        if self.sink:
            rec["tick"] = self.tick_n
            rec["t"] = round(self.time, 3)
            rec["episode"] = self.episode
            self.sink(rec)

    def _new_episode(self):
        w = self.w
        self.cursor = {a: w.tip(a) for a in ARMS}
        self.cmd = {a: w.tip(a) for a in ARMS}
        self.qcmd = {a: w.q(a) for a in ARMS}
        self.phase = {a: "hover" for a in ARMS}
        self.mode = {a: "lift" for a in ARMS}       # lift, slide or stack
        self.carry = {a: CARRY_DEFAULT for a in ARMS}
        self.t_phase = {a: 0.0 for a in ARMS}
        self.reach_flash = {a: 0.0 for a in ARMS}
        self.reach_limited = {a: False for a in ARMS}
        self.yaw_cmd = {a: w.jaw_yaw(a) for a in ARMS}   # where the jaws should point (wheel turns it)
        self.roll = {a: float(w.roll(a)) for a in ARMS}
        self.pending_flow = None                          # (time, cloth copy, grabs so far) after a drop
        self.last_flowback = None                         # (avg mm, worst-5% mm)
        self.next_frame_t = self.next_cloth_t = self.time

    def cloth_state(self):
        # an array, not a list: turning 14,746 points into text takes ~10 ms, so the sink does it off the control
        # loop (it is written out as the same JSON list)
        return np.round(self.w.cloth.x * 100, 2)

    def snapshot(self):
        return dict(world=self.w.get_state(), cursor={a: self.cursor[a].copy() for a in ARMS},
                    cmd={a: self.cmd[a].copy() for a in ARMS}, qcmd={a: self.qcmd[a].copy() for a in ARMS},
                    roll=dict(self.roll), yaw=dict(self.yaw_cmd), score=self.score)

    def do_undo(self):
        if not self.undo:
            self.msg = "nothing to undo"; return
        s = self.undo.pop()
        self.w.set_state(s["world"])
        for a in ARMS:
            self.cursor[a], self.cmd[a], self.qcmd[a] = s["cursor"][a], s["cmd"][a], s["qcmd"][a]
            self.roll[a], self.yaw_cmd[a] = s["roll"][a], s["yaw"][a]
            self.phase[a], self.t_phase[a] = "hover", 0.0
        self.score = s["score"]
        self.pending_flow = None
        self.write({"undo": True})
        self.msg = "undone"

    def do_reset(self):
        self.w.reset()
        self.undo.clear()
        self.episode += 1
        self._new_episode()
        self.score = self.best = self.w.score() or {}
        self.write({"reset": True})
        self.msg = "new shirt"

    # ---------------------------------------------------------------- live tuning
    def tune_op(self, op):
        """op: ["adjust", key, +1|-1] | ["reset", key] | ["set", key, value] | ["apply_cloth"]."""
        t = self.tune
        if op[0] == "apply_cloth":
            if not hasattr(self.w, "apply_cloth"):
                return
            self.w.apply_cloth(t)          # rebuilds the shirt with the new physics (several seconds in Isaac)
            t.mark_cloth_applied()
            self.undo.clear()
            self.episode += 1
            self._new_episode()
            self.score = self.best = self.w.score() or {}
            self.write({"reset": True, "tune_cloth": {k: t[k] for k in t.applied_cloth}})
            self.msg = "new cloth settings applied - fresh shirt"
            return
        k = op[1]
        ok = {"adjust": lambda: t.adjust(k, int(op[2])), "reset": lambda: t.reset(k),
              "set": lambda: t.set(k, op[2])}.get(op[0], lambda: False)()
        if ok:
            self.write({"tune": {k: t[k]}})
            if k in t.applied_cloth and t.cloth_pending():
                self.msg = "cloth setting changed - press C in the panel to rebuild the shirt with it"

    # ---------------------------------------------------------------- keys
    def key(self, k):
        """k: one of 'T','Z','R','A','ENTER','S','+','-','[',']'."""
        if k == "T":
            self.tandem = (self.tandem + 1) % 3
            self.msg = f"tandem: {TANDEM[self.tandem]}" + (" - either mouse moves BOTH arms" if self.tandem else "")
        elif k == "Z":
            self.do_undo()
        elif k == "R":
            self.do_reset()
        elif k == "A":
            for a in ARMS:
                if self.w.held[a]:
                    self.w.release(a)
                self.phase[a] = "rising"
            self.devmap, self.assigning = {}, True
            self.msg = "click the LEFT button on the mouse for the LEFT (orange) arm"
        elif k == "ENTER" and self.assigning and self.devmap:
            self.assigning = False
            self.msg = "one mouse: press S to switch it to the other arm, T to drive both"
        elif k == "S":
            self.devmap = {h: ("R" if a == "L" else "L") for h, a in self.devmap.items()}
            self.msg = "arms swapped"
        elif k in ("+", "-") and self.last_moved in self.gain:
            f = 1.25 if k == "+" else 0.8
            self.gain[self.last_moved] = float(np.clip(self.gain[self.last_moved] * f, 2e-5, 5e-4))
            self.msg = f"{self.devmap.get(self.last_moved, '?')} mouse speed {self.gain[self.last_moved] * 1e5:.0f}"
        elif k in ("[", "]"):  # carry height of the arm whose mouse moved last (both in tandem)
            arms = ARMS if (self.tandem or self.last_moved not in self.devmap) else (self.devmap[self.last_moved],)
            for b in arms:
                self.carry[b] = float(np.clip(self.carry[b] + (0.005 if k == "]" else -0.005), 0.01, 0.08))
            self.msg = "carry height " + ", ".join(f"{b} {self.carry[b] * 100:.1f} cm" for b in ARMS)

    # ---------------------------------------------------------------- one control tick
    def tick(self, mice, dt=1 / 60, is_external=lambda h: True):
        """mice: {handle: (dx, dy, wheel, left_held, right_held, events)}. Advances the world by dt."""
        w = self.w
        if not self._tune_logged:          # every demo log starts with the settings it was played with
            self.write({"tune": self.tune.values()}); self._tune_logged = True
        self.tick_n += 1
        self.time += dt
        now = time.time()
        for h, (dx, dy, wh, l, r, ev) in mice.items():
            if dx or dy or wh or ev:
                self.last_input[h] = now
            if dx or dy:
                self.last_moved = h
        if self.assigning:
            for h, (dx, dy, wh, l, r, ev) in mice.items():
                if "left_down" in ev and h not in self.devmap:
                    if not is_external(h):
                        self.msg = "that was the touchpad - click one of the USB mice"; continue
                    self.devmap[h] = "L" if not self.devmap else "R"
                    self.gain.setdefault(h, 8e-5)
                    self.write({"assign": {str(k): v for k, v in self.devmap.items()}})
                    if len(self.devmap) == 1:
                        self.msg = "now click the OTHER mouse for the RIGHT (green) arm   (Enter = one mouse)"
                    else:
                        self.assigning = False
                        self.msg = "go!  hold LEFT = grab and lift   hold RIGHT = drag along table   T = tandem"
        else:
            self._apply_mice(mice)
        self._move_arms(dt)
        w.step(dt)
        self._log_frame()

    def _apply_mice(self, mice):
        moves = {a: np.zeros(2) for a in ARMS}
        presses, releases = [], []
        for h, (dx, dy, wh, l, r, ev) in mice.items():
            a = self.devmap.get(h)
            if a is None:
                continue
            g = self.gain.get(h, 8e-5)
            d = np.array([dx * g, -dy * g])
            if self.tandem == 0:
                moves[a] += d
            else:  # either mouse drives both arms
                moves["L"] += d
                moves["R"] += d * (np.array([-1, 1]) if self.tandem == 2 else 1)
            if wh:  # wheel turns the jaws (and the cloth they hold), wheel_deg per notch
                for b in (ARMS if self.tandem else (a,)):
                    sgn = -1 if (self.tandem == 2 and b != a) else 1
                    self.yaw_cmd[b] += sgn * wh * np.radians(self.tune["wheel_deg"])
            for e, kind in (("left_down", "lift"), ("right_down", "slide"), ("middle_down", "stack")):
                if e in ev:
                    presses += [(b, kind) for b in (ARMS if self.tandem else (a,))]
            if "left_up" in ev or "right_up" in ev or "middle_up" in ev:
                releases += list(ARMS if self.tandem else (a,))
        for a in ARMS:
            c = self.cursor[a]
            c[0] = np.clip(c[0] + moves[a][0], *self.xlim)
            c[1] = np.clip(c[1] + moves[a][1], *self.ylim)
        if presses and any(self.phase[b] in ("hover", "rising") for b, _ in presses):
            self.undo.append(self.snapshot())
            del self.undo[:-30]
        for b, kind in presses:
            if self.phase[b] in ("hover", "rising"):
                self.phase[b], self.mode[b], self.t_phase[b] = "down", kind, 0.0
        for b in set(releases):
            if self.phase[b] == "down":
                self.phase[b], self.t_phase[b] = "rising", 0.0
            elif self.phase[b] in ("carry", "closing"):
                self.phase[b], self.t_phase[b] = ("place" if self.phase[b] == "carry" else "drop_after_close"), 0.0

    def _busy(self, a):
        b = getattr(self.w, "busy", None)
        return bool(b(a)) if b else False

    def _resync(self, a):
        """After a scripted grab/release move (Isaac) the arm is somewhere else: continue from where it really is,
        instead of snapping back to the pre-move aim point."""
        self.cmd[a] = self.w.tip(a)
        self.qcmd[a] = self.w.q(a)
        self.cursor[a][:2] = self.cmd[a][:2]

    def _z_target(self, a):
        w, c = self.w, self.cursor[a]
        ph = self.phase[a]
        if ph in ("closing", "opening", "drop_after_close"):
            return self.cmd[a][2]                       # hold still while the jaws close / open
        top = w.surface_under(c[:2])
        if ph == "down":
            return top + self.tune["down_offset"]
        if ph == "place":
            return top + self.tune["place_offset"]
        if ph == "carry" and self.mode[a] == "slide":
            return top + self.tune["slide_h"]
        return top + self.carry[a]

    def _reachable(self, a, p, seeds):
        # the aim point did not move (arm resting, no input): reuse the last answer instead of solving IK again
        key = (round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5), round(self.roll[a], 4), seeds)
        cache = self.__dict__.setdefault("_ik_cache", {})
        q_in = np.asarray(self.qcmd[a], float)
        hit = cache.get(a)
        if hit and hit[0] == key and np.allclose(hit[1], q_in) and hit[2] == self.tune["max_tilt"]:
            return None if hit[3] is None else hit[3].copy()
        q, err, tilt = self.w.ik(a, p, q_in, self.roll[a], seeds=seeds)
        q = q if (err < 0.004 and tilt < self.tune["max_tilt"]) else None
        cache[a] = (key, q_in.copy(), self.tune["max_tilt"], None if q is None else np.asarray(q, float).copy())
        return q

    def _move_arms(self, dt):
        w = self.w
        choice, zts = {}, {}
        for a in ARMS:  # propose this tick's aim point; if the arm can't reach it, find the closest thing it can do
            self.t_phase[a] += dt
            zt = zts[a] = self._z_target(a)
            c = self.cursor[a]
            holding_still = self.phase[a] in ("closing", "opening", "drop_after_close")
            d_xy = np.zeros(3) if holding_still else np.r_[c[:2] - self.cmd[a][:2], 0.0]
            n = np.linalg.norm(d_xy)
            sp = getattr(self, "speed_scale", 1.0)        # >1 when the sim runs slower than real time (see server.py)
            v_xy = self.tune["arm_speed"] * sp
            if n > v_xy * dt:
                d_xy *= v_xy * dt / n
            vz = sp * (self.tune["vz_fast"] if self.phase[a] in ("down", "place", "rising") else self.tune["vz_carry"])
            dz = np.r_[0, 0, np.clip(zt - self.cmd[a][2], -vz * dt, vz * dt)]
            base = self.cmd[a]
            floor = w.surface_under(base[:2]) + 0.012   # never auto-lower below 1.2 cm over the cloth
            cands = [("full", base + d_xy + dz)]
            for k in (1, 2, 3):  # reach further by lowering the gripper (a raised arm reaches less far)
                p = base + d_xy + dz - np.r_[0, 0, 0.01 * k]
                if p[2] >= floor and p[2] < base[2] + dz[2]:
                    cands.append(("lowered", p))
            if not self.tandem and n > 1e-6:  # slide along the edge of the reachable area instead of freezing
                cands += [("slide", base + np.r_[d_xy[0], 0, 0] + dz), ("slide", base + np.r_[0, d_xy[1], 0] + dz)]
            choice[a] = (None, base, None)
            for i, (kind, p) in enumerate(cands):
                q = self._reachable(a, p, seeds=(i == 0))
                if q is not None:
                    choice[a] = (kind, p, q)
                    break
        both_stop = self.tandem and any(choice[a][0] is None for a in ARMS)
        for a in ARMS:
            kind, p, q = choice[a]
            if kind is not None and not both_stop:
                self.cmd[a], self.qcmd[a] = p, q
            if kind in ("slide", None):  # at the reach limit: the aim point stays with the arm instead of running off
                self.cursor[a][:2] = self.cmd[a][:2]
            self.reach_limited[a] = kind != "full"
            if kind is None:
                self.reach_flash[a] = 0.5
            self.reach_flash[a] = max(0.0, self.reach_flash[a] - dt)
            # wrist: point the jaws where the wheel says (held cloth turns with them)
            self.roll[a], achieved = w.roll_for_yaw(a, self.qcmd[a], self.yaw_cmd[a])
            if abs(((self.yaw_cmd[a] - achieved + np.pi) % (2 * np.pi)) - np.pi) > 1e-3:
                self.yaw_cmd[a] = achieved  # at the wrist's joint limit: don't wind up past it
            w.set_arm(a, self.qcmd[a], self.roll[a])
        for a in ARMS:
            zt = zts[a]
            # phase transitions (time-outs guarantee nothing ever gets stuck)
            tip_z = w.tip(a)[2]
            arrived = abs(tip_z - zt) < 0.003 and abs(self.cmd[a][2] - zt) < 0.001
            ph = self.phase[a]
            if ph == "down" and (arrived or self.t_phase[a] > 0.8):
                w.grab(a, all_layers=self.mode[a] == "stack")   # instant (MuJoCo) or a closing move (Isaac)
                self.phase[a], self.t_phase[a] = "closing", 0.0
            elif ph in ("closing", "drop_after_close") and (not self._busy(a) or self.t_phase[a] > 3.0):
                self._resync(a)
                n = w.held[a]
                self.n_grabs += 1
                self.write({"event": "grab", "arm": a, "mode": self.mode[a], "n": n,
                            "tip_cm": list(np.round(w.tip(a) * 100, 2)), "score": self.score, "cloth_cm": self.cloth_state()})
                if ph == "drop_after_close":  # button was released while the jaws were still closing
                    self.phase[a], self.t_phase[a] = "place", 0.0
                elif n or getattr(w, "busy", None):
                    # Isaac (grab assist): the jaws stay shut until the player lets go, even when the pad sensor counts
                    # no cloth (the assist test lifted the cloth in 20/20 grabs; the sensor saw nothing in 2 of them)
                    self.phase[a], self.t_phase[a] = "carry", 0.0
                    self.msg = (f"{a} holding the cloth - move it, let go to drop" if self.mode[a] == "lift"
                                else f"{a} dragging along the table")
                else:
                    w.release(a)
                    self.phase[a], self.t_phase[a] = "rising", 0.0
                    self.msg = f"{a} missed - no cloth under the gripper"
                    if self.undo:
                        self.undo.pop()
            elif ph == "place" and (arrived or self.t_phase[a] > 0.8):
                w.release(a)                      # instant (MuJoCo) or open + back off + slow lift (Isaac)
                self.phase[a], self.t_phase[a] = "opening", 0.0
            elif ph == "opening" and (not self._busy(a) or self.t_phase[a] > 3.0):
                self._resync(a)
                self.phase[a], self.t_phase[a] = "rising", 0.0
                self.score = w.score() or {}
                if self.score.get("fold_score", 0) > self.best.get("fold_score", 0):
                    self.best = self.score
                self.write({"event": "drop", "arm": a, "tip_cm": list(np.round(w.tip(a) * 100, 2)),
                            "score": self.score, "cloth_cm": self.cloth_state()})
                fs = self.score.get("fold_score")
                self.msg = f"{a} dropped" + (f" - fold score {fs:.0f}" if fs is not None else "")
                self.pending_flow = (self.time, w.cloth.x.copy(), self.n_grabs)
            elif ph == "rising" and (arrived or self.t_phase[a] > 0.8):
                self.phase[a] = "hover"
        self._measure_flowback()

    def _measure_flowback(self):
        """How far the cloth moves in the 1.5 s after a drop (the 'flowback'): shown on screen and logged,
        so it can be learned from. Measured early if you grab again sooner (but at least 0.5 s after)."""
        if self.pending_flow is None:
            return
        t0, x0, grabs0 = self.pending_flow
        age = self.time - t0
        if age < 1.5 and self.n_grabs == grabs0:
            return
        self.pending_flow = None
        if age < 0.5:
            return
        d = np.linalg.norm(self.w.cloth.x - x0, axis=1) * 1000
        self.last_flowback = (float(d.mean()), float(np.percentile(d, 95)))
        self.write({"event": "flowback", "avg_mm": round(self.last_flowback[0], 2),
                    "worst5_mm": round(self.last_flowback[1], 2), "seconds": round(age, 2)})

    def _log_frame(self):
        if self.time < self.next_frame_t:   # 20 records per simulated second
            return
        self.next_frame_t = self.time + 0.05
        w = self.w
        rec = {"frame": True,
               "cursor_cm": {a: list(np.round(self.cursor[a] * 100, 2)) for a in ARMS},
               "target_cm": {a: list(np.round(self.cmd[a] * 100, 2)) for a in ARMS},
               "tip_cm": {a: list(np.round(w.tip(a) * 100, 2)) for a in ARMS},
               "q": {a: list(np.round(self.qcmd[a], 4)) for a in ARMS},
               "grip": {a: bool(w.held[a]) for a in ARMS},
               "phase": dict(self.phase), "mode": dict(self.mode),
               "carry_cm": {a: round(self.carry[a] * 100, 2) for a in ARMS}, "tandem": TANDEM[self.tandem],
               "jaw_deg": {a: round(float(np.degrees(self.yaw_cmd[a])), 1) for a in ARMS},
               "roll": {a: round(self.roll[a], 4) for a in ARMS},
               "reach_limited": dict(self.reach_limited)}
        period = getattr(self.w, "CLOTH_LOG_S", 1.0)   # None: only on grab/drop (Isaac: a full record stalls ~24 ms)
        if period and self.time >= self.next_cloth_t:
            self.next_cloth_t = self.time + period
            rec["cloth_cm"] = self.cloth_state()
        self.write(rec)

    def hud(self):
        """Small dict for a remote display (no cloth arrays)."""
        now = time.time()
        arms = {}
        for a in ARMS:
            hs = [h for h, b in self.devmap.items() if b == a]
            arms[a] = dict(mouse=("none" if not hs else ("moving" if now - self.last_input.get(hs[0], 0) < 0.4 else "idle")),
                           speed=round(self.gain.get(hs[0], 0) * 1e5) if hs else None, phase=self.phase[a],
                           carry_cm=round(self.carry[a] * 100, 1), jaw_deg=round(float(np.degrees(self.yaw_cmd[a]))),
                           reach_limited=bool(self.reach_limited[a]), held=bool(self.w.held[a]),
                           aim=[round(float(v), 4) for v in self.cmd[a]],
                           cursor=[round(float(v), 4) for v in self.cursor[a][:2]])
        return dict(msg=self.msg, tandem=TANDEM[self.tandem], assigning=self.assigning, arms=arms, episode=self.episode,
                    flowback=self.last_flowback, score={k: v for k, v in self.score.items() if not isinstance(v, (list, dict))},
                    tune=self.tune.values(), cloth_pending=self.tune.cloth_pending(),
                    mice={str(h): [a, self.gain.get(h, 8e-5)] for h, a in self.devmap.items()},
                    lim=[list(self.xlim), list(self.ylim)])

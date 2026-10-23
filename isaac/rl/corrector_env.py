"""RL corrector for the half fold (docs/results.md, 2026-10-22: the folds that pass are settled folds that one push
tipped over; a scripted push is close to a coin flip and once broke a fold that already passed).

An episode starts from a settled scripted fold (latch + g-fold base, both arms retreated) and makes up to K decisions.
Each decision is, per arm, a push of the top corner (xy, PUSH_SCALE m per unit) at a press depth (PRESS_RANGE), done
by isaac.experts.place_servo.IsaacPlaceServoExpert.start_touchup, or stop. After the pushes the fold resettles and is scored:
the reward is the change in `score_s` (log mean and edge against the bar), plus PASS_BONUS for ending on a passing
fold; a push that drags the bottom layer DRAG_FAIL or blows the cloth up ends the episode with -FAIL.

The zero action pushes nothing and never stops (the settled base after K no-ops); `scripted_action` is the scripted
touch-up (push each corner onto its partner at -2 mm, stop once both are within 8 mm or the fold passes).
"""

from __future__ import annotations

import math

import numpy as np

from isaac.rl.precision_env import ARMS, FOLDS, PAIRS, SETTLE_STEPS, gate

K = 3                          # decisions per episode
PUSH_SCALE = 0.03              # m of push per unit action (x, y)
PRESS_RANGE = (-0.004, 0.002)  # m above the top corner, action -1 .. +1 (start_touchup's press)
MIN_PUSH = 0.004               # m: a shorter push is skipped
PUSH_STEPS_MAX = 220           # sim steps per decision (pushes and resettle)
DRAG_FAIL = 0.10               # m of bottom-layer drift since the episode began
FAIL = 5.0
PASS_BONUS = 10.0
CORNER = {"left_": (10, 0), "right_": (120, 110)}      # (top corner, its partner)
ACT_DIM = 7                    # per arm dx, dy, press; stop
STOP_ABOVE = 0.5               # stop = action[6] > STOP_ABOVE (c1: at 0 the untrained policy's noise stopped half the time)


def _edge_groups():
    from cloth_fold_rl.fold_alignment import groups
    members, edge = groups(11, FOLDS)
    return [m for m, e in zip(members, edge) if e and len(m) > 1]


EDGE_GROUPS = _edge_groups()
OBS_DIM = 3 * len(PAIRS) + 2 * len(CORNER) + len(EDGE_GROUPS) + 2 + 1


def score_s(cloth):
    """(log score against the bar, fold_alignment.score dict)."""
    from cloth_fold_rl.fold_alignment import EDGE_BAR, MEAN_BAR, score
    s = score(cloth, folds=FOLDS)
    v = -float(np.clip(math.log(max(s["mean"], 1e-4) / MEAN_BAR), -1.0, 4.0))
    v -= float(np.clip(math.log(max(s["edge_max"], 1e-4) / EDGE_BAR), -1.0, 4.0))
    return v, s


def features(cloth, k):
    """(OBS_DIM,) float32: pair offsets, corner gaps (partner - corner, xy), outline spreads, mean / edge (dm), k / K."""
    from cloth_fold_rl.fold_alignment import spreads
    _, s = score_s(cloth)
    parts = [(cloth[PAIRS[:, 0]] - cloth[PAIRS[:, 1]]).ravel() * 10.0]
    parts += [(cloth[part, :2] - cloth[top, :2]) * 10.0 for top, part in CORNER.values()]
    parts += [spreads(cloth, EDGE_GROUPS) * 10.0, [s["mean"] * 10.0, s["edge_max"] * 10.0], [k / K]]
    return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)


def decode(action):
    """action (7,) -> ({arm: (push xy m, press m)}, stop)."""
    a = np.clip(np.asarray(action, float), -1.0, 1.0)
    lo, hi = PRESS_RANGE
    pushes = {p: (a[3 * k:3 * k + 2] * PUSH_SCALE, lo + (a[3 * k + 2] + 1.0) / 2.0 * (hi - lo))
              for k, p in enumerate(ARMS)}
    return pushes, bool(a[6] > STOP_ABOVE)


def scripted_action(cloth, tol=0.008, press=-0.002):
    """The scripted touch-up as an action: push each corner onto its partner; stop when both are within tol or the
    fold passes."""
    _, s = score_s(cloth)
    lo, hi = PRESS_RANGE
    a = np.zeros(ACT_DIM)
    near = True
    for k, (top, part) in enumerate(CORNER.values()):
        off = cloth[part, :2] - cloth[top, :2]
        near &= bool(np.linalg.norm(off) <= tol)
        a[3 * k:3 * k + 2] = np.clip(off / PUSH_SCALE, -1.0, 1.0)
        a[3 * k + 2] = 2.0 * (press - lo) / (hi - lo) - 1.0
    a[6] = 1.0 if (near or s["passed"]) else -1.0
    return a


class CorrectorBatch:
    """B copies (isaac.rl.precision_env.PrecisionBatch): settled-fold snapshots, then corrector episodes."""

    def __init__(self, n, cameras=None):
        import isaac.rl.precision_env as pe
        pe.TOUCH_TOL = 0.0                   # the window's own touch-up is off: the corrector does the pushing
        self.pe = pe
        self.pb = pe.PrecisionBatch(n, cameras=cameras)
        self.n = n

    def settled_snapshots(self, seeds):
        """Fold each seed with the base to a settled fold (zero residual) and snapshot it there (None: no window)."""
        snaps, _ = self.pb.scripted_to_window(seeds)
        return self.settle(snaps, seeds)

    def settle(self, snaps, seeds):
        """From carry-start snapshots, run the base window and snapshot each copy the step it settles; settle_cut[i] is
        the length of pb.frames[i] then (demo video)."""
        pb = self.pb
        ok = [i for i in range(self.n) if snaps[i] is not None]
        out, self.settle_cut = [None] * self.n, [0] * self.n
        if not ok:
            return out
        pb.reset_window([snaps[i] if snaps[i] is not None else snaps[ok[0]] for i in range(self.n)])
        while any(pb.state[i]["alive"] for i in range(self.n)):
            _, _, _, _, _, infos = pb.step(np.zeros((self.n, self.pe.ACT_DIM)))
            for i in ok:
                if infos[i] is not None and infos[i]["reason"] == "settled":
                    out[i] = pb.snapshot(i, int(seeds[i]))
                    self.settle_cut[i] = len(pb.frames[i]) if pb.frames is not None else 0
        return out

    def _cloth(self, i):
        return self.pb.subs[i].cloth_positions()

    def reset(self, snaps):
        for i, snap in enumerate(snaps):
            self.pb.restore(i, snap)
        self.k = np.zeros(self.n, int)
        self.alive = np.ones(self.n, bool)
        self.start = [self._cloth(i).copy() for i in range(self.n)]
        self.s_prev = [score_s(self._cloth(i))[0] for i in range(self.n)]
        self.prev_cloth = [self._cloth(i).copy() for i in range(self.n)]
        return np.stack([features(self._cloth(i), 0) for i in range(self.n)])

    def _final(self, i, reason):
        cloth = self._cloth(i)
        _, s = score_s(cloth)
        sub = self.pb.subs[i]
        g = gate(cloth, self.prev_cloth[i], [sub.gripper_position(p) for p in ARMS],
                 [sub.joint_positions(p)[5] for p in ARMS])
        self.alive[i] = False
        return {"reason": reason, "decisions": int(self.k[i]), "mean": round(s["mean"], 4),
                "edge_max": round(s["edge_max"], 4), "passed": bool(s["passed"]), "gate": g,
                "passed_valid": bool(s["passed"] and g["valid"]),
                "cloth": np.round(self._cloth(i), 4).tolist()}

    def step(self, actions):
        """actions (n, 7) -> obs, reward, terminal, done, alive_before, infos (alive_before False: padding)."""
        pb, alive = self.pb, self.alive.copy()
        rew, term, infos = np.zeros(self.n), np.zeros(self.n, bool), [None] * self.n
        pushing = []
        for i in np.flatnonzero(alive):
            pushes, stop = decode(actions[i])
            if stop:
                self.prev_cloth[i] = self._cloth(i).copy()          # nothing moved since the last resettle
                infos[i] = self._final(i, "stopped")
                rew[i], term[i] = (PASS_BONUS if infos[i]["passed_valid"] else 0.0), True
                continue
            for p, (d, press) in pushes.items():
                if np.linalg.norm(d) > MIN_PUSH:
                    pb.experts[i][p].start_touchup(d, press=press)
            self.k[i] += 1
            pushing.append(i)
        settle = {i: 0 for i in pushing}
        for _ in range(PUSH_STEPS_MAX if pushing else 0):
            for i in pushing:
                self.prev_cloth[i] = self._cloth(i).copy()
            pb._scripted_step()
            for i in pushing:
                done = all(e.PHASES[e.phase] == "done" for e in pb.experts[i].values())
                settle[i] = settle[i] + 1 if done else 0
            if all(v >= SETTLE_STEPS for v in settle.values()):
                break
        for i in pushing:
            cloth = self._cloth(i)
            if pb.subs[i]._failed() or self.pe.drift(cloth, self.start[i]) > DRAG_FAIL:
                infos[i] = self._final(i, "unstable" if pb.subs[i]._failed() else "dragged")
                rew[i], term[i] = -FAIL, True
                continue
            s_new = score_s(cloth)[0]
            rew[i], self.s_prev[i] = s_new - self.s_prev[i], s_new
            if self.k[i] >= K:
                infos[i] = self._final(i, "decisions")
                rew[i] += PASS_BONUS if infos[i]["passed_valid"] else 0.0
                term[i] = True
        obs = np.stack([features(np.nan_to_num(self._cloth(i)), self.k[i]) for i in range(self.n)])
        return obs, np.nan_to_num(rew), term, ~self.alive & alive, alive, infos

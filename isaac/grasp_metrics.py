"""Per-arm friction-grasp metrics from a per-step trace (track G, roadmap IG.1-IG.2). Numpy only, so the bench's
summaries can be recomputed and tested outside the Isaac venv.

A trace is what isaac/grasp_bench.py records for one arm over one episode, one row per control step (read before
the step's action): the expert's phase, the gripperframe site, the carried corner, the jaw (gripper joint) angle,
and the arm's anchor vertex (the corner that must not be dragged). From it:

  acquired  the corner went up with the gripper: when the expert enters carry (the lift waypoints are done) the
            corner is at least LIFT_MIN above where it rested and within HOLD_DIST of the site
  held      acquired, and the corner stayed within HOLD_DIST of the site from then until the expert opened the jaw
  placed    after the release, the corner lies within SUCCESS_DIST of its goal (the env's own placement test),
            read SETTLE_WAIT steps after the jaw opened, or at the last recorded step if the episode ended sooner
  released  the jaw let go cleanly: the corner was held, the expert opened the jaw, and by the end of its retreat the corner had moved less than
            RELEASE_MOVE horizontally from where it was when the jaw opened and lies within RELEASE_Z of its rest
            height (not carried up, flung or dragged by the opening jaw or the retreat; falling straight down is fine)
  anchor drift  the largest distance of the arm's anchor vertex from its start over the episode

Each rate is unconditional (out of every episode), so a corner that was never acquired counts against held, placed
and released too.
"""

from __future__ import annotations

import math

import numpy as np

LIFT_MIN = 0.02          # m the corner must have risen by the end of the lift
# m between the corner and the gripperframe site while held. The site is the fixed fingertip and a held corner rides
# 2.5-3.8 cm from it, up between the pads (IG.1 baseline, 20 seeds); a dropped one jumps to 6 cm and more
HOLD_DIST = 0.05
SUCCESS_DIST = 0.05      # = cloth_fold_rl.fold_env.SUCCESS_DIST
SETTLE_WAIT = 15         # = cloth_fold_rl.quarter_fold_expert.SETTLE_WAIT
RELEASE_MOVE = 0.03      # m the corner may move between the jaw opening and the end of the retreat
RELEASE_Z = 0.025        # m above its rest height the released corner may lie (it lies on the near half)
METRICS = ("acquired", "held", "placed", "released")


def _first(phases, name, start=0):
    for t in range(start, len(phases)):
        if phases[t] == name:
            return t
    return None


def arm_metrics(trace: dict, goal, rest_z: float) -> dict:
    """trace: {"phase": [str], "site": (T, 3), "corner": (T, 3), "anchor": (T, 3)} for one arm; goal: (3,) the corner's
    goal; rest_z: the corner's height at the start. Returns the four flags plus diagnostics."""
    phases = list(trace["phase"])
    site = np.asarray(trace["site"], float)
    corner = np.asarray(trace["corner"], float)
    anchor = np.asarray(trace["anchor"], float)
    goal = np.asarray(goal, float)
    dist = np.linalg.norm(corner - site, axis=1)
    t_carry = _first(phases, "carry")
    t_rel = _first(phases, "release")
    out = {"t_carry": t_carry, "t_release": t_rel,
           "anchor_drift": float(np.max(np.linalg.norm(anchor - anchor[0], axis=1))) if len(anchor) else 0.0}
    rise = float(corner[t_carry, 2] - rest_z) if t_carry is not None else None
    acquired = t_carry is not None and rise >= LIFT_MIN and dist[t_carry] < HOLD_DIST
    out["lift_rise"] = rise
    held = False
    if acquired and t_rel is not None and t_rel > t_carry:
        out["carry_max_dist"] = float(np.max(dist[t_carry:t_rel]))
        held = out["carry_max_dist"] < HOLD_DIST
        if not held:
            out["drop_t"] = int(t_carry + np.argmax(dist[t_carry:t_rel] >= HOLD_DIST))
    elif acquired:
        out["carry_max_dist"] = float(np.max(dist[t_carry:]))
    placed = released = False
    if t_rel is not None:
        t_read = min(t_rel + SETTLE_WAIT, len(phases) - 1)
        out["place_err"] = float(np.linalg.norm(corner[t_read] - goal))
        placed = out["place_err"] < SUCCESS_DIST
        t_done = _first(phases, "done", t_rel)
        t_end = t_done if t_done is not None else len(phases) - 1
        # horizontal only: a corner let go a few cm above the table falls straight down, which is a clean release
        out["release_move"] = float(np.linalg.norm(corner[t_end, :2] - corner[t_rel, :2]))
        out["release_z"] = float(corner[t_end, 2] - rest_z)
        # a corner dropped before the jaw opened was never released by it
        released = held and out["release_move"] < RELEASE_MOVE and out["release_z"] < RELEASE_Z
    out.update(acquired=bool(acquired), held=bool(held), placed=bool(placed), released=bool(released))
    out["failed_at"] = next((m for m in METRICS if not out[m]), None)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, in percent (= imitation.verify.wilson)."""
    if n <= 0:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h)))


def summarize(rows: list[dict]) -> dict:
    """rows: one per episode, {"seed", "arms": {prefix: arm_metrics(...)}}. Per arm and metric: k, n, rate %, Wilson
    95 %; plus the anchor drift (mean / max) and where each failure happened."""
    out = {"n": len(rows), "seeds": sorted(r["seed"] for r in rows), "arms": {}}
    prefixes = sorted({p for r in rows for p in r["arms"]})
    for p in prefixes:
        arm = [r["arms"][p] for r in rows]
        stats = {}
        for m in METRICS:
            k = sum(bool(a[m]) for a in arm)
            lo, hi = wilson(k, len(arm))
            stats[m] = {"k": k, "n": len(arm), "rate": round(100.0 * k / max(len(arm), 1), 1),
                        "wilson": [round(lo, 1), round(hi, 1)]}
        drift = [a["anchor_drift"] for a in arm]
        stats["anchor_drift_m"] = {"mean": round(float(np.mean(drift)), 4), "max": round(float(np.max(drift)), 4)}
        fails = {}
        for a in arm:
            if a["failed_at"]:
                fails[a["failed_at"]] = fails.get(a["failed_at"], 0) + 1
        stats["first_failure"] = fails
        out["arms"][p] = stats
    return out


def meets_ig2(summary: dict, held_bar: float = 98.0, placed_bar: float = 95.0) -> bool:
    """IG.2's per-block bar: per arm, acquired / held / released >= 98 % and placed >= 95 %."""
    for stats in summary["arms"].values():
        for m in METRICS:
            if stats[m]["rate"] < (placed_bar if m == "placed" else held_bar):
                return False
    return bool(summary["arms"])

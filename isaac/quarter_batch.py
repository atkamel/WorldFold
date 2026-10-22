"""The quarter fold on the GPU friction profile as three scripted stages over B copies (docs/results.md 2026-10-22).

No two-armed fold 2 is collision-free on any layout, and the one-armed fold 2 from where fold 1 leaves the sheet
slipped 8/8 (38 deg pinch at full reach). So between the folds both arms DRAG the half sheet by DRAG (m): the left arm
pulls the west short edge's middle (cloth_2 under cloth_8), the right arm the east one (cloth_112 under cloth_118),
flat and low. Then the right arm alone carries the west middle onto the east middle (fold 2) while the left arm
parks at home. Offline (scratchpad/drag_search.py): with DRAG (0.12, 0.04) fold 2's pinch is near vertical at 0.29 m.

    python isaac/quarter_batch.py --copies 8 --seeds 0:8 --video --out outputs/isaac/quarter/drag1

Run with WORLDFOLD_PLACE_PARAMS for fold 1's base (latch + g-fold 1.2). One row per seed in <out>/episodes.jsonl
(fold-1 alignment, the sheet's move in the drag, the final quarter-fold alignment), a top video per seed with --video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaac.precision_env import ARMS, SETTLE_STEPS, PrecisionBatch, coordinate  # noqa: E402

WEST, EAST = (2, 8), (112, 118)          # the short edges' middles after fold 1 (bottom vertex, top vertex)
DRAG = (0.12, 0.04)
FLAT_ARC = 0.01                          # the drag's lift (IsaacArmExpert.ARC_HEIGHT for its plan) and no corner arc
CAPS = {"fold1": 650, "drag": 300, "fold2": 400}
PARK_STEPS = 40                          # the left arm goes home before fold 2 starts (it sits where fold 2 grasps)


def park_action(q, home, scale):
    """(6,) action moving joints q toward home at the speed limit, jaw open."""
    a = np.zeros(6, dtype=np.float32)
    d = (np.asarray(home, float) - np.asarray(q, float)) / scale
    a[0:5] = d / max(1.0, float(np.max(np.abs(d))))
    a[5] = 1.0
    return a


class QuarterBatch(PrecisionBatch):

    def __init__(self, n, drag=DRAG, cameras=None):
        self.parked = [set() for _ in range(n)]          # before super(): its warm-up steps through _scripted_step
        self.drag = np.r_[np.asarray(drag, float), 0.0]
        super().__init__(n, cameras=cameras)

    def _configure(self, i, p, corners, goal, live_goal, flat):
        e = self.experts[i][p]
        e.corners = tuple(corners)
        e.goal, e.live_goal = goal, live_goal
        for k in ("ARC_HEIGHT", "ARC_FRAC"):
            e.__dict__.pop(k, None)
        if flat:
            e.ARC_HEIGHT, e.ARC_FRAC = FLAT_ARC, 0.0
        e.reset()
        e.release_allowed = False

    def _scripted_step(self, actions=None):
        from mujuco.cloth_params import JOINT_DELTA_SCALE
        for i, sub in enumerate(self.subs):
            moving = {p: e for p, e in self.experts[i].items() if p not in self.parked[i]}
            coordinate(moving)
            acts = [moving[p].act() if p in moving else
                    park_action(sub.joint_positions(p)[:5], sub._home()[:5], JOINT_DELTA_SCALE) for p in ARMS]
            sub.step_submit(self.tasks[i]._expand(np.concatenate(acts)))
        self.batch.advance()
        for sub in self.subs:
            sub.step_collect()
        if self.frames is not None:
            for i, sub in enumerate(self.subs):
                self.frames[i].append(sub.render_rig()["top"].transpose(1, 2, 0))

    def _run_stage(self, cap):
        """Step until every copy's moving arms have been done SETTLE_STEPS steps (or cap); per-copy settled flags."""
        settle = [0] * self.n
        for _ in range(cap):
            self._scripted_step()
            for i in range(self.n):
                moving = [e for p, e in self.experts[i].items() if p not in self.parked[i]]
                settle[i] = settle[i] + 1 if all(e.PHASES[e.phase] == "done" for e in moving) else 0
            if min(settle) >= SETTLE_STEPS:
                break
        return [s >= SETTLE_STEPS for s in settle]

    def run(self, seeds):
        """Fold 1, drag, fold 2 for copy i from seeds[i]; returns one record per copy."""
        from cloth_fold_rl.fold_alignment import QUARTER_FOLDS, score
        from isaac.precision_env import FOLDS
        recs = [{"seed": int(s)} for s in seeds]
        for i, seed in enumerate(seeds):
            self.tasks[i].reset(seed=int(seed))
            self.parked[i] = set()
            task, sub = self.tasks[i], self.subs[i]
            for p, m in self.moves.items():
                self._configure(i, p, m.corners, lambda m=m, t=task: t.goal(m),
                                lambda m=m, s=sub: s.cloth_positions()[m.goal], flat=False)
        ok1 = self._run_stage(CAPS["fold1"])
        for i, sub in enumerate(self.subs):
            s = score(sub.cloth_positions(), folds=FOLDS)
            recs[i]["fold1"] = {"settled": ok1[i], "mean": round(s["mean"], 4), "edge_max": round(s["edge_max"], 4)}
            cloth = sub.cloth_positions()
            w0, e0 = cloth[list(WEST)].mean(axis=0), cloth[list(EAST)].mean(axis=0)
            recs[i]["drag_from"] = {"west": np.round(w0, 4).tolist(), "east": np.round(e0, 4).tolist()}
            self._configure(i, "left_", WEST, lambda g=w0 + self.drag: g, lambda g=w0 + self.drag: g, flat=True)
            self._configure(i, "right_", EAST, lambda g=e0 + self.drag: g, lambda g=e0 + self.drag: g, flat=True)
        okd = self._run_stage(CAPS["drag"])
        for i, sub in enumerate(self.subs):
            cloth = sub.cloth_positions()
            w1, e1 = cloth[list(WEST)].mean(axis=0), cloth[list(EAST)].mean(axis=0)
            w0, e0 = (np.array(recs[i]["drag_from"][k]) for k in ("west", "east"))
            recs[i]["drag"] = {"settled": okd[i], "west_moved": np.round(w1 - w0, 4).tolist(),
                               "east_moved": np.round(e1 - e0, 4).tolist(),
                               "fold1_after": round(score(cloth, folds=FOLDS)["mean"], 4)}
            self.parked[i] = {"left_"}
        for _ in range(PARK_STEPS):           # the right arm is done (idle) while the left one goes home
            self._scripted_step()
        for i, sub in enumerate(self.subs):
            self._configure(i, "right_", WEST, lambda s=sub: s.cloth_positions()[list(EAST)].mean(axis=0),
                            lambda s=sub: s.cloth_positions()[list(EAST)].mean(axis=0), flat=False)
        ok2 = self._run_stage(CAPS["fold2"])
        for i, sub in enumerate(self.subs):
            cloth = sub.cloth_positions()
            s = score(cloth, folds=QUARTER_FOLDS)
            off = cloth[list(WEST)].mean(axis=0) - cloth[list(EAST)].mean(axis=0)
            recs[i]["fold2"] = {"settled": ok2[i], "west_on_east_xy": np.round(off[:2], 4).tolist(),
                                "dropped": self.experts[i]["right_"].dropped}
            recs[i]["final"] = {"mean": round(s["mean"], 4), "edge_max": round(s["edge_max"], 4),
                                "passed": bool(s["passed"]), "unstable": bool(sub._failed())}
            recs[i]["cloth"] = np.round(cloth, 4).tolist()
        return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--seeds", default="0:8")
    ap.add_argument("--drag", default="0.12,0.04")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.argv = sys.argv[:1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qb = QuarterBatch(args.copies, drag=[float(v) for v in args.drag.split(",")],
                      cameras={"top": 400} if args.video else None)
    a, b = (int(v) for v in args.seeds.split(":"))
    for start in range(a, b, qb.n):
        seeds = list(range(start, start + qb.n))
        qb.frames = [[] for _ in range(qb.n)] if args.video else None
        for rec in qb.run(seeds):
            print("QUARTER", json.dumps({k: v for k, v in rec.items() if k != "cloth"}), flush=True)
            with open(out / "episodes.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        if args.video:
            import imageio
            for i, seed in enumerate(seeds):
                imageio.mimwrite(str(out / f"quarter_seed{seed}.mp4"), qb.frames[i] + qb.frames[i][-1:] * 20, fps=20,
                                 macro_block_size=1)
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()

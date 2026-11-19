"""The half fold with the iPhone watching throughout (RGB and LiDAR over Record3D USB), everything recorded.

    python fold_live.py --left arm2 --right arm1 [--grasp-check] [--speed 0.5] [--no-depth] [--dry-run] [--yes]

Needs fold_camera.py calibrate first (camera.json). Then, per run:
  look      the cloth's corners from the first frame, the fold planned from them (fold_camera.plan). With depth, the
            table plane is fitted once and every pixel standing more than 2 cm above it (the arms, lifted cloth) is
            left out of the cloth's outline, so the arms may be in view.
  track     a thread reads every frame, records it, and follows the cloth's near edge: the bottom layer's edge, in
            view through the carry. How far it has moved is how far the bottom layer has slid.
  correct   at SWITCH of the carry the place point moves with the near edge (at most MAX_SHIFT): the fold is planned
            again to the new place and the arms go on along it.
  record    runs/<time>/: every frame (jpg + depth), every control step's targets and measured joints, the look, both
            plans, the grasp check's readings, the correction, and the fold's score from the first and last frames.

Not tracked: the folded-over half (dark on dark) and the held corners (under the grippers).
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

import fold_camera
import fold_score
import fold_traj
import recorder
import replay
import rgbd

HERE = Path(__file__).parent
SWITCH = 0.6          # fraction of the carry at which the place is corrected: the near edge is still in view
MAX_SHIFT = 0.03      # m: the place follows the near edge at most this far
MIN_SHIFT = 0.003     # m: a smaller movement is left alone
EDGE_INSET = 0.03     # m: the near edge is read this far in from the cloth's sides, where the arms come down
SIDES = (("left", "near_left"), ("right", "near_right"))


class Tracker(threading.Thread):
    """Reads frames as they come, records each, and keeps the near edge's latest estimates."""

    def __init__(self, phone, cal, run, plane, x_range):
        super().__init__(daemon=True)
        self.phone, self.cal, self.run_dir, self.plane, self.x_range = phone, cal, run, plane, x_range
        self.grid = fold_camera.search_region(cal)
        self.edges, self.lock, self.stopped = [], threading.Lock(), threading.Event()

    def run(self):
        last = None
        while not self.stopped.is_set():
            f = self.phone.wait()
            if f.t == last:                    # a replayed stream repeating its last frame
                time.sleep(0.01)
                continue
            last = f.t
            self.run_dir.frame(f.t, f.rgb, f.depth, f.K)
            edge = fold_camera.near_edge(f.rgb, self.cal, self.grid, self.x_range, keep_mask(f, self.plane))
            if edge is not None:
                with self.lock:
                    self.edges.append((f.t, *edge))

    def edge_now(self, n=5):
        """(slope, Y at X = 0) of the near edge: the median of the last n estimates, None before the first."""
        with self.lock:
            recent = np.array(self.edges[-n:])
        return None if len(recent) == 0 else tuple(np.median(recent[:, 1:], axis=0))


def keep_mask(f, plane):
    """Pixels with nothing standing above the table (rgbd.flat), or None without depth."""
    if plane is None or f.depth is None:
        return None
    return rgbd.flat(rgbd.height(f.depth, f.K, f.rgb.shape, plane), f.rgb.shape)


class StepLog:
    """replay.play's log: each control step goes straight to the recorder with its targets (model radians)."""

    def __init__(self, run, traj, followers, name):
        self.run, self.traj, self.followers, self.name = run, traj, followers, name

    def append(self, entry):
        row, t, present = entry
        self.run.step(t, row, {side: self.traj["arms"][side]["target"][row] for side in self.followers},
                      {side: [f.to_rad(m, present[side][m]) for m in replay.MOTOR_IDS]
                       for side, f in self.followers.items()}, plan=self.name)


def to_ticks(followers, traj, allow_clamp=False):
    ticks = {}
    for side, f in followers.items():
        ticks[side], stats = replay.convert(f, traj["arms"][side]["target"])
        worst = max(w for _, w in stats.values())
        if worst > replay.CLAMP_REFUSE_DEG and not allow_clamp:
            raise ValueError(f"the {side} arm's limits clamp this fold by {worst:.1f} deg")
    return ticks


def carry_row(marks, fraction):
    return marks["check"] + int(round(fraction * (marks["placed"] - marks["check"])))


def corrected_points(points, corners, ref, edge):
    """`points` with each arm's place moved as far as the near edge has (m, forward), and those movements."""
    moved, shifts = {}, {}
    for side, corner in SIDES:
        x = corners[corner][0]
        dy = float(np.clip((edge[0] * x + edge[1]) - (ref[0] * x + ref[1]), -MAX_SHIFT, MAX_SHIFT))
        pinch, place = points[side]
        moved[side], shifts[side] = (pinch, (place[0] + dy, place[1])), dy
    return moved, shifts


def fold(followers, phone, cal, run, args):
    """One fold on `followers` ({"left": replay.Follower, "right": ...}); returns 0, or 1 if it did not fold."""
    first = phone.wait()
    plane = None
    if first.depth is not None and not args.no_depth:
        plane = rgbd.fit_plane(rgbd.points(first.depth, first.K, first.rgb.shape))
    keep = keep_mask(first, plane)
    try:
        traj, corners, px, points = fold_camera.plan(first.rgb, cal, args.spacing, args.closed, keep,
                                                     args.grasp_check)
        ticks = to_ticks(followers, traj, args.allow_clamp)
    except ValueError as err:
        print(f"Look failed: {err}")
        run.event(time.monotonic(), "look_failed", error=str(err))
        return 1
    run.save("look.png", fold_camera.overlay(first.rgb, cal, px, points, args.spacing))
    run.save("plan_1.json", traj)
    run.event(first.t, "look", corners=corners, points=points, depth=plane is not None, plane=plane)
    print(f"Cloth corners on the table (drawn in {run.path / 'look.png'}):")
    if not fold_camera.report(traj, corners):
        print("Not folding: move the cloth closer.")
        return 1
    xs = [corners[k][0] for k in ("near_left", "far_left", "near_right", "far_right")]
    x_range = (max(xs[:2]) + EDGE_INSET, min(xs[2:]) - EDGE_INSET)
    ref = fold_camera.near_edge(first.rgb, cal, fold_camera.search_region(cal), x_range, keep)
    tracker = Tracker(phone, cal, run, plane, x_range)
    tracker.start()
    if args.dry_run:                           # no motion: record a few seconds of frames and the tracked edge
        time.sleep(args.settle)
        tracker.stopped.set()
        tracker.join(timeout=2)
        run.event(time.monotonic(), "dry_run", reference=ref, edge=tracker.edge_now())
        return 0
    if not args.yes:
        input("Check look.png. Enter to fold, Ctrl-C to stop ")

    for f in followers.values():
        f.connect()
    present = {side: f.read_present() for side, f in followers.items()}
    for side, f in followers.items():
        f.torque_on_holding(present[side])
    motors, marks, progress = replay.MOTOR_IDS, traj["marks"], {}

    def ramp(start, goal):
        steps = [replay.ramp_points(start[s], {m: goal[s][m - 1] for m in motors}) for s in followers]
        n = max(len(p) for p in steps)
        replay.run_poses(followers, [{s: p[min(i, len(p) - 1)] for s, p in zip(followers, steps) if p}
                                     for i in range(n)])

    def player(plan_ticks, plan, name):
        log = StepLog(run, plan, followers, name)
        return lambda rows: replay.play(followers, plan_ticks, plan["dt"], args.speed, motors, progress, rows, log)

    ramp(present, {s: ticks[s][0] for s in followers})
    play, switch = player(ticks, traj, "plan_1"), carry_row(marks, SWITCH)
    if args.grasp_check:
        note = lambda attempt, empty, lifted, held: run.event(   # noqa: E731
            time.monotonic(), "grasp_check", attempt=attempt, empty=empty, lifted=lifted, held=held)
        replay.play_with_grasp_check(followers, marks, switch + 1, play, args.grasp_margin, args.retries, note)
    else:
        play(range(switch + 1))

    edge, moved, shifts = tracker.edge_now(), points, {}
    if ref is not None and edge is not None:
        moved, shifts = corrected_points(points, corners, ref, edge)
    run.event(time.monotonic(), "place_correction", reference=ref, edge=edge, shifts=shifts, points=moved)
    if shifts and max(abs(d) for d in shifts.values()) >= MIN_SHIFT:
        print("Bottom layer moved: place shifted " + ", ".join(f"{s} {d * 100:+.1f} cm" for s, d in shifts.items()))
        try:
            traj2 = fold_traj.fold(moved, args.closed, args.grasp_check)
            ticks2 = to_ticks(followers, traj2, args.allow_clamp)
        except ValueError as err:
            print(f"Corrected place not usable ({err}); keeping the first plan")
            traj2 = None
        if traj2 is not None:
            run.save("plan_2.json", traj2)
            switch2 = carry_row(traj2["marks"], SWITCH)
            ramp({s: {m: ticks[s][switch][m - 1] for m in motors} for s in followers},
                 {s: ticks2[s][switch2] for s in followers})
            traj, ticks, switch = traj2, ticks2, switch2
            play = player(ticks, traj, "plan_2")
    play(range(switch + 1, len(ticks["left"])))

    time.sleep(args.settle)
    tracker.stopped.set()
    tracker.join(timeout=2)
    last = phone.latest()
    try:
        result = fold_score.score(first.rgb, last.rgb, cal["side"], cal["near"], cal["dark"],
                                  cal.get("colour", "dark"))
        print("Fold score (cm):", json.dumps({k: round(v, 2) for k, v in result.items()}))
    except Exception as err:        # a score is a bonus: an arm in the last frame must not fail the run
        result = {"error": str(err)}
    run.event(time.monotonic(), "score", **result)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--device", type=int, default=0, help="Record3D device index")
    ap.add_argument("--spacing", type=float, default=0.30, help="between the base origins (m)")
    ap.add_argument("--closed", type=float, default=-0.17, help="jaw target at the pinch (rad)")
    ap.add_argument("--speed", type=replay.speed_arg, default=0.5)
    ap.add_argument("--grasp-check", action="store_true")
    ap.add_argument("--grasp-margin", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--allow-clamp", action="store_true")
    ap.add_argument("--no-depth", action="store_true", help="RGB only: the arms must be out of the cloth's outline")
    ap.add_argument("--dry-run", action="store_true", help="look, plan and record, no motion")
    ap.add_argument("--yes", action="store_true", help="do not wait for Enter after the look")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds before the last frame")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    args = ap.parse_args(argv)
    cal = json.loads(fold_camera.CAMERA.read_text())
    if "arms" in cal:       # the edge tracking and the place correction here work in the cloth-made table frame
        sys.exit("camera.json was calibrated from the arms (arm_calibration.py), which fold_live.py does not read "
                 "yet: use fold_camera.py fold.")
    followers = {"left": replay.Follower(args.left), "right": replay.Follower(args.right)}
    phone = rgbd.Phone(args.device)
    phone.start()
    progress = {"step": "before playing"}
    with recorder.Recorder(args.runs, {"args": vars(args), "camera": cal}) as run:
        try:
            return fold(followers, phone, cal, run, args)
        except (KeyboardInterrupt, OSError) as err:
            reason = "Ctrl-C" if isinstance(err, KeyboardInterrupt) else f"bus error: {err}"
            run.event(time.monotonic(), "stopped", reason=reason)
            print(f"\nStopped ({reason}) at step {progress['step']}. No more targets sent; torque left on, arms hold.")
            return 1
        finally:
            phone.stop()
            print("Run recorded in", run.path)


if __name__ == "__main__":
    sys.exit(main())

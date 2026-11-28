"""The camera's map from the arms themselves. Each arm touches the table at known points of its own frame, a picture
is kept of every touch, the fingertip is clicked in each picture, and a pixel-to-arm-frame map is fitted per arm.
Nothing here looks at the cloth, so the cloth may lie anywhere afterwards and the bases may stand anywhere, at any
angle: fold_camera.py reads each arm's pinch and place points straight through that arm's own map.

    python arm_calibration.py touch --arm left --left arm2 --right arm1 --speed 0.3 --torque-limit 50
    python arm_calibration.py touch --arm right --left arm2 --right arm1 --speed 0.3 --torque-limit 50
    python arm_calibration.py click        # a page on this machine: click the fingertip in each picture
    python arm_calibration.py fit          # the maps go into camera.json; prints how well the touches fit

Redo it when the phone or a base moves. The table should be clear, so the fingertip shows against it. `touch` plays
its trajectory through replay.py (flags after its own go there: the same checks, glides, torque cap and log as a
fold), with `--only` set to the arm being calibrated. The point commanded is what is paired with the pixel, so a
joint zero that is a little off is folded into the map: the arm is sent where the picture says.

The fingertip rests at the pinch's height (fold_traj.PINCH_Z, 5 mm in the arm model), and the click is on its tip.
A cloth corner lies on the table a few mm lower, which the camera's slant turns into a few mm along the table; not
corrected.
"""

import argparse
import json
import shutil
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

import collision
import fold_traj
from kinematics import Arm

HERE = Path(__file__).parent
CAMERA = HERE / "camera.json"
FOLDER = HERE / "runs" / "arm_calibration"      # <FOLDER>/<arm>/: the touch pictures and touches.json
RADII = (0.36, 0.16)        # m from the base origin: the far ring first, then the near one on the way back
WIDEST = 25.0               # degrees to either side of straight ahead that the rings span: with the bases side by
                            # side 30 cm apart, the widest that keeps the touching arm out of the other arm's place
MIDDLE = (0.26, 0.0)
HOLD = 2.0                  # s at full speed the fingertip rests on the table for its picture
CLOSED = -0.17              # jaw: shut the whole time, so the fingertip is one point in the picture
MIN_TOUCHES = 4             # the least a map can be fitted to
MIN_CLICKED = 6             # the least `fit` takes: with fewer, one wrong click cannot be told from the rest
MAX_FIT_ERROR = 0.015       # m: a touch this far from the map the OTHER touches make means a wrong click or a moved
                            # phone. Clicks 1.5 px off give up to about 14 mm here; a click 20 mm off is caught 19
                            # times in 20, one 10 mm off mostly is not (results.md, 2026-11-19)
CLICK_PORT = 8011


def touch_points(widest=WIDEST, outer=None):
    """The points an arm touches, (x forward, y left) m in its own frame: the far ring left to right, the middle,
    the near ring right to left, so consecutive points are neighbours. The rings run from `widest` degrees on the
    arm's right (the left arm's inner side) to `outer` degrees on its left, `widest` when not given: the wider, the
    more of the table the map is fitted over, and the more room the arm needs beside it. An `outer` of 0 keeps
    every touch between the centre line and the inner edge, for a camera that does not see the outer side."""
    if outer is None:
        outer = widest
    if outer <= -widest:
        raise ValueError(f"the fan's outer edge ({outer} degrees) must be left of its inner edge ({-widest})")
    bearings = []
    for k in range(4):
        bearings.append(-widest + (outer + widest) * k / 3)
    far = []
    for bearing in reversed(bearings):
        far.append((RADII[0] * np.cos(np.radians(bearing)), RADII[0] * np.sin(np.radians(bearing))))
    near = []
    for bearing in bearings:
        near.append((RADII[1] * np.cos(np.radians(bearing)), RADII[1] * np.sin(np.radians(bearing))))
    centre = np.radians((outer - widest) / 2)
    reach = float(np.hypot(MIDDLE[0], MIDDLE[1]))
    middle = (reach * np.cos(centre), reach * np.sin(centre))
    return far + [middle] + near


def touch_plan(arm, points, step=fold_traj.STEP):
    """One arm's rows [T, 6] touching each of `points` in turn, and the (first, last) row of each rest on the table.
    It starts above the first point and ends above the last. Between points the fingertip travels HOVER up, along
    a path whose distance from the base and bearing change evenly: a straight line between two points on a ring
    would cut across the base. ValueError when a point is out of reach."""
    up, down = fold_traj.HOVER, fold_traj.PINCH_Z
    hold = int(round(HOLD / fold_traj.DT))
    q = np.radians([0.0, 40.0, -30.0, 80.0, 0.0])
    rows, rests = [], []

    def go(p):
        nonlocal q
        q, err, _ = arm.solve(np.asarray(p, float), q)
        if err > 1e-3:
            raise ValueError(f"the arm cannot reach {np.round(p, 3).tolist()} (misses by {err * 1000:.0f} mm)")
        rows.append([*q, CLOSED])

    def line(a, b):
        n = max(1, int(np.ceil(np.linalg.norm(np.subtract(b, a)) / step)))
        for k in range(1, n + 1):
            go(np.asarray(a) + np.subtract(b, a) * k / n)

    first = points[0]
    q = arm.solve(np.array([*first, down]), q)[0]        # the touch first, then up from it: one branch (as the fold)
    go((*first, up))
    for i, p in enumerate(points):
        if i:
            a, b = np.asarray(points[i - 1]), np.asarray(p)
            r, th = np.hypot(*a), np.arctan2(a[1], a[0])
            dr, dth = np.hypot(*b) - r, np.arctan2(b[1], b[0]) - th
            n = max(1, int(np.ceil(max(abs(dr), max(r, r + dr) * abs(dth)) / step)))
            for k in range(1, n + 1):
                go(((r + dr * k / n) * np.cos(th + dth * k / n), (r + dr * k / n) * np.sin(th + dth * k / n), up))
        line((*p, up), (*p, down))
        rests.append((len(rows) - 1, len(rows) - 1 + hold))
        rows.extend([rows[-1]] * hold)
        line((*p, down), (*p, up))
    return np.array(rows), rests


def trajectory(points=None):
    """The touch sequence as a trajectory for replay.py (both arms get the same rows in their own frames; `touch`
    plays one of them), with marks["rests"] and source["points"]. Checked here as a fold is: every point reached,
    the arm clear of itself, no joint asked for more than the fold's rate."""
    arm, points = Arm(), touch_points() if points is None else points
    step = fold_traj.STEP
    while True:
        rows, rests = touch_plan(arm, points, step)
        fastest = float(np.abs(np.diff(rows[:, :5], axis=0)).max())
        if fastest <= fold_traj.JOINT_RATE:
            break
        step /= 2                               # a path the arm cannot follow at this pace: take it slower
        if step < fold_traj.STEP / 8:
            raise ValueError(f"a joint would have to move {np.degrees(fastest):.1f} deg in one row")
    clearance = collision.closest(rows.tolist())[0]
    if clearance < collision.CLEARANCE:
        raise ValueError(f"the arm comes within {clearance * 1000:.0f} mm of itself")
    q = np.vstack([rows[:1], rows])
    tips = [[-p[1], p[0], p[2]] for p in (arm.tip(r[:5])[0] for r in q)]
    side = {"target": rows.tolist(), "q": q.tolist(), "tip": tips}
    return {"dt": fold_traj.DT, "joints": fold_traj.JOINTS, "arms": {"left": side, "right": side},
            "marks": {"rests": [list(r) for r in rests]},
            "source": {"tool": "hardware/so101/arm_calibration.py", "points": [list(map(float, p)) for p in points],
                       "self_clearance_mm": round(clearance * 1000, 1)}}


class Capture(threading.Thread):
    """Keeps a picture every PERIOD seconds while the arm moves: <folder>/frames/<time.time() in ms>.jpg."""

    PERIOD = 0.25

    def __init__(self, phone, folder):
        super().__init__(daemon=True)
        self.phone, self.folder, self.stopped = phone, Path(folder) / "frames", threading.Event()
        self.folder.mkdir(parents=True, exist_ok=True)

    def run(self):
        from PIL import Image
        seen = None
        while not self.stopped.wait(self.PERIOD):
            frame = self.phone.latest()
            count = getattr(self.phone, "count", None)      # rgbd.Phone counts the frames it was sent
            if frame is None or (count is not None and count == seen):
                continue                                    # a stalled phone repeats its last frame: not a new picture
            seen = count
            Image.fromarray(np.asarray(frame.rgb, np.uint8)).save(self.folder / f"{int(time.time() * 1000)}.jpg")


def rest_times(log_lines, side, rests, settle=0.5):
    """{touch index: (from, to)} from replay.py's step log: the time.time() window in which the arm was resting on
    that point, from `settle` seconds after it arrived to the rest's last row (the row after starts the lift). A
    rest the run never finished, or shorter than the settle, is left out."""
    seen = {}
    for line in log_lines:
        if line.get("phase") == "fold" and side in line.get("target", {}):
            seen.setdefault(line["row"], line["t"])
    out = {}
    for k, (a, b) in enumerate(rests):
        if a in seen and b in seen and seen[b] - seen[a] > settle:
            out[k] = (seen[a] + settle, seen[b])
    return out


def keep_touches(folder, side, traj, log_path):
    """Pair each finished touch with the picture taken nearest its moment: <folder>/touch_<k>.jpg and touches.json."""
    folder = Path(folder)
    lines = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()] if Path(log_path).exists() else []
    times = rest_times(lines, side, traj["marks"]["rests"])
    frames = sorted((int(p.stem), p) for p in (folder / "frames").glob("*.jpg"))
    touches = []
    for k, (start, end) in sorted(times.items()):
        inside = [f for f in frames if start <= f[0] / 1000 <= end]
        if not inside:
            continue                            # no picture while it rested there: the phone stalled
        _, picture = min(inside, key=lambda f: abs(f[0] / 1000 - (start + end) / 2))
        shutil.copy(picture, folder / f"touch_{k}.jpg")
        touches.append({"point": traj["source"]["points"][k], "frame": f"touch_{k}.jpg", "px": None})
    (folder / "touches.json").write_text(json.dumps({"arm": side, "touches": touches}, indent=1))
    return touches


def fit_map(px, xy):
    """The 3x3 map taking pixels (u, v) to arm-frame metres (x, y), by least squares over four or more touches."""
    px, xy = np.asarray(px, float), np.asarray(xy, float)
    if len(px) < MIN_TOUCHES:
        raise ValueError(f"{len(px)} touches, at least {MIN_TOUCHES} are needed")

    def scaling(p):                             # centre and scale each set: the equations are badly conditioned raw
        centre, scale = p.mean(0), np.sqrt(2) / np.linalg.norm(p - p.mean(0), axis=1).mean()
        return np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])
    A, B = scaling(px), scaling(xy)
    p, q = apply(A, px), apply(B, xy)
    rows = []
    for (u, v), (x, y) in zip(p, q):
        rows.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
        rows.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
    h = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(B) @ h @ A
    return H / H[2, 2]


def apply(H, pts):
    q = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.asarray(H).T
    return q[:, :2] / q[:, 2:]


def fit_report(px, xy):
    """(map, per-touch error m with every touch in the fit, per-touch error m with that touch left out, and for
    each touch the worst error among the OTHERS when it is left out). The first error flatters: one wrong click
    bends the whole map toward itself. The second is how far the map puts a point it was not fitted to. The third
    says which touch is the odd one: the others fit best without it."""
    px, xy = np.asarray(px, float), np.asarray(xy, float)
    H = fit_map(px, xy)
    inside = np.linalg.norm(apply(H, px) - xy, axis=1)
    left_out, others = np.full(len(px), np.nan), np.full(len(px), np.nan)
    if len(px) > MIN_TOUCHES:
        for i in range(len(px)):
            rest = np.arange(len(px)) != i
            without = fit_map(px[rest], xy[rest])
            left_out[i] = np.linalg.norm(apply(without, px[i:i + 1])[0] - xy[i])
            others[i] = np.linalg.norm(apply(without, px[rest]) - xy[rest], axis=1).max()
    return H, inside, left_out, others


def load_touches(side):
    path = FOLDER / side / "touches.json"
    return json.loads(path.read_text())["touches"] if path.exists() else []


def fit():
    """Fit both arms' maps from their clicked touches and write them into camera.json (its other entries are kept).
    Returns {arm: summary}. ValueError, and nothing written, when an arm has fewer than MIN_CLICKED clicked touches,
    when a touch lies further than MAX_FIT_ERROR from the map the others make, or when the two arms' pictures are
    not the same size (the phone was changed between them). Both arms or neither: fold_camera.py reads both."""
    from PIL import Image
    maps, summary, shapes = {}, {}, set()
    for side in ("left", "right"):
        clicked = [t for t in load_touches(side) if t["px"] is not None]
        if len(clicked) < MIN_CLICKED:
            raise ValueError(f"the {side} arm has {len(clicked)} clicked touches, {MIN_CLICKED} are needed: run "
                             f"`touch --arm {side}` and `click`")
        H, inside, left_out, others = fit_report([t["px"] for t in clicked], [t["point"] for t in clicked])
        if left_out.max() > MAX_FIT_ERROR:
            odd = int(others.argmin())          # the touch the others fit best without
            raise ValueError(f"the {side} arm's touches do not agree: one lies {left_out.max() * 1000:.0f} mm from "
                             f"the map the others make. The others fit best without {clicked[odd]['frame']} (within "
                             f"{others[odd] * 1000:.0f} mm): click it again first, or the phone moved")
        with Image.open(FOLDER / side / clicked[0]["frame"]) as picture:
            shapes.add((picture.height, picture.width))
        summary[side] = {"touches": len(clicked), "rms_mm": round(float(np.sqrt((inside ** 2).mean())) * 1000, 1),
                         "max_mm": round(float(inside.max()) * 1000, 1),
                         "left_out_max_mm": round(float(left_out.max()) * 1000, 1)}
        maps[side] = {"H": H.tolist(), **summary[side]}
    if len(shapes) != 1:
        raise ValueError(f"the two arms' pictures are not the same size ({sorted(shapes)})")
    cal = json.loads(CAMERA.read_text()) if CAMERA.exists() else {}
    cal["arms"] = maps | {"fitted": datetime.now().isoformat(timespec="minutes"), "shape": list(shapes.pop())}
    CAMERA.write_text(json.dumps(cal))
    return summary


PAGE = """<!doctype html><meta charset="utf-8"><title>Arm calibration</title>
<style>body{font:15px sans-serif;margin:12px}canvas{max-width:100%;cursor:crosshair;border:1px solid #888}
button{font:15px sans-serif;margin-right:8px}</style>
<p id="what"></p><canvas id="c"></canvas>
<p><button id="skip">Fingertip not in the picture</button><button id="back">Back</button><span id="left"></span></p>
<script>
let list = [], i = 0; const c = document.getElementById('c'), g = c.getContext('2d'), img = new Image();
function show() {
  if (i >= list.length) { document.getElementById('what').textContent = 'All touches done. Close this page and run: arm_calibration.py fit'; c.style.display = 'none'; return; }
  const t = list[i]; c.style.display = '';
  document.getElementById('what').textContent = t.arm + ' arm, touch ' + (i + 1) + ' of ' + list.length + ': click the very tip of the closed gripper, where it meets the table.';
  img.onload = () => { c.width = img.naturalWidth; c.height = img.naturalHeight; g.drawImage(img, 0, 0);
    if (t.px) { g.strokeStyle = 'red'; g.lineWidth = 2; g.beginPath(); g.arc(t.px[0], t.px[1], 8, 0, 7); g.stroke(); } };
  img.src = '/picture/' + t.arm + '/' + t.frame + '?' + Date.now();
}
async function save(px) { const t = list[i];
  await fetch('/click', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({arm: t.arm, frame: t.frame, px: px})});
  t.px = px; i += 1; show(); }
c.onclick = e => { const r = c.getBoundingClientRect(); save([(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height]); };
document.getElementById('skip').onclick = () => save(null);
document.getElementById('back').onclick = () => { i = Math.max(0, i - 1); show(); };
fetch('/touches').then(r => r.json()).then(l => { list = l; i = Math.max(0, l.findIndex(t => t.px === null)); show(); });
</script>"""


def is_touch_picture(side, name):
    """Whether (side, name) can name one arm's touch picture: nothing else is read from or written to disk."""
    if side not in ("left", "right"):
        return False
    if not isinstance(name, str):
        return False
    if "/" in name or "\\" in name:
        return False
    return name.startswith("touch_") and name.endswith(".jpg")


def is_pixel(px):
    """Whether `px` is a click: None (fingertip not in the picture) or two numbers."""
    if px is None:
        return True
    if not isinstance(px, list) or len(px) != 2:
        return False
    for value in px:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
    return True


def click_handler():
    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, kind="application/json", status=200):
            body = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def from_this_machine(self):
            """False (after replying 403) for a request not addressed to this machine by name: a web page
            elsewhere that points its own name at 127.0.0.1 arrives with that other name."""
            port = self.server.server_address[1]
            if self.headers.get("Host", "") in (f"localhost:{port}", f"127.0.0.1:{port}"):
                return True
            self.reply({"error": "this page only answers on localhost"}, status=403)
            return False

        def do_GET(self):
            if not self.from_this_machine():
                return
            path = self.path.split("?")[0]
            if path == "/touches":
                self.reply([{"arm": side, **t} for side in ("left", "right") for t in load_touches(side)])
            elif path.startswith("/picture/"):
                parts = path.split("/")
                if len(parts) == 4 and is_touch_picture(parts[2], parts[3]) and (FOLDER / parts[2] / parts[3]).exists():
                    self.reply((FOLDER / parts[2] / parts[3]).read_bytes(), "image/jpeg")
                else:
                    self.reply({"error": "no such picture"}, status=404)
            else:
                self.reply(PAGE.encode(), "text/html")

        def do_POST(self):
            if not self.from_this_machine():
                return
            # A browser lets any web page send a plain form to this port without asking; JSON it must ask about
            # first, and this server never says yes.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self.reply({"error": "send JSON"}, status=415)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except ValueError:
                body = None
            if not isinstance(body, dict):
                self.reply({"error": "send a JSON object"}, status=400)
                return
            if not is_touch_picture(body.get("arm"), body.get("frame")) or not is_pixel(body.get("px")):
                self.reply({"error": "not a click on a touch picture"}, status=400)
                return
            record_click(body["arm"], body["frame"], body["px"])
            self.reply({"ok": True})

        def log_message(self, *args):
            pass
    return Handler


def record_click(side, frame, px):
    """Store one click (pixel (u, v), or None for a fingertip that is not in the picture) in the arm's touches.json."""
    path = FOLDER / side / "touches.json"
    data = json.loads(path.read_text())
    for t in data["touches"]:
        if t["frame"] == frame:
            t["px"] = None if px is None else [float(px[0]), float(px[1])]
    path.write_text(json.dumps(data, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["touch", "click", "fit"])
    ap.add_argument("--arm", choices=["left", "right"], help="touch: the arm to calibrate")
    ap.add_argument("--device", type=int, default=0, help="touch: Record3D device index")
    ap.add_argument("--widest", type=float, default=WIDEST, help="touch: degrees to either side of straight ahead "
                    "that the touches span. Nothing checks for the other arm. The default keeps the touching arm "
                    "within 18 cm of its own centre line, for bases side by side 30 cm apart; with the bases far "
                    "apart (arms at the cloth's sides) 55 fits the map over more of the table, sweeping 31 cm")
    ap.add_argument("--outer", type=float, help="touch: degrees the touches span on the arm's outer side (default: "
                    "as --widest). 0 keeps every touch between the arm's centre line and its inner edge, for a "
                    "phone that does not see the outer side; the map is then only fitted over that strip")
    args, replay_args = ap.parse_known_args(argv)
    if args.mode == "click":
        if not load_touches("left") and not load_touches("right"):
            sys.exit("No touches yet: run `arm_calibration.py touch --arm left ...` first.")
        print(f"Open http://localhost:{CLICK_PORT} and click the fingertip in each picture. Ctrl-C here when done.")
        try:
            HTTPServer(("127.0.0.1", CLICK_PORT), click_handler()).serve_forever()
        except KeyboardInterrupt:
            return 0
    if args.mode == "fit":
        try:
            summary = fit()
        except ValueError as err:
            sys.exit(f"Not saved: {err}")
        for side, s in summary.items():
            print(f"{side} arm: {s['touches']} touches, fit error {s['rms_mm']} mm rms, {s['max_mm']} mm worst; a "
                  f"touch left out of the fit lands within {s['left_out_max_mm']} mm")
        print(f"Saved in {CAMERA.name}. fold_camera.py now reads each arm's points through its own map; its "
              "`calibrate` only learns the cloth's colour from here on.")
        return 0
    if args.arm is None:
        sys.exit("touch needs --arm left or --arm right")
    import replay
    import rgbd
    try:
        points = touch_points(args.widest, args.outer)
    except ValueError as err:
        sys.exit(f"No touch sequence: {err}")
    if args.arm == "right":                         # mirrored, so each arm's first touch (the one replay.py glides
        points = [(x, -y) for x, y in points]       # to before it asks) is on its outer side, away from the other
    try:
        traj = trajectory(points)
    except ValueError as err:
        sys.exit(f"No touch sequence: {err}")
    arm = Arm()
    sideways = max(abs(p[1]) for row in traj["arms"]["left"]["target"][::5] for p in arm.joint_points(row[:5]))
    print(f"The {args.arm} arm touches {len(traj['source']['points'])} points and swings its wrist and fingertip up "
          f"to {sideways * 100:.0f} cm to either side of its centre line (the gripper is a few cm wider). Keep that "
          "clear: the other arm is not checked for.")
    # everything goes into a folder of its own first: the arm's earlier touches and clicks are only replaced once
    # this run has pictures of its own (a dry run, a refused trajectory or a stop before the first rest keeps them)
    folder, work = FOLDER / args.arm, FOLDER / f"{args.arm}.new"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "touch_traj.json").write_text(json.dumps(traj))
    log = work / "steps.jsonl"
    capture = None
    if "--dry-run" not in replay_args:
        phone = rgbd.Phone(args.device)
        phone.start()
        phone.wait()
        capture = Capture(phone, work)
        capture.start()
    try:
        code = replay.main([str(work / "touch_traj.json"), "--only", args.arm, "--log", str(log), *replay_args])
    finally:
        if capture is not None:
            capture.stopped.set()
            capture.join(timeout=2)
    touches = [] if capture is None else keep_touches(work, args.arm, traj, log)
    if not touches:
        shutil.rmtree(work)
        if capture is not None:
            print(f"No touch was finished with a picture. The {args.arm} arm's earlier touches, if any, are kept.")
        return code
    if folder.exists():
        shutil.rmtree(folder)
    work.rename(folder)
    print(f"{len(touches)} of {len(traj['source']['points'])} touches have a picture, in {folder}. "
          "Next: the other arm, then `arm_calibration.py click`.")
    return code


if __name__ == "__main__":
    sys.exit(main())

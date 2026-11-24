"""The half fold with one look from the iPhone: find the dark cloth's four corners in a frame, plan the fold to where
the cloth lies (fold_traj.fold), and play it (replay.py). The cloth may be shifted and turned; it is not tracked
after the look.

    python fold_camera.py calibrate            # with the cloth at the ruler position, again whenever the phone
                                               # moves (after arm_calibration.py: cloth anywhere; see below)
    python fold_camera.py look                 # no motion: where it sees the cloth and whether the arms reach it
    python fold_camera.py fold --left arm2 --right arm1 [--grasp-check] [--log run.json] [--speed 0.5] [--dry-run]

Frames come from the Record3D app over USB, as hardware/iphone_lidar/usb_viewer.py connects
(`pip install record3d==1.4.1`, the app open in USB Streaming mode and not recording), or from
--image FILE. The phone sits behind the arms looking over them; --near is the side of the image nearest the arms.

calibrate takes the cloth at the place fold_traj.py's ruler layout puts it (square, centred between the bases, near
edge --near-edge in front of the base origins) and stores the pixel-to-table map in camera.json. Table frame: X to
the right and Y forward seen from behind the arms, origin midway between the two base origins, metres.

After arm_calibration.py has put each arm's own pixel-to-arm map in camera.json ("arms"), the look reads every pinch
and place point through the map of the arm that goes there. The cloth's place at `calibrate` then no longer matters
(it only teaches the cloth's colour), --spacing and --near-edge go unused, and the bases may stand anywhere. When
the phone or a base moves, the arms' maps are stale and `calibrate` does NOT renew them: run arm_calibration.py again.
Every look prints when the maps were fitted.

Both arms should stand over their bases during a look, out of the cloth's outline in the image. Every look writes
look.png (the corners it found, and each arm's pinch and place points): check it before letting the arms move.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import fold_score
import fold_traj
import rgbd
from kinematics import Arm

HERE = Path(__file__).parent
CAMERA = HERE / "camera.json"
INSET = 0.01          # m: the pinch lands this far in from the corner along the cloth's diagonal, unless the
                      # calibration holds its own ("inset", from calibrate --inset)
SIDE_TOL = 0.15       # a detected side this far off the cloth's side (fraction) is not the cloth
REACH_TOL = 0.006     # m: a pinch the arm model misses by more than this is out of reach
OVERSHOOT = (0.0, 0.0)    # m (forward, outward): how far past the point they are sent to the fingertips land, unless
                         # the calibration holds its own ("overshoot"); outward is away from the other arm
MAX_OVERSHOOT = 0.03
PINCH_HEIGHTS = (-0.02, 0.02)    # m: the lowest and highest fingertip height at the pinch that is accepted. The
                                 # model's table is not the real one: on the rig one arm met it at about -0.2 cm
                                 # and the other was still above it at -0.5 (results.md, 2026-11-19)
PINCH_TILTS = (0.0, 45.0)        # deg: the range a saved pinch tilt may take
MAX_PULL = 0.10                  # m: the furthest a pinch is pulled toward its arm to keep the fingers steep
PULL_STEP = 0.0025
MAX_TILT = 30.0       # deg from upright: past it the fingers lie too flat to pinch a corner. On the rig 25 to 27
                      # held the cloth and 34 to 36 did not (results.md, 2026-11-19)
# Table area searched for the cloth (m): in front of the bases, clear of them
REGION = {"x": (-0.35, 0.35), "y": (fold_traj.BASE_FRONT + 0.005, 0.55)}


def record3d_frame(device=0, timeout=5.0):
    """One RGB frame [H, W, 3] from Record3D over USB."""
    phone = rgbd.Phone(device)
    phone.start()
    return phone.wait(timeout).rgb


def frame(args):
    return np.asarray(Image.open(args.image).convert("RGB")) if args.image else record3d_frame(args.device)


def table_grid(H, shape):
    """Table (X, Y) of every pixel, [h, w, 2], and whether the pixel looks at the table at all (below the horizon)."""
    v, u = np.mgrid[:shape[0], :shape[1]]
    q = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1) @ H.T
    centre = H @ np.array([shape[1] / 2, shape[0] / 2, 1.0])
    below = (q[:, 2] * centre[2] > 0).reshape(shape[:2])
    return (q[:, :2] / q[:, 2:]).reshape(shape[0], shape[1], 2), below


def calibrate(rgb, side, near_edge, near="bottom", dark=None, colour="dark"):
    """The camera map from a frame of the cloth at the ruler position. `colour` is what sets the cloth apart from the
    table ("dark" or "blue"); `dark` is the threshold on fold_score.cloth_value for it."""
    H_square, corners, dark = fold_score.calibrate(rgb, side, near, dark, colour)
    shift = np.array([[1.0, 0.0, -side / 2], [0.0, 1.0, near_edge], [0.0, 0.0, 1.0]])   # the square's frame to the table's
    return {"H": (shift @ H_square).tolist(), "dark": float(dark), "colour": colour, "near": near, "side": side,
            "shape": list(rgb.shape[:2]), "px": corners.tolist()}


def search_region(cal):
    """Per pixel: its table (X, Y) [h, w, 2], and whether it lies in the table area searched for the cloth."""
    xy, below = table_grid(np.array(cal["H"]), cal["shape"])
    inside = (below & (xy[..., 0] > REGION["x"][0]) & (xy[..., 0] < REGION["x"][1])
              & (xy[..., 1] > REGION["y"][0]) & (xy[..., 1] < REGION["y"][1]))
    return xy, inside


def cloth_pixels(rgb, cal, inside, keep=None):
    """Cloth-coloured pixels in the searched area; `keep` (rgbd.flat: nothing standing above the table) drops the
    arms. A calibration saved without a colour is for a dark cloth."""
    if list(rgb.shape[:2]) != cal["shape"]:
        raise ValueError(f"frame is {rgb.shape[:2]}, the calibration was {cal['shape']}")
    dark = (fold_score.cloth_value(rgb, cal.get("colour", "dark")) < cal["dark"]) & inside
    return dark if keep is None else dark & keep


def near_edge(rgb, cal, grid, x_range, keep=None):
    """Where the cloth's near edge lies now: (slope, Y at X = 0) of a line through the nearest cloth pixel of each
    1 cm column of X within x_range (m), or None with under 8 columns. It stays in view through the carry (the
    bottom layer's edge, nearest the camera), so its movement is how far the bottom layer has slid."""
    xy, inside = grid
    cloth = cloth_pixels(rgb, cal, inside, keep)
    x, y = xy[..., 0][cloth], xy[..., 1][cloth]
    cols = np.floor((x - x_range[0]) / 0.01).astype(int)
    n = int(np.ceil((x_range[1] - x_range[0]) / 0.01))
    centres, nearest = [], []
    for c in range(n):
        ys = y[cols == c]
        if len(ys) >= 5:
            centres.append(x_range[0] + (c + 0.5) * 0.01)
            nearest.append(np.percentile(ys, 2))
    if len(centres) < 8:
        return None
    centres, nearest = np.array(centres), np.array(nearest)
    slope, y0 = np.polyfit(centres, nearest, 1)
    good = np.abs(nearest - (slope * centres + y0)) < 0.01           # one refit without the columns an arm clipped
    if good.sum() >= 8:
        slope, y0 = np.polyfit(centres[good], nearest[good], 1)
    return float(slope), float(y0)


def look(rgb, cal, keep=None):
    """The cloth's corners on the table (m): {"near_left", "near_right", "far_right", "far_left"}, and in pixels."""
    H = np.array(cal["H"])
    labels, n = ndimage.label(cloth_pixels(rgb, cal, search_region(cal)[1], keep))
    if n == 0:
        raise ValueError("no dark cloth in front of the arms")
    mask = ndimage.binary_fill_holes(labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    px = fold_score.square_corners(mask, cal["near"])
    table = fold_score.apply(H, px)
    sides = np.linalg.norm(table - np.roll(table, -1, axis=0), axis=1)
    if np.any(np.abs(sides / cal["side"] - 1.0) > SIDE_TOL):
        raise ValueError(f"the dark shape has sides {np.round(sides * 100, 1)} cm, not a {cal['side'] * 100:.0f} cm "
                         "cloth: an arm or a shadow may be touching its outline")
    names = ["near_left", "near_right", "far_right", "far_left"]
    return dict(zip(names, table)), dict(zip(names, px))


REACH = 0.60          # m from a base: further than this is not searched for the cloth


def arm_maps(cal):
    return {side: np.array(cal["arms"][side]["H"]) for side in ("left", "right")}


def arms_region(cal):
    """Whether each pixel lies in the table area searched for the cloth, from the arms' own maps: in front of both
    bases' front edges and within REACH of one of them."""
    in_front, near = np.ones(cal["shape"], bool), np.zeros(cal["shape"], bool)
    for H in arm_maps(cal).values():
        xy, below = table_grid(H, cal["shape"])
        in_front &= below & (xy[..., 0] > fold_traj.BASE_FRONT + 0.005)
        near |= np.hypot(xy[..., 0], xy[..., 1]) < REACH
    return in_front & near


def look_arms(rgb, cal, keep=None):
    """The cloth's corners in each arm's own frame, {arm: {"near_left", ...: (x forward, y left) m}}, and in pixels.
    For a calibration made from the arms (arm_calibration.py)."""
    labels, n = ndimage.label(cloth_pixels(rgb, cal, arms_region(cal), keep))
    if n == 0:
        raise ValueError("no cloth in front of the arms")
    mask = ndimage.binary_fill_holes(labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    px = fold_score.square_corners(mask, cal["near"])
    names = ["near_left", "near_right", "far_right", "far_left"]
    by_arm = {}
    for side, H in arm_maps(cal).items():
        frame = fold_score.apply(H, px)
        sides = np.linalg.norm(frame - np.roll(frame, -1, axis=0), axis=1)
        if np.any(np.abs(sides / cal["side"] - 1.0) > SIDE_TOL):
            raise ValueError(f"the shape has sides {np.round(sides * 100, 1)} cm in the {side} arm's map, not a "
                             f"{cal['side'] * 100:.0f} cm cloth: an arm or a shadow may be touching its outline")
        by_arm[side] = dict(zip(names, frame))
    return by_arm, dict(zip(names, px))


def fold_points_arms(by_arm, inset=INSET):
    """Each arm's (pinch, place) from the corners in its own frame (look_arms): as fold_points, with no layout."""
    points = {}
    for side, far, near, across in (("left", "far_left", "near_left", "near_right"),
                                    ("right", "far_right", "near_right", "near_left")):
        corners = by_arm[side]
        diagonal = corners[across] - corners[far]
        offset = inset * diagonal / np.linalg.norm(diagonal)
        points[side] = tuple((float(p[0]), float(p[1])) for p in (corners[far] + offset, corners[near] + offset))
    return points


def fold_points(corners, spacing, inset=INSET):
    """Each arm's (pinch, place) in its own base frame (x forward, y left): the far corner on its side onto the near
    one, both moved `inset` in along the far corner's diagonal so the corner lands on the corner."""
    points = {}
    for side, base_x, far, near, across in (("left", -spacing / 2, "far_left", "near_left", "near_right"),
                                            ("right", spacing / 2, "far_right", "near_right", "near_left")):
        diagonal = corners[across] - corners[far]
        offset = inset * diagonal / np.linalg.norm(diagonal)
        points[side] = tuple((float(p[1]), float(-(p[0] - base_x))) for p in (corners[far] + offset,
                                                                               corners[near] + offset))
    return points


def pinch_heights(value):
    """{"left": m, "right": m} from a saved pinch height: one number for both arms, or [left, right]."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"the pinch height is {value!r}, not one number or a left and a right one")
        return {"left": value[0], "right": value[1]}
    return {"left": value, "right": value}


def overshoots(value):
    """{"left": (forward, outward), "right": (forward, outward)} m from a saved overshoot: one pair for both arms,
    or [left pair, right pair]."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"the calibration's overshoot is {value!r}, not a pair or a left and a right pair")
    if isinstance(value[0], (list, tuple)):
        pairs = {"left": value[0], "right": value[1]}
    else:
        pairs = {"left": value, "right": value}
    for side in ("left", "right"):
        if not isinstance(pairs[side], (list, tuple)) or len(pairs[side]) != 2:
            raise ValueError(f"the {side} overshoot is {pairs[side]!r}, not two numbers")
    return pairs


def steepened(points, overshoot, heights, limit):
    """`points` ({side: (pinch, place)}) with each pinch pulled straight toward its arm's base, as little as leaves
    the fingers within `limit` degrees of upright there, and how far each was pulled (m). The arm cannot hold its
    fingers upright past about 30 cm: it leans them about 3.5 degrees for every cm further. ValueError when MAX_PULL
    is not enough."""
    arm = Arm()
    seed = np.radians([0.0, 40.0, -30.0, 80.0, 0.0])
    moved = {}
    pulled = {}
    for side in ("left", "right"):
        pinch, place = points[side]
        distance = float(np.hypot(pinch[0], pinch[1]))
        pull = 0.0
        while True:
            scale = (distance - pull) / distance
            candidate = (pinch[0] * scale, pinch[1] * scale)
            sent = aimed({"left": (candidate, place), "right": (candidate, place)}, overshoot)[side][0]
            tilt = arm.solve(np.array([sent[0], sent[1], heights[side]]), seed)[2]
            if tilt <= limit:
                break
            pull = pull + PULL_STEP
            if pull > MAX_PULL:
                raise ValueError(f"the {side} arm's fingers lean {tilt:.0f} deg even with the pinch {MAX_PULL * 100:.0f} "
                                 f"cm nearer the arm; the limit is {limit:.0f}: move the cloth closer")
        moved[side] = (candidate, place)
        pulled[side] = pull
    return moved, pulled


def settings(cal):
    """(inset, overshoot, each arm's pinch height) of a calibration, each inside its limits. ValueError otherwise:
    these send the arms, and camera.json can hold anything."""
    inset = cal.get("inset", INSET)
    overshoot = overshoots(cal.get("overshoot", OVERSHOOT))
    heights = pinch_heights(cal.get("pinch_height", fold_traj.PINCH_Z))
    numbers = [inset, heights["left"], heights["right"]] + list(overshoot["left"]) + list(overshoot["right"])
    for number in numbers:
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number):
            raise ValueError(f"the calibration holds {number!r} where a number belongs")
    if not 0.0 <= inset < cal["side"] / 2:
        raise ValueError(f"the inset {inset} m is not between 0 and half the cloth's side")
    for side in ("left", "right"):
        if max(abs(overshoot[side][0]), abs(overshoot[side][1])) > MAX_OVERSHOOT:
            raise ValueError(f"the {side} overshoot {list(overshoot[side])} m is over {MAX_OVERSHOOT} m")
    for side in ("left", "right"):
        if not PINCH_HEIGHTS[0] <= heights[side] <= PINCH_HEIGHTS[1]:
            raise ValueError(f"the {side} pinch height {heights[side]} m is not between {PINCH_HEIGHTS[0]} and "
                             f"{PINCH_HEIGHTS[1]} m")
    tilt = cal.get("pinch_tilt")
    if tilt is not None:
        if isinstance(tilt, bool) or not isinstance(tilt, (int, float)) or not PINCH_TILTS[0] <= tilt <= PINCH_TILTS[1]:
            raise ValueError(f"the pinch tilt {tilt!r} is not between {PINCH_TILTS[0]} and {PINCH_TILTS[1]} deg")
    return inset, overshoot, heights


def aimed(points, overshoot):
    """Where each arm is sent so that its fingertip lands on `points` ({side: (pinch, place)}, each arm's own frame):
    each arm's `overshoot[side]` (forward, outward) m short of them. Outward is to the left for the left arm, the
    right for the right."""
    sent = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        forward, outward = overshoot[side]
        moved = []
        for x, y in points[side]:
            moved.append((x - forward, y - sign * outward))
        sent[side] = tuple(moved)
    return sent


def plan(rgb, cal, spacing, closed, keep=None, empty_reading=False):
    """The fold for the cloth in `rgb`: (trajectory, corners, corners in pixels, each arm's points). With a
    calibration from the arms the corners returned are in the left arm's frame (x forward, y left). The points
    returned are where the fingertips should land; the trajectory is sent the calibration's overshoot short of them."""
    inset, overshoot, pinch_z = settings(cal)
    if "arms" in cal:
        by_arm, px = look_arms(rgb, cal, keep)
        points = fold_points_arms(by_arm, inset)
        pulled = {"left": 0.0, "right": 0.0}
        if cal.get("pinch_tilt") is not None:
            points, pulled = steepened(points, overshoot, pinch_z, cal["pinch_tilt"])
        traj = fold_traj.fold(aimed(points, overshoot), closed, empty_reading, pinch_z)
        traj["layout"] = {"frame": "each arm's base, from its own camera map", "pinch_pulled_m": pulled, "overshoot": {side: list(pair) for side, pair in overshoot.items()}, "corners":
                          {side: {k: np.round(v, 4).tolist() for k, v in c.items()} for side, c in by_arm.items()}}
        return traj, by_arm["left"], px, points
    corners, px = look(rgb, cal, keep)
    points = fold_points(corners, spacing, inset)
    pulled = {"left": 0.0, "right": 0.0}
    if cal.get("pinch_tilt") is not None:
        points, pulled = steepened(points, overshoot, pinch_z, cal["pinch_tilt"])
    traj = fold_traj.fold(aimed(points, overshoot), closed, empty_reading, pinch_z)
    traj["layout"] = {"frame": "table: X right, Y forward from the line of the base origins", "base_spacing": spacing,
                      "pinch_pulled_m": pulled,
                      "overshoot": {side: list(pair) for side, pair in overshoot.items()},
                      "corners": {k: np.round(v, 4).tolist() for k, v in corners.items()}}
    return traj, corners, px, points


def overlay(rgb, cal, px, points, spacing):
    """The frame with the cloth's outline (green) and each arm's pinch (red) and place (blue) points."""
    img = Image.fromarray(np.asarray(rgb, np.uint8))
    draw = ImageDraw.Draw(img)
    quad = [tuple(px[k]) for k in ("near_left", "near_right", "far_right", "far_left")]
    draw.line(quad + quad[:1], fill=(0, 255, 0), width=3)
    to_px = None if "arms" in cal else np.linalg.inv(np.array(cal["H"]))
    for side, base_x in (("left", -spacing / 2), ("right", spacing / 2)):
        for (x, y), colour in zip(points[side], ((255, 0, 0), (0, 80, 255))):
            if to_px is None:
                u, v = fold_score.apply(np.linalg.inv(arm_maps(cal)[side]), np.array([[x, y]]))[0]
            else:
                u, v = fold_score.apply(to_px, np.array([[base_x - y, x]]))[0]
            draw.ellipse([u - 7, v - 7, u + 7, v + 7], outline=colour, width=3)
    return img


def report(traj, corners, max_tilt=MAX_TILT):
    """Print the corners and each arm's reach. False when an arm misses its pinch or its fingers lie flatter than
    `max_tilt` degrees from upright anywhere in the fold."""
    for name, p in corners.items():
        print(f"  {name:10s} X {p[0] * 100:6.1f}  Y {p[1] * 100:6.1f} cm")
    ok = True
    for side in ("left", "right"):
        r = traj["source"][side]
        note = ""
        if r["pinch_error_mm"] > REACH_TOL * 1000:
            ok = False
            note = "  OUT OF REACH"
        elif r["max_tilt_deg"] > max_tilt:
            ok = False
            note = f"  TOO FLAT (over {max_tilt:.0f} deg)"
        print(f"  {side} arm: pinch missed by {r['pinch_error_mm']} mm, place by {r['place_error_mm']} mm, fingers "
              f"tilted up to {r['max_tilt_deg']} deg{note}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["calibrate", "look", "fold"])
    ap.add_argument("--image", help="a photo instead of the live phone")
    ap.add_argument("--device", type=int, default=0, help="Record3D device index")
    ap.add_argument("--near", default="bottom", choices=list(fold_score.NEAR_DIR), help="image side nearest the arms")
    ap.add_argument("--cloth", type=float, default=0.30, help="cloth side (m)")
    ap.add_argument("--near-edge", type=float, default=0.08, help="calibrate: near edge in front of the base origins (m)")
    ap.add_argument("--spacing", type=float, default=0.30, help="between the base origins (m)")
    ap.add_argument("--dark", type=float, help="calibrate: grey level below which a pixel is cloth (default: Otsu)")
    ap.add_argument("--cloth-colour", choices=fold_score.COLOURS, default="dark",
                    help="calibrate: what sets the cloth apart from the table; saved in the calibration")
    ap.add_argument("--closed", type=float, default=-0.17, help="jaw target at the pinch (rad)")
    ap.add_argument("--inset", type=float, help="how far in from each corner, along the cloth's diagonal, the pinch "
                    "lands (m, default 0.01). calibrate saves it; on look and fold it is for that run only. Raise it "
                    "when the grippers come down past the corners")
    ap.add_argument("--overshoot", type=float, nargs="+", metavar="M", help="how far past the point they are sent to "
                    "the fingertips land (m, default 0 0): FORWARD OUTWARD for both arms, or four numbers, the left "
                    "arm's pair then the right's. The arms are sent that much short. Outward is away from the "
                    "other arm. Saved by calibrate, for one run on look and fold")
    ap.add_argument("--pinch-height", type=float, nargs="+", metavar="M", help="fingertip height at the pinch in "
                    "the arm model (m, default 0.005): one number for both arms, or LEFT RIGHT. Lower an arm's "
                    "when its fingers stop above the cloth. Saved like --overshoot")
    ap.add_argument("--pinch-tilt", type=float, metavar="DEG", help="keep the fingers within DEG of upright at the "
                    "pinch: each pinch is pulled toward its arm, further into the cloth, by as much as that takes. "
                    "Saved like --overshoot; without it the pinch stays where the inset puts it")
    ap.add_argument("--max-tilt", type=float, default=MAX_TILT, help="deg from upright the fingers may lean anywhere "
                    "in the fold; a fold that needs more is refused")
    ap.add_argument("--out", default=str(HERE / "look.png"))
    args, replay_args = ap.parse_known_args(argv)
    if args.inset is not None and not 0.0 <= args.inset < args.cloth / 2:
        sys.exit(f"--inset {args.inset} m is not between 0 and half the cloth's side ({args.cloth / 2} m).")
    if args.overshoot is not None:
        if len(args.overshoot) not in (2, 4):
            sys.exit("--overshoot takes FORWARD OUTWARD for both arms, or four numbers: the left pair, then the right.")
        for number in args.overshoot:
            if abs(number) > MAX_OVERSHOOT:
                sys.exit(f"--overshoot {number} m: no number may be over {MAX_OVERSHOOT} m.")
    if args.pinch_height is not None:
        if len(args.pinch_height) > 2:
            sys.exit("--pinch-height takes one number for both arms, or a left and a right one.")
        for height in args.pinch_height:
            if not PINCH_HEIGHTS[0] <= height <= PINCH_HEIGHTS[1]:
                sys.exit(f"--pinch-height {height} m is not between {PINCH_HEIGHTS[0]} and {PINCH_HEIGHTS[1]} m.")
    if args.pinch_tilt is not None and not PINCH_TILTS[0] <= args.pinch_tilt <= PINCH_TILTS[1]:
        sys.exit(f"--pinch-tilt {args.pinch_tilt} deg is not between {PINCH_TILTS[0]} and {PINCH_TILTS[1]}.")
    given = {}                  # the settings given on this command line: saved by calibrate, else for this run
    if args.pinch_tilt is not None:
        given["pinch_tilt"] = args.pinch_tilt
    if args.inset is not None:
        given["inset"] = args.inset
    if args.overshoot is not None:
        if len(args.overshoot) == 2:
            given["overshoot"] = list(args.overshoot)
        else:
            given["overshoot"] = [list(args.overshoot[:2]), list(args.overshoot[2:])]
    if args.pinch_height is not None:
        if len(args.pinch_height) == 1:
            given["pinch_height"] = args.pinch_height[0]
        else:
            given["pinch_height"] = list(args.pinch_height)
    rgb = frame(args)
    if args.mode == "calibrate":
        cal = calibrate(rgb, args.cloth, args.near_edge, args.near, args.dark, args.cloth_colour)
        try:
            kept = json.loads(CAMERA.read_text()).get("arms") if CAMERA.exists() else None
        except ValueError:          # an unreadable camera.json is written over, as before
            kept = None
        if kept is not None:
            cal["arms"] = kept      # the map came from the arms: this look at the cloth must not move it
        cal.update(given)
        CAMERA.write_text(json.dumps(cal))
        print(f"Saved {CAMERA.name}: {cal['colour']} cloth, threshold {cal['dark']:.0f}, "
              f"corners (px) {np.round(cal['px']).tolist()}")
        if kept is not None:
            print("The arms' own maps (arm_calibration.py) are kept: only the cloth's colour was learned.")
    if not CAMERA.exists():
        sys.exit(f"No {CAMERA.name} yet: run `fold_camera.py calibrate` first, with the cloth at the ruler position.")
    cal = json.loads(CAMERA.read_text())
    cal.update(given)
    if "dark" not in cal:
        sys.exit(f"{CAMERA.name} has the arms' maps but not the cloth's colour yet: run `fold_camera.py calibrate` "
                 "once with the cloth in view, anywhere.")
    if "arms" in cal:
        maps = cal["arms"]
        if "left" not in maps or "right" not in maps:
            sys.exit(f"{CAMERA.name} holds a map for one arm only: run `arm_calibration.py fit` with both arms' touches.")
        if maps.get("shape") not in (None, list(rgb.shape[:2])):
            sys.exit(f"The arms' maps were fitted on {maps['shape']} pictures and this one is {list(rgb.shape[:2])}: "
                     "run arm_calibration.py again.")
        print(f"Reading the cloth through the arms' own maps, fitted {maps.get('fitted', 'at an unknown time')}. "
              "They hold only while the phone and the bases stay where they were.")
    try:
        traj, corners, px, points = plan(rgb, cal, args.spacing, args.closed,
                                         empty_reading="--grasp-check" in replay_args)
    except ValueError as err:
        sys.exit(f"Look failed: {err}")
    overlay(rgb, cal, px, points, args.spacing).save(args.out)
    print(f"Cloth corners {'in the left arm frame, X forward and Y left' if 'arms' in cal else 'on the table'} "
          f"(drawn in {args.out}):")
    reachable = report(traj, corners, args.max_tilt)
    inset, overshoot, heights = settings(cal)
    print(f"  pinching {inset * 100:.1f} cm in from each corner")
    for side in ("left", "right"):
        print(f"  {side} arm sent {overshoot[side][0] * 100:.1f} cm short of it and {overshoot[side][1] * 100:.1f} cm "
              f"toward the other arm, fingertip {heights[side] * 100:.1f} cm above the model's table")
        pull = traj["layout"]["pinch_pulled_m"][side]
        if pull > 0:
            print(f"  {side} pinch pulled {pull * 100:.1f} cm toward its arm to keep the fingers within "
                  f"{cal['pinch_tilt']:.0f} deg of upright")
    if args.mode != "fold":
        return 0
    if not reachable:
        sys.exit("Not folding: move the cloth closer to the arms.")
    input(f"Check {args.out}. Enter to fold, Ctrl-C to stop ")
    import replay
    path = HERE / "last_look.json"
    path.write_text(json.dumps(traj))
    return replay.main([str(path), *replay_args])


if __name__ == "__main__":
    sys.exit(main())

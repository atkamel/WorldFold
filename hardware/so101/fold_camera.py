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

HERE = Path(__file__).parent
CAMERA = HERE / "camera.json"
INSET = 0.01          # m: the pinch lands this far in from the corner along the cloth's diagonal
SIDE_TOL = 0.15       # a detected side this far off the cloth's side (fraction) is not the cloth
REACH_TOL = 0.006     # m: a pinch the arm model misses by more than this is out of reach
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


def fold_points_arms(by_arm):
    """Each arm's (pinch, place) from the corners in its own frame (look_arms): as fold_points, with no layout."""
    points = {}
    for side, far, near, across in (("left", "far_left", "near_left", "near_right"),
                                    ("right", "far_right", "near_right", "near_left")):
        corners = by_arm[side]
        diagonal = corners[across] - corners[far]
        offset = INSET * diagonal / np.linalg.norm(diagonal)
        points[side] = tuple((float(p[0]), float(p[1])) for p in (corners[far] + offset, corners[near] + offset))
    return points


def fold_points(corners, spacing):
    """Each arm's (pinch, place) in its own base frame (x forward, y left): the far corner on its side onto the near
    one, both moved INSET in along the far corner's diagonal so the corner lands on the corner."""
    points = {}
    for side, base_x, far, near, across in (("left", -spacing / 2, "far_left", "near_left", "near_right"),
                                            ("right", spacing / 2, "far_right", "near_right", "near_left")):
        diagonal = corners[across] - corners[far]
        offset = INSET * diagonal / np.linalg.norm(diagonal)
        points[side] = tuple((float(p[1]), float(-(p[0] - base_x))) for p in (corners[far] + offset,
                                                                               corners[near] + offset))
    return points


def plan(rgb, cal, spacing, closed, keep=None, empty_reading=False):
    """The fold for the cloth in `rgb`: (trajectory, corners, corners in pixels, each arm's points). With a
    calibration from the arms the corners returned are in the left arm's frame (x forward, y left)."""
    if "arms" in cal:
        by_arm, px = look_arms(rgb, cal, keep)
        points = fold_points_arms(by_arm)
        traj = fold_traj.fold(points, closed, empty_reading)
        traj["layout"] = {"frame": "each arm's base, from its own camera map", "corners":
                          {side: {k: np.round(v, 4).tolist() for k, v in c.items()} for side, c in by_arm.items()}}
        return traj, by_arm["left"], px, points
    corners, px = look(rgb, cal, keep)
    points = fold_points(corners, spacing)
    traj = fold_traj.fold(points, closed, empty_reading)
    traj["layout"] = {"frame": "table: X right, Y forward from the line of the base origins", "base_spacing": spacing,
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


def report(traj, corners):
    for name, p in corners.items():
        print(f"  {name:10s} X {p[0] * 100:6.1f}  Y {p[1] * 100:6.1f} cm")
    ok = True
    for side in ("left", "right"):
        r = traj["source"][side]
        reach = r["pinch_error_mm"] <= REACH_TOL * 1000
        ok &= reach
        print(f"  {side} arm: pinch missed by {r['pinch_error_mm']} mm, place by {r['place_error_mm']} mm, fingers "
              f"tilted up to {r['max_tilt_deg']} deg{'' if reach else '  OUT OF REACH'}")
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
    ap.add_argument("--out", default=str(HERE / "look.png"))
    args, replay_args = ap.parse_known_args(argv)
    rgb = frame(args)
    if args.mode == "calibrate":
        cal = calibrate(rgb, args.cloth, args.near_edge, args.near, args.dark, args.cloth_colour)
        try:
            kept = json.loads(CAMERA.read_text()).get("arms") if CAMERA.exists() else None
        except ValueError:          # an unreadable camera.json is written over, as before
            kept = None
        if kept is not None:
            cal["arms"] = kept      # the map came from the arms: this look at the cloth must not move it
        CAMERA.write_text(json.dumps(cal))
        print(f"Saved {CAMERA.name}: {cal['colour']} cloth, threshold {cal['dark']:.0f}, "
              f"corners (px) {np.round(cal['px']).tolist()}")
        if kept is not None:
            print("The arms' own maps (arm_calibration.py) are kept: only the cloth's colour was learned.")
    if not CAMERA.exists():
        sys.exit(f"No {CAMERA.name} yet: run `fold_camera.py calibrate` first, with the cloth at the ruler position.")
    cal = json.loads(CAMERA.read_text())
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
    reachable = report(traj, corners)
    if args.mode != "fold":
        return 0
    if not reachable:
        sys.exit("Not folding: move the cloth closer.")
    input(f"Check {args.out}. Enter to fold, Ctrl-C to stop ")
    import replay
    path = HERE / "last_look.json"
    path.write_text(json.dumps(traj))
    return replay.main([str(path), *replay_args])


if __name__ == "__main__":
    sys.exit(main())

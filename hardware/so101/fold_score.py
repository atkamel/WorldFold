"""Scores a cloth half fold from two photos of a fixed camera: one before the fold (the dark cloth flat, a full
square) and one after. The before photo gives the cloth's four corners, hence a homography from pixels to the table
in metres; the after silhouette is mapped through it and compared with the ideal rectangle (side x side/2 on the
near half of the square). No other calibration. Prints one JSON line, all lengths in cm.

    .venv/bin/python hardware/so101/fold_score.py before.jpg after.jpg [--side 0.30] [--near bottom] [--dark N] \
        [--out overlay.png]

--near is the side of the image that the near edge (the one closest to the robot, onto which the cloth folds) is on.
--dark overrides the cloth threshold (grey level); the default is Otsu's on the before photo, reused for the after one.
"""

import argparse
import itertools
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

NEAR_DIR = {"bottom": (0, 1), "top": (0, -1), "left": (-1, 0), "right": (1, 0)}   # image (u, v) direction
MAX_HULL = 40         # hull vertices kept before the brute-force search for the largest quadrilateral
BIN = 0.01            # m, width of the x-bins along the folded edge
RASTER = 0.002        # m, cell of the coverage raster


def otsu(grey):
    hist = np.bincount(np.round(grey).astype(int).ravel(), minlength=256).astype(float)
    levels = np.arange(256)
    best, dark = -1.0, 128
    for t in range(1, 256):
        w0, w1 = hist[:t].sum(), hist[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (hist[:t] * levels[:t]).sum() / w0, (hist[t:] * levels[t:]).sum() / w1
        if w0 * w1 * (m0 - m1) ** 2 > best:
            best, dark = w0 * w1 * (m0 - m1) ** 2, t
    return dark


COLOURS = ["dark", "blue"]


def cloth_value(rgb, colour="dark"):
    """Per pixel, 0 to 255, low where the cloth is: the grey level for a dark cloth, and for a blue one how far blue
    falls short of standing out over red (white, grey, black and brown things all read 255)."""
    pixels = np.asarray(rgb, float)
    if colour == "dark":
        return pixels.mean(axis=2)
    if colour == "blue":
        blueness = np.clip(pixels[:, :, 2] - pixels[:, :, 0], 0, 255)
        return 255 - np.clip(2 * blueness, 0, 255)
    raise ValueError(f"cloth colour {colour!r} is not one of {COLOURS}")


def cloth_mask(rgb, dark=None, colour="dark"):
    """Boolean mask of the largest blob of cloth-coloured pixels (holes filled) and the threshold used."""
    grey = cloth_value(rgb, colour)
    if dark is None:
        dark = otsu(grey)
    labels, n = ndimage.label(grey < dark)
    if n == 0:
        raise ValueError("no cloth pixels below the threshold %s" % dark)
    biggest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(labels == biggest), dark


def boundary(mask):
    v, u = np.nonzero(mask & ~ndimage.binary_erosion(mask))
    return np.stack([u, v], axis=1).astype(float)


def shoelace(p):
    return 0.5 * (p[..., 0] * np.roll(p[..., 1], -1, axis=-1) - np.roll(p[..., 0], -1, axis=-1) * p[..., 1]).sum(-1)


def square_corners(mask, near="bottom"):
    """Pixel corners (u, v) of the cloth, ordered [near-left, near-right, far-right, far-left]."""
    pts = boundary(mask)
    hull = pts[ConvexHull(pts).vertices]
    while len(hull) > MAX_HULL:   # drop the vertex whose removal changes the outline least
        tri = np.stack([np.roll(hull, 1, 0), hull, np.roll(hull, -1, 0)], axis=1)
        hull = np.delete(hull, np.argmin(np.abs(shoelace(tri))), axis=0)
    combos = np.array(list(itertools.combinations(range(len(hull)), 4)))
    quad = hull[combos[np.argmax(np.abs(shoelace(hull[combos])))]]
    if shoelace(quad) > 0:        # visually counter-clockwise (v points down): near-left, near-right, far-right, far-left
        quad = quad[::-1]
    mids = (quad + np.roll(quad, -1, 0)) / 2
    return np.roll(quad, -int(np.argmax(mids @ np.array(NEAR_DIR[near]))), 0)


def homography(px, side):
    """3x3 matrix taking pixels (u, v) to table metres, the corners px going to the square [0, side]^2."""
    table = np.array([[0, 0], [side, 0], [side, side], [0, side]], float)
    a = []
    for (u, v), (x, y) in zip(px, table):
        a.append([u, v, 1, 0, 0, 0, -x * u, -x * v])
        a.append([0, 0, 0, u, v, 1, -y * u, -y * v])
    return np.append(np.linalg.solve(a, table.ravel()), 1).reshape(3, 3)


def apply(h, pts):
    q = np.c_[pts, np.ones(len(pts))] @ h.T
    return q[:, :2] / q[:, 2:]


def calibrate(before_rgb, side=0.30, near="bottom", dark=None, colour="dark"):
    mask, dark = cloth_mask(before_rgb, dark, colour)
    corners = square_corners(mask, near)
    return homography(corners, side), corners, dark


def score(before_rgb, after_rgb, side=0.30, near="bottom", dark=None, colour="dark"):
    h, _, dark = calibrate(before_rgb, side, near, dark, colour)
    mask, _ = cloth_mask(after_rgb, dark, colour)
    x, y = apply(h, boundary(mask)).T

    far = (y > side / 4) & (x >= 0) & (x <= side)
    nbins = round(side / BIN)
    top = np.full(nbins, -np.inf)
    np.maximum.at(top, np.minimum((x[far] / BIN).astype(int), nbins - 1), y[far])
    err = np.abs(top[np.isfinite(top)] - side / 2)

    outside = np.hypot(np.maximum(0, np.maximum(-x, x - side)), np.maximum(0, np.maximum(-y, y - side)))

    v, u = np.nonzero(mask)
    scale = abs(np.linalg.det(h)) / (h[2] @ np.stack([u, v, np.ones(len(u))])) ** 3   # table m^2 per pixel
    area = scale.sum()

    gx, gy = np.meshgrid(np.arange(RASTER / 2, side, RASTER), np.arange(RASTER / 2, side / 2, RASTER))
    pu, pv = np.round(apply(np.linalg.inv(h), np.c_[gx.ravel(), gy.ravel()])).astype(int).T
    inside = (pu >= 0) & (pu < mask.shape[1]) & (pv >= 0) & (pv < mask.shape[0])
    covered = np.zeros(len(pu), bool)
    covered[inside] = mask[pv[inside], pu[inside]]

    return {"far_edge_mean": 100 * err.mean(), "far_edge_max": 100 * err.max(), "overhang_max": 100 * outside.max(),
            "area_ratio": area / (side * side / 2), "uncovered_near": 1 - covered.mean()}


def overlay(after_rgb, corners, h, side):
    img = Image.fromarray(np.asarray(after_rgb, np.uint8))
    draw = ImageDraw.Draw(img)
    draw.line([tuple(c) for c in corners] + [tuple(corners[0])], fill=(0, 255, 0), width=3)
    ends = apply(np.linalg.inv(h), np.array([[0, side / 2], [side, side / 2]]))
    draw.line([tuple(e) for e in ends], fill=(255, 0, 0), width=3)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--side", type=float, default=0.30, help="cloth side (m)")
    ap.add_argument("--near", choices=list(NEAR_DIR), default="bottom")
    ap.add_argument("--dark", type=float, default=None, help="grey threshold; default Otsu on the before photo")
    ap.add_argument("--cloth-colour", choices=COLOURS, default="dark", help="what sets the cloth apart from the table")
    ap.add_argument("--out", help="save the after photo with the before square (green) and the ideal far edge (red)")
    args = ap.parse_args(argv)
    before, after = (np.asarray(Image.open(p).convert("RGB")) for p in (args.before, args.after))
    res = score(before, after, args.side, args.near, args.dark, args.cloth_colour)
    print(json.dumps({k: round(float(v), 4) for k, v in res.items()}))
    if args.out:
        h, corners, _ = calibrate(before, args.side, args.near, args.dark, args.cloth_colour)
        overlay(after, corners, h, args.side).save(args.out)


if __name__ == "__main__":
    sys.exit(main())

"""Dump a few frames of an mp4 as PNGs so a demo can be checked by eye (QD, 2026-10-16).
    .venv/Scripts/python.exe scripts/video_frames.py video.mp4 out_dir [--at 0.1 0.5 0.95]  (fractions of the length)
"""
import argparse
from pathlib import Path

import imageio.v3 as iio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--at", type=float, nargs="+", default=[0.05, 0.5, 0.98])
    a = ap.parse_args()
    frames = iio.imread(a.video, plugin="FFMPEG", index=None)
    n = len(frames)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.at:
        i = min(n - 1, int(f * n))
        p = out / f"{Path(a.video).stem}_{i:05d}.png"
        iio.imwrite(p, frames[i])
        print(p, f"{i}/{n}")


if __name__ == "__main__":
    main()

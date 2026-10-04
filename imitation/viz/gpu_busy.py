"""GPU busy fraction over a time window, from an nvidia-smi sample log (roadmap M5c.3).

    python -m imitation.viz.gpu_busy --samples runs/m5c3_gpu.csv --window runs/m5c3_window.csv

`samples`: `epoch,util_pct,mem_mib` rows (nvidia-smi utilization.gpu every few seconds).
`window`: `dagger_start,<epoch>` / `dagger_end,<epoch>` rows. Prints the mean utilisation and
the share of samples above a busy threshold inside the window, as JSON.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def busy(samples, start, end, threshold=10.0):
    t, u = samples[:, 0], samples[:, 1]
    inside = (t >= start) & (t <= end)
    u = u[inside]
    if not len(u):
        return {"n_samples": 0}
    return {"n_samples": int(len(u)), "window_s": float(end - start), "mean_util_pct": round(float(u.mean()), 1),
            "busy_share": round(float((u > threshold).mean()), 3), "busy_threshold_pct": threshold,
            "p50_util_pct": float(np.median(u))}


def read_samples(path):
    rows = []
    for line in open(path):
        parts = line.strip().split(",")
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except (ValueError, IndexError):
            continue                                          # header or a failed nvidia-smi read
    return np.array(rows).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--threshold", type=float, default=10.0)
    args = ap.parse_args()
    marks = dict(line.strip().split(",") for line in open(args.window) if "," in line)
    print(json.dumps(busy(read_samples(args.samples), float(marks["dagger_start"]), float(marks["dagger_end"]),
                          args.threshold), indent=1))


if __name__ == "__main__":
    main()

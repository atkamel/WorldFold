"""Figures for a recovery_replay trace (Phase F3b): per arm, the corner and the gripper over the episode.

Left panel: top-down (x, y) -- the corner's path, the gripper's path, the arm base, the drop / knock point, and
where every descend ended. Right panel: the gripper's distance to the corner and its height above the corner vs
time, with the expert's phases as coloured bands.

    python isaac/plot_trace.py outputs/isaac/recovery_friction/drop_combo/traces/630002.json --out docs/reports/media
"""

import argparse
import json
from itertools import groupby
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = {"left_": (-0.22, -0.30), "right_": (0.22, -0.30)}       # mujuco.cloth_params ARM_BASE_*
PHASE_COLOURS = {"approach": "#cfe2f3", "descend": "#f9cb9c", "close": "#ea9999", "lift": "#b6d7a8",
                 "carry": "#d9ead3", "place": "#d0e0e3", "hold": "#fff2cc", "release": "#ead1dc",
                 "retreat": "#eeeeee", "done": "#ffffff"}


def plot(trace_path, out_dir):
    tr = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    seed, (t0, k) = tr["seed"], tr["knock"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, (p, a) in enumerate(tr["arms"].items()):
        c, s = np.asarray(a["corner"]), np.asarray(a["site"])
        ph = [x.lstrip("K") for x in a["phase"]]
        ax = axes[row, 0]
        ax.plot(100 * c[:, 0], 100 * c[:, 1], color="#c0392b", lw=1.5, label="corner")
        ax.plot(100 * s[:, 0], 100 * s[:, 1], color="#2c3e50", lw=0.8, alpha=0.7, label="gripper")
        ax.scatter([100 * BASE[p][0]], [100 * BASE[p][1]], marker="s", s=80, color="k", label="arm base")
        ax.scatter([100 * c[0, 0]], [100 * c[0, 1]], marker="o", color="#c0392b", label="corner start")
        if t0 < len(c):
            ax.scatter([100 * c[t0, 0]], [100 * c[t0, 1]], marker="X", s=90, color="#e67e22", label="drop / knock")
        i = 0
        for key, g in groupby(ph):
            n = len(list(g))
            if key == "descend" and i + n - 1 >= t0:
                e = i + n - 1
                ax.scatter([100 * s[e, 0]], [100 * s[e, 1]], marker="v", color="#8e44ad", s=40)
            i += n
        ax.scatter([], [], marker="v", color="#8e44ad", label="descend end")
        ax.set_title(f"seed {seed} {p.rstrip('_')}: top-down (cm)")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        ax = axes[row, 1]
        i = 0
        for key, g in groupby(ph):
            n = len(list(g))
            ax.axvspan(i, i + n, color=PHASE_COLOURS.get(key, "#ffffff"), alpha=0.8, lw=0)
            if n > 12:
                ax.text(i + n / 2, 0.97, key, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
            i += n
        t = np.arange(len(c))
        ax.plot(t, 100 * np.linalg.norm(s[:, :2] - c[:, :2], axis=1), color="#2c3e50", label="gripper-corner xy (cm)")
        ax.plot(t, 100 * (s[:, 2] - c[:, 2]), color="#16a085", label="gripper above corner (cm)")
        ax.axvline(t0, color="#e67e22", ls="--", lw=1)
        ax.set_ylim(-5, 30)
        ax.set_xlabel("control step")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title("distance to the corner, phases shaded")
    fig.tight_layout()
    out = Path(out_dir) / f"f3b_trace_{Path(trace_path).parent.parent.name}_{seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default="docs/reports/media")
    args = ap.parse_args()
    for path in args.traces:
        print(plot(path, args.out))

"""Independent recheck of the Newton held-out score: re-reads the raw Motive CSVs and recomputes the paper metric for
results/newton/final_pred.npz without importing coltraro_data / replay / newton_cloth (own parser, own marker
ordering, own resampling, own weights). Prints per-tag means next to final.json's.

  python verify_held.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "coltraro", "raw", "tracking_dataset", "Free-hanging")
RES = os.path.join(HERE, "results", "newton")


def read_motive(path):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    hdr = next(i for i, r in enumerate(rows) if len(r) > 1 and r[0].strip() == "Frame")
    mrow = next(i for i, r in enumerate(rows) if len(r) > 2 and r[2].strip() == "Marker")
    lab = rows[mrow + 1]                                                        # the marker-name row
    names = [lab[c].strip() for c in range(2, len(lab), 3)]
    data = np.array([[float(x) if x.strip() else np.nan for x in r[:2 + 3 * len(names)]] for r in rows[hdr + 1:] if r],
                    float)
    t = data[:, 1]
    P = data[:, 2:].reshape(len(t), len(names), 3)[:, :, [0, 2, 1]]          # Y-up -> Z-up
    return t, P, names


def grid_order(P0, n):
    """Rows by height (top first), then along the top edge's direction."""
    R, C = {20: (4, 5), 12: (3, 4)}[n]
    by_z = np.argsort(-P0[:, 2])
    top = by_z[:C]
    a, b = P0[top[np.argmin(P0[top, 1])]], P0[top[np.argmax(P0[top, 1])]]
    axis = (b - a)[:2] / np.linalg.norm((b - a)[:2])
    order = []
    for r in range(R):
        row = by_z[r * C:(r + 1) * C]
        order += list(row[np.argsort(P0[row, :2] @ axis)])
    return np.array(order), R, C


def lumped(X, R, C):
    w = np.zeros(len(X))
    for i in range(R - 1):
        for j in range(C - 1):
            q = [i * C + j, i * C + j + 1, (i + 1) * C + j + 1, (i + 1) * C + j]
            a, b, c, d = X[q]
            w[q] += (0.5 * np.linalg.norm(np.cross(b - a, d - a)) + 0.5 * np.linalg.norm(np.cross(b - c, d - c))) / 4
    return w


def resample(t, P, hz=100.0):
    tn = np.arange(0.0, t[-1] + 1e-9, 1.0 / hz)
    j = np.clip(np.searchsorted(t, tn, side="right") - 1, 0, len(t) - 2)
    w = ((tn - t[j]) / (t[j + 1] - t[j]))[:, None, None]
    return (1 - w) * P[j] + w * P[j + 1]


def real_track(name):
    t, P, names = read_motive(os.path.join(RAW, name + ".csv"))
    n = len(names)
    full = np.where(~np.isnan(P).any(axis=(1, 2)))[0][0]
    t, P = t[full:] - t[full], P[full:]
    if all(re.fullmatch(r"\d+", s) for s in names):
        P = P[:, np.argsort([int(s) for s in names])]
    order, R, C = grid_order(P[0], n)
    if not all(re.fullmatch(r"\d+", s) for s in names):
        P = P[:, order]
    else:
        same = np.array_equal(order, np.arange(n))
        if not same:
            print(f"  note {name}: label order and geometric order differ (labels used)")
    return resample(t, P), R, C


def lit_marker(pred, real, w):
    d = pred[1:] - real[1:]
    ok = ~np.isnan(d).any(axis=2)
    d2 = np.where(ok, np.nansum(d * d, axis=2), 0.0)
    return float(np.mean(100.0 * np.sqrt((ok * w[None] * d2).sum(1))))


def main(which="final"):
    """which = final (newton_defaults, newton_fitted) or final_physbend (the hand-off model)."""
    pred = np.load(os.path.join(RES, f"{which}_pred.npz"))
    fin = {f"{r['tag']}|{r['name']}": r for r in json.load(open(os.path.join(RES, f"{which}.json")))["runs"]}
    by_tag = {}
    worst, skipped = 0.0, 0
    if set(pred.files) != set(fin):
        print(f"  key sets differ: {len(pred.files)} predictions vs {len(fin)} scored runs")
        return 1
    for key in pred.files:
        tag, name = key.split("|")
        real, R, C = real_track(name)
        p = pred[key].astype(float)
        if len(p) != len(real):
            print(f"  length mismatch {key}: pred {len(p)} real {len(real)}")
            skipped += 1
            continue
        # frame 0 of a prediction is the recorded frame 0, so it fixes the marker correspondence (the harness numbers
        # 3 unlabelled polyester files as the column mirror of this script's grid; same geometry, same physics)
        dd = np.linalg.norm(p[0][:, None] - real[0][None], axis=2)
        perm = dd.argmin(axis=1)
        if len(set(perm)) != len(perm) or dd.min(axis=1).max() > 1e-5:
            print(f"  cannot align {key}")
            skipped += 1
            continue
        if not np.array_equal(perm, np.arange(len(perm))):
            real = real[:, perm]
            if not np.allclose(lumped(real[0], R, C).sum(), lumped(real[0][np.argsort(perm)], R, C).sum()):
                print(f"  note {key}: aligned by frame 0 (non-mirror permutation)")
        v = lit_marker(p, real, lumped(real[0], R, C)) if not fin[key]["nan"] else fin[key]["lit_marker"]
        worst = max(worst, abs(v - fin[key]["lit_marker"]))
        by_tag.setdefault(tag, []).append((v, fin[key]["lit_marker"]))
    for tag, vs in by_tag.items():
        a = np.array(vs)
        print(f"{tag:18s} recomputed mean {a[:, 0].mean():.4f}  final.json {a[:, 1].mean():.4f}  ({len(a)} recordings)")
    n_ok = sum(len(v) for v in by_tag.values())
    print(f"max per-recording difference {worst:.2e}; scored {n_ok}, skipped {skipped}")
    return 0 if worst < 1e-3 and skipped == 0 and all(len(v) == 32 for v in by_tag.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "final"))

"""Coltraro et al. (IJRR 2025) cloth mocap dataset: download, parse, marker grid, paper reference tables.

Data: Zenodo 14644526 tracking_dataset.zip (CC BY-NC-SA per its License.txt; research use only, never committed).
Free-hanging recordings: 4 fabrics x A2/A3 x shake/twist x slow/fast x hanger (repetition I) / hands (repetition II),
120 Hz OptiTrack, metres, Y-up in the file (we return z-up). Markers are labelled 1..N row-major from the top row,
so labels 1 and C are the two held upper corners.
"""
from __future__ import annotations

import hashlib
import os
import re
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "coltraro")
ZIP = os.path.join(DATA, "tracking_dataset.zip")
ZIP_URL = "https://zenodo.org/api/records/14644526/files/tracking_dataset.zip/content"
ZIP_SHA256 = "14916efa89a26d991c024024cc9449397d3a6f654311e621bb91e9602e231e1a"
RAW = os.path.join(DATA, "raw", "tracking_dataset")

# areal density kg/m^2 (paper Fig. 1); the files call the paper's "Stiff-cotton" just "cotton"
RHO = {"polyester": 0.1042, "wool": 0.1804, "denim": 0.3046, "cotton": 0.3046}
GRID = {20: (4, 5), 12: (3, 4)}            # markers -> (rows, cols); A2 4x5, A3 3x4
SIZE_M = {"A2": (0.594, 0.42), "A3": (0.42, 0.297)}   # (width along the held edge, height)
NAME_RE = re.compile(r"(cotton|denim|polyester|wool)_(A2|A3)_(shake|twist)_(slow|fast)_(hanger|hands)\.csv$")


def ensure_data():
    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(ZIP):
        urllib.request.urlretrieve(ZIP_URL, ZIP)
    with open(ZIP, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    if h != ZIP_SHA256:
        raise RuntimeError(f"zip sha256 {h} != {ZIP_SHA256}")
    if not os.path.isdir(RAW):
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(os.path.join(DATA, "raw"))
    return RAW


def parse_name(path):
    m = NAME_RE.search(os.path.basename(path))
    if not m:
        return None
    fabric, size, motion, speed, mode = m.groups()
    # The paper calls repetition I "with a special hanger", but its Table 2 speeds (A2 twist, 16 values) match our
    # *_hands files as repetition I and *_hanger files as repetition II to within 0.003 m/s (checked 2026-10-16), so the
    # released suffixes and the paper's labels are swapped. `rep` is the paper's table key; `mode` is the file suffix.
    return {"fabric": fabric, "size": size, "motion": motion, "speed": speed, "mode": mode,
            "rep": "I" if mode == "hands" else "II",
            "key": f"{fabric}_{size}_{motion}_{speed}", "name": os.path.basename(path)[:-4]}


def free_hanging():
    """All 64 free-hanging recordings as dicts with path + parsed name, sorted."""
    d = os.path.join(ensure_data(), "Free-hanging")
    out = []
    for f in sorted(os.listdir(d)):
        info = parse_name(f)
        if info:
            info["path"] = os.path.join(d, f)
            out.append(info)
    return out


def load_csv(path):
    """Return t (s, from 0), P (T, N, 3) z-up metres ordered by label 1..N (NaN where a marker is missing)."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    head = next(k for k, l in enumerate(lines) if l.startswith("Frame,Time"))
    ids = lines[3].split(",")[2:][0::3]
    A = np.genfromtxt(path, delimiter=",", skip_header=head + 1)
    t = A[:, 1] - A[0, 1]
    P = A[:, 2:].reshape(len(t), -1, 3)[:, :, [0, 2, 1]]       # file is Y-up: (x, y, z) <- (X, Z, Y)
    if all(x.strip().isdigit() for x in ids):
        labels = np.array([int(x) for x in ids])
        order = np.argsort(labels)
        if not np.array_equal(labels[order], np.arange(1, len(labels) + 1)):
            raise ValueError(f"{path}: labels are not 1..N")
    else:                                                      # unlabelled export (polyester, bare hands)
        order = geometric_order(P)
    return t, P[:, order, :].copy()


def load_raw(path):
    """t, P (T, K, 3) z-up in file column order, no labelling (for unlabelled exports with extra rod markers)."""
    with open(path, encoding="utf-8") as fh:
        head = next(k for k, l in enumerate(fh.read().splitlines()) if l.startswith("Frame,Time"))
    A = np.genfromtxt(path, delimiter=",", skip_header=head + 1)
    return A[:, 1] - A[0, 1], A[:, 2:].reshape(len(A), -1, 3)[:, :, [0, 2, 1]]


def geometric_order(P):
    """Row-major grid order from geometry on the first full frame: rows by height (top first), columns by position
    along the top row's horizontal direction. Used only where the file has no 1..N labels; check_grid validates."""
    X = P[first_full_frame(P)]
    R, C = GRID[X.shape[0]]
    by_z = np.argsort(-X[:, 2])
    rows = [by_z[r * C:(r + 1) * C] for r in range(R)]
    top = X[rows[0], :2]
    axis = np.linalg.svd(top - top.mean(0))[2][0]
    axis = np.array([axis[0], axis[1], 0.0])
    return np.concatenate([row[np.argsort(X[row] @ axis)] for row in rows])


def grid_shape(n_markers):
    return GRID[n_markers]


def corner_ids(n_markers):
    R, C = grid_shape(n_markers)
    return [0, C - 1]


def coarse_quads(R, C):
    """Quad connectivity of the marker grid (row-major vertex ids), cyclic order as clothilde expects."""
    T = []
    for r in range(R - 1):
        for c in range(C - 1):
            v = r * C + c
            T.append([v, v + 1, v + C + 1, v + C])
    return np.array(T, dtype=int)


def lumped_areas(X, T):
    """Lumped (row-sum) bilinear mass of a quad mesh = quarter of each quad's area per vertex (exact for planar
    parallelograms; we use the two-triangle area of each quad)."""
    m = np.zeros(len(X))
    for q in T:
        a, b, c, d = X[q]
        area = 0.5 * np.linalg.norm(np.cross(b - a, d - a)) + 0.5 * np.linalg.norm(np.cross(b - c, d - c))
        m[q] += area / 4.0
    return m


def first_full_frame(P):
    ok = ~np.isnan(P).any(axis=(1, 2))
    if not ok.any():
        raise ValueError("no frame with every marker present")
    return int(np.argmax(ok))


def fill_track(t, x):
    """Linear-in-time fill of NaN gaps in one marker track (T, 3); edges held. Returns filled copy + gap count."""
    x = x.copy()
    bad = np.isnan(x).any(axis=1)
    if bad.any():
        good = ~bad
        for k in range(3):
            x[bad, k] = np.interp(t[bad], t[good], x[good, k])
    return x, int(bad.sum())


def check_grid(P0, R, C):
    """Sanity of the label -> grid mapping on a full frame: rows go down in z, neighbours ~ equal spacing."""
    G = P0.reshape(R, C, 3)
    rows_down = bool(np.all(np.diff(G[:, :, 2].mean(axis=1)) < 0))
    dh = np.linalg.norm(np.diff(G, axis=1), axis=2)
    dv = np.linalg.norm(np.diff(G, axis=0), axis=2)
    return {"rows_down": rows_down, "h_mean_m": float(dh.mean()), "h_cv": float(dh.std() / dh.mean()),
            "v_mean_m": float(dv.mean()), "v_cv": float(dv.std() / dv.mean())}


# Paper (Coltraro et al., MMT 2025 preprint, IRI scidoc 3007) Tables 3 (repetition I, hanger) and 4 (II, hands):
# per recording (fabric, size, motion, speed) -> (error cm, spatial std cm, alpha, delta), fitted in-sample by sweep.
_T3 = """cotton A2 shake slow .433 .661 .336 .16
cotton A2 shake fast .516 .77 .657 .181
cotton A2 twist slow .324 .533 .496 .222
cotton A2 twist fast .562 .918 .817 .284
cotton A3 shake slow .268 .711 .336 .201
cotton A3 shake fast .336 .798 .817 .222
cotton A3 twist slow .267 .711 .336 .243
cotton A3 twist fast .315 .823 .898 .263
wool A2 shake slow .511 .836 .199 .058
wool A2 shake fast .991 1.485 .437 .082
wool A2 twist slow .451 .738 .437 .131
wool A2 twist fast .844 1.314 .532 .095
wool A3 shake slow .289 .747 .294 .082
wool A3 shake fast .447 1.083 .579 .107
wool A3 twist slow .219 .563 .294 .107
wool A3 twist fast .361 .919 .579 .107
polyester A2 shake slow .887 1.187 .17 .026
polyester A2 shake fast 1.36 1.819 .307 .041
polyester A2 twist slow .444 .685 .197 .041
polyester A2 twist fast 1.152 1.617 .307 .048
polyester A3 shake slow .463 .903 .142 .033
polyester A3 shake fast .471 .99 .28 .041
polyester A3 twist slow .243 .689 .115 .048
polyester A3 twist fast .359 .854 .307 .055
denim A2 shake slow .628 1.138 .416 .139
denim A2 shake fast .841 1.252 .898 .201
denim A2 twist slow .438 .932 .416 .222
denim A2 twist fast .985 1.35 .978 .263
denim A3 shake slow .306 .877 .416 .181
denim A3 shake fast .334 .959 .737 .201
denim A3 twist slow .266 .764 .416 .243
denim A3 twist fast .288 .804 .737 .305"""
_T4 = """cotton A2 shake slow .446 .682 .416 .139
cotton A2 shake fast .561 .898 .737 .181
cotton A2 twist slow .351 .569 .496 .222
cotton A2 twist fast .681 1.098 .898 .305
cotton A3 shake slow .256 .651 .336 .181
cotton A3 shake fast .331 .885 .737 .222
cotton A3 twist slow .282 .716 .496 .284
cotton A3 twist fast .329 .838 .978 .305
wool A2 shake slow .559 .842 .247 .07
wool A2 shake fast .964 1.456 .532 .095
wool A2 twist slow .421 .72 .294 .095
wool A2 twist fast .958 1.408 .627 .107
wool A3 shake slow .316 .796 .342 .082
wool A3 shake fast .403 1.028 .532 .107
wool A3 twist slow .235 .627 .294 .095
wool A3 twist fast .468 1.076 .627 .144
polyester A2 shake slow 1.025 1.327 .142 .026
polyester A2 shake fast 1.378 1.761 .28 .041
polyester A2 twist slow .52 .811 .17 .033
polyester A2 twist fast 1.22 1.696 .252 .041
polyester A3 shake slow .467 1.227 .17 .033
polyester A3 shake fast .575 1.247 .28 .041
polyester A3 twist slow .543 1.429 .17 .033
polyester A3 twist fast .528 1.166 .417 .055
denim A2 shake slow .512 .978 .336 .139
denim A2 shake fast .739 1.083 .817 .181
denim A2 twist slow .526 .983 .336 .222
denim A2 twist fast .876 1.318 .978 .243
denim A3 shake slow .376 1.194 .416 .181
denim A3 shake fast .368 1.15 .336 .201
denim A3 twist slow .334 .991 .978 .222
denim A3 twist fast .375 1.121 .496 .222"""


def _table(txt):
    out = {}
    for line in txt.splitlines():
        f, s, m, sp, e, sd, a, d = line.split()
        out[f"{f}_{s}_{m}_{sp}"] = {"err_cm": float(e), "std_cm": float(sd), "alpha": float(a), "delta": float(d)}
    return out


PAPER = {"I": _table(_T3), "II": _table(_T4)}
PAPER_T5_REP_I = {"polyester": 0.67, "wool": 0.51, "denim": 0.50, "cotton": 0.38, "A2": 0.71, "A3": 0.33,
                  "shake": 0.57, "twist": 0.47, "slow": 0.40, "fast": 0.63, "global": 0.51}

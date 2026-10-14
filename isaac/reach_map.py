"""IK reach map (Phase Q2): for each arm, the x range it can pinch top-down along rows of the table at a given height.
Used to size which second folds the two arms can carry (isaac/reach_check_qf.py). Isaac venv, repo root:

    python isaac/reach_map.py > outputs/isaac/reach_map.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaac.reach_check import PINCH_HEIGHT, SEED_Q  # noqa: E402
from mujuco.cloth_params import TABLE_TOP_Z  # noqa: E402

TOL = 0.005


def main():
    from isaac.pinch import PinchIK
    ik = PinchIK()
    xs = np.round(np.arange(-0.30, 0.31, 0.03), 2)
    for z_off, label in ((PINCH_HEIGHT, "table"), (0.10, "+10cm")):
        for y in (-0.285, -0.21, -0.135, -0.06):
            for p in ("left_", "right_"):
                ok = []
                for x in xs:
                    errs = [ik.solve(p, np.array([x, y, TABLE_TOP_Z + z_off]), np.array(j, float), SEED_Q[p])[1]
                            for j in ((0, 1, 0), (1, 0, 0), (-1, 0, 0))]
                    ok.append(min(errs) < TOL)
                r = xs[np.array(ok)]
                print(f"{label:6s} y={y:+.3f} {p:6s} x reach [{r.min() if len(r) else None}, "
                      f"{r.max() if len(r) else None}]  {''.join('#' if o else '.' for o in ok)}", flush=True)


if __name__ == "__main__":
    main()

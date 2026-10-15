"""R2 probe (2026-10-15): the fold-over's left arm on the v0+v10 stack from the clean start (isaac.diag_clean_fold),
alone or after the right arm has closed on v5. Separates reach / IK from contact with the right arm.

Logs every IK solve made while planning (target, error) and per step the left arm's phase, site, pinch point and joint
error to its target. Isaac venv, repo root, WORLDFOLD_QF_SECOND=fold_over WORLDFOLD_QF_START=clean:

    python -u isaac/r2_left_probe.py --mode alone --seeds 660000:660002 --out outputs/imitation/isaac_friction/r2/probe.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("alone", "both"), default="alone",
                    help="alone: the right arm stays at home; both: the expert's own sequence (right first)")
    ap.add_argument("--seeds", default="660000:660002")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="ATTR=VALUE",
                    help="IsaacArmExpert class attribute override, e.g. PINCH_INSET=0.02")
    args = ap.parse_args()
    assert os.environ.get("WORLDFOLD_QF_START") == "clean", "needs WORLDFOLD_QF_START=clean"
    from imitation.tasks import make_env
    from imitation.teachers.scripted import ScriptedTeacher
    from isaac.fold_expert import IsaacArmExpert
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(IsaacArmExpert, k, float(v))
    env = make_env("isaac_friction", task="quarter_fold")
    qfe = ScriptedTeacher(env, seed=0).expert
    base = env.unwrapped
    solves = []
    orig = IsaacArmExpert._solve

    def logged(self, tip, jaw, q, tol=None, **kw):
        q1, err = orig(self, tip, jaw, q, tol=tol, **kw)
        solves.append({"arm": self.prefix[0], "stage": self._stage(), "tip": np.round(tip, 4).tolist(),
                       "err": round(float(err), 4)})
        return q1, err
    IsaacArmExpert._solve = logged

    a, b = (int(v) for v in args.seeds.split(":"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in range(a, b):
        env.reset(seed=seed)
        qfe.reset()
        left = qfe.experts[(1, "left_")]
        solves.clear()
        steps, plans, grasped_at = [], [], None
        for t in range(args.steps):
            if args.mode == "alone":
                act = np.zeros(12, dtype=np.float32)
                act[0:6] = left.act()
                act[11] = 1.0
            else:
                act = qfe.act()
            ph = left.PHASES[left.phase]
            site = np.asarray(base.gripper_position("left_"), dtype=float)
            stack = np.mean([base.cloth_positions()[v] for v in left.corners], axis=0)
            qerr = (float(np.max(np.abs(left.q_target - left._q()))) if left.q_target is not None else None)
            if left.plan is not None and (not plans or plans[-1] is not left.plan):
                plans.append(left.plan)
            links = {}
            if t % 25 == 0 or (grasped_at is None and base.grasp_active("left_")):
                # every arm link's height and the cloth beneath it (R2: is the reaching arm lying on the strip?)
                cp = base.lab.copies[base._copy]
                cloth = base.cloth_positions()
                for pr, arm in cp.arms.items():
                    pos = arm.data.body_link_pos_w[0].cpu().numpy() - (cp.offset if base._copy else 0.0)
                    rows = []
                    for name, p in zip(arm.body_names, pos):
                        near = np.linalg.norm(cloth[:, :2] - p[:2], axis=1) < 0.03
                        top = float(cloth[near, 2].max()) if near.any() else None
                        rows.append([name, np.round(p, 4).tolist(), None if top is None else round(top, 4)])
                    links[pr] = rows
            tilt = {}
            for pr in ("left_", "right_"):        # the gripper frame's z axis vs straight down, degrees
                _, R, _, _ = base.lab.site_pose(pr, base._copy)
                tilt[pr] = round(float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))), 1)
            rph = qfe.experts[(1, "right_")]
            steps.append({"t": t, "ph": ph, "links": links, "tilt": tilt, "rph": rph.PHASES[rph.phase],
                          "rg": int(base.grasp_active("right_")),
                          "rsite": np.round(base.gripper_position("right_"), 4).tolist(), "w": int(left.waiting), "site": np.round(site, 4).tolist(),
                          "stack": np.round(stack, 4).tolist(), "qerr": None if qerr is None else round(qerr, 3),
                          "g": int(base.grasp_active("left_"))})
            if grasped_at is None and base.grasp_active("left_"):
                grasped_at = t
            env.step(act)
            if grasped_at is not None and t > grasped_at + 40:
                break
        row = {"seed": seed, "mode": args.mode, "knobs": args.set, "grasped_at": grasped_at, "solves": list(solves),
               "n_plans": len(plans), "steps": steps}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        worst = max((s["err"] for s in solves), default=None)
        print("EP", json.dumps({"seed": seed, "mode": args.mode, "grasped_at": grasped_at, "n_solves": len(solves),
                                "worst_err": worst, "n_plans": len(plans)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

"""Scripted quarter fold on Isaac (profile "quarter"): QuarterFoldExpert with an IsaacArmExpert per move. Per episode it
logs the stage reached, the termination reason, each move's grasp and placement, why fold 2 failed (FOLD2_CAUSES), and
the layer alignment after fold 1 and at the end (cloth_fold_rl.fold_alignment); optionally a video.

    python isaac/quarter_fold_run.py --seeds 0,1 --out outputs/isaac/quarter/run --video
    WORLDFOLD_EXPERT_PARAMS="SQUEEZE=-0.05" python isaac/quarter_fold_run.py ...   # IsaacArmExpert overrides
    python isaac/quarter_fold_run.py --profile roy --fold2 stack ...   # the Roy-port step-3 stack test

Each fold-2 move that carries a stack (two corners) is scored "lifted" when, at some step, both corners are at least
LIFT_MIN above where they lay when fold 2 began, both within HOLD_DIST of the gripper site, and less than STACK_GAP
apart in height: both layers up in the jaws (isaac/grasp_metrics.py has the constants).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STACK_SPLIT = 0.03      # the stack's two corners further apart than this at the end: one layer was carried
STACK_GAP = 0.01        # m in height between a lifted stack's corners: both layers in the jaws (plan step 3)
FOLD2_CAUSES = ("no_grasp", "slip", "layers_split", "misplaced", "cloth_dragged", "unstable", "timeout")


def stack_step(a, corners, rest_z, site):
    """Update a stack move's score a = {"lifted", "rise"} with one step: corners (2, 3), their fold-2 start heights
    rest_z (2,), the gripper site (3,)."""
    from isaac.grasp_metrics import HOLD_DIST, LIFT_MIN
    rise = corners[:, 2] - rest_z
    near = np.linalg.norm(corners - site, axis=1) < HOLD_DIST
    a["rise"] = [round(max(r0, float(r)), 4) for r0, r in zip(a["rise"], rise)]
    a["lifted"] |= bool((rise >= LIFT_MIN).all() and near.all() and abs(corners[0, 2] - corners[1, 2]) < STACK_GAP)
    return a


def fold2_cause(rec):
    """The first thing that went wrong in fold 2, from the episode record."""
    if rec["success"]:
        return None
    if rec["reason"] in ("cloth_dragged", "unstable"):
        return rec["reason"]
    m = rec["moves"].get("1")
    if m is None:
        return "timeout"
    if not all(a["held"] for a in m.values()):
        return "no_grasp"
    if rec["stack_gap"] > STACK_SPLIT:
        return "layers_split"
    if not all(a["held_to_place"] for a in m.values()):
        return "slip"
    if not all(a["final_dist"] < 0.05 for a in m.values()):
        return "misplaced"
    return "timeout"


def run(env, expert, seed, frames=None, trace=None):
    from cloth_fold_rl.fold_alignment import score
    from cloth_fold_rl.quarter_fold_env import CLOTH_0, CLOTH_10
    base = env.unwrapped
    _, info = env.reset(seed=seed)
    expert.reset()
    moves, fold1, rest, stack = {}, None, None, {}
    for t in range(base.max_episode_steps):
        stage = env.stage
        _, _, terminated, truncated, info = env.step(expert.act())
        if frames is not None:
            frames.append(base.render_rig()["top"].transpose(1, 2, 0))
        phases = expert.phases()
        for k, mv in enumerate(env.stages[stage].moves):
            a = moves.setdefault(str(stage), {}).setdefault(mv.prefix, {"held": False, "held_to_place": False,
                                                                         "final_dist": None})
            held = bool(info["grasped"][mv.prefix]) if env.stage == stage else a["held"]
            a["held"] |= held
            if held and phases[mv.prefix] in ("place", "hold", "release"):
                a["held_to_place"] = True
            if env.stage == stage:
                a["final_dist"] = round(float(info["move_distance"][k]), 4)
        if trace is not None and t % 5 == 0:
            row = {"t": t, "stage": env.stage}
            for k, mv in enumerate(env.stages[env.stage].moves):
                p = mv.prefix
                row[p] = {"phase": phases[p], "grip": np.round(base.gripper_position(p), 3).tolist(),
                          "jaw": round(float(base.joint_positions(p)[5]), 3), "held": bool(info["grasped"][p]),
                          "carried": np.round(base.cloth_positions()[list(mv.corners)].mean(axis=0), 3).tolist(),
                          "goal": np.round(env.goal(mv), 3).tolist(),
                          "q": np.round(base.joint_positions(p)[:5], 3).tolist(),
                          "q_target": None if expert.experts[(env.stage, p)].q_target is None else
                          np.round(expert.experts[(env.stage, p)].q_target, 3).tolist()}
            trace.append(row)
        if env.stage == 1:
            pos = base.cloth_positions()
            rest = pos[:, 2].copy() if rest is None else rest
            for mv in env.stages[1].moves:
                if len(mv.corners) != 2:
                    continue
                c = pos[list(mv.corners)]
                if mv.prefix not in stack:     # start_gap: how far apart fold 1 left the two corners
                    stack[mv.prefix] = {"lifted": False, "rise": [0.0, 0.0],
                                        "start_gap": round(float(np.linalg.norm(c[0, :2] - c[1, :2])), 4)}
                stack_step(stack[mv.prefix], c, rest[list(mv.corners)], base.gripper_position(mv.prefix))
        if env.stage == 1 and fold1 is None:
            s = score(base.cloth_positions(), folds=(("y", 5),))
            fold1 = {"mean": round(s["mean"], 4), "edge_max": round(s["edge_max"], 4)}
        if terminated or truncated:
            break
    pos = base.cloth_positions()
    s = score(pos, folds=env.alignment_folds)
    return {"seed": seed, "steps": t + 1, "stage": info["stage"], "success": info["success"],
            "reason": info["termination_reason"] or "truncated", "retries": sum(expert.retries.values()),
            "anchor_drift": round(info["anchor_drift"], 4), "moves": moves, "fold1_align": fold1,
            "align": {"mean": round(s["mean"], 4), "edge_max": round(s["edge_max"], 4), "passed": s["passed"]},
            "stack_gap": round(float(np.linalg.norm(pos[CLOTH_0, :2] - pos[CLOTH_10, :2])), 4), "stack": stack}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="outputs/isaac/quarter/run")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--profile", default="quarter")
    ap.add_argument("--fold2", default=None, help="stack: fold 2 lifts each fold-1 stack (make_quarter_fold_env)")
    ap.add_argument("--overshoot", default=None, help="fold-1 'lx,ly,rx,ry' (m) placement offsets, as grasp_bench's")
    ap.add_argument("--no-retries", action="store_true", help="the expert's placement retries off (one attempt)")
    ap.add_argument("--max-steps", type=int, default=None, help="episode cap (default QUARTER_MAX_STEPS)")
    args = ap.parse_args()
    from isaac.isaac_env import QUARTER_MAX_STEPS, make_quarter_fold_env
    env = make_quarter_fold_env(profile=args.profile, cameras={"top": 400} if args.video else None, fold2=args.fold2,
                                max_episode_steps=args.max_steps or QUARTER_MAX_STEPS)
    if args.no_retries:
        import cloth_fold_rl.quarter_fold_expert as qfe
        qfe.MAX_RETRIES = 0
    from isaac.fold_expert import OVERSHOOTS, IsaacQuarterFoldExpert, _apply_env_params
    _apply_env_params()
    if args.overshoot:
        lx, ly, rx, ry = (float(v) for v in args.overshoot.split(","))
        OVERSHOOTS[args.profile][(0, "left_")][:] = (lx, ly, 0.0)
        OVERSHOOTS[args.profile][(0, "right_")][:] = (rx, ry, 0.0)
    expert = IsaacQuarterFoldExpert(env, profile=args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        frames = [] if args.video else None
        trace = []
        rec = run(env, expert, seed, frames, trace)
        (out / f"trace_seed{seed}.json").write_text(json.dumps(trace))
        rec["fold2_cause"] = fold2_cause(rec)
        print("EPISODE", json.dumps(rec), flush=True)
        with open(out / "episodes.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if frames:
            import imageio
            imageio.mimwrite(str(out / f"quarter_seed{seed}.mp4"), frames + [frames[-1]] * 15, fps=20, macro_block_size=1)
    env.close()


if __name__ == "__main__":
    main()

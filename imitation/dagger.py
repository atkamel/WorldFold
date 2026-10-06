"""DAgger rounds (design doc 8.2): the student drives, the teacher labels the states
the student actually visits, the labels are aggregated, the student is retrained.

Each round r:
  1. roll out the current best student on fresh seeds. At every replan point the
     scripted teacher labels a chunk from that exact state (snapshot -> expert
     K steps -> restore); with probability beta_r the teacher's chunk is executed
     instead (beta decays per round). A fraction of episodes are knocked off
     course mid-way so recovery states get labelled too.
  2. freeze those episodes as a new dataset version on top of the previous one
     (aggregate, never replace).
  3. retrain from the best checkpoint on the aggregated version.
  4. evaluate on the fixed sets; keep the checkpoint if it beats the best.
The teacher is the scripted expert by default; `--teacher <state checkpoint>` makes a
trained privileged policy the teacher instead (`imitation.teachers.PolicyTeacher`), which
is how the sensor-only student is distilled (roadmap M4.2) -- same loop, nothing else
changes. An image student (`vision` checkpoint) gets a camera-rendering pool
automatically; `--relabel` then retrains on *every* visited step labelled by the teacher
(`train --teacher`) instead of only the replan-point labels. `--shift-fraction` rolls a
share of each round out from shifted cloth poses (M5b.2), and `--score-sets` picks the
sets the keep/stop rule sums.

Keep/stop decisions are in standard errors of the score difference (n >= 200 per set,
M3.1): a round is kept if it gains at all, and the loop stops once a round's gain is
under --min-gain-se. `--resume` continues from history.json after a kill.

    python -m imitation.dagger --init runs/mlp_v1/final.pt --dataset v1 --out runs/dagger_mlp --rounds 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from imitation.data.collect import DEFAULT_ROOT, recovery_perturbation
from imitation.data.dataset import clamp_fraction  # noqa: F401 (re-exported)
from imitation.data.schema import DatasetWriter, load_manifest
from imitation.evaluate import evaluate
from imitation.policies.common import load_policy
from imitation.rollout import EnvPool, PolicyController, rollout
from imitation.seeds import DAGGER_SEED_BASE, SHIFT_SEED_BASE, shifted_pose
from imitation.spec import ACTION_DIM, OBS_DIM
from imitation.train import train


SCORE_SETS = ("id_easy", "recovery")


def score(results, sets=SCORE_SETS):
    """Model selection: success summed over `sets`, equally weighted (default id_easy +
    recovery; M5b.2 on adds id_hard so a round that trades shift for recovery is seen)."""
    return sum(results[k]["success_rate"] for k in sets if k in results)


def gain_in_se(new, old, sets=SCORE_SETS):
    """score(new) - score(old) in standard errors of that difference (independent
    binomial sets, so the variances add)."""
    var = sum(r[k]["success_rate"] * (1 - r[k]["success_rate"]) / r[k]["n"]
              for r in (new, old) for k in sets if k in r)
    return (score(new, sets) - score(old, sets)) / max(var, 1e-12) ** 0.5


def round_seeds(r, episodes, shift_fraction):
    """Round r's rollout seeds: DAgger seeds, plus a share on shifted cloth poses from the
    disjoint SHIFT range (M5b.2). Returns (seeds, reset_options fn or None)."""
    n_shift = int(round(shift_fraction * episodes))
    base = DAGGER_SEED_BASE + 1000 * r
    seeds = list(range(base, base + episodes - n_shift))
    seeds += list(range(SHIFT_SEED_BASE + 1000 * r, SHIFT_SEED_BASE + 1000 * r + n_shift))
    if not n_shift:
        return seeds, None
    return seeds, lambda seed: shifted_pose(seed) if seed >= SHIFT_SEED_BASE else None


def resume_state(out):
    """(best_ckpt, best_eval, dataset, next_round, history) from out/history.json."""
    history = json.loads((Path(out) / "history.json").read_text())
    kept = [h for h in history if h["kept"]][-1]
    return Path(kept["checkpoint"]), kept["eval"], history[-1]["dataset"], history[-1]["round"] + 1, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="imitation-baseline checkpoint to start from")
    ap.add_argument("--dataset", default="v1", help="frozen version the baseline was trained on")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=64, help="student rollouts per round")
    ap.add_argument("--beta0", type=float, default=0.3)
    ap.add_argument("--beta-decay", type=float, default=0.5)
    ap.add_argument("--recovery-fraction", type=float, default=0.3)
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=15_000)
    ap.add_argument("--dagger-weight", type=float, default=4.0)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--eval-sets", nargs="+", default=["id_easy", "id_hard", "recovery"])
    ap.add_argument("--min-gain-se", type=float, default=1.0, help="stop when a round gains less (in SE)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/history.json")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--teacher", default="expert", help="'expert' or a privileged state-policy checkpoint")
    ap.add_argument("--relabel", action="store_true",
                    help="policy teacher: retrain on every visited step relabelled by it")
    ap.add_argument("--shift-fraction", type=float, default=0.0, help="share of rollouts from shifted poses")
    ap.add_argument("--score-sets", nargs="+", default=list(SCORE_SETS))
    ap.add_argument("--allow-failures", action="store_true", help="also train on failed expert demos")
    args = ap.parse_args()
    sets = tuple(args.score_sets)
    teacher_ckpt = None if args.teacher == "expert" else args.teacher
    if args.relabel and not teacher_ckpt:
        raise SystemExit("--relabel needs a policy --teacher")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "history.json"
    log_file = open(out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    tag = out.name
    student = load_policy(args.init, "cpu")
    # a policy teacher labels on the GPU in this process (PolicyController.teacher_policy);
    # the workers only simulate
    teacher_policy = load_policy(teacher_ckpt) if teacher_ckpt else None
    pool_kwargs = {}
    if student.needs_images:
        pool_kwargs.update(render=True, cameras=dict(student.cameras))
    log(f"== teacher: {args.teacher} | student: {student.kind} | score sets {sets} | "
        f"shift fraction {args.shift_fraction} | relabel {args.relabel}")
    with EnvPool(args.workers, pool_kwargs) as pool:
        if args.resume and history_path.exists():
            best_ckpt, best, dataset, first, history = resume_state(out)
            log(f"== resuming at round {first} from {best_ckpt} on {dataset}")
        else:
            best_ckpt, dataset, first = Path(args.init), args.dataset, 1
            log(f"== round 0: evaluating {best_ckpt}")
            best = evaluate(best_ckpt, args.eval_sets, args.eval_n, pool=pool, replan_every=args.replan_every, log=log)
            history = [{"round": 0, "checkpoint": str(best_ckpt), "dataset": dataset, "eval": best, "kept": True}]
            with open(history_path, "w") as f:
                json.dump(history, f, indent=1)
        for r in range(first, args.rounds + 1):
            beta = args.beta0 * args.beta_decay ** (r - 1)
            log(f"== round {r}: beta {beta:.3f}, {args.episodes} student rollouts from {best_ckpt}")
            t0 = time.time()
            policy = load_policy(best_ckpt)
            version = f"{args.dataset}_{tag}_r{r}"
            if (Path(args.root) / version / "manifest.json").exists():     # killed after freezing
                m = load_manifest(args.root, version)
                log(f"   reusing frozen {version} ({m['n_episodes']} episodes)")
            else:
                controller = PolicyController(policy, replan_every=args.replan_every, beta=beta, label=True,
                                              source="dagger", teacher_policy=teacher_policy)
                writer = DatasetWriter(args.root, version, parent=dataset, resume=args.resume,
                                       config={"round": r, "beta": beta, "policy": str(best_ckpt),
                                               "replan_every": args.replan_every, "episodes": args.episodes,
                                               "recovery_fraction": args.recovery_fraction,
                                               "teacher": args.teacher, "shift_fraction": args.shift_fraction})
                all_seeds, reset_options = round_seeds(r, args.episodes, args.shift_fraction)
                seeds = [s for s in all_seeds if s not in writer.done_seeds]
                rollout(pool, seeds, controller, reset_options=reset_options,
                        perturb_fn=recovery_perturbation(args.recovery_fraction),
                        meta_extra={"round": r, "beta": beta, "policy": str(best_ckpt)},
                        on_done=lambda e: writer.add(e, obs_dim=OBS_DIM, action_dim=ACTION_DIM))
                m = writer.freeze()
            new = [e for e in m["episodes"] if e["file"].startswith(f"{version}/")]
            n_labels = sum(e["n_labels"] for e in new)
            log(f"   rollouts: student success {sum(e['success'] for e in new)}/{len(new)}, "
                f"{n_labels} teacher labels, froze {version} ({m['n_episodes']} episodes), {time.time() - t0:.0f}s")

            ckpt, _ = train(policy.kind, version, out / f"round_{r}", root=args.root, steps=args.train_steps,
                            init_from=best_ckpt, dagger_weight=args.dagger_weight,
                            teacher=teacher_ckpt if args.relabel else None,
                            allow_failures=args.allow_failures or bool(args.relabel), log=log)
            res = evaluate(ckpt, args.eval_sets, args.eval_n, pool=pool, replan_every=args.replan_every, log=log)
            gain, gain_se = score(res, sets) - score(best, sets), gain_in_se(res, best, sets)
            kept = gain > 0
            history.append({"round": r, "checkpoint": str(ckpt), "dataset": version, "beta": beta,
                            "n_labels": n_labels, "eval": res, "gain": gain, "gain_se": gain_se, "kept": kept})
            with open(history_path, "w") as f:
                json.dump(history, f, indent=1)
            dataset = version          # always aggregate, even when the retrained model is not kept
            if kept:
                best_ckpt, best = ckpt, res
            log(f"   round {r}: score {score(res, sets):.3f} (gain {gain:+.3f} = {gain_se:+.1f} SE) -> best {best_ckpt}")
            if gain_se < args.min_gain_se and r > 1:
                log(f"   gain below {args.min_gain_se} SE: stopping")
                break
    log(f"best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

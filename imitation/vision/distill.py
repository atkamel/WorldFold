"""Privileged -> sensor-only distillation (roadmap M4.2): DAgger with a policy teacher.

    python -m imitation.vision.distill --teacher runs/dagger_v1/round_2/final.pt \
        --dataset v1_img --out runs/distill_v1 --rounds 3

The same loop as `imitation.dagger`, with the student and teacher swapped for:
  student  VisionChunkPolicy: cameras + the 48 sensor-available proprio dims
  teacher  the best privileged state policy (139-D obs), not the scripted expert
Because every episode already stores the full privileged observation, the teacher's
label for any state the student visited is just `teacher.predict(obs history)` -- no
sim snapshot/restore. So training relabels *every* step of every episode densely
(`imitation.data.dataset.teacher_samples`), and rollouts only need the teacher online
for the beta-mixture (executing the teacher's chunk with probability beta).

Round 0 trains the student on the rendered expert data (`v1_img`) relabelled by the
teacher. Each round r then rolls the student out with cameras rendering on fresh seeds
(DISTILL_SEED_BASE + 1000 r), freezes them on top of the previous version, retrains from
the best student, and evaluates at n>=200 with the same keep/stop rule as DAgger.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from imitation.dagger import gain_in_se, resume_state, score
from imitation.data.collect import DEFAULT_ROOT, recovery_perturbation
from imitation.data.schema import ACTOR_STUDENT, ACTOR_TEACHER, DatasetWriter, load_manifest
from imitation.evaluate import evaluate
from imitation.policies.common import load_policy
from imitation.rollout import Controller, EnvPool, Plan, padded_predict, rollout
from imitation.seeds import DISTILL_SEED_BASE
from imitation.spec import ACTION_DIM, OBS_DIM
from imitation.train import train


class DistillController(Controller):
    """Vision student drives; the privileged teacher labels every replan state and, with
    probability beta, its chunk is executed instead."""
    needs_images = True
    source = "student"

    def __init__(self, student, teacher, replan_every=8, beta=0.0):
        self.student, self.teacher, self.replan_every, self.beta = student, teacher, replan_every, beta
        self.horizon = max(student.obs_horizon, teacher.obs_horizon)

    def plan(self, slots, obs_hist, labels, rngs, images=None):
        s_chunks = padded_predict(self.student, obs_hist[:, -self.student.obs_horizon:], images)
        t_chunks = padded_predict(self.teacher, obs_hist[:, -self.teacher.obs_horizon:])
        plans = []
        for j, slot in enumerate(slots):
            use_teacher = self.beta > 0 and rngs[slot].random() < self.beta
            chunk = t_chunks[j] if use_teacher else s_chunks[j]
            plans.append(Plan(actions=chunk[:self.replan_every], label=t_chunks[j],
                              actor=ACTOR_TEACHER if use_teacher else ACTOR_STUDENT))
        return plans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="privileged (state) checkpoint")
    ap.add_argument("--dataset", default="v1_img", help="frozen version with images")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="round-0 student checkpoint (skip round-0 training)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--beta0", type=float, default=0.3)
    ap.add_argument("--beta-decay", type=float, default=0.5)
    ap.add_argument("--recovery-fraction", type=float, default=0.3)
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=30_000)
    ap.add_argument("--round-train-steps", type=int, default=15_000)
    ap.add_argument("--student-weight", type=float, default=2.0, help="oversampling of student-visited steps")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--eval-sets", nargs="+", default=["id_easy", "id_hard", "recovery"])
    ap.add_argument("--min-gain-se", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "history.json"
    log_file = open(out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    def save(history):
        with open(history_path, "w") as f:
            json.dump(history, f, indent=1)

    teacher = load_policy(args.teacher)
    with EnvPool(args.workers, {"render": True}) as pool:
        if args.resume and history_path.exists():
            best_ckpt, best, dataset, first, history = resume_state(out)
            log(f"== resuming at round {first} from {best_ckpt} on {dataset}")
        else:
            if args.init:
                best_ckpt = Path(args.init)
            else:
                log(f"== round 0: training the vision student on {args.dataset}, labels from {args.teacher}")
                best_ckpt, _ = train("vision", args.dataset, out / "round_0", root=args.root,
                                     steps=args.train_steps, teacher=args.teacher, log=log)
            best = evaluate(best_ckpt, args.eval_sets, args.eval_n, pool=pool, replan_every=args.replan_every, log=log)
            dataset, first = args.dataset, 1
            history = [{"round": 0, "checkpoint": str(best_ckpt), "dataset": dataset, "eval": best, "kept": True}]
            save(history)
        for r in range(first, args.rounds + 1):
            beta = args.beta0 * args.beta_decay ** (r - 1)
            log(f"== round {r}: beta {beta:.3f}, {args.episodes} student rollouts from {best_ckpt}")
            t0 = time.time()
            version = f"{args.dataset}_{out.name}_r{r}"
            if (Path(args.root) / version / "manifest.json").exists():
                m = load_manifest(args.root, version)
                log(f"   reusing frozen {version}")
            else:
                controller = DistillController(load_policy(best_ckpt), teacher, args.replan_every, beta)
                writer = DatasetWriter(args.root, version, parent=dataset, resume=args.resume,
                                       config={"round": r, "beta": beta, "student": str(best_ckpt),
                                               "teacher": args.teacher, "episodes": args.episodes})
                seeds = [s for s in range(DISTILL_SEED_BASE + 1000 * r, DISTILL_SEED_BASE + 1000 * r + args.episodes)
                         if s not in writer.done_seeds]
                rollout(pool, seeds, controller, perturb_fn=recovery_perturbation(args.recovery_fraction),
                        meta_extra={"round": r, "beta": beta, "policy": str(best_ckpt)},
                        on_done=lambda e: writer.add(e, obs_dim=OBS_DIM, action_dim=ACTION_DIM))
                m = writer.freeze()
            new = [e for e in m["episodes"] if e["file"].startswith(f"{version}/")]
            log(f"   rollouts: student success {sum(e['success'] for e in new)}/{len(new)}, "
                f"froze {version} ({m['n_episodes']} episodes), {time.time() - t0:.0f}s")
            ckpt, _ = train("vision", version, out / f"round_{r}", root=args.root, steps=args.round_train_steps,
                            init_from=best_ckpt, teacher=args.teacher, dagger_weight=args.student_weight,
                            allow_failures=True, log=log)
            res = evaluate(ckpt, args.eval_sets, args.eval_n, pool=pool, replan_every=args.replan_every, log=log)
            gain, gain_se = score(res) - score(best), gain_in_se(res, best)
            kept = gain > 0
            history.append({"round": r, "checkpoint": str(ckpt), "dataset": version, "beta": beta, "eval": res,
                            "gain": gain, "gain_se": gain_se, "kept": kept})
            save(history)
            dataset = version
            if kept:
                best_ckpt, best = ckpt, res
            log(f"   round {r}: score {score(res):.3f} (gain {gain:+.3f} = {gain_se:+.1f} SE) -> best {best_ckpt}")
            if gain_se < args.min_gain_se and r > 1:
                log(f"   gain below {args.min_gain_se} SE: stopping")
                break
    log(f"best student: {best_ckpt}")


if __name__ == "__main__":
    main()

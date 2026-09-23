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
Stops after --rounds, or when a round gains less than --min-gain.

    python -m imitation.dagger --init runs/mlp_v1/final.pt --dataset v1 --out runs/dagger_mlp --rounds 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from imitation.data.collect import DEFAULT_ROOT, recovery_perturbation
from imitation.data.schema import DatasetWriter, load_manifest
from imitation.evaluate import evaluate
from imitation.policies.common import load_policy
from imitation.rollout import EnvPool, PolicyController, rollout
from imitation.seeds import DAGGER_SEED_BASE
from imitation.spec import ACTION_DIM, OBS_DIM
from imitation.train import train


def score(results):
    """Model selection: in-distribution and recovery success, equally weighted."""
    return sum(results[k]["success_rate"] for k in ("id_easy", "recovery") if k in results)


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
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--eval-sets", nargs="+", default=["id_easy", "id_hard", "recovery"])
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "history.json"
    log_file = open(out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    tag = out.name
    with EnvPool(args.workers) as pool:
        best_ckpt, dataset = Path(args.init), args.dataset
        log(f"== round 0: evaluating {best_ckpt}")
        best = evaluate(best_ckpt, args.eval_sets, args.eval_n, pool=pool, replan_every=args.replan_every, log=log)
        history = [{"round": 0, "checkpoint": str(best_ckpt), "dataset": dataset, "eval": best, "kept": True}]
        for r in range(1, args.rounds + 1):
            beta = args.beta0 * args.beta_decay ** (r - 1)
            log(f"== round {r}: beta {beta:.3f}, {args.episodes} student rollouts from {best_ckpt}")
            t0 = time.time()
            policy = load_policy(best_ckpt)
            controller = PolicyController(policy, replan_every=args.replan_every, beta=beta, label=True,
                                          source="dagger")
            seeds = range(DAGGER_SEED_BASE + 1000 * r, DAGGER_SEED_BASE + 1000 * r + args.episodes)
            eps = rollout(pool, seeds, controller, perturb_fn=recovery_perturbation(args.recovery_fraction),
                          meta_extra={"round": r, "beta": beta, "policy": str(best_ckpt)})
            version = f"{args.dataset}_{tag}_r{r}"
            writer = DatasetWriter(args.root, version, parent=dataset,
                                   config={"round": r, "beta": beta, "policy": str(best_ckpt),
                                           "replan_every": args.replan_every, "episodes": args.episodes,
                                           "recovery_fraction": args.recovery_fraction})
            for e in eps:
                writer.add(e, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
            m = writer.freeze()
            n_labels = sum(len(e.label_steps) for e in eps)
            log(f"   rollouts: student success {sum(e.meta['success'] for e in eps)}/{len(eps)}, "
                f"{n_labels} teacher labels, froze {version} ({m['n_episodes']} episodes), {time.time() - t0:.0f}s")

            ckpt, _ = train(policy.kind, version, out / f"round_{r}", root=args.root, steps=args.train_steps,
                            init_from=best_ckpt, dagger_weight=args.dagger_weight, log=log)
            res = evaluate(ckpt, args.eval_sets, args.eval_n, pool=pool, replan_every=args.replan_every, log=log)
            gain = score(res) - score(best)
            kept = gain > 0
            history.append({"round": r, "checkpoint": str(ckpt), "dataset": version, "beta": beta,
                            "n_labels": n_labels, "eval": res, "gain": gain, "kept": kept})
            with open(history_path, "w") as f:
                json.dump(history, f, indent=1)
            dataset = version          # always aggregate, even when the retrained model is not kept
            if kept:
                best_ckpt, best = ckpt, res
            log(f"   round {r}: score {score(res):.3f} (gain {gain:+.3f}) -> best {best_ckpt}")
            if gain < args.min_gain and r > 1:
                log(f"   gain below {args.min_gain}: stopping")
                break
    log(f"best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

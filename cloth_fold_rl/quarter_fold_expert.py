"""Scripted expert for the quarter fold: one FoldExpert per (stage, arm) move,
staged. Both stages run the same synchronized-pair pattern: two FoldExperts
act every step, both hold their placed corners until the other has placed too,
then both release and retreat together.

A released corner springs back: the fold's bend pulls it a few centimetres
toward the fold line and the flap narrows across it. Placed exactly on the
goal, corners settle away from it and the fold fails, so each move places past
its goal by an OVERSHOOT (the mean spring-back measured on the stock cloth).
Stage 1 carries both west-edge points east onto the east edge (right arm: the
two-layer south stack -> cloth_110; left arm: the crease-end vertex ->
cloth_115), so both overshoot a little further east (+x) than their goal.
Feasibility check:

    python -m cloth_fold_rl.quarter_fold_expert --episodes 5
"""

from __future__ import annotations

import argparse

import numpy as np

from cloth_fold_rl.quarter_fold_env import HalfFoldEnv, QuarterFoldEnv

# per (stage, arm): metres past the goal to place the corner at (the mean
# spring-back measured on the stock cloth). Stage 1 carries both west-edge
# points east, so both overshoot a little further east than their goal.
# A fixed overshoot can't track the +-30% mass/friction/damping domain
# randomization exactly, so some episodes place a corner just outside
# SUCCESS_DIST -- the retry loop below (measured error, MAX_RETRIES) closes
# that last centimetre instead of a bigger fixed margin risking overshoot the
# other way.
OVERSHOOT = {(0, "left_"): np.array([-0.04, -0.03, 0.0]), (0, "right_"): np.array([0.02, -0.03, 0.0]),
             (1, "right_"): np.array([0.02, 0.0, 0.0]), (1, "left_"): np.array([0.02, 0.0, 0.0])}
MAX_RETRIES = 2
SETTLE_WAIT = 15   # steps to wait in "done" before judging placement -- the
                    # released corner is still springing back right after release


def _gripping(e):
    """Phase index from which an arm counts as having its point: "close", or an anchor's "hold" (R2)."""
    return e.PHASES.index("close" if "close" in e.PHASES else "hold")


class QuarterFoldExpert:
    # Q2: per stage, the order in which the arms approach and close. The second fold on Isaac (the fold-over) puts
    # both arms on the west edge, and approaching together they tangle: the right arm reaches across over the left
    # (demo 2026-10-14). So the right arm goes first while the left arm waits parked at home. The left arm
    # approaches once the right arm has closed, and the lift gate then lifts both together.
    # A fully sequential fold (one arm folds and releases, then the other) cannot work: with the crease end left
    # behind, the stack cannot reach the far corner (they are 15 cm apart in the cloth; trace 2026-10-14).
    # Only for per-arm experts with a `waiting` flag (isaac.fold_expert.IsaacArmExpert); MuJoCo's FoldExpert and
    # stage 0 are unchanged.
    # R2 left_fold: the right arm presses its anchor first, then the left arm approaches the west edge.
    SEQUENCE = {"fold_over": ("right_", "left_"), "left_fold": ("right_", "left_")}  # keyed by stage role (_stage_role)
    ANCHOR_RELEASE_PHASE = "carry"     # the carrying arms' phase at which an anchor lifts off (R2 left_fold)

    def __init__(self, env, seed=0, expert_cls=None, overshoot=None):
        # expert_cls: the per-arm phase machine (default the MuJoCo FoldExpert; isaac.fold_expert.IsaacArmExpert on
        # Isaac); overshoot: per (stage, arm) placement offset, MuJoCo's measured spring-back by default
        if expert_cls is None:
            from cloth_fold_rl.expert import FoldExpert as expert_cls
        self.overshoot = OVERSHOOT if overshoot is None else overshoot
        self.env = env
        self.base = env.unwrapped
        self.experts = {}
        self.moves = {}
        self.correction = {}
        for s, stage in enumerate(env.stages):
            for k, move in enumerate(stage.moves):
                self.moves[(s, move.prefix)] = move
                goal = lambda m=move, s=s: (env.goal(m) + self.overshoot.get((s, m.prefix), np.zeros(3))
                                            + self.correction[(s, m.prefix)])
                self.experts[(s, move.prefix)] = expert_cls(env, seed=seed + 10 * s + k, prefix=move.prefix,
                                                            corner=list(move.corners), goal=goal, release=True,
                                                            raw_vertex=True)
                if getattr(move, "kind", "carry") == "anchor":
                    self.experts[(s, move.prefix)].set_anchor()     # Isaac per-arm experts only (R2)
        self.reset()

    def reset(self):
        for key, expert in self.experts.items():
            expert.reset()
            expert.rng = np.random.default_rng(expert.seed)   # per episode, not per worker
            expert.release_allowed = False
            self.correction[key] = np.zeros(3)
        self.retries = {key: 0 for key in self.experts}
        self.done_steps = {key: 0 for key in self.experts}
        self.turn_done = set()      # sequenced stages: (stage, arm) that have finished their turn once

    def _arms(self):
        return {p: e for (s, p), e in self.experts.items() if s == self.env.stage}

    def phases(self):
        return {p: e.PHASES[e.phase] for p, e in self._arms().items()}

    def _maybe_retry(self, key, expert):
        if expert.PHASES[expert.phase] != "done":
            self.done_steps[key] = 0
            return
        self.done_steps[key] += 1
        if (self.done_steps[key] < SETTLE_WAIT or self.env._placed(self.moves[key])
                or self.retries[key] >= MAX_RETRIES):
            return
        move = self.moves[key]
        carried = np.mean([self.env._vertex(c) for c in move.corners], axis=0)
        self.correction[key] += self.env.goal(move) - carried   # nudge by the measured miss
        expert.reset()
        self.retries[key] += 1
        self.done_steps[key] = 0

    def resync(self):
        """Re-infer every current-stage arm's phase from the sim state, for when a
        learner has been driving (DAgger labelling). Retry bookkeeping restarts."""
        for (s, p), expert in self.experts.items():
            if s == self.env.stage:
                expert.infer_phase(placed=self.env._placed(self.moves[(s, p)]))
                self.done_steps[(s, p)] = 0
        self._update_release_gate()
        return self.phases()

    def _stage_role(self):
        """"fold_over" for the Isaac quarter fold's second fold (whatever its index), else the stage index."""
        try:
            from cloth_fold_rl.quarter_fold_env import STAGE1_FOLD_OVER, STAGE1_LEFT_FOLD
            if self.env.stages[self.env.stage] == STAGE1_FOLD_OVER:
                return "fold_over"
            if self.env.stages[self.env.stage] == STAGE1_LEFT_FOLD:
                return "left_fold"
        except (AttributeError, IndexError, TypeError):
            pass
        return self.env.stage

    def _sequenced(self):
        arms = self._arms()
        order = self.SEQUENCE.get(self._stage_role())
        if not order or not all(hasattr(e, "waiting") for e in arms.values()) or set(order) != set(arms):
            return None
        return order

    def _update_sequence(self):
        """In a sequenced stage an arm waits, parked, until every arm before it has closed on its point. The turn is
        latched, so a later regrasp of an earlier arm does not send the next one back."""
        order = self._sequenced()
        if order is None:
            return
        arms = self._arms()
        ahead = True
        for p in order:
            e = arms[p]
            if e.phase >= _gripping(e):
                self.turn_done.add((self.env.stage, p))
            e.waiting = not ahead
            ahead = ahead and (self.env.stage, p) in self.turn_done

    def _update_lift_gate(self):
        # Q2, from stage 1 on (the bimanual fold-over): no arm lifts until every arm of the stage has closed on its
        # point; one arm lifting alone drags the folded cloth out from under the other (traces 2026-10-14). Stage 0
        # and per-arm experts without the gate (MuJoCo's FoldExpert) are unchanged.
        arms = self._arms()
        if self.env.stage == 0 or not all(hasattr(e, "lift_allowed") for e in arms.values()):
            return
        ready = all(e.phase >= _gripping(e) for e in arms.values())
        for e in arms.values():
            e.lift_allowed = ready

    def _update_release_gate(self):
        # release once every arm has reached hold. An arm already past hold (a
        # learner released it) counts too, or the other arm would hold forever.
        arms = self._arms()
        carry = {p: e for p, e in arms.items() if not getattr(e, "anchor", False)}
        anchors = [e for e in arms.values() if getattr(e, "anchor", False)]
        if all(e.PHASES[e.phase] in ("hold",) + e.RELEASE_PHASES for e in carry.values()):
            for e in carry.values():
                e.release_allowed = True
        # R2: an anchor lifts off once every carrying arm has lifted its point and starts the carry, so the anchor arm
        # is gone before the carry crosses the east half (its links reach over it; trace 660001, attempt E)
        if anchors and all(e.phase >= e.PHASES.index(self.ANCHOR_RELEASE_PHASE) for e in carry.values()):
            for e in anchors:
                e.release_allowed = True

    def act(self):
        arms = self._arms()
        self._update_release_gate()
        self._update_lift_gate()
        self._update_sequence()
        for (s, p), expert in self.experts.items():
            if s == self.env.stage:
                self._maybe_retry((s, p), expert)
        return np.concatenate([arms["left_"].act(), arms["right_"].act()])


def run_episode(env, expert, seed, verbose=True):
    obs, info = env.reset(seed=seed)
    expert.reset()
    total = 0.0
    for t in range(env.unwrapped.max_episode_steps):
        obs, reward, terminated, truncated, info = env.step(expert.act())
        total += reward
        if verbose and t % 25 == 0:
            phases = expert.phases()
            print(f"  t{t:3d} stage {info['stage']} L {phases['left_']:<8} R {phases['right_']:<8} "
                  f"score {info['fold_score']:.3f} d {'/'.join(f'{d:.3f}' for d in info['move_distance'])} "
                  f"grasp {int(info['grasped']['left_'])}{int(info['grasped']['right_'])} settle {info['settle_steps']}")
        if terminated or truncated:
            break
    return {"seed": seed, "steps": t + 1, "reward": round(total, 2), "fold_score": round(info["fold_score"], 3),
            "stage": info["stage"], "retries": sum(expert.retries.values()), "success": info["success"],
            "reason": info["termination_reason"] or "truncated", "anchor_drift": round(info["anchor_drift"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--half", action="store_true", help="stage 0 only (HalfFoldEnv)")
    args = ap.parse_args()
    env = HalfFoldEnv() if args.half else QuarterFoldEnv()
    env.unwrapped.domain_randomization = True
    expert = QuarterFoldExpert(env)
    rows = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        if not args.quiet:
            print(f"episode seed={seed}")
        row = run_episode(env, expert, seed, verbose=not args.quiet)
        rows.append(row)
        print(f"  -> {row}")
    print(f"success {sum(r['success'] for r in rows)}/{len(rows)}, reached stage 1 in "
          f"{sum(r['stage'] >= 1 for r in rows)}, mean steps {np.mean([r['steps'] for r in rows]):.0f}, "
          f"mean score {np.mean([r['fold_score'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()

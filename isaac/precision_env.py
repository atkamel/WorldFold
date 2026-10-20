"""The residual-RL learning window on Isaac (docs/superpowers/plans/2026-10-20-closed-loop-rl.md, R1-R2).

B copies of the half fold on the GPU `friction` profile (isaac_env.IsaacClothFoldBatch), each driven by two
isaac.place_servo.IsaacPlaceServoExpert arms. `scripted_to_window` runs the scripted fold from a seed to the start
of the carry (both arms steering, both corners in the jaws) and snapshots the copy; `reset_window` restores
snapshots and `step` runs the window: every control step the policy's action moves each arm's residual (DELTA m per
unit, clipped to RES_MAX) and sets its release vote, the base does the rest. All copies step together (one
`batch.advance()` per step, no scheduler threads); a copy whose episode has ended keeps stepping with a zero action
and its transitions are flagged dead.

Observation, shaping and terminal score are pure numpy (module functions) so they are tested without Isaac.
"""

from __future__ import annotations

import copy
import math

import numpy as np

N = 11
# (top, bottom) vertex pairs the half fold (north onto south about iy = 5) must stack: west edge, centre line and east
# edge at three depths. Index = ix * N + iy.
PAIRS = np.array([(ix * N + iy, ix * N + (N - 1 - iy)) for ix in (0, 5, 10) for iy in (10, 8, 6)])
DRIFT_IDX = np.array([0, 5, 110, 115])     # bottom-layer corners and crease ends: they should not move
FOLDS = (("y", 5),)
WINDOW_PHASES = ("carry", "place", "hold", "release", "retreat", "done")
ARMS = ("left_", "right_")
SCALE = 10.0                 # m -> dm in the observation
DELTA = 0.004                # m of residual per unit action per step
RES_MAX = 0.06
VOTE_BELOW = -0.5            # vote = action > VOTE_BELOW, so the zero action releases as the base does
OBS_DIM = 3 * len(PAIRS) + 2 * len(DRIFT_IDX) + len(ARMS) * (3 + 3 + 3 + 1 + 3 + len(WINDOW_PHASES) + 1) + 2 * 4
ACT_DIM = 2 * 4              # per arm: dx, dy, dz, release vote
EPS = 0.005                  # m inside the potential's log
W_ACT = 0.01
W_DRIFT = 5.0                # per m of new bottom-layer drift
FAIL = 10.0
PASS_BONUS = 10.0
SETTLE_STEPS = 15            # window ends this long after both arms are done
T_MAX = 200                  # window cap (a truncation, not a terminal)
PRE_MAX = 220                # scripted steps allowed to reach the window
DRAG_LIMIT = 0.20            # = cloth_fold_rl.quarter_fold_env.DRAG_LIMIT
HOLD_DIST = 0.05             # = isaac.isaac_env.HOLD_DIST


def features(cloth, start, arms, prev_action):
    """(OBS_DIM,) float32. cloth / start: (121, 3) grid now / at the window start. arms: (left, right) dicts with corner,
    partner, site, site_vel, jaw, residual, phase, phase_steps. prev_action: (ACT_DIM,)."""
    parts = [(cloth[PAIRS[:, 0]] - cloth[PAIRS[:, 1]]).ravel() * SCALE,
             (cloth[DRIFT_IDX, :2] - start[DRIFT_IDX, :2]).ravel() * SCALE]
    for a in arms:
        onehot = np.zeros(len(WINDOW_PHASES))
        onehot[WINDOW_PHASES.index(a["phase"])] = 1.0
        parts += [(a["corner"] - a["partner"]) * SCALE, (a["site"] - a["corner"]) * SCALE, a["site_vel"], [a["jaw"]],
                  a["residual"] * SCALE, onehot, [min(a["phase_steps"], 100) / 50.0]]
    parts.append(prev_action)
    return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)


def potential(cloth):
    """Higher when the pairs stack: -mean log(xy gap + EPS)."""
    gap = np.linalg.norm(cloth[PAIRS[:, 0], :2] - cloth[PAIRS[:, 1], :2], axis=1)
    return -float(np.mean(np.log(gap + EPS)))


def drift(cloth, start):
    return float(np.max(np.linalg.norm(cloth[DRIFT_IDX, :2] - start[DRIFT_IDX, :2], axis=1)))


def step_reward(phi_prev, phi, action, drift_prev, drift_now):
    """Shaping (sums to phi_T - phi_0 over a window), an action cost on the residual moves, new drift."""
    a = np.asarray(action, float).reshape(len(ARMS), 4)[:, :3]
    return phi - phi_prev - W_ACT * float(np.sum(a ** 2)) - W_DRIFT * max(0.0, drift_now - drift_prev)


def terminal_reward(cloth):
    """The fold's score at the end of the window: log-scaled against the bar, plus PASS_BONUS when it passes."""
    from cloth_fold_rl.fold_alignment import EDGE_BAR, MEAN_BAR, score
    s = score(cloth, folds=FOLDS)
    r = -float(np.clip(math.log(max(s["mean"], 1e-4) / MEAN_BAR), -2.0, 5.0))
    r -= float(np.clip(math.log(max(s["edge_max"], 1e-4) / EDGE_BAR), -2.0, 5.0))
    return r + (PASS_BONUS if s["passed"] else 0.0), s


def apply_action(experts, action):
    """Move each arm's residual by its action and set its release vote."""
    a = np.clip(np.asarray(action, float).reshape(len(ARMS), 4), -1.0, 1.0)
    for k, p in enumerate(ARMS):
        e = experts[p]
        e.residual = np.clip(e.residual + DELTA * a[k, :3], -RES_MAX, RES_MAX)
        e.release_vote = bool(a[k, 3] > VOTE_BELOW)


def coordinate(experts):
    """IsaacQuarterFoldExpert's gates: no arm lifts until both have closed; both release only once both hold."""
    ready = all(e.phase >= e.PHASES.index("close") for e in experts.values())
    for e in experts.values():
        e.lift_allowed = ready
    if all(e.PHASES[e.phase] in ("hold",) + e.RELEASE_PHASES for e in experts.values()):
        for e in experts.values():
            e.release_allowed = True


class PrecisionBatch:
    """B half-fold copies in one Isaac scene, scripted to the window and stepped through it together."""

    def __init__(self, n, profile="friction", cameras=None):
        from cloth_fold_rl.quarter_fold_env import CLOTH_10, CLOTH_120, GRASP_RADIUS, STAGES, QuarterFoldEnv
        from isaac.isaac_env import CLOTH_JITTER, IsaacClothFoldBatch
        from isaac.place_servo import IsaacPlaceServoExpert
        self.n = n
        self.batch = IsaacClothFoldBatch(n, profile=profile, observation_mode="state", max_episode_steps=10 ** 6,
                                         grasp_corners={"left_": (CLOTH_10,), "right_": (CLOTH_120,)},
                                         grasp_radius=GRASP_RADIUS, cameras=cameras)
        self.subs = self.batch.envs
        self.tasks = [QuarterFoldEnv(base_env=s, cloth_jitter=CLOTH_JITTER, stages=(STAGES[0],)) for s in self.subs]
        self.moves = {m.prefix: m for m in STAGES[0].moves}
        self.experts = []
        for task, sub in zip(self.tasks, self.subs):
            self.experts.append({p: IsaacPlaceServoExpert(task, prefix=p, corner=list(m.corners), release=True,
                                                          goal=lambda m=m, t=task: t.goal(m),
                                                          live_goal=lambda m=m, s=sub: s.cloth_positions()[m.goal])
                                 for p, m in self.moves.items()})
        self.state = [None] * n

    # ---- scripted phase ----
    def _reset_experts(self, i, seed):
        for k, e in enumerate(self.experts[i].values()):
            e.reset()
            e.rng = np.random.default_rng(seed + k)
            e.release_allowed = False

    def _scripted_step(self, actions=None):
        """One control step of every copy: the base (plus actions where given) acts, physics advances once."""
        for i, sub in enumerate(self.subs):
            if actions is not None and actions[i] is not None:
                apply_action(self.experts[i], actions[i])
            coordinate(self.experts[i])
            sub.step_submit(np.concatenate([self.experts[i][p].act() for p in ARMS]))
        self.batch.advance()
        for sub in self.subs:
            sub.step_collect()

    def _held(self, i):
        sub = self.subs[i]
        return all(np.linalg.norm(e._corner() - sub.gripper_position(p)) < HOLD_DIST
                   for p, e in self.experts[i].items())

    def snapshot(self, i, seed):
        exp = {p: {f: copy.deepcopy(getattr(e, f)) for f in e.PHASE_FIELDS + ("lift_allowed",)}
               for p, e in self.experts[i].items()}
        return {"seed": seed, "env": self.subs[i].snapshot(), "experts": exp,
                "task_start": self.tasks[i]._start.copy(), "start": self.subs[i].cloth_positions().copy()}

    def scripted_to_window(self, seeds, finish=False):
        """Reset copy i with seeds[i], run the base to the window start, snapshot it (None: it never got there with
        both corners held). finish: also run each copy's base on through the window (zero action), returning the
        final alignment per copy, the reference for the restore check."""
        for i, seed in enumerate(seeds):
            self.tasks[i].reset(seed=int(seed))
            self._reset_experts(i, int(seed))
        snaps, finals = [None] * self.n, [None] * self.n
        pending = set(range(self.n))
        for _ in range(PRE_MAX):
            self._scripted_step()
            for i in list(pending):
                if all(e.PHASES[e.phase] == "carry" for e in self.experts[i].values()):
                    pending.discard(i)
                    if self._held(i):
                        snaps[i] = self.snapshot(i, int(seeds[i]))
            if not pending:
                break
        if finish:
            live = [i for i in range(self.n) if snaps[i] is not None]
            self._begin(live)
            while any(self.state[i]["alive"] for i in live):
                self.step(np.zeros((self.n, ACT_DIM)))
            for i in live:
                finals[i] = self.state[i]["final"]
        return snaps, finals

    # ---- the window ----
    def _arm_obs(self, i):
        sub, out = self.subs[i], []
        st = self.state[i]
        for p in ARMS:
            e = self.experts[i][p]
            site = np.asarray(sub.gripper_position(p), float)
            name = e.PHASES[e.phase]
            out.append({"corner": e._corner(), "partner": e._goal_now(), "site": site,
                        "site_vel": (site - st["prev_site"][p]) / sub.control_dt, "jaw": float(sub.joint_positions(p)[5]),
                        "residual": e.residual, "phase": name if name in WINDOW_PHASES else "carry",
                        "phase_steps": e.phase_steps})
            st["prev_site"][p] = site
        return out

    def _obs(self, i):
        st = self.state[i]
        return features(self.subs[i].cloth_positions(), st["start"], self._arm_obs(i), st["prev_action"])

    def _begin(self, idx):
        for i in idx:
            cloth = self.subs[i].cloth_positions()
            self.state[i] = {"alive": True, "t": 0, "settle": 0, "start": cloth.copy(), "phi": potential(cloth),
                             "drift": 0.0, "prev_action": np.zeros(ACT_DIM),
                             "prev_site": {p: np.asarray(self.subs[i].gripper_position(p), float) for p in ARMS},
                             "final": None}

    def restore(self, i, snap):
        self.subs[i].restore(snap["env"])
        for p, fields in snap["experts"].items():
            for f, v in fields.items():
                setattr(self.experts[i][p], f, copy.deepcopy(v))
        self.tasks[i]._start = snap["task_start"].copy()

    def reset_window(self, snaps):
        """Restore snapshot snaps[i] into copy i (len(snaps) == n); returns the (n, OBS_DIM) observation."""
        for i, snap in enumerate(snaps):
            self.restore(i, snap)
        self._begin(range(self.n))
        return np.stack([self._obs(i) for i in range(self.n)])

    def step(self, actions):
        """actions (n, ACT_DIM) -> obs (n, OBS_DIM), reward (n,), terminal (n,), done (n,), alive_before (n,), infos.
        Rows where alive_before is False are padding (the copy's episode had already ended)."""
        alive = np.array([self.state[i] is not None and self.state[i]["alive"] for i in range(self.n)])
        acts = [np.asarray(actions[i], float) if alive[i] else np.zeros(ACT_DIM) for i in range(self.n)]
        self._scripted_step(acts)
        obs, rew = np.zeros((self.n, OBS_DIM), np.float32), np.zeros(self.n)
        term, done, infos = np.zeros(self.n, bool), np.zeros(self.n, bool), [None] * self.n
        for i in range(self.n):
            st = self.state[i]
            if not alive[i]:
                continue
            sub, ex = self.subs[i], self.experts[i]
            cloth = sub.cloth_positions()
            st["t"] += 1
            phi, d = potential(cloth), drift(cloth, st["start"])
            rew[i] = step_reward(st["phi"], phi, acts[i], st["drift"], d)
            st["phi"], st["drift"], st["prev_action"] = phi, max(st["drift"], d), acts[i]
            st["settle"] = st["settle"] + 1 if all(e.PHASES[e.phase] == "done" for e in ex.values()) else 0
            reason = None
            if any(e.dropped for e in ex.values()):
                reason = "dropped"
            elif sub._failed():
                reason = "unstable"
            elif d > DRAG_LIMIT:
                reason = "cloth_dragged"
            elif st["settle"] >= SETTLE_STEPS:
                reason = "settled"
            elif st["t"] >= T_MAX:
                reason = "truncated"
            if reason is not None:
                st["alive"], done[i] = False, True
                term[i] = reason != "truncated"
                r_end, s = terminal_reward(cloth)
                rew[i] += -FAIL if reason in ("dropped", "unstable", "cloth_dragged") else (r_end if term[i] else 0.0)
                st["final"] = {"reason": reason, "steps": st["t"], "mean": round(s["mean"], 4),
                               "edge_max": round(s["edge_max"], 4), "passed": bool(s["passed"]),
                               "residual": {p: np.round(ex[p].residual, 4).tolist() for p in ARMS}}
                infos[i] = st["final"]
            obs[i] = self._obs(i)
        return obs, rew, term, done, alive, infos

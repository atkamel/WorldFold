"""The residual-RL learning window on Isaac (docs/superpowers/plans/2026-10-20-closed-loop-rl.md, R1-R2).

B copies of the half fold on the GPU `friction` profile (isaac_env.IsaacClothFoldBatch), each driven by two
isaac.experts.place_servo.IsaacPlaceServoExpert arms. `scripted_to_window` runs the scripted fold from a seed to the start
of the carry (both arms steering, both corners in the jaws) and snapshots the copy; `reset_window` restores
snapshots and `step` runs the window: every control step the policy's action sets each arm's residual target
(RES_SCALE m per unit, so bounded; the residual follows it with RES_SMOOTH) and its release vote, the base does the
rest. (p3, results.md: a residual that integrated the action let any bias in the policy build up to the bound and
drag the cloth.) All copies step together (one
`batch.advance()` per step, no scheduler threads); a copy whose episode has ended keeps stepping with a zero action
and its transitions are flagged dead.

Observation, shaping and terminal score are pure numpy (module functions) so they are tested without Isaac.
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np

N = 11
# (top, bottom) vertex pairs the half fold (north onto south about iy = 5) must stack: west edge, centre line and east
# edge at three depths. Index = ix * N + iy.
PAIRS = np.array([(ix * N + iy, ix * N + (N - 1 - iy)) for ix in (0, 5, 10) for iy in (10, 8, 6)])
DRIFT_IDX = np.array([0, 5, 110, 115])     # bottom-layer corners and crease ends: they should not move
FOLDS = (("y", 5),)
WINDOW_PHASES = ("carry", "place", "hold", "release", "retreat", "done", "touch")
ARMS = ("left_", "right_")
SCALE = 10.0                 # m -> dm in the observation
RES_SCALE = np.array([0.03, 0.03, 0.02])    # m of residual per unit action (x, y, z): the residual's bound
RES_SMOOTH = 0.3             # the residual moves this fraction of the way to its target each step
VOTE_BELOW = -0.5            # vote = action > VOTE_BELOW, so the zero action releases as the base does
OBS_DIM = 3 * len(PAIRS) + 2 * len(DRIFT_IDX) + len(ARMS) * (3 + 3 + 3 + 1 + 3 + len(WINDOW_PHASES) + 1) + 2 * 4
ACT_DIM = 2 * 4              # per arm: dx, dy, dz, release vote
EPS = 0.005                  # m inside the potential's log
W_ACT = 0.01
W_DRIFT = 5.0                # per m of new bottom-layer drift
W_BOTTOM = 0.3               # terminal: per cm of mean bottom-layer displacement (an ideal half fold moves it none)
BOTTOM_IDX = np.array([ix * N + iy for ix in range(N) for iy in range(N // 2)])   # the half that stays (iy < 5)
FAIL = 10.0
PASS_BONUS = 10.0
SETTLE_STEPS = 15            # window ends this long after both arms are done
T_MAX = int(os.environ.get("WORLDFOLD_WINDOW_STEPS", 200))   # window cap (a failure); slower bases need more
PRE_MAX = 220                # scripted steps allowed to reach the window
DRAG_LIMIT = float(os.environ.get("WORLDFOLD_DRAG_LIMIT", 0.20))   # = quarter_fold_env.DRAG_LIMIT by default
# m: once settled, each arm whose top corner is further than this from its partner touches it up once
# (IsaacPlaceServoExpert.start_touchup) and the window settles again; unset: no touch-up
TOUCH_TOL = float(os.environ.get("WORLDFOLD_TOUCHUP") or 0.0)
TOUCH_ROUNDS = int(os.environ.get("WORLDFOLD_TOUCH_ROUNDS") or 1)   # touch-up rounds, each after the fold resettles
# "k=v,..." friction-grasp knobs on top of the profile's (isaac_env.FRICTION_GRASP), e.g. particle_friction=0.6: the
# cloth's friction against every rigid body, table and jaws alike (PhysX PBD has no table-only coefficient)
GRASP_KNOBS = os.environ.get("WORLDFOLD_GRASP_KNOBS", "")
HOLD_DIST = 0.05             # = isaac.sim.isaac_env.HOLD_DIST
# Validity gate on every score (re-evaluation 2026-10-22: a cloth hanging from a gripper scores 0.0 cm on
# fold_alignment, which is how carried-off sheets looked like near-successes): every grid vertex within GATE_Z of the
# table and on it, both jaws open and both gripper sites at least GATE_CLEAR from the cloth, the cloth moving less
# than GATE_STILL per step
# m above the table top: catches a lifted or hanging cloth (carried-off sheets sat ~30 cm up) but not ridges: stored
# settled half folds peak at a median 2.2-2.8 cm with a tail to ~5 cm (isaac/tools/rescore_gate.py, 2026-10-22); a cloth
# still on a finger is the clearance / jaw checks' job
GATE_Z = 0.05
GATE_CLEAR = 0.03
GATE_STILL = 0.001
GATE_JAW_OPEN = 0.5          # rad: a jaw above this counts as open (closed ~0.05, open ~1.0)
TABLE_HALF = 0.35            # = isaac_env.TABLE_SIZE / 2, the table centred on the copy's origin


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


def bottom_shift(cloth, start):
    """m: mean xy displacement of the bottom half since the window began."""
    return float(np.mean(np.linalg.norm(cloth[BOTTOM_IDX, :2] - start[BOTTOM_IDX, :2], axis=1)))


def terminal_reward(cloth, start):
    """The fold's score at the end of the window: log-scaled against the bar, plus PASS_BONUS when it passes, minus
    W_BOTTOM per cm the bottom half moved (scrunching the cloth also brings pairs together; this charges for it)."""
    from cloth_fold_rl.fold_alignment import EDGE_BAR, MEAN_BAR, score
    s = score(cloth, folds=FOLDS)
    r = -float(np.clip(math.log(max(s["mean"], 1e-4) / MEAN_BAR), -2.0, 5.0))
    r -= float(np.clip(math.log(max(s["edge_max"], 1e-4) / EDGE_BAR), -2.0, 5.0))
    r -= W_BOTTOM * 100.0 * bottom_shift(cloth, start)
    return r + (PASS_BONUS if s["passed"] else 0.0), s


def instability(sub):
    """Which of IsaacClothFoldEnv._failed's checks fire, and by how much (particles in the copy's frame)."""
    from isaac.sim.isaac_env import CLOTH_SPEED_LIMIT, WORKSPACE_XY
    pos, vel = sub._particle_positions(), sub._particle_vel
    finite = np.all(np.isfinite(pos), axis=1)
    speed = np.linalg.norm(np.nan_to_num(vel), axis=1)
    return {"nan": int((~finite).sum()), "out": int((np.abs(np.nan_to_num(pos[:, :2])) > WORKSPACE_XY).any(axis=1).sum()),
            "fast": int((speed > CLOTH_SPEED_LIMIT).sum()), "vmax": round(float(speed.max()), 2),
            "zmin": round(float(np.nanmin(pos[:, 2])), 3), "zmax": round(float(np.nanmax(pos[:, 2])), 3)}


def gate(cloth, prev_cloth, sites, jaws):
    """Is this final cloth state scoreable? dict(valid, z_max_cm, on_table, clear_cm, jaws_open, still_mm)."""
    from mujoco_sim.cloth_params import TABLE_TOP_Z
    cloth, prev = np.asarray(cloth, float), np.asarray(prev_cloth, float)
    z_max = float(cloth[:, 2].max() - TABLE_TOP_Z)
    on_table = bool(np.all(np.abs(cloth[:, :2]) <= TABLE_HALF))
    clear = min(float(np.min(np.linalg.norm(cloth - np.asarray(s, float), axis=1))) for s in sites)
    jaws_open = bool(all(float(j) > GATE_JAW_OPEN for j in jaws))
    still = float(np.max(np.linalg.norm(cloth - prev, axis=1)))
    valid = z_max <= GATE_Z and on_table and clear >= GATE_CLEAR and jaws_open and still <= GATE_STILL
    return {"valid": bool(valid), "z_max_cm": round(z_max * 100, 2), "on_table": on_table,
            "clear_cm": round(clear * 100, 2), "jaws_open": jaws_open, "still_mm": round(still * 1000, 2)}


def apply_action(experts, action):
    """Each arm's residual moves RES_SMOOTH of the way to its target (action * RES_SCALE); the 4th entry votes."""
    a = np.clip(np.asarray(action, float).reshape(len(ARMS), 4), -1.0, 1.0)
    for k, p in enumerate(ARMS):
        e = experts[p]
        e.residual = e.residual + RES_SMOOTH * (a[k, :3] * RES_SCALE - e.residual)
        e.release_vote = bool(a[k, 3] > VOTE_BELOW)


def coordinate(experts):
    """The fold's gates: no arm lifts until both have closed; both release only once both hold."""
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
        from isaac.tools.grasp_bench import parse_kv
        from isaac.sim.isaac_env import CLOTH_JITTER, IsaacClothFoldBatch
        from isaac.experts.place_servo import IsaacPlaceServoExpert
        self.n = n
        self.batch = IsaacClothFoldBatch(n, profile=profile, observation_mode="state", max_episode_steps=10 ** 6,
                                         grasp_corners={"left_": (CLOTH_10,), "right_": (CLOTH_120,)},
                                         grasp_radius=GRASP_RADIUS, cameras=cameras,
                                         grasp_knobs=parse_kv(GRASP_KNOBS) or None)
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
        self.frames, self.cut = None, [None] * n     # demo video: per copy, the "top" frame after every step
        self.warm_up()

    def warm_up(self, steps=5):
        """Reset and step every copy once before any restore: in a scene that has never stepped, the first restores
        lost their cloth to the spawn pose on some copies (p4's version 0: 10.4 cm, sd 7.6, against 3.22 warm)."""
        for i, task in enumerate(self.tasks):
            task.reset(seed=i)
            self._reset_experts(i, i)
        for _ in range(steps):
            self._scripted_step()

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
            sub.step_submit(self.tasks[i]._expand(np.concatenate([self.experts[i][p].act() for p in ARMS])))
        self.batch.advance()
        for sub in self.subs:
            sub.step_collect()
        if self.frames is not None:
            for i, sub in enumerate(self.subs):
                self.frames[i].append(sub.render_rig()["top"].transpose(1, 2, 0))

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
                        if self.frames is not None:
                            self.cut[i] = len(self.frames[i])
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
            self.state[i] = {"alive": True, "t": 0, "settle": 0, "start": cloth.copy(), "prev_cloth": cloth.copy(),
                             "phi": potential(cloth),
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
            st["t"] += 1
            if sub._failed():
                # a blown-up cloth: park it at once so it can't drift into the other copies, its arms idle
                st["final"] = {"reason": "unstable", "steps": st["t"], "mean": None, "edge_max": None, "passed": False,
                               "dropped": [p for p in ARMS if ex[p].dropped], "instability": instability(sub),
                               "residual": {p: np.round(ex[p].residual, 4).tolist() for p in ARMS}}
                sub.park()
                for e in ex.values():
                    e.phase = e.PHASES.index("done")
                st["alive"], done[i], term[i], rew[i], infos[i] = False, True, True, -FAIL, st["final"]
                obs[i] = self._obs(i)
                continue
            cloth = sub.cloth_positions()
            phi, d = potential(cloth), drift(cloth, st["start"])
            rew[i] = step_reward(st["phi"], phi, acts[i], st["drift"], d)
            st["phi"], st["drift"], st["prev_action"] = phi, max(st["drift"], d), acts[i]
            prev_cloth, st["prev_cloth"] = st["prev_cloth"], cloth.copy()
            st["settle"] = st["settle"] + 1 if all(e.PHASES[e.phase] == "done" for e in ex.values()) else 0
            rounds = st.setdefault("touch", [])
            if TOUCH_TOL and st["settle"] >= SETTLE_STEPS and len(rounds) < TOUCH_ROUNDS \
                    and not (rounds and not rounds[-1]["arms"]):
                from cloth_fold_rl.fold_alignment import score
                pre = score(cloth, folds=FOLDS)
                rounds.append({"pre_mean": round(pre["mean"], 4), "pre_edge": round(pre["edge_max"], 4), "arms": {}})
                for p, e in ex.items():
                    off = (np.asarray(e.live_goal(), float) - e._corner())[:2]
                    if np.linalg.norm(off) > TOUCH_TOL:
                        rounds[-1]["arms"][p] = np.round(off, 4).tolist()
                        e.start_touchup(off)
                if rounds[-1]["arms"]:
                    st["settle"] = 0
            # a corner leaving the jaws is not a failure in itself: in R1 (results.md) most did so at the end of the
            # place and landed well, so the final alignment scores it
            reason = None
            if d > DRAG_LIMIT:
                reason = "cloth_dragged"
            elif st["settle"] >= SETTLE_STEPS:
                reason = "settled"
            elif st["t"] >= T_MAX:
                reason = "truncated"
            if reason is not None:
                # the window is the whole task, and only a released, settled fold completes it: T_MAX is a failure
                # like a drag (p3: an unscored truncation bootstrapped on the critic; p5: a scored one paid the
                # policy to stall the carry and never release)
                st["alive"], done[i], term[i] = False, True, True
                r_end, s = terminal_reward(cloth, st["start"])
                g = gate(cloth, prev_cloth, [sub.gripper_position(p) for p in ARMS],
                         [sub.joint_positions(p)[5] for p in ARMS])
                # only a settled fold that passes the gate scores; an invalid one is a failure, so the score can't
                # be gamed by a lifted or still-held cloth
                rew[i] += r_end if (reason == "settled" and g["valid"]) else -FAIL
                st["final"] = {"reason": reason, "steps": st["t"], "mean": round(s["mean"], 4),
                               "bottom_shift": round(bottom_shift(cloth, st["start"]), 4),
                               "dropped": [p for p in ARMS if ex[p].dropped], "instability": None,
                               "touch": st.get("touch") or None,
                               "edge_max": round(s["edge_max"], 4), "passed": bool(s["passed"]),
                               "gate": g, "passed_valid": bool(s["passed"] and g["valid"] and reason == "settled"),
                               "residual": {p: np.round(ex[p].residual, 4).tolist() for p in ARMS},
                               "cloth": np.round(cloth, 4).tolist(), "start": np.round(st["start"], 4).tolist()}
                infos[i] = st["final"]
            obs[i] = self._obs(i)
        return np.nan_to_num(obs), np.nan_to_num(rew), term, done, alive, infos

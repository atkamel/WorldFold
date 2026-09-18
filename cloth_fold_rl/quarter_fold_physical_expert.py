"""Scripted two-arm scoop-fold expert for the physical grasp (no weld).

This is physical_expert.ScoopExpert generalised to two synchronised arms, the
same way quarter_fold_expert.QuarterFoldExpert generalises the weld FoldExpert:
one ScoopArm phase machine per arm, both act every step, both hold their placed
corner until the other has placed too, then both release and retreat together. A
retry loop (measured miss, re-scoop) closes near-misses under domain
randomization, exactly like the weld quarter-fold expert.

Each ScoopArm runs the physical-grasp routine proven in prove_grabber, adapted
to the re-calibrated ramp mount and reach-down posture of this geometry:

    hover -> down -> slide -> close -> lift -> carry -> place -> hold
             -> release -> retreat -> done

  * hover/down: bring the ramp (bolted to the gripper) to the approach point,
    the leading edge skimming the table just OUTSIDE the corner tip along the
    corner's outward diagonal,
  * slide: creep along that diagonal so the corner tip cams UP the ramp into the
    pocket (closed-loop on the corner riding up AND sitting over the ramp centre,
    so it tolerates the per-episode cloth jitter),
  * close: pinch; wait for the physical-grasp signal; capture the jaw orientation,
  * lift/carry/place: hold that jaw orientation (a 5-DOF arm cannot, so it is a
    weighted wish -- the same reason the carry is short, see the env docstring)
    and lay the corner down on the goal.

    python -m cloth_fold_rl.quarter_fold_physical_expert --episodes 5
"""

from __future__ import annotations

import argparse

import numpy as np
import mujoco

from cloth_fold_rl.expert import JOINT_DELTA_SCALE
from cloth_fold_rl.fold_env import SUCCESS_DIST
from cloth_fold_rl.quarter_fold_physical_env import (
    MOVES, QuarterFoldPhysicalEnv, RAMP_SIZE, TABLE_TOP_Z, ARM_JOINTS,
)

# scoop geometry (metres; heights above the table top)
HOVER_Z = 0.10
RAMP_SKIM_Z = 0.003    # ramp CENTRE height while sliding under the corner
APPROACH_BACK = 0.04   # ramp centre starts this far back along -slide from the corner
SLIDE_HOP = 0.003      # slide step along the outward diagonal per hop...
SLIDE_HOLD = 6         # ...held this many control steps so the cloth cams up
LIFT_Z = 0.06
CARRY_SPEED = 0.006    # m per control step the carry target creeps
TRACK_TOL = 0.03       # ...and only while the site is tracking this close
CARRY_ROT_W = 0.4      # weight on holding the grasp-time jaw orientation through the carry
MAX_RETRIES = 2
SETTLE_WAIT = 15       # steps in "done" before judging placement (corner still settling)


def _ik_pos(m, d, target_id, is_geom, qpos_adr, dof_adr, lo, hi, target,
            seed_q, q_ref=None, iters=300, damping=0.05, posture_gain=0.3, tol=0.001):
    """Position-only damped-least-squares IK on a scratch MjData, targeting a
    site or a geom, with an optional nullspace pull toward q_ref. (ramp targeting
    is why this can aim at a geom -- the ramp is bolted to the gripper.)"""
    scratch = mujoco.MjData(m); scratch.qpos[:] = d.qpos; scratch.qvel[:] = 0.0
    n = len(qpos_adr); q = np.clip(np.asarray(seed_q, float), lo, hi); target = np.asarray(target, float)
    for _ in range(iters):
        for k, adr in enumerate(qpos_adr):
            scratch.qpos[adr] = q[k]
        mujoco.mj_kinematics(m, scratch); mujoco.mj_comPos(m, scratch)
        pos = scratch.geom_xpos[target_id] if is_geom else scratch.site_xpos[target_id]
        err = target - pos
        if float(np.linalg.norm(err)) < tol:
            break
        jacp = np.zeros((3, m.nv))
        if is_geom:
            mujoco.mj_jacGeom(m, scratch, jacp, None, target_id)
        else:
            mujoco.mj_jacSite(m, scratch, jacp, None, target_id)
        J = jacp[:, dof_adr]; Jp = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), np.eye(3))
        dq = Jp @ err
        if q_ref is not None:
            dq += posture_gain * (np.eye(n) - Jp @ J) @ (q_ref - q)
        s = float(np.max(np.abs(dq)))
        if s > 0.1:
            dq *= 0.1 / s
        q = np.clip(q + dq, lo, hi)
    return q


class ScoopArm:
    """Physical-grasp scoop-fold phase machine for one arm. Same act()/reset()
    role as FoldExpert, driving 6 values (5 joint deltas + gripper)."""

    PHASES = ("hover", "down", "slide", "close", "lift", "carry", "place",
              "hold", "release", "retreat", "done")
    PATIENCE = {"hover": 45, "down": 35, "slide": 55, "close": 30, "lift": 18,
                "carry": 90, "place": 45, "hold": 10 ** 9, "release": 8, "retreat": 25, "done": 10 ** 9}

    def __init__(self, env, move, correction_fn, seed=0):
        self.env = env
        self.base = env.unwrapped
        self.move = move
        self.prefix = move.prefix
        self.slide_dir = np.asarray(move.slide_dir, float); self.slide_dir /= np.linalg.norm(self.slide_dir)
        self.correction_fn = correction_fn      # returns a live goal-xy correction (retry loop)
        self.rng = np.random.default_rng(seed)
        m = self.base.model
        self.m = m; self.d = self.base.data
        self.site_id = self.base._site_id[self.prefix]
        self.ramp_gid = self.base._ramp_gid[self.prefix]
        self.qpos_adr = self.base._arm_qpos_adr[self.prefix]
        self.dof_adr = self.base._arm_dof_adr[self.prefix]
        jr = [m.joint(f"{self.prefix}{n}").range for n in ARM_JOINTS]
        self.lo = np.array([r[0] for r in jr]); self.hi = np.array([r[1] for r in jr])
        self.q_scoop = np.asarray(env.reach_q[self.prefix], float)
        self.release_allowed = True
        self.reset()

    def reset(self):
        self.phase = 0
        self.phase_steps = 0
        self.approach_xy = None
        self.slide_dist = 0.0
        self.on_ramp = False
        self.seat = 0
        self.carry_quat = None
        self.carry_target = None
        self.grasp_steps = 0
        self.place_z = LIFT_Z

    # ---- state helpers ----------------------------------------------------

    def _q(self):
        return np.array([self.d.qpos[a] for a in self.qpos_adr])

    def _site(self):
        return self.d.site_xpos[self.site_id].copy()

    def _rampc(self):
        return self.d.geom_xpos[self.ramp_gid].copy()

    def _corner(self):
        return self.d.xpos[self.base._cloth_body_ids[self.move.corner]].copy()

    def _lift(self):
        return float(self._corner()[2] - TABLE_TOP_Z)

    def _grasped(self):
        return self.env._grasped(self.move)

    def _goal_xy(self):
        return np.asarray(self.move.goal_xy, float) + self.correction_fn()

    # ---- IK wrappers ------------------------------------------------------

    def _ramp_ik(self, target):
        return _ik_pos(self.m, self.d, self.ramp_gid, True, self.qpos_adr, self.dof_adr,
                       self.lo, self.hi, target, seed_q=self.q_scoop, q_ref=self.q_scoop)

    def _site_ik(self, target):
        return _ik_pos(self.m, self.d, self.site_id, False, self.qpos_adr, self.dof_adr,
                       self.lo, self.hi, target, seed_q=self._q(), q_ref=None, iters=200)

    def _track_ik(self, target):
        """6-DoF step toward target holding the captured jaw orientation (weighted
        by CARRY_ROT_W). Same idea as physical_expert.ScoopExpert._track."""
        scratch = mujoco.MjData(self.m); scratch.qpos[:] = self.d.qpos
        q = self._q().copy()
        for _ in range(40):
            for k, adr in enumerate(self.qpos_adr):
                scratch.qpos[adr] = q[k]
            mujoco.mj_kinematics(self.m, scratch); mujoco.mj_comPos(self.m, scratch)
            ep = target - scratch.site_xpos[self.site_id]
            R = scratch.site_xmat[self.site_id].reshape(3, 3)
            sq = np.zeros(4); mujoco.mju_mat2Quat(sq, scratch.site_xmat[self.site_id])
            el = np.zeros(3); mujoco.mju_subQuat(el, self.carry_quat, sq); er = R @ el
            if np.linalg.norm(ep) < 0.002 and np.linalg.norm(er) < 0.02:
                break
            jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, scratch, jp, jr, self.site_id)
            J = np.vstack([jp[:, self.dof_adr], CARRY_ROT_W * jr[:, self.dof_adr]])
            e = np.concatenate([ep, CARRY_ROT_W * er])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(6), e)
            s = float(np.max(np.abs(dq)))
            if s > 0.05:
                dq *= 0.05 / s
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def _carry_creep(self, goal_site_xyz):
        if self.carry_target is None:
            self.carry_target = self._site()
        if float(np.linalg.norm(self._site() - self.carry_target)) < TRACK_TOL:
            d = goal_site_xyz - self.carry_target
            n = float(np.linalg.norm(d))
            if n > CARRY_SPEED:
                d *= CARRY_SPEED / n
            self.carry_target = self.carry_target + d
        return self.carry_target.copy()

    # ---- plan / act -------------------------------------------------------

    def _q_target(self):
        name = self.PHASES[self.phase]
        if self.approach_xy is None:
            c = self._corner()
            self.approach_xy = c[:2] - self.slide_dir * APPROACH_BACK
        if name == "hover":
            return self._ramp_ik(np.array([self.approach_xy[0], self.approach_xy[1], TABLE_TOP_Z + HOVER_Z]))
        if name == "down":
            return self._ramp_ik(np.array([self.approach_xy[0], self.approach_xy[1], TABLE_TOP_Z + RAMP_SKIM_Z]))
        if name == "slide":
            tgt = self.approach_xy + self.slide_dir * self.slide_dist
            return self._ramp_ik(np.array([tgt[0], tgt[1], TABLE_TOP_Z + RAMP_SKIM_Z]))
        if name == "close":
            return self._q()                       # hold posture while the jaw shuts
        if name == "lift":
            s = self._site()
            return self._track_ik(np.array([s[0], s[1], TABLE_TOP_Z + LIFT_Z]))
        if name == "carry":
            s = self._site(); off = s[:2] - self._corner()[:2]
            g = self._goal_xy()
            goal_site = np.array([g[0] + off[0], g[1] + off[1], TABLE_TOP_Z + LIFT_Z])
            return self._track_ik(self._carry_creep(goal_site))
        if name == "place":
            ct = self.carry_target if self.carry_target is not None else self._site()
            self.place_z = max(self.place_z - 0.004, 0.012)
            return self._track_ik(np.array([ct[0], ct[1], TABLE_TOP_Z + self.place_z]))
        if name == "retreat":
            s = self._site()
            return self._site_ik(np.array([s[0], s[1], TABLE_TOP_Z + 0.12]))
        return None    # hold / release / done: stop moving

    def act(self):
        name = self.PHASES[self.phase]
        action = np.zeros(6, dtype=np.float32)
        # gripper: open while approaching and after release, closed once pinching
        action[5] = 1.0 if name in ("hover", "down", "slide", "release", "retreat", "done") else -1.0

        if name == "slide" and self.phase_steps % SLIDE_HOLD == 0:
            self.slide_dist += SLIDE_HOP

        q_target = self._q_target()
        if q_target is not None:
            action[0:5] = np.clip((q_target - self._q()) / JOINT_DELTA_SCALE, -1.0, 1.0)

        self.phase_steps += 1
        self._maybe_advance(name)
        return action

    def _maybe_advance(self, name):
        if name in ("done",) or (name == "hold" and not self.release_allowed):
            return
        advance = False
        if name in ("hover", "down"):
            advance = self.phase_steps >= 22
        elif name == "slide":
            horiz = float(np.linalg.norm((self._corner() - self._rampc())[:2]))
            if self._lift() > 0.015 and horiz < 0.026:
                self.on_ramp = True
            if self.on_ramp:
                self.seat += 1
            advance = (self.on_ramp and (horiz < 0.017 or self.seat >= 4 * SLIDE_HOLD))
        elif name == "close":
            self.grasp_steps = self.grasp_steps + 1 if self._grasped() else 0
            advance = self.grasp_steps >= 3
        elif name == "lift":
            advance = self.phase_steps >= 12
        elif name == "carry":
            advance = float(np.linalg.norm(self._corner()[:2] - self._goal_xy())) < 0.03
        elif name == "place":
            advance = self._lift() < 0.016
        elif name == "hold":
            advance = self.release_allowed
        elif name == "release":
            advance = (not self._grasped()) and self.phase_steps >= 3
        elif name == "retreat":
            advance = self.phase_steps >= 20

        if advance or self.phase_steps > self.PATIENCE[name]:
            if name == "close":
                q = np.zeros(4); mujoco.mju_mat2Quat(q, self.d.site_xmat[self.site_id])
                self.carry_quat = q          # jaw orientation to hold through the carry
            if name in ("close", "lift"):
                self.carry_target = None     # re-anchor the carry rate limiter on the live site
            self.phase = min(self.phase + 1, len(self.PHASES) - 1)
            self.phase_steps = 0


class QuarterFoldPhysicalExpert:
    """Two synchronised ScoopArms + a retry loop, mirroring QuarterFoldExpert."""

    def __init__(self, env, seed=0):
        self.env = env
        self.base = env.unwrapped
        self.correction = {m.prefix: np.zeros(2) for m in MOVES}
        self.arms = {m.prefix: ScoopArm(env, m, (lambda p=m.prefix: self.correction[p]), seed=seed + i)
                     for i, m in enumerate(MOVES)}
        self.moves = {m.prefix: m for m in MOVES}
        self.reset()

    def reset(self):
        for p, arm in self.arms.items():
            arm.reset(); arm.release_allowed = False
            self.correction[p] = np.zeros(2)
        self.retries = {p: 0 for p in self.arms}
        self.done_steps = {p: 0 for p in self.arms}

    def phases(self):
        return {p: a.PHASES[a.phase] for p, a in self.arms.items()}

    def _maybe_retry(self, p, arm):
        if arm.PHASES[arm.phase] != "done":
            self.done_steps[p] = 0
            return
        self.done_steps[p] += 1
        if (self.done_steps[p] < SETTLE_WAIT or self.env._placed(self.moves[p])
                or self.retries[p] >= MAX_RETRIES):
            return
        move = self.moves[p]
        carried = self.env._vertex(move.corner)[:2]
        self.correction[p] += self.env.goal(move)[:2] - carried   # nudge by the measured miss
        arm.reset(); arm.release_allowed = True
        self.retries[p] += 1
        self.done_steps[p] = 0

    def act(self):
        # both arms hold their placed corner until BOTH are holding, then release
        if all(a.PHASES[a.phase] == "hold" for a in self.arms.values()):
            for a in self.arms.values():
                a.release_allowed = True
        for p, arm in self.arms.items():
            self._maybe_retry(p, arm)
        return np.concatenate([self.arms["left_"].act(), self.arms["right_"].act()])


def run_episode(env, expert, seed, verbose=True):
    obs, info = env.reset(seed=seed)
    expert.reset()
    total = 0.0
    for t in range(env.unwrapped.max_episode_steps):
        obs, reward, terminated, truncated, info = env.step(expert.act())
        total += reward
        if verbose and t % 25 == 0:
            ph = expert.phases()
            print(f"  t{t:3d} L {ph['left_']:<8} R {ph['right_']:<8} score {info['fold_score']:.3f} "
                  f"d {'/'.join(f'{x:.3f}' for x in info['move_distance'])} "
                  f"grasp {int(info['grasped']['left_'])}{int(info['grasped']['right_'])} "
                  f"settle {info['settle_steps']} drift {info['anchor_drift']:.3f}")
        if terminated or truncated:
            break
    return {"seed": seed, "steps": t + 1, "reward": round(total, 2),
            "fold_score": round(info["fold_score"], 3), "retries": sum(expert.retries.values()),
            "success": info["success"], "reason": info["termination_reason"] or "truncated",
            "move_distance": [round(x, 3) for x in info["move_distance"]],
            "anchor_drift": round(info["anchor_drift"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--domain-random", action="store_true", default=True)
    ap.add_argument("--no-domain-random", dest="domain_random", action="store_false")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    env = QuarterFoldPhysicalEnv()
    env.unwrapped.domain_randomization = args.domain_random
    expert = QuarterFoldPhysicalExpert(env)
    rows = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        if not args.quiet:
            print(f"episode seed={seed}")
        row = run_episode(env, expert, seed, verbose=not args.quiet)
        rows.append(row); print(f"  -> {row}")
    print(f"success {sum(r['success'] for r in rows)}/{len(rows)}, "
          f"mean steps {np.mean([r['steps'] for r in rows]):.0f}, "
          f"mean score {np.mean([r['fold_score'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()

"""Two-arm PHYSICAL-GRASP fold (no weld), the quarter-fold task ported off the
weld cheat onto the real scoop-ramp + paddle contact grasp.

Relationship to the rest of the package
---------------------------------------
* quarter_fold_env.QuarterFoldEnv folds the cloth with the WELD cheat: closing a
  gripper near a listed cloth vertex instantly welds it to the hand. Two stages,
  full 0.30 m folds, far (north) corners grabbed from above.
* physical_env.PhysicalGraspEnv is the single-corner physical grasp: grabber
  plates bolted to the jaws, weld disabled, the corner held by contact+friction.
* THIS file is the intersection: the two-arm task on the physical grasp.

Why this is NOT a byte-for-byte port of the weld quarter fold (the scope cut)
----------------------------------------------------------------------------
The weld quarter fold grabs the FAR (north, +y) corners and carries them 0.30 m
south, and folds the whole west edge east. Re-measuring reachability for the
PHYSICAL grasp in the raised (+6 cm) side-by-side arm geometry showed two hard
walls that the weld cheat never hit (it teleports the cloth to the hand and
carries it welded, so reach in a specific low posture never mattered):

  1. The scoop can only reach the NEAR (south) corners. The bolted ramp has to
     skim the table under a corner while the gripper body clears the tabletop;
     that reach-down posture is only achievable for cloth_0 (near the LEFT arm)
     and cloth_110 (near the RIGHT arm). The far north corners sit ~0.45 m out --
     unreachable in any table-clear scoop posture (measured, FK/IK sweep).
  2. The carry can only hold ~0.10-0.12 m. Keeping the corner in the pocket needs
     the jaw ORIENTATION held (see physical_expert._track / this file's expert);
     a 5-DOF arm cannot hold that orientation and track a long carry, so the
     corner drops past ~0.12 m of travel. This is the same wall physical_env.py
     documents as FOLD_FRACTION = 0.5 for the single-corner task.

So the physical two-arm fold this env scores is: each arm SCOOPS its near south
corner and carries it diagonally inward-and-up onto the sheet, the two bottom
corners folded in together (a real simultaneous two-gripper fold). It is a
scope-reduced quarter fold -- one stage, near corners, ~0.10 m carries -- chosen
to fit the physical grasp's real reach envelope rather than forcing the weld
version's exact geometry onto a grasp that cannot reach it.

Grasp mechanism
---------------
The ramp MOUNT (its pose in the gripper frame) is re-calibrated per arm here --
not just the arm posture. The stock mount (prove_grabber.RAMP_POS/QUAT) was
calibrated for the OLD diagonal, table-height arm layout; in the raised
side-by-side layout that mount forces the gripper near-horizontal to keep the
ramp level, which drives the gripper/paddle into the tabletop before the ramp
can skim. build_scoop_mounts() instead picks a table-clear reach-down posture
per arm, reads the achieved gripper world orientation, and derives the ramp mount
that makes the ramp sit level-inclined at the table under the corner (leading
edge skimming, top face up, slide axis along the corner's outward diagonal so it
catches the corner TIP instead of plowing the whole edge). The paddle rides at
PADDLE_GAP along the ramp normal, so the closed jaw pinches the corner in the
pocket. Weld stays disabled throughout; "grasp" is contact + friction, reported
by physical_grasp_vertex() (jaw commanded closed AND the carried vertex sitting
in the closed jaw's pocket).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import mujoco

from cloth_fold_rl.fold_env import (
    CLOTH_JITTER, CTRL_COST, DRAG_LIMIT, GRASP_BONUS, SUCCESS_BONUS, SUCCESS_DIST, W_CARRY, W_REACH,
)
from cloth_fold_rl.expert import solve_ik
from sim_main import (
    ClothFoldEnv, StateOnlyWrapper, TABLE_TOP_Z, CLOTH_COUNT, ARM_JOINTS,
    GRIPPER_OPEN, GRIPPER_CLOSED,
)
from prove_grabber import RAMP_SIZE, PADDLE_GAP

N = CLOTH_COUNT
CLOTH_0, CLOTH_10, CLOTH_110, CLOTH_120 = 0, N - 1, (N - 1) * N, N * N - 1

# --- physical grasp / pocket thresholds (same spirit as physical_env.py) ------
PINCH_RADIUS = 0.032   # carried vertex within this of the closed jaw's pocket = held...
PINCH_MIN_DZ = -0.020  # ...and not ejected below the ramp

# --- scoop-mount calibration constants (see module docstring) -----------------
SCOOP_INCLINE = np.radians(24)   # ramp incline: leading edge low, root high, so the corner tip cams UP
REACH_DOWN_Z = 0.05              # gripper site height above the corner for the reach-down calibration posture
RAMP_LEAD_SKIM = 0.004           # ramp leading edge target height above the table

# --- fold scope (see docstring for why these are near corners / short carries) -
# each arm scoops its near south corner and carries it diagonally inward+north.
# goals are absolute table xy; kept within the measured ~0.10 m carry envelope.
LEFT_CORNER, RIGHT_CORNER = CLOTH_0, CLOTH_110
LEFT_GOAL_XY = np.array([-0.06, -0.075])
RIGHT_GOAL_XY = np.array([0.06, -0.075])
# outward diagonal each arm slides along to catch its corner tip (toward interior)
LEFT_SLIDE_DIR = np.array([1.0, 1.0])
RIGHT_SLIDE_DIR = np.array([-1.0, 1.0])

# anchors that must not be dragged: the far (north) corners the arms never touch
ANCHORS = ((CLOTH_10, CLOTH_10), (CLOTH_120, CLOTH_120))

SETTLE_STEPS = 20
MAX_STEPS = 500
ACTION_DIM = 12


def _desired_ramp_world_R(slide_xy):
    """Ramp world rotation: columns = local axes. x=normal (up, tilted back),
    y=slide (along slide_xy, tilted down toward the leading edge), z=wide."""
    s = np.asarray(slide_xy, float); s = s / (np.linalg.norm(s) + 1e-9)
    slide = np.array([s[0] * np.cos(SCOOP_INCLINE), s[1] * np.cos(SCOOP_INCLINE), -np.sin(SCOOP_INCLINE)])
    normal = np.array([s[0] * np.sin(SCOOP_INCLINE), s[1] * np.sin(SCOOP_INCLINE), np.cos(SCOOP_INCLINE)])
    wide = np.cross(normal, slide)
    return np.column_stack([normal, slide, wide])


def build_scoop_mounts(corners, slide_dirs):
    """Per-arm ramp mount (pos,quat in the gripper frame) and the reach-down
    calibration posture, derived for the CURRENT geometry. Deterministic (uses a
    stock env reset at seed 0), so it is safe to call once at env construction.

    Returns (mounts, reach_q) where mounts[prefix] = {"ramp_pos", "ramp_quat"}
    and reach_q[prefix] is the 5-joint reach-down posture used for the scoop.
    """
    base = ClothFoldEnv(observation_mode="state", action_mode="joint_delta", max_episode_steps=50)
    base.reset(seed=0)
    m, d = base.model, base.data
    mounts, reach_q = {}, {}
    for p in ["left_", "right_"]:
        c = d.xpos[base._cloth_body_ids[corners[p]]].copy()
        sdir = np.asarray(slide_dirs[p], float); sdir = sdir / (np.linalg.norm(sdir) + 1e-9)
        jr = [m.joint(f"{p}{n}").range for n in ARM_JOINTS]
        q, _ = solve_ik(m, d, base._site_id[p], base._arm_qpos_adr[p], base._arm_dof_adr[p],
                        jr, np.array([c[0], c[1], TABLE_TOP_Z + REACH_DOWN_Z]), rng=np.random.default_rng(0))
        reach_q[p] = q
        scratch = mujoco.MjData(m); scratch.qpos[:] = d.qpos
        for k, adr in enumerate(base._arm_qpos_adr[p]):
            scratch.qpos[adr] = q[k]
        mujoco.mj_kinematics(m, scratch); mujoco.mj_comPos(m, scratch)
        gb = m.body(f"{p}gripper").id
        grip_pos = scratch.xpos[gb].copy(); Rg = scratch.xmat[gb].reshape(3, 3).copy()
        Rr = _desired_ramp_world_R(sdir)
        lead_world = np.array([c[0] - sdir[0] * 0.01, c[1] - sdir[1] * 0.01, TABLE_TOP_Z + RAMP_LEAD_SKIM])
        ramp_center_world = lead_world - Rr[:, 1] * RAMP_SIZE[1]
        ramp_pos_local = Rg.T @ (ramp_center_world - grip_pos)
        ramp_quat_local = np.zeros(4)
        mujoco.mju_mat2Quat(ramp_quat_local, np.ascontiguousarray(Rg.T @ Rr).ravel())
        mounts[p] = {"ramp_pos": ramp_pos_local.tolist(), "ramp_quat": ramp_quat_local.tolist()}
    return mounts, reach_q


def make_scoop_hook(mounts):
    """spec_hook that bolts the re-calibrated ramp + paddle onto both jaws. Both
    plates go on the gripper body so the pocket (paddle at PADDLE_GAP along the
    ramp normal) stays fixed relative to the ramp. Weld is left compiled but is
    never activated (see TwoArmScoopEnv.set_gripper)."""
    def hook(spec):
        for p in ["left_", "right_"]:
            rp = np.array(mounts[p]["ramp_pos"]); rq = np.array(mounts[p]["ramp_quat"])
            Rr = np.zeros(9); mujoco.mju_quat2Mat(Rr, rq); Rr = Rr.reshape(3, 3)
            pp = rp + Rr[:, 0] * PADDLE_GAP        # paddle offset along the ramp normal
            for name, pos, rgba in [("ramp", rp, [0.2, 0.9, 0.4, 1.0]),
                                    ("paddle", pp, [0.9, 0.4, 0.2, 1.0])]:
                g = spec.body(f"{p}gripper").add_geom()
                g.name = f"{p}{name}"; g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size = RAMP_SIZE; g.pos = pos.tolist(); g.quat = rq.tolist(); g.rgba = rgba
                g.friction = [1.2, 0.05, 0.001]; g.condim = 3
                g.solref = [0.02, 1]; g.solimp = [0.8, 0.9, 0.01, 0.5, 2]
    return hook


class TwoArmScoopEnv(ClothFoldEnv):
    """ClothFoldEnv with the re-calibrated grabber plates on both jaws and the
    weld cheat disabled -- the two-arm analogue of physical_env.PhysicalGraspEnv."""

    def __init__(self, mounts, **kw):
        super().__init__(spec_hook=make_scoop_hook(mounts), **kw)
        self._ramp_gid = {p: self.model.geom(f"{p}ramp").id for p in self.prefixes}
        self._paddle_gid = {p: self.model.geom(f"{p}paddle").id for p in self.prefixes}

    def set_gripper(self, prefix, command):
        # close/open the gripper with the same hysteresis as the stock env, but
        # NEVER weld -- holding the cloth is contact + friction only.
        if command < -0.3:
            self._gripper_closed[prefix] = True
        elif command > 0.3:
            self._gripper_closed[prefix] = False
        closed = self._gripper_closed[prefix]
        self.data.ctrl[self._gripper_act[prefix]] = GRIPPER_CLOSED if closed else GRIPPER_OPEN
        for eqid in self._weld_ids[prefix]:
            self.data.eq_active[eqid] = 0

    def pinch_point(self, prefix):
        return 0.5 * (self.data.geom_xpos[self._ramp_gid[prefix]]
                      + self.data.geom_xpos[self._paddle_gid[prefix]])

    def physical_grasp_vertex(self, prefix, vertex):
        """True when the jaw is commanded closed AND cloth `vertex` sits in the
        closed jaw's pocket (the two-arm generalisation of
        PhysicalGraspEnv.physical_grasp, which only ever checked one corner)."""
        if not self._gripper_closed[prefix]:
            return False
        v = self.data.xpos[self._cloth_body_ids[vertex]]
        pk = self.pinch_point(prefix)
        return bool(np.linalg.norm(v - pk) < PINCH_RADIUS and v[2] - pk[2] > PINCH_MIN_DZ)


@dataclass(frozen=True)
class Move:
    prefix: str
    corner: int           # cloth vertex this arm scoops
    goal_xy: np.ndarray   # absolute table xy the corner is carried to
    slide_dir: np.ndarray # outward diagonal the ramp slides along to catch the tip


MOVES = (
    Move("left_", LEFT_CORNER, LEFT_GOAL_XY, LEFT_SLIDE_DIR),
    Move("right_", RIGHT_CORNER, RIGHT_GOAL_XY, RIGHT_SLIDE_DIR),
)


class QuarterFoldPhysicalEnv(gym.Wrapper):
    """Two-arm physical-grasp fold. Both arms scoop their near south corner and
    carry it diagonally inward; the fold has to hold after release. See docstring."""

    def __init__(self, max_episode_steps=MAX_STEPS, seed=None, cloth_jitter=CLOTH_JITTER):
        corners = {m.prefix: m.corner for m in MOVES}
        slide_dirs = {m.prefix: m.slide_dir for m in MOVES}
        self.scoop_mounts, self.reach_q = build_scoop_mounts(corners, slide_dirs)
        env = TwoArmScoopEnv(self.scoop_mounts, observation_mode="state", action_mode="joint_delta",
                             max_episode_steps=max_episode_steps)
        super().__init__(env)
        self.cloth_jitter = cloth_jitter
        self._rng = np.random.default_rng(seed)
        self._flat = StateOnlyWrapper(env)
        self.observation_space = self._flat.observation_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self._start = None
        self._settle_steps = 0
        self._prev_potential = 0.0

    # ---- geometry helpers -------------------------------------------------

    def _vertex(self, index):
        return self.env.data.xpos[self.env._cloth_body_ids[index]]

    def goal(self, move):
        # absolute goal on the table at the corner's resting height
        return np.array([move.goal_xy[0], move.goal_xy[1], self._start[move.corner][2]])

    def _grasped(self, move):
        return self.env.physical_grasp_vertex(move.prefix, move.corner)

    def _move_distance(self, move):
        # measured in the table plane: the corner springs a little in z after
        # release, so xy placement is what defines the fold landing on target
        return float(np.linalg.norm((self._vertex(move.corner) - self.goal(move))[:2]))

    def _placed(self, move):
        return self._move_distance(move) < SUCCESS_DIST

    def _start_distance(self, move):
        return max(float(np.linalg.norm((self._start[move.corner] - self.goal(move))[:2])), 1e-6)

    def _anchor_drift(self):
        return max(float(np.linalg.norm(self._vertex(v) - self._start[ref])) for v, ref in ANCHORS)

    def fold_score(self):
        done = sum(self._start_distance(m) - min(self._move_distance(m), self._start_distance(m)) for m in MOVES)
        total = sum(self._start_distance(m) for m in MOVES)
        return float(done / total)

    # ---- reward (potential shaping, same shape as QuarterFoldEnv) ----------

    def _move_potential(self, move):
        d = self._move_distance(move)
        if self._grasped(move):
            return GRASP_BONUS - W_CARRY * d
        if self._placed(move):
            return GRASP_BONUS + 3.0 - W_CARRY * d
        gripper = self.env.data.site_xpos[self.env._site_id[move.prefix]]
        reach = float(np.linalg.norm(gripper - self._vertex(move.corner)))
        return -W_REACH * reach - W_CARRY * self._start_distance(move)

    def _potential(self):
        return sum(self._move_potential(m) for m in MOVES)

    def _settled(self):
        return all(not self._grasped(m) and self._placed(m) for m in MOVES)

    # ---- gym API ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.cloth_jitter > 0 and "cloth_pose" not in opts:
            opts["cloth_pose"] = self._rng.uniform(-self.cloth_jitter, self.cloth_jitter, size=2)
        obs, info = self.env.reset(seed=seed, options=opts)
        self._start = self.env.data.xpos[self.env._cloth_body_ids].copy()
        self._settle_steps = 0
        self._prev_potential = self._potential()
        info = dict(info); info.update(self._info(None))
        return self._flat.observation(obs), info

    def _expand(self, action):
        a = np.zeros(14, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).ravel()
        a[0:5], a[6] = action[0:5], action[5]       # left joint deltas, left gripper
        a[7:12], a[13] = action[6:11], action[11]   # right joint deltas, right gripper
        return a

    def _info(self, reason):
        return {"fold_score": self.fold_score(), "settle_steps": self._settle_steps,
                "grasped": {m.prefix: self._grasped(m) for m in MOVES},
                "move_distance": [self._move_distance(m) for m in MOVES],
                "anchor_drift": self._anchor_drift(), "success": reason == "success",
                "termination_reason": reason}

    def step(self, action):
        obs, _, _, _, info = self.env.step(self._expand(action))
        potential = self._potential()
        reward = potential - self._prev_potential - CTRL_COST * float(np.square(action).sum())
        self._prev_potential = potential
        self._settle_steps = self._settle_steps + 1 if self._settled() else 0

        terminated, reason = False, None
        if self._settle_steps >= SETTLE_STEPS:
            reward += SUCCESS_BONUS
            terminated, reason = True, "success"
        elif self._anchor_drift() > DRAG_LIMIT:
            reward -= 5.0
            terminated, reason = True, "cloth_dragged"
        elif self.env._failed():
            reward -= 5.0
            terminated, reason = True, "unstable"
        truncated = (not terminated) and self.env._step_count >= self.env.max_episode_steps
        info = dict(info); info.update(self._info(reason))
        return self._flat.observation(obs), float(reward), terminated, truncated, info

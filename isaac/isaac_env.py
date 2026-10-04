"""WorldFold's cloth-fold env on LeHome's Isaac stack: a drop-in for mujuco.sim_main.ClothFoldEnv in joint_delta mode.

Same 14-dim action and observation dict as ClothFoldEnv, so cloth_fold_rl's wrappers (SingleCornerFoldEnv,
HalfFoldEnv) run on it unchanged. The scene (isaac/lab_scene.py) is IsaacLab with LeHome's SO101 arms, particle
cloth parameters and simulation setup (lehome-challenge a805ad2); the table, cloth size, arm base poses and
camera come from mujuco/cloth_params.py. Nothing attaches the cloth: the jaws hold it by friction and adhesion.

Call start_app() (or construct the env, which does it) before importing anything else from Isaac Sim or IsaacLab.
"""

import time

import numpy as np
import gymnasium as gym

from mujuco.cloth_params import (
    CLOTH_COUNT, CLOTH_SPACING, CLOTH_MASS, ARM_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED, JOINT_DELTA_SCALE,
    GRASP_CORNERS, GRASP_RADIUS, HOLD_STEPS, SETTLE_STEPS, ARM_TIMESTEP, WORKSPACE_XY, SUCCESS_FOLD_SCORE,
    CORNER_PLACED_DIST, TASK_NAMES, DEPTH_MAX, CAMERA_FOVY_DEG, sensor_depth,
)

# Isaac-only knobs (MuJoCo has its own equivalents inside the MJCF / flexcomp)
PHYSICS_DT              = 1.0 / 100.0         # LeHome steps at 1/90; 1/100 divides the 0.05 s control step
TABLE_SIZE              = 0.70                # room for the moved cloth and its reset jitter
CLOTH_SUBDIV            = 10                  # particles per grid gap: 101x101 at 3 mm, ~LeHome's garment particle count
# cloth centre on the table, x y: 13.5 cm toward the arms. Pointing straight down, the gripper reaches the table
# between about 8 and 32 cm from its arm's base; this puts the half fold's far corners 32 cm out and its near corners
# 7 cm out, the 30 cm fold spanning that whole range
CLOTH_CENTER            = (0.0, -0.135)
# HalfFoldEnv's reset offset of the cloth, +-m in x and y (MuJoCo: fold_env.CLOTH_JITTER, 2.5 cm). With the fold
# spanning the whole straight-down reach, 2.5 cm pushes one end out of it; at 1 cm the scripted fold's IK still lands
# every pinch within 4 mm and every place within 2 cm
CLOTH_JITTER            = 0.01
# LeHome's particle_garment_cfg.yaml, except: its particle_mass (10 g per particle) would make this 10,201-particle
# sheet ~100 kg, so the sheet weighs CLOTH_MASS in total; its gravity_scale of 2 is undocumented, real gravity is 1;
# its adhesion of 0.1 pulls the cloth onto every rigid surface it touches, the table included, and real cloth on a
# dry table has none
CLOTH_GRAVITY_SCALE     = 1.0
CLOTH_ADHESION          = 0.0
# Reset drops the cloth LeHome-style so it lands with some slack instead of perfectly flat: from DROP_HEIGHT above
# its resting height with a random roll and pitch of up to DROP_TILT_DEG (LeHome drops garments from ~0.6 m, +-36 deg)
DROP_HEIGHT             = 0.05
DROP_TILT_DEG           = 10.0
# High-friction lining on the inner face of each finger, as on the real gripper: a flat pad PAD_LINING_THICKNESS
# thick over LeHome's capsule pads. Each mm on both sides closes the jaw gap 2 mm earlier; at 3 mm the pads meet at
# about -0.015 rad, so closing to GRIPPER_CLOSED squeezes.
PAD_LINING_THICKNESS    = 0.003
PAD_LINING_FRICTION     = 1.5
# LeHome's USD root frame in the MJCF base frame (ARM_BASE_* poses): same kinematics, with the root offset and
# turned +90 deg about z. Fitted on six joint poses, residual under 0.3 mm.
USD_ROOT_OFFSET         = np.array([0.0164, -0.0208, -0.0324])
USD_ROOT_YAW            = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
# gripperframe site in the gripper link frame: the so101_new_calib MJCF's site (LeHome's URDF gripper_frame_link)
GRIPPERFRAME_POS        = np.array([-0.0079, -0.000218121, -0.0981274])
GRIPPERFRAME_QUAT       = np.array([0.0, 0.0, 1.0, 0.0])
# weld grasp (grasp_mode="weld", Phase W): particles within this distance of a welded grid vertex are pinned with it.
# MuJoCo's grid vertex is one body standing for a 3 cm cell; one 3 mm particle alone would tear out of the sheet.
WELD_PATCH              = 0.012
# MuJoCo's weld is an equality constraint at the default solref (0.02 s, damping ratio 1): soft. The Isaac weld steers
# the patch with that time constant (lab_scene._drive_pins); a rigid zero-mass pin (None) stored the fold's tension and
# snapped the corner back up to 12 cm on release (W3).
WELD_TAU                = 0.02
WELD_MASS               = 10.0                # soft weld: the pinned patch's mass multiple, calibrated so weld_check matches MuJoCo's weld (lag 27 vs 23-27 mm, rise 8.8 vs 9.0 cm)
# The so101_new_calib MJCF's sts3215 class, which MuJoCo used for every joint: position actuator kp 998.22, kv 2.731,
# plus joint damping 0.60 (both damp joint velocity, so they add), armature 0.028, forcerange 3.35 on each
# actuator. LeHome's SO101 drives are kp 17.8 / kd 0.60 / 10 N m, about 56x softer, so the arm lags its targets.
# The MJCF's frictionloss (0.052 N m) has no equivalent: PhysX joint friction is a coefficient, not a torque.
MUJOCO_ARM_DRIVE        = {"stiffness": 998.22, "damping": 2.731 + 0.60, "effort_limit_sim": 3.35, "armature": 0.028}
# Scene profiles. "lehome" is this env as ported from LeHome (friction grasp on the CPU device, cloth toward the arms,
# tilted drop). "weld" carries over the MuJoCo setup the imitation pipeline was built on (Phase W): the weld grasp
# (GPU pipeline), the cloth centred, a flat drop, MuJoCo's arm drives and its dynamics DR (reset, ×U(0.7, 1.3) on
# cloth mass, cloth-table friction and cloth damping while domain_randomization is on).
PROFILES = {
    "lehome": {"cloth_center": CLOTH_CENTER, "drop_height": DROP_HEIGHT, "drop_tilt_deg": DROP_TILT_DEG,
               "arm_drive": None, "dynamics_dr": False, "grasp_mode": "friction", "device": "cpu", "weld_tau": None},
    "weld": {"cloth_center": (0.0, 0.0), "drop_height": 0.005, "drop_tilt_deg": 0.0,
               "arm_drive": MUJOCO_ARM_DRIVE, "dynamics_dr": True, "grasp_mode": "weld", "device": "cuda:0",
               "weld_tau": WELD_TAU},
}
DR_RANGE                = (0.7, 1.3)          # mujuco/sim_main.py reset
CLOTH_SPEED_LIMIT       = 20.0                # m/s; replaces MuJoCo's qacc explosion check
KIT_TICK_PERIOD_S       = 30.0                # state mode ticks Kit this often; its hang detector allows 120 s
CAMERA_CLIP             = (0.05, 10.0)
# lights mirror build_cloth_xml's two <light>s plus MuJoCo's default headlight (diffuse 0.4, ambient 0.1):
# same directions, same relative strengths. LIGHT_INTENSITY (Isaac units per unit of MuJoCo diffuse) is a knob.
LIGHT_INTENSITY         = 1000.0
LIGHTS                  = (((0.0, 0.0, -1.0), 0.9), ((-0.5, 0.5, -1.0), 0.4))   # (direction, MuJoCo diffuse)
HEADLIGHT_DIFFUSE       = 0.4
HEADLIGHT_AMBIENT       = 0.1
FLOOR_SIZE              = 4.0                 # MJCF floor plane size="2 2", half-extents
FLOOR_COLOR             = (0.3, 0.3, 0.35)

GRIPPER_JOINT = "gripper"

_app = None


def start_app(headless=True, cameras=False, device="cpu"):
    """Launches Isaac Sim once through IsaacLab's AppLauncher on the CPU device, as LeHome's scripts do."""
    global _app
    if _app is None:
        import os
        from isaaclab.app import AppLauncher
        # extra Kit settings, e.g. the local install's telemetry-off / no-registry / portable-root flags
        # (isaac/env_windows.ps1); unset on Modal
        kit_args = os.environ.get("WORLDFOLD_KIT_ARGS", "")
        _app = AppLauncher(headless=headless, enable_cameras=cameras, device=device, kit_args=kit_args).app
    return _app


def _npy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)



def _quat_from_matrix(R):
    # wxyz quaternion from a 3x3 rotation matrix, picking the largest component first so 180 deg cases stay exact
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array([s / 4.0, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, s / 4.0, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4.0, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4.0])


def _matrix_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_facing(direction):
    # wxyz rotation whose local -Z points along direction (USD lights and cameras look down -Z)
    z = -np.asarray(direction, dtype=float)
    z = z / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x = x / np.linalg.norm(x)
    return _quat_from_matrix(np.column_stack([x, np.cross(z, x), z]))


def cloth_grid_mesh(subdiv=1):
    # row-major grid, index = ix * n + iy, same as MuJoCo's flexcomp; subdiv splits each grid gap into finer particles
    n = (CLOTH_COUNT - 1) * subdiv + 1
    step = CLOTH_SPACING / subdiv
    half = (CLOTH_COUNT - 1) * CLOTH_SPACING / 2.0
    points = []
    for ix in range(n):
        for iy in range(n):
            points.append((ix * step - half, iy * step - half, 0.0))
    faces = []
    for ix in range(n - 1):
        for iy in range(n - 1):
            a = ix * n + iy
            b = a + n
            c = a + 1
            d = b + 1
            faces += [a, b, c, c, b, d]
    return points, faces


def usd_root_pose(base_pos, base_quat):
    """World pose (pos, wxyz) of LeHome's SO101 USD root for an arm whose MJCF base sits at base_pos, base_quat."""
    R = _matrix_from_quat(base_quat)
    return np.asarray(base_pos) + R @ USD_ROOT_OFFSET, _quat_from_matrix(R @ _matrix_from_quat(USD_ROOT_YAW))


def grid_particles(subdiv):
    # particle index of each MuJoCo grid vertex (cloth_0..cloth_120, row-major) in the subdivided mesh
    n = (CLOTH_COUNT - 1) * subdiv + 1
    return np.array([(ix * subdiv) * n + iy * subdiv for ix in range(CLOTH_COUNT) for iy in range(CLOTH_COUNT)])


class IsaacClothFoldEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, control_dt=0.05, max_episode_steps=200, observation_mode="state", image_size=(84, 84),
                 camera_names=None, n_cloth_samples=9, n_tasks=4, grasp_corners=None, grasp_radius=GRASP_RADIUS,
                 headless=True, cameras=None, grasp_mode=None, device=None, profile="lehome"):
        # profile: a PROFILES key; grasp_mode and device default to the profile's
        # device: "cpu" (LeHome's choice: on the CUDA device the grippers pass through the cloth) or "cuda:0", whose
        # particle tensor view the weld grasp needs to pin particles exactly (zero mass + set positions, Phase W)
        # grasp_mode: "friction" (the jaws hold the cloth, LeHome's way) or "weld" (MuJoCo's: on close, every allowed
        # grasp corner within grasp_radius attaches to the gripper until it opens -- the Phase W baseline)
        self.profile = profile
        self._prof = PROFILES[profile]
        grasp_mode = grasp_mode or self._prof["grasp_mode"]
        device = device or self._prof["device"]
        if self._prof["dynamics_dr"] and device == "cpu":
            raise ValueError("the dynamics DR needs the GPU pipeline (device='cuda:0')")
        # MuJoCo's switch (ClothFoldEnv.domain_randomization; HalfFoldEnv sets it): only the "weld" profile has DR
        self.domain_randomization = False
        if grasp_mode not in ("friction", "weld"):
            raise ValueError(grasp_mode)
        if grasp_mode == "weld" and device == "cpu":
            raise ValueError("grasp_mode='weld' needs the GPU pipeline (device='cuda:0'); see lab_scene.site_pose")
        self.grasp_mode = grasp_mode
        self._pinned = {"left_": {}, "right_": {}}    # weld: {grid vertex: (particle idx, offsets)}
        # cameras: {name: square size} -- the imitation pipeline's camera rig (main + wrists), read with render_rig();
        # independent of observation_mode's own 84x84 main image
        self.rig = dict(cameras) if cameras else {}
        if camera_names is not None and list(camera_names) != ["main"]:
            raise NotImplementedError("only the shared 'main' camera is ported")
        self.observation_mode = observation_mode
        self.image_size = image_size
        self._use_image = observation_mode in ("pixels", "hybrid")
        self._use_cloth = observation_mode in ("state", "hybrid")
        self.sim_device = device
        start_app(headless, cameras=self._use_image or bool(self.rig), device=device)
        import torch
        from isaac.lab_scene import SceneEnv, make_cfg
        self._torch = torch

        self.control_dt = control_dt
        self.max_episode_steps = max_episode_steps
        self.n_substeps = int(round(control_dt / PHYSICS_DT))
        self.settle_steps = int(round(SETTLE_STEPS * ARM_TIMESTEP / control_dt))
        self.grasp_corners = dict(GRASP_CORNERS if grasp_corners is None else grasp_corners)
        self.grasp_radius = grasp_radius
        self.prefixes = ["left_", "right_"]
        self.action_mode = "joint_delta"
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        self.camera_names = ["main"]
        self.n_cloth_samples = n_cloth_samples
        self.n_tasks = n_tasks

        per_arm = len(ARM_JOINTS) * 2 + 1 + 7 + 6 + 1
        P = per_arm * len(self.prefixes)
        C = (4 * 3) + (4 * 3) + (self.n_cloth_samples * 3) + 3 + 3 + (4 * 3)
        T = self.n_tasks + 1 + (4 * 3) + 4 + 1
        H, W = self.image_size
        spaces = {
            "proprio": gym.spaces.Box(-np.inf, np.inf, shape=(P,), dtype=np.float32),
            "task": gym.spaces.Box(-np.inf, np.inf, shape=(T,), dtype=np.float32),
        }
        if self._use_cloth:
            spaces["cloth_state"] = gym.spaces.Box(-np.inf, np.inf, shape=(C,), dtype=np.float32)
        if self._use_image:
            spaces["image"] = gym.spaces.Box(0, 255, shape=(H, W, 3), dtype=np.uint8)
            spaces["depth"] = gym.spaces.Box(0.0, DEPTH_MAX, shape=(H, W, 1), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self._pixel_angle = np.radians(CAMERA_FOVY_DEG) / H

        n_vert = CLOTH_COUNT * CLOTH_COUNT
        self._corner_idx = [0, CLOTH_COUNT - 1, (CLOTH_COUNT - 1) * CLOTH_COUNT, n_vert - 1]
        self._sample_idx = np.linspace(0, n_vert - 1, self.n_cloth_samples).astype(int)
        self._grid = grid_particles(CLOTH_SUBDIV)
        self.weld_mask = {p: None for p in self.prefixes}

        cfg = make_cfg(PHYSICS_DT, self.n_substeps, image_size if self._use_image else None, rig=self.rig, device=device,
                       arm_drive=self._prof["arm_drive"])
        self.lab = SceneEnv(cfg, self._prof["cloth_center"])
        self.lab.weld_tau = self._prof["weld_tau"]
        self.lab.weld_mass = WELD_MASS
        self.arms = self.lab.arms
        left = self.arms["left_"]
        self._dof_limits = _npy(left.data.soft_joint_pos_limits[0, self.lab.joint_ids["left_"]])   # (6, 2), both arms
        self._particles = self.lab.cloth_rest.copy()
        self._kit_ticked = time.time()
        self._particle_vel = np.zeros_like(self._particles)

        self._joint_targets = {p: self._home() for p in self.prefixes}
        self._gripper_closed = {p: False for p in self.prefixes}
        self._step_count = 0
        self._task_id = 0
        self._goal_corners = np.zeros((4, 3))
        self._goal_scale = np.ones(4)
        self._success_steps = 0
        self._action_clipped = False
        self._domain_params = {}

    @staticmethod
    def _home():
        q = np.zeros(6)
        q[5] = GRIPPER_OPEN
        return q

    # ---- state readers ----
    def _refresh_cloth(self):
        # particle velocities as the mean over the last control step (the CPU device has no particle velocity view)
        positions = self.lab.particle_positions()
        self._particle_vel = (positions - self._particles) / self.control_dt
        self._particles = positions

    def _particle_positions(self):
        return self._particles

    def cloth_positions(self):
        # the MuJoCo grid's 121 vertices, picked out of the denser simulated mesh
        return self._particles[self._grid]

    def cloth_velocities(self):
        return self._particle_vel[self._grid]

    def corner_positions(self):
        return self.cloth_positions()[self._corner_idx]

    def _gripper_link(self, prefix):
        data = self.arms[prefix].data
        b = self.lab.gripper_body[prefix]
        return data, b, _npy(data.body_link_pos_w[0, b]), _matrix_from_quat(_npy(data.body_link_quat_w[0, b]))

    def gripper_pose(self, prefix):
        # gripperframe site = gripper link pose composed with the site offset. The quaternion goes through a
        # rotation matrix so its sign follows mju_mat2Quat, as in MuJoCo's proprio.
        _, _, link_pos, link_R = self._gripper_link(prefix)
        pos = link_pos + link_R @ GRIPPERFRAME_POS
        quat = _quat_from_matrix(link_R @ _matrix_from_quat(GRIPPERFRAME_QUAT))
        return pos, quat

    def gripper_velocity(self, prefix):
        # [angular(3), linear(3)] of the gripperframe site in world axes, mj_objectVelocity(flg_local=0)'s layout
        data, b, link_pos, _ = self._gripper_link(prefix)
        lin = _npy(data.body_link_lin_vel_w[0, b])
        ang = _npy(data.body_link_ang_vel_w[0, b])
        return np.concatenate([ang, lin + np.cross(ang, self.gripper_position(prefix) - link_pos)])

    def gripper_position(self, prefix):
        return self.gripper_pose(prefix)[0]

    def joint_positions(self, prefix):
        return _npy(self.arms[prefix].data.joint_pos[0, self.lab.joint_ids[prefix]])

    def joint_velocities(self, prefix):
        return _npy(self.arms[prefix].data.joint_vel[0, self.lab.joint_ids[prefix]])

    def grasp_active(self, prefix):
        if self.grasp_mode == "weld":
            return bool(self._pinned[prefix])
        # nothing attaches the cloth, so this reports what the wrappers ask: the gripper is closed with one of its
        # corners (grasp_corners, filtered by weld_mask) within grasp_radius of the gripper frame
        if not self._gripper_closed[prefix]:
            return False
        pos = self.gripper_position(prefix)
        allc = self.cloth_positions()
        allowed = self.weld_mask.get(prefix)
        return any(np.linalg.norm(allc[vtx] - pos) < self.grasp_radius for vtx in self.grasp_corners[prefix]
                   if allowed is None or vtx in allowed)

    # ---- task helpers (mirror ClothFoldEnv) ----
    # ponytail: duplicated from sim_main; extract shared fold-task functions when a second task variant lands
    def _corner_dists(self):
        return np.linalg.norm(self._goal_corners - self.corner_positions(), axis=1)

    def _corner_progress(self):
        return (self._corner_dists() < CORNER_PLACED_DIST).astype(np.float32)

    def _fold_score(self):
        return float(np.clip(1.0 - self._corner_dists() / self._goal_scale, 0.0, 1.0).mean())

    def _failed(self):
        pos = self._particle_positions()
        if not np.all(np.isfinite(pos)):
            return True
        if np.any(np.abs(pos[:, :2]) > WORKSPACE_XY):
            return True
        speed = np.linalg.norm(self._particle_vel, axis=1)
        if np.max(speed) > CLOTH_SPEED_LIMIT:
            return True
        return False

    def _reward(self):
        dist_term = -float(self._corner_dists().mean())
        success_term = 10.0 if self._fold_score() >= SUCCESS_FOLD_SCORE else 0.0
        terms = {"corner_dist": dist_term, "success_bonus": success_term}
        return dist_term + success_term, terms

    def _step_info(self, terms, reason=None):
        return {
            "reward_terms": terms,
            "success": self._success_steps >= HOLD_STEPS,
            "fold_score": self._fold_score(),
            "action_clipped": self._action_clipped,
            "left_ik_success": True,
            "right_ik_success": True,
            "left_grasp_active": self.grasp_active("left_"),
            "right_grasp_active": self.grasp_active("right_"),
            "termination_reason": reason,
        }

    # ---- observations ----
    def _render_image(self):
        # the scene renders the camera once per control step
        out = self.lab.camera.data.output
        rgb = _npy(out["rgb"][0])[:, :, :3].astype(np.uint8)
        raw = _npy(out["distance_to_image_plane"][0])[:, :, 0].astype(np.float32)
        raw = np.where(np.isfinite(raw), raw, DEPTH_MAX + 1.0)
        depth = sensor_depth(raw, self._pixel_angle, self.np_random)
        return rgb, depth[:, :, None].astype(np.float32)

    def _get_obs(self):
        proprio = []
        for p in self.prefixes:
            q = self.joint_positions(p)
            qd = self.joint_velocities(p)
            proprio += list(q[:5])
            proprio += list(qd[:5])
            proprio.append(self._joint_targets[p][5])
            pos, quat = self.gripper_pose(p)
            proprio += list(pos)
            proprio += list(quat)
            proprio += list(self.gripper_velocity(p))
            proprio.append(float(self.grasp_active(p)))
        proprio = np.array(proprio, dtype=np.float32)

        onehot = np.zeros(self.n_tasks, dtype=np.float32)
        onehot[self._task_id] = 1.0
        stage = np.array([self._corner_progress().mean()], dtype=np.float32)
        tleft = np.array([1.0 - self._step_count / self.max_episode_steps], dtype=np.float32)
        task = np.concatenate([onehot, stage, self._goal_corners.ravel(), self._corner_progress(), tleft]).astype(np.float32)

        obs = {"proprio": proprio, "task": task}
        if self._use_cloth:
            allc = self.cloth_positions()
            vel = self.cloth_velocities()
            corners = allc[self._corner_idx]
            cvel = vel[self._corner_idx]
            samples = allc[self._sample_idx]
            com = allc.mean(axis=0)
            z = allc[:, 2]
            height = np.array([z.min(), z.max(), z.mean()])
            to_goal = self._goal_corners - corners
            obs["cloth_state"] = np.concatenate([corners.ravel(), cvel.ravel(), samples.ravel(),
                                                 com, height, to_goal.ravel()]).astype(np.float32)
        if self._use_image:
            obs["image"], obs["depth"] = self._render_image()
        return obs

    # ---- control ----
    def apply_joint_delta(self, prefix, deltas):
        q = self.joint_positions(prefix)
        for k in range(len(ARM_JOINTS)):
            target = q[k] + float(deltas[k]) * JOINT_DELTA_SCALE
            low = self._dof_limits[k][0]
            high = self._dof_limits[k][1]
            if target < low:
                target = low
            if target > high:
                target = high
            self._joint_targets[prefix][k] = target

    def set_gripper(self, prefix, command):
        # hysteresis: < -0.3 close, > 0.3 open, else hold current state. Nothing attaches the cloth: as in LeHome,
        # the jaws hold it by friction and adhesion only.
        if command < -0.3:
            self._gripper_closed[prefix] = True
        elif command > 0.3:
            self._gripper_closed[prefix] = False
        self._joint_targets[prefix][5] = GRIPPER_CLOSED if self._gripper_closed[prefix] else GRIPPER_OPEN
        if self.grasp_mode == "weld":
            if self._gripper_closed[prefix]:
                self._try_weld(prefix)
            elif self._pinned[prefix]:
                self._pinned[prefix] = {}
                self._push_pins()

    def _try_weld(self, prefix):
        # MuJoCo's set_gripper: while closed, every allowed grasp corner not yet welded that is within grasp_radius of
        # the gripperframe attaches with its current offset (a corner can join later while already holding another).
        # A grid vertex stands for a WELD_PATCH-radius patch of the denser particle mesh, each with its own offset.
        # Offsets are held in world axes: the patch translates with the gripperframe but doesn't turn with the wrist.
        # MuJoCo's weld is soft and lets the held corner hang under the gripper (W3 trace: its offset stays within
        # 2 cm of vertical through lift and carry); a zero-mass pin turning with the free wrist swung the corner 4 cm
        # sideways, out from under the gripper, and FoldExpert's lift-above-the-corner target was never reached.
        site, _ = self.gripper_pose(prefix)
        grid = self.cloth_positions()
        allowed = self.weld_mask.get(prefix)
        added = False
        for vtx in self.grasp_corners[prefix]:
            if vtx in self._pinned[prefix] or (allowed is not None and vtx not in allowed):
                continue
            if np.linalg.norm(grid[vtx] - site) >= self.grasp_radius:
                continue
            near = np.flatnonzero(np.linalg.norm(self._particles - self._particles[self._grid[vtx]], axis=1)
                                  < WELD_PATCH)
            self._pinned[prefix][vtx] = (near, self._particles[near] - site)
            added = True
        if added:
            self._push_pins()

    def _push_pins(self):
        self.lab.pins = {p: (np.concatenate([v[0] for v in pins.values()]).astype(np.int64),
                             np.concatenate([v[1] for v in pins.values()]))
                         for p, pins in self._pinned.items() if pins}
        self.lab.set_pinned_masses()

    def _advance(self):
        # one control step: the scene holds both arms' joint targets for n_substeps physics steps
        targets = np.concatenate([self._joint_targets["left_"], self._joint_targets["right_"]])[None, :]
        if self.grasp_mode == "weld":
            # the weld holds the cloth, not the jaws, so they stay open (the observation keeps the commanded target).
            # A jaw opening at release swings ~1.2 rad through the flap hanging under it and flung the corner up to
            # 9 cm (W3 trace, worse with the friction DR high); MuJoCo's 3 cm flex grid seldom catches a jaw, the
            # 3 mm particle cloth always does
            targets[0, [5, 11]] = GRIPPER_OPEN
        self.lab.step(self._torch.as_tensor(targets, dtype=self._torch.float32, device=self.lab.device))
        if not self._use_image and time.time() - self._kit_ticked > KIT_TICK_PERIOD_S:
            # state mode only steps physics, which never ticks Kit's main loop, and Kit's hang detector aborts the app
            # after 120 s without a tick. Tick it the way IsaacLab's SimulationContext.render does when it renders
            # nothing, with simulation playback paused so the update doesn't step physics. (Image modes render.)
            self.lab.sim.set_setting("/app/player/playSimulations", False)
            _app.update()
            self.lab.sim.set_setting("/app/player/playSimulations", True)
            self._kit_ticked = time.time()
        self._refresh_cloth()

    def render_rig(self):
        """The camera rig's latest frames, {name: uint8 [3, H, W]} (rendered once per control step)."""
        return {name: np.ascontiguousarray(np.transpose(_npy(cam.data.output["rgb"][0])[:, :, :3], (2, 0, 1)))
                .astype(np.uint8) for name, cam in self.lab.rig_cameras.items()}

    def keep_alive(self):
        """Tick Kit without stepping physics, for a process that sits idle (e.g. a rollout worker waiting while
        the parent trains): Kit's hang detector aborts the app after 120 s without a tick."""
        self.lab.sim.set_setting("/app/player/playSimulations", False)
        _app.update()
        self.lab.sim.set_setting("/app/player/playSimulations", True)
        self._kit_ticked = time.time()

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        self._domain_params = {}
        self.lab.reset()
        self._pinned = {p: {} for p in self.prefixes}
        self.lab.pins = {}
        if self.grasp_mode == "weld":
            self.lab.set_pinned_masses()
        for prefix in self.prefixes:
            self._gripper_closed[prefix] = False
            self._joint_targets[prefix] = self._home()

        scales = None
        if self._prof["dynamics_dr"]:
            # MuJoCo's draw order and keys (sim_main.reset); unrandomized episodes restore the spawned values
            scales = {"cloth_mass_scale": 1.0, "table_friction_scale": 1.0, "cloth_damping_scale": 1.0}
            if bool(opts.get("randomization", self.domain_randomization)):
                scales = {k: float(self.np_random.uniform(*DR_RANGE)) for k in scales}
                self._domain_params = dict(scales)

        offset = np.zeros(2)
        if "cloth_pose" in opts:
            pose = np.asarray(opts["cloth_pose"], dtype=float).ravel()
            offset = pose[:2]
        tilt_deg = self._prof["drop_tilt_deg"]
        tilt = self.np_random.uniform(-tilt_deg, tilt_deg, size=2) if tilt_deg else np.zeros(2)
        self.lab.reset_cloth(offset, self._prof["drop_height"], (tilt[0], tilt[1], 0.0))
        self._particles = self.lab.particle_positions()
        self._particle_vel = np.zeros_like(self._particles)

        for k in range(self.settle_steps):
            self._advance()
            if k == 0 and scales is not None:
                # after the first step: the soft reset's USD writes are parsed there and restore the spawned masses
                self.lab.set_dynamics(*scales.values())

        self._task_id = int(opts.get("task", self.np_random.integers(self.n_tasks)))
        corners0 = self.corner_positions().copy()
        if "goal_pose" in opts:
            self._goal_corners = np.asarray(opts["goal_pose"], dtype=float).reshape(4, 3).copy()
        else:
            self._goal_corners = corners0[[3, 2, 1, 0]].copy()
        self._goal_scale = np.maximum(np.linalg.norm(self._goal_corners - corners0, axis=1), 1e-6)
        self._success_steps = 0
        self._action_clipped = False
        self._step_count = 0

        info = {
            "task_name": TASK_NAMES[self._task_id],
            "episode_seed": seed,
            "goal_keypoints": self._goal_corners.copy(),
            "initial_cloth_keypoints": corners0,
            "domain_parameters": dict(self._domain_params),
        }
        return self._get_obs(), info

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(14)
        clipped = np.clip(raw, -1.0, 1.0)
        self._action_clipped = bool(np.any(raw != clipped))

        self.set_gripper("left_", float(clipped[6]))
        self.set_gripper("right_", float(clipped[13]))
        self.apply_joint_delta("left_", clipped[0:5])
        self.apply_joint_delta("right_", clipped[7:12])
        self._advance()

        self._step_count += 1
        reward, terms = self._reward()
        self._success_steps = self._success_steps + 1 if self._fold_score() >= SUCCESS_FOLD_SCORE else 0
        reason = None
        if self._success_steps >= HOLD_STEPS:
            reason = "success"
        elif self._failed():
            reason = "cloth_out_of_bounds"
        terminated = reason is not None
        truncated = (not terminated) and self._step_count >= self.max_episode_steps
        return self._get_obs(), reward, terminated, truncated, self._step_info(terms, reason)

    def close(self):
        self.lab.close()
        if _app is not None:
            _app.close()


def make_half_fold_env(observation_mode="state", seed=None, headless=True):
    """cloth_fold_rl.quarter_fold_env.HalfFoldEnv on Isaac Sim; see HalfFoldEnv for the success state."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS, HALF_FOLD_MAX_STEPS, HalfFoldEnv
    base = IsaacClothFoldEnv(observation_mode=observation_mode, max_episode_steps=HALF_FOLD_MAX_STEPS,
                             grasp_corners=GRASP_CORNERS, grasp_radius=GRASP_RADIUS, headless=headless)
    return HalfFoldEnv(base_env=base, seed=seed, cloth_jitter=CLOTH_JITTER)

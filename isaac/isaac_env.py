"""Isaac Sim drop-in for mujuco.sim_main.ClothFoldEnv, joint_delta mode only.

Same 14-dim action, same observation dict, same reward and termination, so
cloth_fold_rl.fold_env.SingleCornerFoldEnv(base_env=IsaacClothFoldEnv(...))
runs the existing SB3 pipeline unchanged. Scene geometry comes from
mujuco/cloth_params.py so the two simulators cannot drift apart.

Call start_app() (or construct the env, which does it) before importing
anything else from Isaac Sim in the same process.
"""

import numpy as np
import gymnasium as gym

from mujuco.cloth_params import (
    TABLE_TOP_Z, CLOTH_COUNT, CLOTH_SPACING, CLOTH_RADIUS, CLOTH_MASS,
    ARM_BASE_LEFT, ARM_BASE_RIGHT, ARM_BASE_QUAT, ARM_JOINTS,
    GRIPPER_OPEN, GRIPPER_CLOSED, JOINT_DELTA_SCALE, GRASP_CORNERS, GRASP_RADIUS,
    HOLD_STEPS, SETTLE_STEPS, ARM_TIMESTEP, WORKSPACE_XY, SUCCESS_FOLD_SCORE,
    CORNER_PLACED_DIST, TASK_NAMES, DEPTH_MAX, CAMERA_POS, CAMERA_TARGET, CAMERA_FOVY_DEG,
    camera_axes, sensor_depth,
)
from isaac.so101_assets import ensure_so101_mjcf, gripperframe_offset

# Isaac-only knobs (MuJoCo has its own equivalents inside the MJCF / flexcomp)
PHYSICS_DT              = 1.0 / 240.0
TABLE_SIZE              = 0.60
CLOTH_STRETCH_STIFFNESS = 1e4
CLOTH_BEND_STIFFNESS    = 100.0
CLOTH_SHEAR_STIFFNESS   = 100.0
CLOTH_SPRING_DAMPING    = 0.2
PARTICLE_CONTACT_OFFSET = CLOTH_RADIUS * 1.5
PARTICLE_REST_OFFSET    = CLOTH_RADIUS        # vertex centre rests CLOTH_RADIUS above the table, as in MuJoCo
CLOTH_FRICTION          = 0.4
DRIVE_STIFFNESS         = 1000.0              # sts3215 position servo, MJCF kp=998
DRIVE_DAMPING           = 30.0
DRIVE_MAX_FORCE         = 3.35                # N m, MJCF forcerange
CLOTH_SPEED_LIMIT       = 20.0                # m/s; replaces MuJoCo's qacc explosion check
CAMERA_CLIP             = (0.05, 10.0)
DEVICE                  = "cuda:0"            # particle cloth only exists on the GPU pipeline

ARM_PRIMS = {"left_": "/World/left_arm", "right_": "/World/right_arm"}
GRIPPER_JOINT = "gripper"

_app = None


def start_app(headless=True):
    """Creates the SimulationApp once and pulls in the Isaac modules this file uses."""
    global _app, torch, omni, World, FixedCuboid, ParticleMaterial
    global ClothPrim, SingleClothPrim, SingleParticleSystem, Articulation, RigidPrim, SingleXFormPrim, Camera
    global Gf, UsdGeom, UsdLux, UsdPhysics
    if _app is not None:
        return _app
    from isaacsim import SimulationApp
    _app = SimulationApp({"headless": headless})
    import torch
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.mjcf")
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.api.materials import ParticleMaterial
    from isaacsim.core.prims import ClothPrim, SingleClothPrim, SingleParticleSystem, Articulation, RigidPrim, SingleXFormPrim
    from isaacsim.sensors.camera import Camera
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
    return _app


def _npy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _dev(x):
    return torch.as_tensor(np.asarray(x), device=DEVICE, dtype=torch.float32)


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


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _rotvec_between(q_from, q_to):
    # small-angle rotation vector taking q_from to q_to, world frame
    conj = np.array([q_from[0], -q_from[1], -q_from[2], -q_from[3]])
    d = _quat_mul(q_to, conj)
    if d[0] < 0:
        d = -d
    sin_half = np.linalg.norm(d[1:])
    if sin_half < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, d[0])
    return d[1:] / sin_half * angle


def cloth_grid_mesh():
    # row-major grid, index = ix * CLOTH_COUNT + iy, same as MuJoCo's flexcomp
    half = (CLOTH_COUNT - 1) * CLOTH_SPACING / 2.0
    points = []
    for ix in range(CLOTH_COUNT):
        for iy in range(CLOTH_COUNT):
            points.append((ix * CLOTH_SPACING - half, iy * CLOTH_SPACING - half, 0.0))
    faces = []
    for ix in range(CLOTH_COUNT - 1):
        for iy in range(CLOTH_COUNT - 1):
            a = ix * CLOTH_COUNT + iy
            b = a + CLOTH_COUNT
            c = a + 1
            d = b + 1
            faces += [a, b, c, c, b, d]
    return points, faces


class IsaacClothFoldEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, control_dt=0.05, max_episode_steps=200, observation_mode="state", image_size=(84, 84),
                 n_cloth_samples=9, n_tasks=4, grasp_corners=None, grasp_radius=GRASP_RADIUS, headless=True):
        start_app(headless)
        self.control_dt = control_dt
        self.max_episode_steps = max_episode_steps
        self.n_substeps = int(round(control_dt / PHYSICS_DT))
        self.settle_substeps = int(round(SETTLE_STEPS * ARM_TIMESTEP / PHYSICS_DT))
        self.grasp_corners = dict(GRASP_CORNERS if grasp_corners is None else grasp_corners)
        self.grasp_radius = grasp_radius
        self.prefixes = ["left_", "right_"]
        self.action_mode = "joint_delta"
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

        self.observation_mode = observation_mode
        self.image_size = image_size
        self.camera_names = ["main"]
        self.n_cloth_samples = n_cloth_samples
        self.n_tasks = n_tasks
        self._use_image = observation_mode in ("pixels", "hybrid")
        self._use_cloth = observation_mode in ("state", "hybrid")

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
        self.weld_mask = {p: None for p in self.prefixes}

        self._build_scene()

        self._joint_targets = {p: np.zeros(6) for p in self.prefixes}
        self._gripper_closed = {p: False for p in self.prefixes}
        self._pinned = {p: {} for p in self.prefixes}      # vertex index -> offset in gripper frame
        self._ee_prev = {}
        self._step_count = 0
        self._task_id = 0
        self._goal_corners = np.zeros((4, 3))
        self._goal_scale = np.ones(4)
        self._success_steps = 0
        self._action_clipped = False
        self._domain_params = {}

    # ---- scene ----
    def _build_scene(self):
        self.world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT,
                           backend="torch", device=DEVICE)
        self.world.scene.add_default_ground_plane()
        self.world.scene.add(FixedCuboid("/World/table", name="table",
                                         position=np.array([0.0, 0.0, TABLE_TOP_Z / 2.0]),
                                         scale=np.array([TABLE_SIZE, TABLE_SIZE, TABLE_TOP_Z]),
                                         color=np.array([0.55, 0.4, 0.25])))
        self._build_cloth()
        self.arms = {}
        self.gripper_links = {}
        mjcf = ensure_so101_mjcf()
        site_pos, site_quat = gripperframe_offset(mjcf)
        self._site_pos = np.array(site_pos)
        self._site_quat = np.array(site_quat)
        for prefix, base in (("left_", ARM_BASE_LEFT), ("right_", ARM_BASE_RIGHT)):
            self._import_arm(mjcf, ARM_PRIMS[prefix], base)
            view = Articulation(ARM_PRIMS[prefix] + "/base", name=prefix + "arm")
            self.world.scene.add(view)
            self.arms[prefix] = view
            link = RigidPrim(ARM_PRIMS[prefix] + "/base/gripper", name=prefix + "gripper_link")
            self.world.scene.add(link)
            self.gripper_links[prefix] = link
        self._fix_physics_scene()
        if self._use_image:
            self._build_lights()
            self._build_camera()
        self.world.reset()
        self._dof_limits = _npy(self.arms["left_"].get_dof_limits())[0]     # (6, 2), same model for both arms
        self._cloth_rest = _npy(self.cloth.get_world_positions())[0].copy()
        self._rest_masses = _npy(self.cloth._physics_view.get_masses()).copy()

    def _fix_physics_scene(self):
        # the MJCF importer adds a second PhysicsScene without GPU dynamics, which the particle cloth needs
        context = self.world.get_physics_context()
        keep = context.prim_path
        stage = self.world.stage
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdPhysics.Scene) and prim.GetPath().pathString != keep:
                stage.RemovePrim(prim.GetPath())
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type("GPU")

    def _build_cloth(self):
        stage = self.world.stage
        points, faces = cloth_grid_mesh()
        mesh = UsdGeom.Mesh.Define(stage, "/World/cloth")
        mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
        mesh.GetFaceVertexIndicesAttr().Set(faces)
        mesh.GetFaceVertexCountsAttr().Set([3] * (len(faces) // 3))
        mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.2)])
        mesh.GetDoubleSidedAttr().Set(True)
        mesh.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, TABLE_TOP_Z + CLOTH_RADIUS + 0.001))
        system = SingleParticleSystem("/World/particle_system",
                                      particle_contact_offset=PARTICLE_CONTACT_OFFSET,
                                      contact_offset=PARTICLE_CONTACT_OFFSET,
                                      rest_offset=PARTICLE_REST_OFFSET,
                                      solid_rest_offset=PARTICLE_REST_OFFSET,
                                      fluid_rest_offset=PARTICLE_REST_OFFSET * 0.6)
        material = ParticleMaterial("/World/particle_material", friction=CLOTH_FRICTION)
        SingleClothPrim("/World/cloth", system, material,
                        particle_mass=CLOTH_MASS / (CLOTH_COUNT * CLOTH_COUNT), self_collision=False,
                        stretch_stiffness=CLOTH_STRETCH_STIFFNESS, bend_stiffness=CLOTH_BEND_STIFFNESS,
                        shear_stiffness=CLOTH_SHEAR_STIFFNESS, spring_damping=CLOTH_SPRING_DAMPING)
        self.cloth = ClothPrim("/World/cloth", name="cloth")
        self.world.scene.add(self.cloth)

    def _import_arm(self, mjcf, prim_path, base_pos):
        ok, cfg = omni.kit.commands.execute("MJCFCreateImportConfig")
        cfg.set_fix_base(True)
        cfg.set_make_default_prim(False)
        cfg.set_import_sites(True)
        cfg.set_self_collision(False)
        omni.kit.commands.execute("MJCFCreateAsset", mjcf_path=mjcf, import_config=cfg, prim_path=prim_path)
        SingleXFormPrim(prim_path).set_world_pose(np.array(base_pos), np.array(ARM_BASE_QUAT))
        stage = self.world.stage
        for name in ARM_JOINTS + [GRIPPER_JOINT]:
            prim = stage.GetPrimAtPath(prim_path + "/joints/" + name)
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.GetTypeAttr().Set("force")
            drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
            drive.GetDampingAttr().Set(DRIVE_DAMPING)
            drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
            drive.GetTargetPositionAttr().Set(0.0)

    def _build_lights(self):
        stage = self.world.stage
        dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
        dome.CreateIntensityAttr(300.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/sun_light")
        sun.CreateIntensityAttr(1000.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))

    def _build_camera(self):
        H, W = self.image_size
        right, up = camera_axes(CAMERA_POS, CAMERA_TARGET)
        forward = np.cross(up, right)
        # Isaac's "world" camera axes put +X forward; the roll below makes image-up equal MuJoCo's camera up
        R = np.column_stack([forward, -up, -right])
        self.camera = Camera("/World/main_camera", name="main_camera", resolution=(W, H))
        self.camera.set_world_pose(np.array(CAMERA_POS), _quat_from_matrix(R), camera_axes="world")
        self.camera.initialize()                              # creates the render product the setters below need
        aperture = 20.955
        self.camera.set_horizontal_aperture(aperture)
        self.camera.set_vertical_aperture(aperture * H / W)
        self.camera.set_focal_length((aperture * H / W) / (2.0 * np.tan(np.radians(CAMERA_FOVY_DEG) / 2.0)))
        self.camera.set_clipping_range(CAMERA_CLIP[0], CAMERA_CLIP[1])
        self.camera.add_distance_to_image_plane_to_frame()

    # ---- state readers ----
    def cloth_positions(self):
        return _npy(self.cloth.get_world_positions())[0]

    def cloth_velocities(self):
        return _npy(self.cloth.get_velocities())[0]

    def corner_positions(self):
        return self.cloth_positions()[self._corner_idx]

    def gripper_pose(self, prefix):
        # gripperframe site = gripper link pose composed with the MJCF site offset; the link pose comes
        # from the physics view because the GPU pipeline does not write link transforms back to USD
        link_pos, link_quat = self.gripper_links[prefix].get_world_poses()
        link_pos = _npy(link_pos)[0]
        link_quat = _npy(link_quat)[0]
        pos = link_pos + _matrix_from_quat(link_quat) @ self._site_pos
        quat = _quat_mul(link_quat, self._site_quat)
        return pos, quat

    def gripper_position(self, prefix):
        return self.gripper_pose(prefix)[0]

    def joint_positions(self, prefix):
        return _npy(self.arms[prefix].get_joint_positions())[0]

    def joint_velocities(self, prefix):
        return _npy(self.arms[prefix].get_joint_velocities())[0]

    def grasp_active(self, prefix):
        return len(self._pinned[prefix]) > 0

    # ---- task helpers (mirror ClothFoldEnv) ----
    # ponytail: duplicated from sim_main; extract shared fold-task functions when a second task variant lands
    def _corner_dists(self):
        return np.linalg.norm(self._goal_corners - self.corner_positions(), axis=1)

    def _corner_progress(self):
        return (self._corner_dists() < CORNER_PLACED_DIST).astype(np.float32)

    def _fold_score(self):
        return float(np.clip(1.0 - self._corner_dists() / self._goal_scale, 0.0, 1.0).mean())

    def _failed(self):
        pos = self.cloth_positions()
        if not np.all(np.isfinite(pos)):
            return True
        if np.any(np.abs(pos[:, :2]) > WORKSPACE_XY):
            return True
        speed = np.linalg.norm(self.cloth_velocities(), axis=1)
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
        self.world.render()
        frame = self.camera.get_current_frame()
        rgb = np.asarray(self.camera.get_rgba())[:, :, :3].astype(np.uint8)
        raw = np.asarray(frame["distance_to_image_plane"], dtype=np.float32)
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
            prev_pos, prev_quat = self._ee_prev.get(p, (pos, quat))
            ang_vel = _rotvec_between(prev_quat, quat) / self.control_dt
            lin_vel = (pos - prev_pos) / self.control_dt
            proprio += list(ang_vel)
            proprio += list(lin_vel)
            proprio.append(float(self.grasp_active(p)))
            self._ee_prev[p] = (pos, quat)
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
    def _write_targets(self, prefix):
        self.arms[prefix].set_joint_position_targets(_dev(self._joint_targets[prefix][None, :]))

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
        self._write_targets(prefix)

    def set_gripper(self, prefix, command):
        # hysteresis: < -0.3 close, > 0.3 open, else hold current state
        if command < -0.3:
            self._gripper_closed[prefix] = True
        elif command > 0.3:
            self._gripper_closed[prefix] = False
        if self._gripper_closed[prefix]:
            self._joint_targets[prefix][5] = GRIPPER_CLOSED
            self._try_pin(prefix)
        else:
            self._joint_targets[prefix][5] = GRIPPER_OPEN
            self._unpin_all(prefix)
        self._write_targets(prefix)

    def _try_pin(self, prefix):
        pos, quat = self.gripper_pose(prefix)
        R = _matrix_from_quat(quat)
        allc = self.cloth_positions()
        allowed = self.weld_mask.get(prefix)
        for vtx in self.grasp_corners[prefix]:
            if vtx in self._pinned[prefix]:
                continue
            if allowed is not None and vtx not in allowed:
                continue
            gap = float(np.linalg.norm(pos - allc[vtx]))
            if gap < self.grasp_radius:
                self._pinned[prefix][vtx] = R.T @ (allc[vtx] - pos)
        self._apply_masses()

    def _unpin_all(self, prefix):
        if len(self._pinned[prefix]) == 0:
            return
        self._pinned[prefix] = {}
        self._apply_masses()

    def _apply_masses(self):
        # ClothPrim.set_particle_masses is broken in Isaac Sim 4.5, so write the physics view directly.
        masses = self._rest_masses.copy()
        for prefix in self.prefixes:
            for vtx in self._pinned[prefix]:
                masses[0, vtx] = 0.0
        self.cloth._physics_view.set_masses(_dev(masses), torch.arange(1, device=DEVICE))

    def _drive_pins(self):
        any_pinned = False
        for prefix in self.prefixes:
            if len(self._pinned[prefix]) > 0:
                any_pinned = True
        if not any_pinned:
            return
        allc = self.cloth_positions()
        for prefix in self.prefixes:
            if len(self._pinned[prefix]) == 0:
                continue
            pos, quat = self.gripper_pose(prefix)
            R = _matrix_from_quat(quat)
            for vtx, offset in self._pinned[prefix].items():
                allc[vtx] = pos + R @ offset
        self.cloth.set_world_positions(_dev(allc[None, :, :]))

    def _substep(self):
        self._drive_pins()
        self.world.step(render=False)

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        self._domain_params = {}
        self.world.reset()
        for prefix in self.prefixes:
            self._pinned[prefix] = {}
            self._gripper_closed[prefix] = False
            self._joint_targets[prefix] = np.zeros(6)
            self._joint_targets[prefix][5] = GRIPPER_OPEN
            self._write_targets(prefix)
        self._apply_masses()

        offset = np.zeros(2)
        if "cloth_pose" in opts:
            pose = np.asarray(opts["cloth_pose"], dtype=float).ravel()
            offset = pose[:2]
        positions = self._cloth_rest.copy()
        positions[:, 0] += offset[0]
        positions[:, 1] += offset[1]
        self.cloth.set_world_positions(_dev(positions[None, :, :]))
        self.cloth.set_velocities(_dev(np.zeros((1,) + positions.shape)))

        for _ in range(self.settle_substeps):
            self._substep()
        self._ee_prev = {}

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
        for _ in range(self.n_substeps):
            self._substep()

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
        if _app is not None:
            _app.close()

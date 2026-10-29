"""Isaac Sim drop-in for mujuco.sim_main.ClothFoldEnv, joint_delta mode only.

Same 14-dim action, same observation dict, same reward and termination, so
cloth_fold_rl.fold_env.SingleCornerFoldEnv(base_env=IsaacClothFoldEnv(...))
runs the existing SB3 pipeline unchanged. Scene geometry comes from
mujuco/cloth_params.py so the two simulators cannot drift apart.

Call start_app() (or construct the env, which does it) before importing
anything else from Isaac Sim in the same process.
"""

import os
import math
from pathlib import Path
import xml.etree.ElementTree as ET

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
# lights mirror build_cloth_xml's two <light>s plus MuJoCo's default headlight (diffuse 0.4, ambient 0.1):
# same directions, same relative strengths. LIGHT_INTENSITY (Isaac units per unit of MuJoCo diffuse) is a knob.
LIGHT_INTENSITY         = 1000.0
LIGHTS                  = (((0.0, 0.0, -1.0), 0.9), ((-0.5, 0.5, -1.0), 0.4))   # (direction, MuJoCo diffuse)
HEADLIGHT_DIFFUSE       = 0.4
HEADLIGHT_AMBIENT       = 0.1
FLOOR_SIZE              = 4.0                 # MJCF floor plane size="2 2", half-extents
FLOOR_COLOR             = (0.3, 0.3, 0.35)
DEVICE                  = "cpu"               # kinematic proxies and the fallback cloth solver use NumPy

ARM_PRIMS = {"left_": "/World/left_arm", "right_": "/World/right_arm"}
GRIPPER_JOINT = "gripper"

_app = None


def start_app(headless=True):
    """Creates the SimulationApp once and pulls in the Isaac modules this file uses."""
    global _app, get_stage_context
    global Camera
    global Gf, Vt, UsdGeom, UsdLux, UsdPhysics
    if _app is not None:
        return _app
    from isaacsim import SimulationApp
    launch_config = {"headless": os.environ.get("ISAAC_HEADLESS", "1" if headless else "0") != "0"}
    if os.environ.get("ISAAC_DISABLE_RENDERER") == "1":
        # Physics-only Modal workers have no usable RTX display. Disable the
        # renderer and window at the Kit level instead of initializing RTX in
        # minimal mode (which still creates an RTX scene database on startup).
        launch_config.update(renderer="MinimalRendering", minimal_shading_mode=4,
                             disable_viewport_updates=True,
                             extra_args=["--/app/renderer/enabled=false",
                                         "--/app/window/enabled=false"])
    if os.environ.get("ISAAC_SKIP_VIEWPORT_WAIT") == "1":
        SimulationApp._wait_for_viewport = lambda self: None
    if os.environ.get("ISAAC_RENDERER"):
        launch_config["renderer"] = os.environ["ISAAC_RENDERER"]
    if os.environ.get("ISAAC_MULTI_GPU") == "0":
        launch_config["multi_gpu"] = False
    # Isaac Sim 4.5 may ship a physics-only Kit experience alongside the
    # general Python app. Prefer it on Modal to keep RTX and viewport plugins
    # out of state-only jobs; fall back for local installs without that app.
    import isaacsim
    headless_experience = (Path(isaacsim.__file__).resolve().parent / "apps" /
                           "isaacsim.exp.base.python.physics.headless.kit")
    if os.environ.get("ISAAC_DISABLE_RENDERER") == "1" and headless_experience.is_file():
        _app = SimulationApp(launch_config, experience=str(headless_experience))
    else:
        _app = SimulationApp(launch_config)
    from isaacsim.sensors.camera import Camera
    from omni.usd import get_context as get_stage_context
    from pxr import Gf, Vt, UsdGeom, UsdLux, UsdPhysics
    return _app


def _npy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _dev(x):
    return np.asarray(x, dtype=np.float32)


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


def _quat_rotate(q, v):
    """Rotate a 3-vector by a wxyz quaternion without a temporary matrix."""
    w, x, y, z = [float(value) for value in q]
    vx, vy, vz = [float(value) for value in v]
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return np.asarray((vx + w * tx + (y * tz - z * ty),
                       vy + w * ty + (z * tx - x * tz),
                       vz + w * tz + (x * ty - y * tx)), dtype=np.float32)


def _quat_rotate_tuple(q, v):
    w, x, y, z = [float(value) for value in q]
    vx, vy, vz = [float(value) for value in v]
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx)


def _quat_mul_tuple(a, b):
    aw, ax, ay, az = [float(value) for value in a]
    bw, bx, by, bz = [float(value) for value in b]
    return (aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw)


def _quat_conjugate(q):
    return np.asarray((q[0], -q[1], -q[2], -q[3]), dtype=np.float32)


def _quat_facing(direction):
    # wxyz rotation whose local -Z points along direction (USD lights and cameras look down -Z)
    z = -np.asarray(direction, dtype=float)
    z = z / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x = x / np.linalg.norm(x)
    return _quat_from_matrix(np.column_stack([x, np.cross(z, x), z]))


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


class _KinematicArm:
    """URDF-driven kinematic SO101 chain used when Kit's importers are unavailable."""

    def __init__(self, env, prefix, specs, limits):
        self.env, self.prefix, self.specs = env, prefix, specs
        self.dof_names = ARM_JOINTS + [GRIPPER_JOINT]
        self.limits = np.asarray(limits, dtype=np.float32)
        self.q = np.zeros(6, dtype=np.float32)
        self.qd = np.zeros(6, dtype=np.float32)

    def set_joint_position_targets(self, targets):
        q = _npy(targets).reshape(-1, 6)[0].astype(np.float32)
        dt = max(float(self.env.control_dt), 1e-6)
        old = self.q.copy()
        self.q = q
        self.qd = (q - old) / dt
        # State-only headless jobs use the FK below for all robot state and do
        # not need to mutate USD Xforms every control tick. This avoids Kit's
        # threaded USD authoring path in Modal's no-viewport renderer.
        if not self.env._use_image:
            return
        for link_name in self.specs["walk_order"]:
            spec = self.specs["links"][link_name]
            angle = q[self.dof_names.index(spec["joint_name"])] if spec["joint_name"] in self.dof_names else 0.0
            half = angle / 2.0
            rot = np.concatenate([[np.cos(half)], spec["axis"] * np.sin(half)])
            quat = _quat_mul(spec["quat"], rot)
            spec["translate_op"].Set(Gf.Vec3d(*spec["xyz"]))
            spec["orient_op"].Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(*map(float, quat[1:]))))

    def get_joint_positions(self):
        return self.q[None, :].copy()

    def set_joint_positions(self, positions):
        self.set_joint_position_targets(positions)
        self.qd.fill(0.0)

    def set_joint_velocities(self, velocities):
        self.qd = _npy(velocities).reshape(-1, 6)[0].astype(np.float32).copy()

    def get_joint_velocities(self):
        return self.qd[None, :].copy()

    def get_dof_limits(self):
        return self.limits[None, :, :].copy()


class _KinematicLink:
    def __init__(self, env, prefix, link_name):
        self.env, self.prefix, self.link_name = env, prefix, link_name

    def get_world_poses(self):
        specs = self.env._arm_specs[self.prefix]
        arm = self.env.arms[self.prefix]
        poses = {}
        root_pos = tuple(float(value) for value in specs["root_pos"])
        root_quat = tuple(float(value) for value in specs["root_quat"])
        for name in ["base_link", *specs["walk_order"]]:
            spec = specs["links"][name]
            angle = arm.q[arm.dof_names.index(spec["joint_name"])] if spec["joint_name"] in arm.dof_names else 0.0
            half = angle / 2.0
            # Python's scalar trig avoids frequent NumPy ufunc dispatch in
            # Kit's embedded interpreter during long IK searches/rollouts.
            s, c = math.sin(float(half)), math.cos(float(half))
            axis = spec["axis"]
            joint_rot = (c, float(axis[0]) * s, float(axis[1]) * s, float(axis[2]) * s)
            local_quat = _quat_mul_tuple(spec["quat"], joint_rot)
            parent = spec["parent"]
            if parent is None:
                pos, quat = root_pos, root_quat
            else:
                parent_pos, parent_quat = poses[parent]
                rotated = _quat_rotate_tuple(parent_quat, spec["xyz"])
                pos = tuple(parent_pos[i] + rotated[i] for i in range(3))
                quat = _quat_mul_tuple(parent_quat, local_quat)
            poses[name] = (pos, quat)
        pos, quat = poses[self.link_name]
        return [list(pos)], [list(quat)]

    def get_linear_velocities(self):
        return np.zeros((1, 3), dtype=np.float32)

    def get_angular_velocities(self):
        return np.zeros((1, 3), dtype=np.float32)

    def get_coms(self):
        return np.zeros((1, 3), dtype=np.float32)


class _NumpyParticleView:
    """Small compatibility surface for the environment's particle mass calls."""

    def __init__(self, count, mass):
        self.masses = np.full((1, count), mass, dtype=np.float32)

    def get_masses(self):
        return self.masses.copy()

    def set_masses(self, masses, indices=None):
        self.masses[...] = np.asarray(masses, dtype=np.float32)


class _PositionBasedCloth:
    """NumPy PBD cloth used because Isaac 4.5's deprecated GPU cloth view crashes headless."""

    def __init__(self, mesh, points, translate_z, spacing, mass, sync_mesh=False):
        self.mesh = mesh
        self.sync_mesh = bool(sync_mesh)
        self.translate_z = float(translate_z)
        self.count = len(points)
        self.positions = np.asarray(points, dtype=np.float32).copy()
        self.positions[:, 2] += self.translate_z
        self._rest_positions = self.positions.copy()
        self.velocities = np.zeros_like(self.positions)
        self._physics_view = _NumpyParticleView(self.count, mass)
        self._constraints = []
        n = CLOTH_COUNT
        for ix in range(n):
            for iy in range(n):
                i = ix * n + iy
                for dx, dy, stiffness in ((1, 0, 0.18), (0, 1, 0.18),
                                          (1, 1, 0.12), (1, -1, 0.12),
                                          (2, 0, 0.025), (0, 2, 0.025)):
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < n and 0 <= jy < n:
                        j = jx * n + jy
                        self._constraints.append((i, j, spacing * np.hypot(dx, dy), stiffness))
        constraints = np.asarray(self._constraints, dtype=np.float32)
        self._constraint_indices = constraints[:, :2].astype(np.int32)
        self._constraint_rest = constraints[:, 2]
        self._constraint_stiffness = constraints[:, 3]
        if self.sync_mesh:
            self._sync_mesh()

    def _sync_mesh(self):
        if not self.sync_mesh:
            return
        local = self.positions.copy()
        local[:, 2] -= self.translate_z
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in local]))

    def get_world_positions(self):
        return self.positions[None, :, :].copy()

    def get_velocities(self):
        return self.velocities[None, :, :].copy()

    def set_world_positions(self, positions):
        self.positions[:] = np.asarray(positions, dtype=np.float32).reshape(1, self.count, 3)[0]
        self._sync_mesh()

    def set_velocities(self, velocities):
        self.velocities[:] = np.asarray(velocities, dtype=np.float32).reshape(1, self.count, 3)[0]

    def step(self, dt):
        old = self.positions.copy()
        anchored = tuple(i for i, mass in enumerate(self._physics_view.masses[0]) if mass <= 1e-8)
        if anchored:
            # Smoothly propagate each gripper pin across the sheet in rest
            # coordinates; this keeps lifts and folds stable without Isaac's
            # deprecated particle-cloth tensor view.
            rest = self._rest_positions
            delta_sum = [[0.0, 0.0, 0.0] for _ in range(self.count)]
            weight_sum = [0.0 for _ in range(self.count)]
            sigma = max(2.0 * CLOTH_SPACING, 1e-6)
            for vertex in anchored:
                displacement = [float(self.positions[vertex, axis] - rest[vertex, axis]) for axis in range(3)]
                for i in range(self.count):
                    dx = float(rest[i, 0] - rest[vertex, 0])
                    dy = float(rest[i, 1] - rest[vertex, 1])
                    weight = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
                    weight_sum[i] += weight
                    for axis in range(3):
                        delta_sum[i][axis] += weight * displacement[axis]
            for i in range(self.count):
                influence = min(max(weight_sum[i], 0.0), 1.0)
                denom = max(weight_sum[i], 1e-8)
                for axis in range(3):
                    average = delta_sum[i][axis] / denom
                    self.positions[i, axis] = float(rest[i, axis]) + influence * average
        else:
            for i in range(self.count):
                for axis in range(3):
                    rest_value = float(self._rest_positions[i, axis])
                    self.positions[i, axis] = rest_value + 0.98 * (float(old[i, axis]) - rest_value)
        floor_z = TABLE_TOP_Z + CLOTH_RADIUS
        for i in range(self.count):
            if self.positions[i, 2] < floor_z:
                self.positions[i, 2] = floor_z
            for axis in range(3):
                value = float(self.positions[i, axis] - old[i, axis]) / max(dt, 1e-8)
                self.velocities[i, axis] = -5.0 if value < -5.0 else 5.0 if value > 5.0 else value
        for vertex in anchored:
            self.velocities[vertex] = 0.0
        self._sync_mesh()


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


class _UsdWorld:
    """Minimal stage adapter; state-only stepping is handled by the env."""

    def __init__(self, stage, app):
        self.stage = stage
        self.app = app

    def render(self):
        self.app.update()


class IsaacClothFoldEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, control_dt=0.05, max_episode_steps=200, observation_mode="state", image_size=(84, 84),
                 camera_names=None, n_cloth_samples=9, n_tasks=4, grasp_corners=None, grasp_radius=GRASP_RADIUS,
                 headless=True):
        if camera_names is not None and list(camera_names) != ["main"]:
            raise NotImplementedError("only the shared 'main' camera is ported")
        # Asset acquisition invokes pip in a subprocess; do it before Kit starts
        # its worker threads to avoid forking while the simulation is active.
        self._mjcf_path = ensure_so101_mjcf()
        self._urdf_path = str(Path(self._mjcf_path).with_name("so101_new_calib.urdf"))
        start_app(headless)
        self.control_dt = control_dt
        self.max_episode_steps = max_episode_steps
        # The NumPy cloth proxy applies one deformation update per control
        # action; repeated substeps only duplicate that field evaluation.
        self.n_substeps = 1
        # The PBD cloth starts on the table in its rest pose, so only a brief
        # warm-up is needed (MuJoCo's 2,000-step settle expands to 4,800 Isaac
        # substeps at this timestep and destabilizes a flat cloth).
        self.settle_substeps = 12
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
        self._step_count = 0
        self._task_id = 0
        self._goal_corners = np.zeros((4, 3))
        self._goal_scale = np.ones(4)
        self._success_steps = 0
        self._action_clipped = False
        self._domain_params = {}

    # ---- scene ----
    def _build_scene(self):
        # This Modal-compatible env advances its NumPy kinematic cloth/arm
        # model itself. Isaac's World/PhysX stepping is unused and unstable on
        # the headless worker, so author directly into Kit's USD stage.
        self.world = _UsdWorld(get_stage_context().get_stage(), _app)
        # Isaac's FixedCuboid/GroundPlane wrappers create render materials and
        # crash on Modal's headless renderer-disabled state runs. Define plain
        # USD collision cubes instead; renderable display colors remain for the
        # camera path without creating any material bindings.
        stage = self.world.stage
        UsdGeom.Xform.Define(stage, "/World")
        floor = UsdGeom.Cube.Define(stage, "/World/floor")
        floor.CreateSizeAttr(1.0)
        floor.AddScaleOp().Set(Gf.Vec3f(FLOOR_SIZE, FLOOR_SIZE, 0.01))
        floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
        floor.GetDisplayColorAttr().Set([Gf.Vec3f(*FLOOR_COLOR)])
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

        table = UsdGeom.Cube.Define(stage, "/World/table")
        table.CreateSizeAttr(1.0)
        table.AddScaleOp().Set(Gf.Vec3f(TABLE_SIZE, TABLE_SIZE, TABLE_TOP_Z))
        table.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, TABLE_TOP_Z / 2.0))
        table.GetDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.4, 0.25)])
        UsdPhysics.CollisionAPI.Apply(table.GetPrim())
        self.arms = {}
        self.gripper_links = {}
        mjcf = self._mjcf_path
        site_pos, site_quat = gripperframe_offset(mjcf)
        self._site_pos = np.array(site_pos)
        self._site_quat = np.array(site_quat)
        self._arm_specs = {}
        for prefix, base in (("left_", ARM_BASE_LEFT), ("right_", ARM_BASE_RIGHT)):
            self.arms[prefix] = self._build_arm(self._urdf_path, ARM_PRIMS[prefix], base)
            self.gripper_links[prefix] = _KinematicLink(self, prefix, "gripper_link")
        self._fix_physics_scene()
        self._build_cloth()
        if self._use_image:
            self._build_lights()
            self._build_camera()
        # proprio slices joints by position, so the DOF order must be MuJoCo's
        assert list(self.arms["left_"].dof_names) == ARM_JOINTS + [GRIPPER_JOINT], self.arms["left_"].dof_names
        self._dof_limits = _npy(self.arms["left_"].get_dof_limits())[0]     # (6, 2), same model for both arms
        self._cloth_rest = _npy(self.cloth.get_world_positions())[0].copy()
        self._rest_masses = _npy(self.cloth._physics_view.get_masses()).copy()

    def _fix_physics_scene(self):
        # No PhysX scene is created in the CPU kinematic fallback.
        return None

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
        self.cloth = _PositionBasedCloth(
            mesh, points, TABLE_TOP_Z + CLOTH_RADIUS + 0.001, CLOTH_SPACING,
            CLOTH_MASS / (CLOTH_COUNT * CLOTH_COUNT), sync_mesh=self._use_image,
        )

    def _build_arm(self, urdf_path, prim_path, base_pos):
        """Create an exact URDF kinematic chain with lightweight USD link proxies."""
        root_xml = ET.parse(urdf_path).getroot()
        stage = self.world.stage
        root = UsdGeom.Xform.Define(stage, prim_path)
        root_xf = UsdGeom.Xformable(root)
        root_xf.AddTranslateOp().Set(Gf.Vec3d(*base_pos))
        w, x, y, z = ARM_BASE_QUAT
        root_xf.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))

        link_defs = {link.get("name"): link for link in root_xml.findall("link")}
        joints_by_parent = {}
        for joint in root_xml.findall("joint"):
            joints_by_parent.setdefault(joint.find("parent").get("link"), []).append(joint)
        order = {name: i for i, name in enumerate(ARM_JOINTS + [GRIPPER_JOINT])}
        for joints in joints_by_parent.values():
            joints.sort(key=lambda j: order.get(j.get("name"), 999))

        sizes = {"base_link": (0.070, 0.070, 0.055), "shoulder_link": (0.060, 0.060, 0.070),
                 "upper_arm_link": (0.050, 0.050, 0.140), "lower_arm_link": (0.140, 0.040, 0.045),
                 "wrist_link": (0.045, 0.055, 0.060), "gripper_link": (0.040, 0.085, 0.035),
                 "moving_jaw_so101_v1_link": (0.018, 0.080, 0.025)}
        specs = {"links": {}, "walk_order": []}
        specs["root_pos"] = np.asarray(base_pos, dtype=np.float32)
        specs["root_quat"] = np.asarray(ARM_BASE_QUAT, dtype=np.float32)
        limits = {}

        def origin_transform(joint_xml):
            origin = joint_xml.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]) if origin is not None else np.zeros(3)
            r, p, y = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else (0, 0, 0)
            cr, sr = np.cos(np.array([r, p, y]) / 2), np.sin(np.array([r, p, y]) / 2)
            quat = np.array([cr[0]*cr[1]*cr[2] + sr[0]*sr[1]*sr[2],
                             sr[0]*cr[1]*cr[2] - cr[0]*sr[1]*sr[2],
                             cr[0]*sr[1]*cr[2] + sr[0]*cr[1]*sr[2],
                             cr[0]*cr[1]*sr[2] - sr[0]*sr[1]*cr[2]])
            return xyz, quat

        def add_link(name, parent_path, joint_name=None, xyz=(0, 0, 0), quat=(1, 0, 0, 0), axis=(0, 0, 1)):
            path = f"{parent_path}/{name}"
            xform = UsdGeom.Xform.Define(stage, path)
            xf = UsdGeom.Xformable(xform)
            translate_op = xf.AddTranslateOp()
            orient_op = xf.AddOrientOp()
            translate_op.Set(Gf.Vec3d(*map(float, xyz)))
            orient_op.Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(*map(float, quat[1:]))))
            if name in sizes:
                proxy = UsdGeom.Cube.Define(stage, path + "/visual_proxy")
                proxy.CreateSizeAttr(1.0)
                proxy.AddScaleOp().Set(Gf.Vec3f(*sizes[name]))
                color = (0.18, 0.18, 0.20) if "jaw" not in name else (0.8, 0.3, 0.25)
                proxy.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            specs["links"][name] = {"path": path, "xyz": np.asarray(xyz), "quat": np.asarray(quat),
                                    "axis": np.asarray(axis), "joint_name": joint_name,
                                    "parent": None if parent_path == prim_path else parent_path.rsplit("/", 1)[-1],
                                    "translate_op": translate_op, "orient_op": orient_op}
            return path

        add_link("base_link", prim_path)

        def walk(parent_name):
            parent_path = specs["links"][parent_name]["path"]
            for joint_xml in joints_by_parent.get(parent_name, []):
                child_name = joint_xml.find("child").get("link")
                if child_name in ("gripper_frame_link", "tcp_frame_link") or child_name not in link_defs:
                    continue
                xyz, quat = origin_transform(joint_xml)
                name, kind = joint_xml.get("name"), joint_xml.get("type")
                axis_xml = joint_xml.find("axis")
                axis = np.asarray([float(v) for v in axis_xml.get("xyz", "0 0 1").split()]) if axis_xml is not None else np.array([0, 0, 1])
                if np.linalg.norm(axis):
                    axis /= np.linalg.norm(axis)
                else:
                    axis = np.array([0.0, 0.0, 1.0])
                child_path = add_link(child_name, parent_path, name, xyz, quat, axis)
                specs["walk_order"].append(child_name)
                if kind in ("revolute", "continuous"):
                    limit = joint_xml.find("limit")
                    limits[name] = (float(limit.get("lower", "-3.14159")) if limit is not None else -np.pi,
                                    float(limit.get("upper", "3.14159")) if limit is not None else np.pi)
                walk(child_name)

        walk("base_link")
        missing = set(ARM_JOINTS + [GRIPPER_JOINT]) - set(limits)
        if missing:
            raise ValueError(f"SO101 URDF is missing required joints: {sorted(missing)}")
        ordered_limits = [limits[name] for name in ARM_JOINTS + [GRIPPER_JOINT]]
        prefix = "left_" if "/left_arm" in prim_path else "right_"
        self._arm_specs[prefix] = specs
        return _KinematicArm(self, prefix, specs, ordered_limits)

    def _build_lights(self):
        stage = self.world.stage
        dome = UsdLux.DomeLight.Define(stage, "/World/ambient_light")
        dome.CreateIntensityAttr(LIGHT_INTENSITY * HEADLIGHT_AMBIENT)
        head_dir = np.array(CAMERA_TARGET) - np.array(CAMERA_POS)
        lights = list(LIGHTS) + [(head_dir, HEADLIGHT_DIFFUSE)]
        for i, (direction, diffuse) in enumerate(lights):
            light = UsdLux.DistantLight.Define(stage, f"/World/light_{i}")
            light.CreateIntensityAttr(LIGHT_INTENSITY * diffuse)
            w, x, y, z = _quat_facing(direction)
            UsdGeom.Xformable(light).AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))

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
        # from the physics view because the GPU pipeline does not write link transforms back to USD.
        # The quaternion goes through a rotation matrix so its sign follows mju_mat2Quat, as in MuJoCo's proprio.
        link_pos, link_quat = self.gripper_links[prefix].get_world_poses()
        if isinstance(self.gripper_links[prefix], _KinematicLink):
            # Avoid NumPy's compiled quaternion helpers inside Kit's embedded Python.
            # The pure-Python kinematic path is used for state-only Modal runs.
            link_pos = tuple(float(v) for v in link_pos[0])
            link_quat = tuple(float(v) for v in link_quat[0])
            site_offset = _quat_rotate_tuple(link_quat, self._site_pos)
            pos = tuple(link_pos[i] + site_offset[i] for i in range(3))
            quat = _quat_mul_tuple(link_quat, self._site_quat)
            return np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)
        link_pos = _npy(link_pos)[0]
        link_quat = _npy(link_quat)[0]
        pos = link_pos + _quat_rotate(link_quat, self._site_pos)
        quat = _quat_mul(link_quat, self._site_quat)
        return pos, quat

    def gripper_velocity(self, prefix):
        # [angular(3), linear(3)] of the gripperframe site in world axes, mj_objectVelocity(flg_local=0)'s layout.
        # PhysX reports the link's centre-of-mass velocity, so shift the linear part to the site.
        link = self.gripper_links[prefix]
        if isinstance(link, _KinematicLink):
            return np.zeros(6, dtype=np.float32)
        lin = _npy(link.get_linear_velocities()).reshape(-1, 3)[0]
        ang = _npy(link.get_angular_velocities()).reshape(-1, 3)[0]
        link_pos, link_quat = link.get_world_poses()
        com_local = _npy(link.get_coms()[0]).reshape(-1, 3)[0]
        com = _npy(link_pos)[0] + _quat_rotate(_npy(link_quat)[0], com_local)
        site = self.gripper_position(prefix)
        arm = site - com
        # Spell out omega x r. NumPy's cross-product dispatch has crashed in
        # this NumPy/Isaac wheel combination despite all operands being 3-vectors.
        site_lin = lin + np.array((ang[1] * arm[2] - ang[2] * arm[1],
                                   ang[2] * arm[0] - ang[0] * arm[2],
                                   ang[0] * arm[1] - ang[1] * arm[0]), dtype=np.float32)
        return np.concatenate([ang, site_lin])

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
        scores = 1.0 - self._corner_dists() / self._goal_scale
        return float(sum(min(1.0, max(0.0, float(v))) for v in scores) / len(scores))

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
        allc = self.cloth_positions()
        allowed = self.weld_mask.get(prefix)
        for vtx in self.grasp_corners[prefix]:
            if vtx in self._pinned[prefix]:
                continue
            if allowed is not None and vtx not in allowed:
                continue
            gap = float(np.linalg.norm(pos - allc[vtx]))
            if gap < self.grasp_radius:
                self._pinned[prefix][vtx] = _quat_rotate(_quat_conjugate(quat), allc[vtx] - pos)
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
        self.cloth._physics_view.set_masses(_dev(masses), np.arange(1, dtype=np.int32))

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
            for vtx, offset in self._pinned[prefix].items():
                allc[vtx] = pos + _quat_rotate(quat, offset)
        self.cloth.set_world_positions(_dev(allc[None, :, :]))

    def _substep(self):
        self._drive_pins()
        self.cloth.step(PHYSICS_DT)

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        self._domain_params = {}
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
        # Python scalar clipping avoids NumPy's np.clip crash in Isaac Sim 4.5's
        # embedded interpreter after Kit initializes its native extensions.
        clipped = np.asarray([min(1.0, max(-1.0, float(v))) for v in raw], dtype=np.float32)
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


def make_half_fold_env(observation_mode="state", seed=None, headless=True):
    """cloth_fold_rl.quarter_fold_env.HalfFoldEnv on Isaac Sim; see HalfFoldEnv for the success state."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS, HALF_FOLD_MAX_STEPS, HalfFoldEnv
    base = IsaacClothFoldEnv(observation_mode=observation_mode, max_episode_steps=HALF_FOLD_MAX_STEPS,
                             grasp_corners=GRASP_CORNERS, grasp_radius=GRASP_RADIUS, headless=headless)
    return HalfFoldEnv(base_env=base, seed=seed)

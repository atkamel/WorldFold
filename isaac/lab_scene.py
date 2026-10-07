"""IsaacLab scene behind IsaacClothFoldEnv, built from LeHome's parts (lehome-challenge a805ad2).

From LeHome: the SO101 follower (its USD, with convex-decomposed jaws and capsule finger pads, its actuators,
link gravity off, self-collision on); the cloth, which is WorldFold's square mesh packaged as a LeHome garment
(USD plus garment config) and built, read and reset by its GarmentObject with the particle settings in its
particle_garment_cfg.yaml; and its simulation setup: IsaacLab on the CPU device with PhysX GPU dynamics forced on
by LeHome's IsaacLab fork. The particle cloth needs GPU dynamics, and on the CUDA device the grippers pass through
it (LeHome issue #36). From WorldFold: the table, the cloth's size and resolution, the arm base poses and the
"main" camera (mujuco/cloth_params.py).

Import only after isaac_env.start_app(): IsaacLab modules need the running app.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import lehome
from lehome.assets.object.Garment import GarmentObject
from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG

from mujuco.cloth_params import (
    ARM_BASE_LEFT, ARM_BASE_RIGHT, ARM_BASE_QUAT, ARM_JOINTS, CAMERA_FOVY_DEG, CAMERA_POS, CAMERA_TARGET,
    CLOTH_COUNT, CLOTH_MASS, GRIPPER_OPEN, TABLE_TOP_Z, camera_axes,
)
from isaac.isaac_env import (
    CAMERA_CLIP, CLOTH_ADHESION, CLOTH_GRAVITY_SCALE, CLOTH_SUBDIV, FLOOR_COLOR, FLOOR_SIZE, GRIPPER_JOINT,
    GRIPPERFRAME_POS, GRIPPERFRAME_QUAT,
    HEADLIGHT_AMBIENT, HEADLIGHT_DIFFUSE, LIGHT_INTENSITY, LIGHTS, TABLE_SIZE, PAD_LINING_FRICTION,
    PAD_LINING_THICKNESS, cloth_grid_mesh, usd_root_pose, _matrix_from_quat, _quat_facing, _quat_from_matrix,
)

PARTICLE_CFG = Path(lehome.__file__).parent / "tasks" / "bedroom" / "config_file" / "particle_garment_cfg.yaml"
JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
HOME = {name: 0.0 for name in ARM_JOINTS} | {GRIPPER_JOINT: GRIPPER_OPEN}

# Milestone V (vectorised env): copy 0 is the scene as it always was; copies 1.. sit on a 3x3 grid COPY_SPACING apart
# so all share the one 8x8 m floor (its grid texture scales with its size, so enlarging it would change copy 0's
# pixels). The main camera sees x in [-0.80, 0.33], y in [-0.6, 0.6] m of floor around its copy: no copy sees another.
COPY_SPACING = 2.5
COPY_OFFSETS = [np.array([dx, dy, 0.0]) * COPY_SPACING
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))]
MAX_COPIES = len(COPY_OFFSETS)


def copy_root(c):
    """Prim root of copy c: copy 0 keeps the original absolute paths."""
    return "/World" if c == 0 else f"/World/Copy{c}"


class _Copy:
    """Everything per copy of the scene (milestone V). Positions here are world-frame."""

    def __init__(self, c):
        self.index = c
        self.offset = COPY_OFFSETS[c]
        self.arms = {}
        self.joint_ids = {}
        self.gripper_body = {}
        self.cloth = None
        self.cloth_pose = None
        self.cloth_rest = None
        self.pin_floor = None
        self.pins = {}           # weld grasp, see SceneEnv._drive_pins
        self.mass_scale = 1.0    # dynamics DR (set_dynamics); folded into set_pinned_masses
        self.rest_masses = None
        self.rest_material = None
        self.rig_cameras = {}


# LeHome's capsule finger pads (so101_follower_good.usd), in the gripper and jaw link frames: radius 1 cm, 5 cm
# spine along the capsule's local z; the lining sits on the side facing the other finger (local +x on the fixed
# finger, -x on the jaw), from the fingertip (local z -0.035 / +0.035) 4.3 cm up the finger
PAD_RADIUS = 0.01
LININGS = {"gripper": ((-0.017042, 0.000205, -0.073359), (0.997564, 0.0, -0.069756, 0.0), 1.0, -1.0),
           "jaw": ((-0.001287, -0.050919, 0.018299), (0.709761, 0.696602, -0.069295, -0.078629), -1.0, 1.0)}


def _robot(prim_path, base_pos, drive=None, gripper_drive=None):
    """drive: {ImplicitActuatorCfg field: value} applied to every actuator group (the weld profile's
    MUJOCO_ARM_DRIVE); None keeps LeHome's SO101 drives. gripper_drive: the same for the gripper actuator only,
    applied after drive (a friction-grasp knob, isaac_env.FRICTION_GRASP)."""
    pos, rot = usd_root_pose(base_pos, ARM_BASE_QUAT)
    init = SO101_FOLLOWER_CFG.init_state.replace(pos=tuple(float(v) for v in pos), rot=tuple(float(v) for v in rot),
                                                 joint_pos=HOME)
    cfg = SO101_FOLLOWER_CFG.replace(prim_path=prim_path, init_state=init)
    if drive:
        cfg = cfg.replace(actuators={name: act.replace(**drive) for name, act in cfg.actuators.items()})
    if gripper_drive:
        cfg = cfg.replace(actuators={name: act.replace(**gripper_drive) if name == "sts3215-gripper" else act
                                     for name, act in cfg.actuators.items()})
    return cfg


def _pinhole(H, W, fovy_deg):
    aperture = 20.955
    return sim_utils.PinholeCameraCfg(
        focal_length=(aperture * H / W) / (2.0 * np.tan(np.radians(fovy_deg) / 2.0)),
        horizontal_aperture=aperture, vertical_aperture=aperture * H / W, clipping_range=CAMERA_CLIP)


def _camera(image_size, prim_path="/World/main_camera", data_types=("rgb", "distance_to_image_plane")):
    H, W = image_size
    right, up = camera_axes(CAMERA_POS, CAMERA_TARGET)
    forward = np.cross(up, right)
    # "world" camera axes put +X forward; this roll makes image-up equal MuJoCo's camera up
    rot = _quat_from_matrix(np.column_stack([forward, -up, -right]))
    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(CAMERA_POS), rot=tuple(float(v) for v in rot), convention="world"),
        data_types=list(data_types), spawn=_pinhole(H, W, CAMERA_FOVY_DEG), width=W, height=H)


# The so101-nexus MJCF's wrist_cam, in its `gripper` body (the same frame as LeHome's USD gripper link, see
# GRIPPERFRAME_POS in isaac_env.py): pos="0 0.04 -0.04", euler="-0.5 0 6.28" (xyz, radians), fovy 75.
# MuJoCo cameras look down -Z with +Y up, which is the "opengl" offset convention.
WRIST_CAM_POS = (0.0, 0.04, -0.04)
WRIST_CAM_EULER = (-0.5, 0.0, 6.28)
WRIST_CAM_FOVY_DEG = 75.0


def wrist_cam_quat():
    """wxyz of the MJCF wrist_cam frame in the gripper link frame (intrinsic xyz Euler, MuJoCo's default)."""
    def rot(axis, a):
        c, s = np.cos(a), np.sin(a)
        R = np.eye(3)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        if axis == 1:
            R[0, 2], R[2, 0] = s, -s
        return R
    ex, ey, ez = WRIST_CAM_EULER
    return _quat_from_matrix(rot(0, ex) @ rot(1, ey) @ rot(2, ez))


def _wrist_camera(robot_path, size):
    return TiledCameraCfg(
        prim_path=f"{robot_path}/gripper/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=WRIST_CAM_POS, rot=tuple(float(v) for v in wrist_cam_quat()),
                                        convention="opengl"),
        data_types=["rgb"], spawn=_pinhole(size, size, WRIST_CAM_FOVY_DEG), width=size, height=size)


def rig_camera_cfg(name, size, c=0):
    """The imitation pipeline's camera rig (imitation.vision.render.CAMERAS): main + one camera per wrist, of copy c."""
    root = copy_root(c)
    if name in ("main", "demo"):  # demo: the main view at video resolution, for imitation.demo (not a policy input)
        cfg = _camera((size, size), prim_path=f"{root}/rig_{name}", data_types=("rgb",))
        if c:
            cfg.offset = cfg.offset.replace(pos=tuple(float(v) for v in np.asarray(CAMERA_POS) + COPY_OFFSETS[c]))
        return cfg
    if name in ("left_wrist_cam", "right_wrist_cam"):
        return _wrist_camera(f"{root}/Robot/Left_Robot" if name.startswith("left") else f"{root}/Robot/Right_Robot",
                             size)
    raise ValueError(f"unknown rig camera {name!r}")


@configclass
class SceneCfg(DirectRLEnvCfg):
    decimation = 5
    episode_length_s = 1.0e6        # IsaacClothFoldEnv decides termination; no auto-resets here
    action_space = 12               # absolute joint targets, [left 6, right 6]: LeHome's action
    observation_space = 12
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 100.0, render_interval=5, device="cpu", use_fabric=False)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    left_robot = _robot("/World/Robot/Left_Robot", ARM_BASE_LEFT)
    right_robot = _robot("/World/Robot/Right_Robot", ARM_BASE_RIGHT)
    camera: TiledCameraCfg | None = None
    rig: dict | None = None         # {camera name: square size}, the imitation camera rig (opt-in)
    n_copies: int = 1               # milestone V: independent copies of arms + table + cloth (GPU pipeline)


def make_cfg(physics_dt, decimation, image_size=None, rig=None, device="cpu", arm_drive=None, n_copies=1,
             gripper_drive=None):
    if not 1 <= n_copies <= MAX_COPIES:
        raise ValueError(f"n_copies must be in 1..{MAX_COPIES}, got {n_copies}")
    if n_copies > 1 and device == "cpu":
        raise ValueError("several scene copies need the GPU pipeline (per-cloth particle views)")
    if n_copies > 1 and image_size is not None:
        raise NotImplementedError("the observation-mode camera is copy 0's only; use the rig (cameras=) with copies")
    cfg = SceneCfg()
    cfg.n_copies = n_copies
    cfg.action_space = 12 * n_copies
    if arm_drive or gripper_drive:
        cfg.left_robot = _robot("/World/Robot/Left_Robot", ARM_BASE_LEFT, arm_drive, gripper_drive)
        cfg.right_robot = _robot("/World/Robot/Right_Robot", ARM_BASE_RIGHT, arm_drive, gripper_drive)
    cfg.sim.device = device
    cfg.sim.use_fabric = device != "cpu"        # the GPU pipeline reads state through fabric / tensor views
    cfg.rig = dict(rig) if rig else None
    cfg.decimation = decimation
    cfg.sim.dt = physics_dt
    cfg.sim.render_interval = decimation
    cfg.camera = _camera(image_size) if image_size is not None else None
    return cfg


class SceneEnv(DirectRLEnv):
    """Two LeHome SO101 arms, a table and WorldFold's square cloth. Steps absolute joint targets; the
    observations, reward and termination live in IsaacClothFoldEnv, which reads the scene directly."""

    cfg: SceneCfg

    def __init__(self, cfg, cloth_center, grasp=None):
        self.cloth_center = np.asarray(cloth_center, dtype=float)
        # friction-grasp knobs (isaac_env.FRICTION_GRASP); a missing or None value keeps the scene as ported
        self.grasp = {k: v for k, v in (grasp or {}).items() if v is not None}
        self.copies = [_Copy(c) for c in range(cfg.n_copies)]
        self.weld_tau = None    # None: rigid weld (zero-mass pins); seconds: soft weld, see _drive_pins
        self.weld_mass = 1.0    # soft weld: pinned particles' mass multiple (the solver moves heavier ones less)
        super().__init__(cfg)
        for cp in self.copies:
            cp.joint_ids = {p: arm.find_joints(JOINTS, preserve_order=True)[0] for p, arm in cp.arms.items()}
            cp.gripper_body = {p: arm.find_bodies("gripper")[0][0] for p, arm in cp.arms.items()}
        self.targets = torch.zeros((1, 12 * len(self.copies)), device=self.device)
        for cp in self.copies:
            cp.cloth.initialize()

    # copy 0's state under its pre-V names (probes and checks read these)
    arms = property(lambda self: self.copies[0].arms)
    joint_ids = property(lambda self: self.copies[0].joint_ids)
    gripper_body = property(lambda self: self.copies[0].gripper_body)
    cloth = property(lambda self: self.copies[0].cloth)
    cloth_rest = property(lambda self: self.copies[0].cloth_rest)
    cloth_pose = property(lambda self: self.copies[0].cloth_pose)
    pin_floor = property(lambda self: self.copies[0].pin_floor)
    rig_cameras = property(lambda self: self.copies[0].rig_cameras)
    _rest_masses = property(lambda self: self.copies[0].rest_masses)
    _rest_material = property(lambda self: self.copies[0].rest_material)

    @property
    def pins(self):
        return self.copies[0].pins

    @pins.setter
    def pins(self, value):
        self.copies[0].pins = value

    @property
    def mass_scale(self):
        return self.copies[0].mass_scale

    @mass_scale.setter
    def mass_scale(self, value):
        self.copies[0].mass_scale = value

    def _setup_scene(self):
        pad_mu = float(self.grasp.get("pad_friction", PAD_LINING_FRICTION))
        lining = sim_utils.RigidBodyMaterialCfg(static_friction=pad_mu, dynamic_friction=pad_mu)
        floor = sim_utils.GroundPlaneCfg(size=(2 * FLOOR_SIZE, 2 * FLOOR_SIZE), color=FLOOR_COLOR)
        table = sim_utils.CuboidCfg(size=(TABLE_SIZE, TABLE_SIZE, TABLE_TOP_Z),
                                    collision_props=sim_utils.CollisionPropertiesCfg(),
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.25)))
        for cp in self.copies:
            c, root, dx = cp.index, copy_root(cp.index), cp.offset
            robots = {"left_": self.cfg.left_robot, "right_": self.cfg.right_robot}
            if c:
                robots = {p: r.replace(prim_path=r.prim_path.replace("/World", root, 1),
                                       init_state=r.init_state.replace(
                                           pos=tuple(float(v) for v in np.asarray(r.init_state.pos) + dx)))
                          for p, r in robots.items()}
            cp.arms = {p: Articulation(r) for p, r in robots.items()}
            suffix = "" if c == 0 else f"_{c}"
            self.scene.articulations[f"left_arm{suffix}"] = cp.arms["left_"]
            self.scene.articulations[f"right_arm{suffix}"] = cp.arms["right_"]
            if c == 0:
                lining.func("/World/Looks/pad_lining", lining)
            for r in robots.values():
                self._add_linings(r.prim_path)
            if c == 0:
                floor.func("/World/floor", floor)
            table.func(f"{root}/table", table, translation=(float(dx[0]), float(dx[1]), TABLE_TOP_Z / 2.0))
            cp.cloth, cp.cloth_rest = self._spawn_cloth(cp)
            if c == 0:
                self.camera = None
                if self.cfg.camera is not None:
                    self.camera = TiledCamera(self.cfg.camera)
                    self.scene.sensors["main"] = self.camera
            cp.rig_cameras = {}
            for name, size in (self.cfg.rig or {}).items():
                cp.rig_cameras[name] = TiledCamera(rig_camera_cfg(name, size, c))
                self.scene.sensors[f"rig_{name}{suffix}"] = cp.rig_cameras[name]
        if self.camera is not None or self.rig_cameras:
            self._spawn_lights()

    def _add_linings(self, robot_path):
        # a box on each finger's inner face, over LeHome's capsule pad: PAD_LINING_THICKNESS out from the capsule
        t = float(self.grasp.get("pad_thickness", PAD_LINING_THICKNESS))
        for link, (center, quat, side, tip) in LININGS.items():
            local = np.array([side * (PAD_RADIUS + t / 2.0 - 0.0005), 0.0, tip * (0.035 - 0.0215)])
            R = _matrix_from_quat(quat)
            box = UsdGeom.Cube.Define(self.sim.stage, f"{robot_path}/{link}/pad_lining")
            box.CreateSizeAttr(1.0)
            box.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(center) + R @ local)))
            box.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(*(float(v) for v in quat[1:]))))
            box.AddScaleOp().Set(Gf.Vec3f(t + 0.001, 0.016, 0.043))
            box.GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.1, 0.1)])
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            sim_utils.bind_physics_material(box.GetPath().pathString, "/World/Looks/pad_lining")

    def _spawn_cloth(self, cp):
        # WorldFold's square cloth packaged the way LeHome packages a garment: a USD whose default prim holds a
        # "mesh", plus a garment config. GarmentObject applies LeHome's particle system, material and cloth settings.
        points, faces = cloth_grid_mesh(CLOTH_SUBDIV)
        usd = os.path.join(tempfile.mkdtemp(), "square_cloth.usd")
        stage = Usd.Stage.CreateNew(usd)
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/mesh")
        mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
        mesh.GetFaceVertexIndicesAttr().Set(faces)
        mesh.GetFaceVertexCountsAttr().Set([3] * (len(faces) // 3))
        mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.2)])
        mesh.GetDoubleSidedAttr().Set(True)
        stage.GetRootLayer().Save()
        n = (CLOTH_COUNT - 1) * CLOTH_SUBDIV + 1
        particle_cfg = OmegaConf.load(PARTICLE_CFG)
        particle_cfg.objects.garment_config.particle_mass = CLOTH_MASS / (n * n)
        particle_cfg.objects.particle_material.gravity_scale = CLOTH_GRAVITY_SCALE
        particle_cfg.objects.particle_material.adhesion = CLOTH_ADHESION
        if "particle_friction" in self.grasp:          # friction-grasp knob; adhesion stays CLOTH_ADHESION (0)
            particle_cfg.objects.particle_material.friction = float(self.grasp["particle_friction"])
        z = TABLE_TOP_Z + particle_cfg.objects.particle_system.rest_offset + 0.001
        cp.cloth_pose = np.array([self.cloth_center[0], self.cloth_center[1], z])
        if cp.index:
            cp.cloth_pose = cp.cloth_pose + cp.offset
        cp.pin_floor = z - 0.001            # a particle resting on the table top (weld pins never go lower)
        pose = [float(v) for v in cp.cloth_pose]
        garment_cfg = OmegaConf.create({
            # GarmentObject joins a leading-slash asset_path onto the working directory
            "asset_path": "/" + os.path.relpath(usd, os.getcwd()),
            "visual_usd_paths": [], "scale": [1.0, 1.0, 1.0],
            "check_point": [0, n - 1, (n - 1) * n, n * n - 1], "success_distance": [],   # the four corners
            "initial_pos_range": pose + pose, "initial_rot_range": [0.0] * 6,
            "soft_reset_pos_range": pose + pose, "soft_reset_rot_range": [0.0] * 6,
        })
        cloth = GarmentObject(f"{copy_root(cp.index)}/Object/cloth", particle_cfg, garment_cfg,
                              rng=np.random.RandomState(0))
        return cloth, np.asarray(points) + cp.cloth_pose

    # On the CPU device there is no particle-cloth tensor view: PhysX writes the particles back to the mesh's USD
    # points (use_fabric=False), and GarmentObject reads and resets them there.
    def particle_positions(self, c=0):
        """Copy c's particles in its own frame (world minus the copy's offset; copy 0's frame is the world)."""
        if self.device != "cpu":           # GPU pipeline: the particle-cloth tensor view (world frame)
            cp = self.copies[c]
            pos = cp.cloth._cloth_prim_view.get_world_positions()[0].detach().cpu().numpy().astype(np.float64)
            return pos - cp.offset if c else pos
        # GarmentObject.get_current_mesh_points without its open3d point cloud
        pos, ori = self.cloth.get_world_pose()
        local = self.cloth._get_points_pose().detach().cpu().numpy()
        return self.cloth.transform_points(local, pos.detach().cpu().numpy(), ori.detach().cpu().numpy(),
                                           self.cloth.get_world_scale().detach().cpu().numpy()).astype(np.float64)

    def reset_cloth(self, offset_xy, height, rpy_deg, c=0):
        # LeHome's soft reset puts the initial particles back at the configured pose; then move the cloth up by
        # height, along by offset_xy and tilt it, so it drops onto the table the way LeHome drops its garments
        cp = self.copies[c]
        cp.cloth.reset()
        pose = cp.cloth_pose + np.array([offset_xy[0], offset_xy[1], height])
        cp.cloth.set_world_pose(position=pose, orientation=euler_angles_to_quat(np.asarray(rpy_deg), degrees=True))

    def reset_copy(self, c):
        """Copy c's arms back to their spawn pose and HOME joints at rest, its pins cleared and its rig cameras
        reset, without touching the other copies and without a physics step (milestone V's per-env reset; the
        cloth is reset_cloth's). lab.reset() still resets every copy at once."""
        cp = self.copies[c]
        cp.pins = {}
        for arm in cp.arms.values():
            arm.reset()
            arm.write_root_pose_to_sim(arm.data.default_root_state[:, :7])
            arm.write_joint_state_to_sim(arm.data.default_joint_pos, arm.data.default_joint_vel)
            arm.write_data_to_sim()
        for cam in cp.rig_cameras.values():
            cam.reset()
        self.sim.forward()

    def set_copy_targets(self, c, targets):
        """Copy c's 12 joint targets (left 6, right 6) for the next lab.step."""
        self.targets[:, 12 * c:12 * c + 12] = targets

    def _spawn_lights(self):
        dome = sim_utils.DomeLightCfg(intensity=LIGHT_INTENSITY * HEADLIGHT_AMBIENT)
        dome.func("/World/ambient_light", dome)
        head_dir = np.array(CAMERA_TARGET) - np.array(CAMERA_POS)
        for i, (direction, diffuse) in enumerate(list(LIGHTS) + [(head_dir, HEADLIGHT_DIFFUSE)]):
            light = sim_utils.DistantLightCfg(intensity=LIGHT_INTENSITY * diffuse)
            light.func(f"/World/light_{i}", light, orientation=tuple(float(v) for v in _quat_facing(direction)))

    def _pre_physics_step(self, actions):
        self.targets = actions.clone()

    def _apply_action(self):
        for cp in self.copies:
            k = 12 * cp.index
            cp.arms["left_"].set_joint_position_target(self.targets[:, k:k + 6], joint_ids=cp.joint_ids["left_"])
            cp.arms["right_"].set_joint_position_target(self.targets[:, k + 6:k + 12],
                                                        joint_ids=cp.joint_ids["right_"])
            if cp.pins:
                self._drive_pins(cp.index)

    # ---- weld grasp (IsaacClothFoldEnv grasp_mode="weld", Phase W) ----
    # pins: {prefix: (particle indices, offsets (k, 3) from the gripperframe site, world axes)}. GPU pipeline only: a
    # welded particle gets zero mass and, before every physics substep, its position (site + offset, never below
    # pin_floor) and velocity (the site's) are written through the particle-cloth tensor view. On the
    # CPU device the only write path is the mesh's USD points, which PhysX ignores mid-simulation (W1: 14-36 mm drift
    # per substep), so IsaacClothFoldEnv refuses grasp_mode="weld" there.
    def site_pose(self, prefix, c=0):
        """World pose (pos, R) of the gripperframe site: the gripper link composed with the MJCF site offset."""
        arm = self.copies[c].arms[prefix]
        b = self.copies[c].gripper_body[prefix]
        link_pos = arm.data.body_link_pos_w[0, b].cpu().numpy().astype(np.float64)
        link_R = _matrix_from_quat(arm.data.body_link_quat_w[0, b].cpu().numpy().astype(np.float64))
        return link_pos + link_R @ GRIPPERFRAME_POS, link_R @ _matrix_from_quat(GRIPPERFRAME_QUAT), link_pos, link_R

    def set_pinned_masses(self, c=0):
        """GPU pipeline: pinned particles get zero mass (held exactly where written); everything else its rest mass.
        ClothPrim.set_particle_masses calls a get_masses it doesn't have, so this uses the physics view directly."""
        if self.device == "cpu":
            return
        cp = self.copies[c]
        pv = cp.cloth._cloth_prim_view._physics_view
        if cp.rest_masses is None:
            cp.rest_masses = pv.get_masses().clone()
        masses = cp.rest_masses * cp.mass_scale
        for idx, _ in cp.pins.values():
            ti = torch.as_tensor(idx, device=masses.device)
            flat = masses.view(masses.shape[0], -1)
            flat[0, ti] = 0.0 if self.weld_tau is None else flat[0, ti] * self.weld_mass
        pv.set_masses(masses, torch.arange(masses.shape[0], device=masses.device))

    def set_dynamics(self, mass_scale=1.0, friction_scale=1.0, damping_scale=1.0, c=0):
        """MuJoCo's dynamics DR (sim_main.reset), GPU pipeline: scales the particle masses, the particle material's
        friction (MuJoCo scales the table's; here it is the cloth's against every rigid, the table included) and its
        global velocity damping (MuJoCo: the cloth joints' damping), from their spawned values."""
        if self.device == "cpu":
            raise ValueError("dynamics DR needs the GPU pipeline (per-particle masses)")
        cp = self.copies[c]
        mat = cp.cloth.particle_material
        if cp.rest_material is None:
            cp.rest_material = (float(mat.get_friction()), float(mat.get_damping()))
        cp.mass_scale = float(mass_scale)
        mat.set_friction(cp.rest_material[0] * float(friction_scale))
        mat.set_damping(cp.rest_material[1] * float(damping_scale))
        self.set_pinned_masses(c)

    def _drive_pins(self, c=0):
        # the cloth view's set_positions/set_velocities index whole cloths, not particles, so the full buffer is
        # written back; unpinned rows carry the values just read
        cp = self.copies[c]
        view = cp.cloth._cloth_prim_view
        pos = view.get_world_positions(clone=False)
        vel = view.get_velocities(clone=False)
        for prefix, (idx, offsets) in cp.pins.items():
            if len(idx) == 0:
                continue
            site, _, link_pos, _ = self.site_pose(prefix, c)
            world = site + offsets
            arm, b = cp.arms[prefix], cp.gripper_body[prefix]
            lin = arm.data.body_link_lin_vel_w[0, b].cpu().numpy()
            ang = arm.data.body_link_ang_vel_w[0, b].cpu().numpy()
            v = np.broadcast_to(lin + np.cross(ang, site - link_pos), world.shape).copy()
            # the table holds a pinned particle up, as MuJoCo's contact does against its soft weld: a kinematic pin
            # would otherwise be pushed through it (W3 trace: 2.7 cm below the top as the gripper closed descending)
            low = world[:, 2] < cp.pin_floor
            world[low, 2] = cp.pin_floor
            v[low, 2] = np.maximum(v[low, 2], 0.0)
            ti = torch.as_tensor(idx, device=pos.device)
            if self.weld_tau is None:
                pos[0, ti] = torch.as_tensor(world, dtype=pos.dtype, device=pos.device)
            else:
                # soft weld, MuJoCo's equality with solref (weld_tau, 1): the pinned particles keep their mass and are
                # steered toward the target at the gripper's velocity plus error / weld_tau; the cloth solver can
                # still pull them, so the held corner gives under tension and hangs under gravity as in MuJoCo
                here = pos[0, ti].detach().cpu().numpy().astype(np.float64)
                v = v + (world - here) / self.weld_tau
            vel[0, ti] = torch.as_tensor(v, dtype=vel.dtype, device=vel.device)
        all_idx = torch.arange(pos.shape[0], device=pos.device)
        if self.weld_tau is None:
            view._physics_view.set_positions(pos, all_idx)
        view._physics_view.set_velocities(vel, all_idx)

    def _get_observations(self):
        return {"policy": self.targets}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return done, done

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        for arm in (a for cp in self.copies for a in cp.arms.values()):
            arm.write_root_pose_to_sim(arm.data.default_root_state[env_ids, :7], env_ids)
            arm.write_joint_state_to_sim(arm.data.default_joint_pos[env_ids], arm.data.default_joint_vel[env_ids],
                                         env_ids=env_ids)

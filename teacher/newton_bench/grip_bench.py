#!/usr/bin/env python3
"""Step 1 of the sim-fidelity plan: the real SO-101 gripper pinching the LeHome shirt in Newton.

One Newton VBD solve holds the shirt (cloth), the gripper and the moving jaw (rigid bodies), so the jaws feel the
cloth and the cloth feels the jaws (two-way). The arm is kinematic: it follows the commanded path like the real
position-controlled arm, and the gripper hangs off it on a vertical spring with a force cap, which stands in for
how far the real arm's servos give when the jaws press on the table. Real values, each with its source:
  shirt    LeHome Top_Long_Seen_0 mesh at LeHome's scale 0.45, 160 g/m^2 cotton jersey (about 40 g)
  jaws     SO-101 jaw meshes and inertias from MuJoCo Menagerie (robotstudio_so101/so101.xml), full-surface contact
  servo    the STS3215 model in that file: kp 998.22 N*m/rad, kv 2.731, 2.94 N*m max. LeRobot's so_follower writes
           Max_Torque_Limit 500 (50 %) to the gripper, so the squeeze is capped at 1.47 N*m (--squeeze)
  arm      vertical give of the SO-101 arm from the same servo model (kp 998 N*m/rad per joint, 2.94 N*m max)
           at ~0.1-0.2 m lever arms: about 14 kN/m, saturating near 15 N (--arm-kz, --arm-fz; an estimate)
  gravity  1 g (LeHome uses 2 g)
Tests, with the same moves and measurements as oracle_fold.py so the numbers compare with the Isaac runs:
  settle   the shirt left alone (cloth_settle)
  grip     the sleeve grip test (assay_trial, extra=True)

  python grip_bench.py --test settle --out results/newton/settle
  python grip_bench.py --test grip --squeeze 1.47 --mu-jaw 0.4 --out results/newton/grip
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import warp as wp

import newton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
import garment_keys as gk  # noqa: E402  (numpy only)

MENAGERIE = os.environ.get("SO101_MENAGERIE", os.path.expanduser("~/wf/assets/so101_menagerie"))

# ---- SO-101 gripper, from MuJoCo Menagerie robotstudio_so101/so101.xml ----
FIXED_JAW_GEOM = ((0.0, -0.000218, 0.00095), (1.0, 0.0, 0.0, 0.0))          # in the gripper body; quat x,y,z,w
JAW_JOINT = ((0.0202, 0.0188, -0.0234), (0.70710678, 0.0, 0.0, 0.70710678))  # moving jaw body in the gripper body
MOVING_JAW_GEOM = ((0.0, 0.0, 0.0189), (0.0, 0.0, 0.0, 1.0))                 # in the moving jaw body
GRIPPER_MASS, GRIPPER_COM = 0.087, (0.000214, 0.000245, -0.025187)
GRIPPER_I = (2.75087e-05, 4.33657e-05, 3.45059e-05, -3.35241e-07, -5.7352e-06, -5.17847e-08)  # ixx iyy izz ixy ixz iyz
JAW_MASS, JAW_COM = 0.012, (-0.001575, -0.0300244, 0.0192755)
JAW_I = (6.61427e-06, 1.89032e-06, 5.28738e-06, -3.19807e-07, -5.90717e-09, -1.09945e-07)
JAW_RANGE = (-0.174533, 1.7453292)
SERVO_KP, SERVO_KV, SERVO_MAX = 998.22, 2.731, 2.94    # class sts3215 position actuator
SERVO_DAMPING = 0.60                                    # class sts3215 joint damping (passive)
TCP = (-0.0079, -0.000218121, -0.0981274)               # so101_new_calib.urdf gripper_frame_joint: the Isaac tests' tip
GRIP_OPEN, GRIP_CLOSED = 0.55, -0.17                    # so101_kin.py
TICK = 1.0 / 90.0                                       # the Isaac tests' control tick (physics_dt 1/90)
SCALE = 0.45                                            # LeHome garment json "scale"


def log(*a):
    print("[bench]", *a, flush=True)


def inertia(ixx, iyy, izz, ixy, ixz, iyz):
    return wp.mat33(ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz)


def xf(pq):
    return wp.transform(wp.vec3(*pq[0]), wp.quat(*pq[1]))


def load_mesh(name, sdf_voxel):
    """A Menagerie STL as a Newton mesh with its SDF (needed for full-surface cloth contact)."""
    import trimesh
    m = trimesh.load(os.path.join(MENAGERIE, "assets", name + ".stl"))
    mesh = newton.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32).reshape(-1))
    mesh.build_sdf(target_voxel_size=sdf_voxel, narrow_band_range=(-0.01, 0.01), margin=0.01)
    return mesh, np.asarray(m.vertices)


@wp.kernel
def reset_counter(sub: wp.array[int]):
    sub[0] = 0


@wp.kernel
def kinematic_pose(a: wp.array[wp.transform], b: wp.array[wp.transform], v: wp.array[wp.spatial_vector],
                   sub: wp.array[int], n_sub: int, q0: int, qd0: int,
                   joint_q: wp.array[float], joint_qd: wp.array[float]):
    """The wrist's pose for this substep: a -> b over the frame (the real arm is stiff position control)."""
    k = sub[0] + 1
    sub[0] = k
    t = float(k) / float(n_sub)
    p = wp.lerp(wp.transform_get_translation(a[0]), wp.transform_get_translation(b[0]), t)
    r = wp.quat_slerp(wp.transform_get_rotation(a[0]), wp.transform_get_rotation(b[0]), t)
    for i in range(3):
        joint_q[q0 + i] = p[i]
    for i in range(4):
        joint_q[q0 + 3 + i] = r[i]
    for i in range(6):
        joint_qd[qd0 + i] = v[0][i]


@wp.kernel
def drives(body_q: wp.array[wp.transform], arm: int, g: int, j: int, x_pj: wp.transform, cmd: wp.array[float],
           cap: float, tgt: int, cap_z: float, tgt_z: int, joint_target_q: wp.array[float], q_out: wp.array[float]):
    """Saturating position drives. VBD has no effort limit, so each target is kept within limit / stiffness of the
    joint's current position: the drive then pushes with at most the limit.
    Jaw: STS3215 under LeRobot, stiff position control whose torque saturates at the gripper's torque limit.
    Arm (vertical): the gripper is held at the arm's commanded height with at most the arm's force."""
    anchor = wp.transform_multiply(body_q[g], x_pj)
    rel = wp.quat_inverse(wp.transform_get_rotation(anchor)) * wp.transform_get_rotation(body_q[j])
    q = 2.0 * wp.atan2(rel[2], rel[3])
    joint_target_q[tgt] = q + wp.clamp(cmd[0] - q, -cap, cap)
    d = wp.transform_get_translation(body_q[g])[2] - wp.transform_get_translation(body_q[arm])[2]
    joint_target_q[tgt_z] = d + wp.clamp(-d, -cap_z, cap_z)
    q_out[0] = q
    q_out[1] = d


def pinch_spot(v, k, inset):
    """The straight-down pinch: `inset` in from the cuff along the sleeve; moving finger toward the armpit."""
    cuff, pit = v[k["cuff_mid"], :2], v[k["armpit"], :2]
    axis = (pit - cuff) / np.linalg.norm(pit - cuff)
    return cuff + axis * inset, math.atan2(axis[1], axis[0])


def stacked_pairs(v, xy_tol=0.0015, dz_min=0.002, dz_max=0.010):
    """oracle_fold.stacked_pairs: (top, bottom) particles lying one above the other."""
    from scipy.spatial import cKDTree
    p = np.array(sorted(cKDTree(v[:, :2]).query_pairs(xy_tol)), dtype=np.int64).reshape(-1, 2)
    if not len(p):
        return p[:, 0], p[:, 1]
    dz = v[p[:, 0], 2] - v[p[:, 1], 2]
    keep = (np.abs(dz) > dz_min) & (np.abs(dz) < dz_max)
    p, dz = p[keep], dz[keep]
    return np.where(dz > 0, p[:, 0], p[:, 1]), np.where(dz > 0, p[:, 1], p[:, 0])


def footprint_m2(v, cell=0.005):
    return float(len(np.unique(np.floor(v[:, :2] / cell).astype(np.int64), axis=0)) * cell * cell)


class Bench:
    def __init__(self, a):
        self.a = a
        self.fps, self.n_sub = 60, a.substeps
        self.frame_dt = 1.0 / self.fps
        self.dt = self.frame_dt / self.n_sub
        self.t = 0.0
        g = np.load(a.garment)
        rest = g["points"].astype(np.float64)
        self.keys = gk.top_keys(rest, g["check_point"])
        verts = rest * SCALE
        faces = g["faces"].astype(np.int64)
        for _ in range(a.subdivide):                           # midpoint split: 4x triangles, half the edge length
            import trimesh
            nv, nf = trimesh.remesh.subdivide(verts, faces)
            assert np.allclose(nv[:len(verts)], verts)          # original vertices keep their indices (keys stay valid)
            verts, faces = nv, nf
        if a.rest_from:                                         # a settled flat shirt as the rest shape: nothing creeps
            verts = np.load(os.path.expanduser(a.rest_from)).astype(np.float64)
            assert len(verts) == len(nv if a.subdivide else rest), "rest_from does not match the mesh"
        else:
            verts[:, 2] += a.drop_gap - verts[:, 2].min()      # LeHome's 3D shirt, dropped onto the table (z = 0)
        self.faces = faces
        self.n_cloth = len(verts)

        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        mu_shape = lambda mu_pair: mu_pair ** 2 / a.mu_cloth    # VBD mixes cloth/shape friction as sqrt(mu_c * mu_s)
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu_shape(a.mu_table), ke=a.shape_ke, kd=a.contact_kd))
        if a.no_cloth:
            self.n_cloth = 0
        else:
          b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                         vertices=[wp.vec3(*p) for p in verts], indices=faces.astype(np.int32).reshape(-1).tolist(),
                         density=a.density, tri_ke=a.tri_ke, tri_ka=a.tri_ka, tri_kd=a.tri_kd,
                         edge_ke=a.edge_ke, edge_kd=a.edge_kd, particle_radius=a.radius, color=(0.35, 0.6, 0.9))

        # kinematic arm -> gripper (fixed jaw) on a capped vertical spring -> moving jaw on its servo
        jaw_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu_shape(a.mu_jaw), ke=a.shape_ke, kd=a.contact_kd)
        fixed_mesh, fixed_v = load_mesh("wrist_roll_follower_so101_v1", a.sdf_voxel)
        moving_mesh, moving_v = load_mesh("moving_jaw_so101_v1", a.sdf_voxel)
        self.arm = b.add_link(xform=wp.transform_identity(), inertia=inertia(1e-4, 1e-4, 1e-4, 0, 0, 0), mass=0.1,
                              is_kinematic=True, label="arm")
        self.g = b.add_link(xform=wp.transform_identity(), com=wp.vec3(*GRIPPER_COM), inertia=inertia(*GRIPPER_I),
                            mass=GRIPPER_MASS, label="gripper")
        b.add_shape_mesh(self.g, xform=xf(FIXED_JAW_GEOM), mesh=fixed_mesh, cfg=jaw_cfg, label="fixed_jaw",
                         color=(0.55, 0.55, 0.6))
        self.jaw = b.add_link(xform=wp.transform_identity(), com=wp.vec3(*JAW_COM), inertia=inertia(*JAW_I),
                              mass=JAW_MASS, label="moving_jaw")
        b.add_shape_mesh(self.jaw, xform=xf(MOVING_JAW_GEOM), mesh=moving_mesh, cfg=jaw_cfg, label="moving_jaw",
                         color=(0.95, 0.55, 0.15))
        j_free = b.add_joint_free(self.arm, label="arm")
        j_z = b.add_joint_prismatic(parent=self.arm, child=self.g, axis=(0.0, 0.0, 1.0), target_pos=0.0,
                                    target_ke=a.arm_kz, target_kd=a.arm_kdz, limit_lower=-0.03, limit_upper=0.03,
                                    label="arm_give")
        # The model's joint limit (-10 deg) stops the jaws 2.3 mm apart (measured on the Menagerie meshes); the real
        # gripper closes until the jaws touch. --jaw-contact lets the jaws collide and moves the limit out of the way.
        j_jaw = b.add_joint_revolute(parent=self.g, child=self.jaw, parent_xform=xf(JAW_JOINT), axis=(0.0, 0.0, 1.0),
                                     target_pos=GRIP_OPEN, target_ke=a.servo_ke, target_kd=a.servo_kd,
                                     limit_lower=a.jaw_lower if a.jaw_contact else JAW_RANGE[0],
                                     limit_upper=JAW_RANGE[1], collision_filter_parent=not a.jaw_contact,
                                     label="gripper")
        b.add_articulation([j_free, j_z, j_jaw], label="so101_gripper")
        self.wrist_q0, self.wrist_qd0 = b.joint_q_start[j_free], b.joint_qd_start[j_free]
        self.jaw_q0 = b.joint_q_start[j_jaw]
        b.joint_q[self.jaw_q0] = GRIP_OPEN

        # geometry for the moves: the fixed finger's lowest point below the tip frame
        r = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)      # FIXED_JAW_GEOM rotation (180 deg about x)
        fv = fixed_v @ r.T + np.array(FIXED_JAW_GEOM[0])
        self.finger_drop = TCP[2] - fv[:, 2].min()                     # tip frame -> lowest point of the fixed finger
        self.fixed_v_g, self.moving_v_j = fv, moving_v + np.array(MOVING_JAW_GEOM[0])

        self.park, park_yaw = (0.0, 0.0, 0.30), 0.0
        if a.rest_from and a.grasp == "pinch":                 # start right over the spot, already turned
            spot, park_yaw = pinch_spot(verts, self.keys[a.side], a.inset)
            self.park = (spot[0], spot[1], 0.5925 - 0.52)
        self.pose = self.wrist_pose(self.park, park_yaw)
        b.joint_q[self.wrist_q0:self.wrist_q0 + 7] = list(self.pose[0]) + list(self.pose[1])

        b.color()
        self.model = b.finalize()
        if self.n_cloth:
            self.model.edge_rest_angle.zero_()              # cloth has no shape memory (as Newton's cloth_franka example)
        self.model.soft_contact_ke, self.model.soft_contact_kd = a.contact_ke, a.contact_kd
        self.model.soft_contact_mu = a.mu_cloth             # cloth-cloth
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.pipe = newton.CollisionPipeline(self.model, soft_contact_gap=a.contact_gap,
                                             include_static_kinematic_pairs=False,
                                             enable_rigid_soft_full_surface_contact=not a.particle_contacts_only)
        self.contacts = self.pipe.contacts()
        kw = {}
        if a.self_detect == "pre_init":                         # self-contact detection once per substep
            S = newton.solvers.SolverBase                       # (as Newton's cloth_franka example)
            kw["collision_frequency_type"] = {S.CollisionSlot.SOFT_SELF_CONTACT: S.CollisionFrequencyType.PRE_INIT}
        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=a.iterations, rigid_compliant_alm=True, friction_epsilon=a.friction_eps,
            particle_enable_self_contact=not a.no_cloth, particle_self_contact_margin=a.self_margin,
            particle_self_contact_gap=a.self_gap, particle_rest_shape_contact_exclusion_radius=a.rest_exclusion,
            rigid_body_particle_contact_buffer_size=a.body_contact_buffer, **kw)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.s0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.s1)

        dev = self.model.device
        self.pose_a = wp.array([wp.transform_identity()], dtype=wp.transform, device=dev)
        self.pose_b = wp.array([wp.transform_identity()], dtype=wp.transform, device=dev)
        self.vel = wp.zeros(1, dtype=wp.spatial_vector, device=dev)
        self.sub = wp.zeros(1, dtype=int, device=dev)
        self.cmd = wp.array([GRIP_OPEN], dtype=float, device=dev)
        self.q_jaw = wp.zeros(2, dtype=float, device=dev)              # jaw angle, gripper height above the arm
        self.cap = float(a.squeeze) / a.servo_ke
        self.cap_z = float(a.arm_fz) / a.arm_kz
        self.tgt = int(self.model.joint_target_q_start.numpy()[j_jaw])
        self.tgt_z = int(self.model.joint_target_q_start.numpy()[j_z])
        self.x_pj = xf(JAW_JOINT)
        self.graph = None
        if a.graph and dev.is_cuda:
            self.set_pose(self.pose, self.pose)
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
        log(f"cloth {self.n_cloth} particles, {self.model.tri_count} triangles, "
            f"mass {(self.cloth_mass() if self.n_cloth else 0.0) * 1000:.1f} g; "
            f"substeps {self.n_sub}, iterations {a.iterations}, graph {self.graph is not None}; "
            f"squeeze cap {a.squeeze} N*m; mu jaw/cloth {a.mu_jaw}, cloth/cloth {a.mu_cloth}, table/cloth {a.mu_table}")

    def cloth_mass(self):
        m = self.model.particle_mass.numpy()[:self.n_cloth]
        return float(m.sum())

    # ---- the wrist ----
    def wrist_pose(self, tip, yaw):
        """Gripper body pose with the tip frame at `tip`, jaws pointing down, the moving finger's side (+x) at yaw."""
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p = np.asarray(tip, float) - rot @ np.asarray(TCP)
        return (tuple(p), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))

    def set_pose(self, a, b):
        self.pose_a.assign([wp.transform(wp.vec3(*a[0]), wp.quat(*a[1]))])
        self.pose_b.assign([wp.transform(wp.vec3(*b[0]), wp.quat(*b[1]))])
        v = (np.asarray(b[0]) - np.asarray(a[0])) / self.frame_dt
        ya, yb = 2 * math.atan2(a[1][2], a[1][3]), 2 * math.atan2(b[1][2], b[1][3])
        w = (yb - ya) / self.frame_dt
        self.vel.assign([wp.spatial_vector(float(v[0]), float(v[1]), float(v[2]), 0.0, 0.0, float(w))])

    def simulate(self):
        wp.launch(reset_counter, dim=1, inputs=[self.sub])
        self.solver.rebuild_bvh(self.s0)
        for _ in range(self.n_sub):
            self.s0.clear_forces()
            wp.launch(kinematic_pose, dim=1, inputs=[self.pose_a, self.pose_b, self.vel, self.sub, self.n_sub,
                                                     self.wrist_q0, self.wrist_qd0, self.s0.joint_q, self.s0.joint_qd])
            newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0,
                           body_flag_filter=int(newton.BodyFlags.KINEMATIC))
            wp.launch(drives, dim=1, inputs=[self.s0.body_q, self.arm, self.g, self.jaw, self.x_pj, self.cmd, self.cap,
                                             self.tgt, self.cap_z, self.tgt_z, self.control.joint_target_q, self.q_jaw])
            self.pipe.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
            self.s0, self.s1 = self.s1, self.s0

    def frame(self, tip, yaw, jaw_cmd):
        """Advance one 1/60 s frame with the wrist moving to (tip, yaw) and the servo commanded to jaw_cmd."""
        nxt = self.wrist_pose(tip, yaw)
        self.set_pose(self.pose, nxt)
        self.cmd.assign([float(jaw_cmd)])
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.pose = nxt
        self.t += self.frame_dt

    # ---- reading the state ----
    def cloth(self):
        if not self.n_cloth:
            return np.zeros((0, 3))
        return self.s0.particle_q.numpy()[:self.n_cloth].astype(np.float64)

    def tip(self, commanded=False):
        """The Isaac tests' tip frame: where the arm puts it (commanded) or where the gripper actually is."""
        p, q = self.pose
        if not commanded:
            x = self.s0.body_q.numpy()[self.g]
            p, q = x[:3], x[3:7]
        yaw = 2 * math.atan2(q[2], q[3])
        c, s = math.cos(yaw), math.sin(yaw)
        return np.asarray(p) + np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.asarray(TCP)

    def jaw_angle(self):
        return float(self.q_jaw.numpy()[0])

    def arm_give_mm(self):
        """How far the table/cloth pushed the gripper up from where the arm holds it."""
        return float(self.q_jaw.numpy()[1]) * 1000

    def press_n(self):
        """Downward force the arm puts on the gripper (positive = pressing down), capped like its servos."""
        d = float(self.q_jaw.numpy()[1])
        return float(self.a.arm_kz * max(-self.cap_z, min(self.cap_z, d)))

    def squeeze_nm(self, cmd):
        """Servo torque now (closing positive): ke * (target - q), capped like the real servo."""
        q = self.jaw_angle()
        return float(-self.a.servo_ke * max(-self.cap, min(self.cap, cmd - q)))


class Mover:
    """Straight-line moves of the tip at a fixed speed, one frame at a time (oracle fold_plans._path style)."""

    def __init__(self, bench, rec=None):
        self.b, self.rec = bench, rec
        self.tip = np.asarray(bench.tip(commanded=True), float)
        self.yaw = 2 * math.atan2(bench.pose[1][2], bench.pose[1][3])
        self.jaw = GRIP_OPEN

    def step(self):
        self.b.frame(self.tip, self.yaw, self.jaw)
        if self.rec is not None:
            self.rec(self.b)

    def move(self, to, speed, yaw=None):
        to = np.asarray(to, float)
        d = float(np.linalg.norm(to - self.tip))
        n = max(1, int(math.ceil(d / (speed * self.b.frame_dt))))
        start, y0 = self.tip.copy(), self.yaw
        y1 = y0 if yaw is None else yaw
        for i in range(1, n + 1):
            self.tip = start + (to - start) * i / n
            self.yaw = y0 + (y1 - y0) * i / n
            self.step()

    def hold(self, seconds, jaw=None, jaw_ramp=0.0):
        n = max(1, int(round(seconds / self.b.frame_dt)))
        j0, j1 = self.jaw, (self.jaw if jaw is None else jaw)
        nr = max(1, int(round(jaw_ramp / self.b.frame_dt)))
        for i in range(1, n + 1):
            self.jaw = j0 + (j1 - j0) * min(1.0, i / nr)
            self.step()


class Video:
    """Close-up mp4 of the grasp: Newton's own renderer (ViewerGL, headless), camera beside the gripper looking along
    the jaw hinge so the jaws open and close in the picture. Starts when the tip first goes below `start_z`, runs
    `seconds` of sim time; every 1/60 s sim frame is one 30 fps video frame (2x slow motion)."""

    def __init__(self, b, path, seconds=3.5, start_z=0.07, w=1280, h=720, dist=0.16, height=0.03):
        import imageio
        import pyglet
        pyglet.options["headless"] = True                    # EGL, no X server (must precede the GL import)
        from newton.viewer import ViewerGL
        self.b, self.path, self.seconds, self.start_z, self.dist, self.height = b, path, seconds, start_z, dist, height
        self.v = ViewerGL(width=w, height=h, headless=True)
        self.v.set_model(b.model)
        self.writer = imageio.get_writer(path, fps=30, codec="libx264", quality=8, macro_block_size=8)
        self.t0, self.n = None, 0

    def __call__(self, b):
        tip_cmd = b.tip(commanded=True)
        if self.t0 is None:
            if tip_cmd[2] > self.start_z or b.t < 1.0:
                return
            self.t0 = b.t
        if b.t - self.t0 > self.seconds:
            return
        from PIL import Image, ImageDraw
        tip = b.tip()
        q = b.pose[1]
        yaw = 2 * math.atan2(q[2], q[3])
        side = np.array([math.sin(yaw), -math.cos(yaw), 0.0])         # the gripper's -y: along the jaw hinge
        cam = tip + side * self.dist + np.array([0.0, 0.0, self.height])
        self.v.camera.pos = wp.vec3(*cam)
        self.v.camera.look_at(wp.vec3(*(tip + np.array([0.0, 0.0, -0.004]))))
        self.v.begin_frame(b.t)
        self.v.log_state(b.s0)
        self.v.end_frame()
        img = Image.fromarray(self.v.get_frame().numpy())
        d = ImageDraw.Draw(img)
        lines = [f"t = {b.t - self.t0:4.2f} s (2x slow motion)",
                 f"moving jaw (orange): {math.degrees(b.jaw_angle()):5.1f} deg   squeeze cap {b.a.squeeze:.2f} N*m",
                 f"table pushes the gripper up {b.arm_give_mm():4.1f} mm (arm gives at most {b.a.arm_fz:.0f} N)",
                 f"friction jaw/cloth {b.a.mu_jaw}   cloth/cloth {b.a.mu_cloth}   cloth/table {b.a.mu_table}"]
        for i, line in enumerate(lines):
            d.text((14, 12 + 22 * i), line, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        self.writer.append_data(np.asarray(img))
        self.n += 1

    def close(self):
        self.writer.close()
        log(f"video: {self.n} frames -> {self.path}")


class Recorder:
    """Cloth + wrist + jaw every `every` s, saved like the Isaac cloth tests (int16, 0.1 mm)."""

    def __init__(self, every=0.1):
        self.every, self.next, self.rows = every, 0.0, []

    def __call__(self, b):
        if b.t + 1e-9 >= self.next:
            self.next += self.every
            self.rows.append((b.t, b.cloth(), np.r_[b.tip(), b.arm_give_mm(), b.press_n()], b.jaw_angle()))

    def save(self, path):
        if not self.rows:
            return
        fr = np.stack([r[1] for r in self.rows])
        o = fr[0].mean(axis=0)
        np.savez_compressed(path, t=np.array([r[0] for r in self.rows]), origin=o, scale=1e-4,
                            frames=np.clip(np.round((fr - o) / 1e-4), -32767, 32767).astype(np.int16),
                            wrist=np.stack([r[2] for r in self.rows]), jaw=np.array([r[3] for r in self.rows]))


def settle_test(b, a, rec):
    """oracle_fold.cloth_settle: leave the shirt alone `settle_s` s (the gripper parked 30 cm up)."""
    m = Mover(b, rec)
    frames, t1s, every = [], None, 0.5
    n = int(round(a.settle_s * b.fps))
    k = int(round(every * b.fps))
    t0 = time.time()
    for i in range(n + 1):
        if i % k == 0:
            frames.append(b.cloth())
        if i == b.fps:
            t1s = b.cloth()
        if i < n:
            m.step()
    wall = time.time() - t0
    v = frames[-1]
    if not np.isfinite(v).all():
        return dict(error="nan while settling")
    prev = frames[-1 - int(round(1.0 / every))]
    sp = np.linalg.norm(v - prev, axis=1) * 1000
    table_z = float(np.percentile(v[:, 2], 1)) - 0.002
    out = dict(jitter_mm_s=round(float(sp.mean()), 3), jitter_p95_mm_s=round(float(np.percentile(sp, 95)), 3),
               footprint_change_pct=round((footprint_m2(v) - footprint_m2(t1s)) / footprint_m2(t1s) * 100, 2),
               footprint_m2=round(footprint_m2(v), 4),
               height_spread_mm=round(float((np.percentile(v[:, 2], 95) - np.percentile(v[:, 2], 5)) * 1000), 2),
               lifted_over_15mm_pct=round(float((v[:, 2] > table_z + 0.015).mean() * 100), 2))
    for tag, dz_min in (("", 0.002), ("_thin", 0.0004)):       # Isaac's layer filter (2-10 mm) and one for thin layers
        top, bot = stacked_pairs(v, dz_min=dz_min)
        gap = (v[top, 2] - v[bot, 2]) * 1000 if len(top) else np.array([np.nan])
        out["layer_gap_mm_median" + tag] = round(float(np.median(gap)), 2)
        out["layer_pairs" + tag] = int(len(top))
    out["cloth_z_mm_p5_p50_p95"] = [round(float(x) * 1000, 2) for x in np.percentile(v[:, 2], [5, 50, 95])]
    out["sim_s"], out["wall_s"] = round(a.settle_s, 2), round(wall, 1)
    out["realtime_x"] = round(a.settle_s / wall, 3)
    return out


def jaws_test(b, a, rec):
    """Close the gripper on nothing, 5 cm above the table: where does the moving jaw stop?"""
    m = Mover(b, rec)
    m.move((0.0, 0.0, 0.05), 0.2)
    m.hold(0.3)
    cmd = GRIP_CLOSED if a.close is None else a.close
    m.hold(0.15, jaw=cmd, jaw_ramp=0.15)
    m.hold(1.0)
    return dict(jaw_contact=bool(a.jaw_contact), jaw_limit_deg=round(math.degrees(a.jaw_lower if a.jaw_contact
                                                                                   else JAW_RANGE[0]), 2),
                command_deg=round(math.degrees(cmd), 2), jaw_deg_final=round(math.degrees(b.jaw_angle()), 2),
                squeeze_nm=round(b.squeeze_nm(cmd), 3))


def prep_test(b, a, rec):
    """Drop the 3D shirt, let it flatten, save the cloth points (the stress-free rest shape for --rest-from)."""
    m = Mover(b, rec)
    t0 = time.time()
    m.hold(a.settle_s)
    v = b.cloth()
    np.save(os.path.expanduser(a.rest_out), v)
    top, bot = stacked_pairs(v, dz_min=0.0002)
    return dict(saved=a.rest_out, points=len(v), layer_gap_mm=round(float(np.median(v[top, 2] - v[bot, 2]) * 1000), 2)
                if len(top) else None, z_mm_p5_p50_p95=[round(float(x) * 1000, 2) for x in np.percentile(v[:, 2], [5, 50, 95])],
                sim_s=a.settle_s, wall_s=round(time.time() - t0, 1))


def grip_test(b, a, rec):
    """oracle_fold.assay_trial(extra=True) on the left sleeve: slide in with the moving finger leading, close, lift,
    hold, carry 15 cm, let go, park. Speeds are the Isaac test's, converted from its 1/90 s ticks."""
    side = a.side
    m = Mover(b, rec)
    out = dict(speed_mm_step=round(a.speed * 1000, 2))
    t0 = time.time()
    m.hold(a.settle_first)                                     # the shirt drops and flattens (gripper 30 cm up)
    v0 = b.cloth()
    m.hold(61 * TICK)                                          # Isaac: creep measured over ticks 49 -> 110
    v1 = b.cloth()
    out["creep_mm_s"] = round(float(np.linalg.norm(v1 - v0, axis=1).mean() * 1000 / (61 * TICK)), 2)
    if not np.isfinite(v1).all():
        out["error"] = "nan at settle"
        return out
    table_z = float(np.percentile(v1[:, 2], 1)) - 0.002
    k = b.keys[side]
    cuff, pit = v1[k["cuff_mid"], :2], v1[k["armpit"], :2]
    axis = (pit - cuff) / np.linalg.norm(pit - cuff)
    if a.grasp == "pinch":
        start, yaw = pinch_spot(v1, k, a.inset)
        target = start
    elif a.grasp == "edge_scoop":
        # fixed finger presses the sleeve `inset` inside the cuff, the moving finger waits past the edge on the bare
        # table, then sweeps in along the table and pushes the edge up between the fingers
        start = target = cuff + axis * a.inset
        yaw = math.atan2(-axis[1], -axis[0])                   # moving finger (+x of the gripper) outside the cuff
    else:
        # the Isaac test: slide 3 cm in from 1 cm inside the cuff with the moving finger leading
        start, target = cuff + axis * 0.01, cuff + axis * 0.04
        yaw = math.atan2(axis[1], axis[0])                     # moving finger (+x of the gripper) leads the slide
    out["grasp"] = a.grasp
    z_hover = 0.5925 - 0.52                                    # fold_plans.Z_HOVER above the Isaac table
    # Isaac: tip lands 7 mm above the table, then presses to 1 mm below it (the table stops the fixed finger).
    # Here the arm aims the fixed finger `press_mm` below the table and its capped vertical give does the rest.
    z_land = b.finger_drop + a.land_mm / 1000
    z_press = b.finger_drop + a.press_mm / 1000
    ds = 0.0015 / TICK
    if a.grasp != "pinch":
        m.move((start[0], start[1], 0.30), ds, yaw=yaw)
    m.move((start[0], start[1], z_hover), ds, yaw=yaw)
    m.move((start[0], start[1], z_land), ds)
    m.move((target[0], target[1], z_press), ds)
    out["tip_z_landed_mm"] = round(float(b.tip()[2] - table_z) * 1000, 2)
    out["finger_above_table_mm"] = round(float(b.tip()[2] - b.finger_drop) * 1000, 2)
    out["arm_give_mm_landed"], out["press_n_landed"] = round(b.arm_give_mm(), 2), round(b.press_n(), 2)
    gc = GRIP_CLOSED if a.close is None else a.close
    m.hold(16 * TICK, jaw=gc, jaw_ramp=16 * TICK)
    m.hold(a.close_hold)
    tip = b.tip()
    vc = b.cloth()
    pinched = np.nonzero(np.linalg.norm(vc - tip, axis=1) < 0.02)[0]
    top, bot = stacked_pairs(v1, dz_min=a.layer_dz_min)
    near = np.linalg.norm(v1[top, :2] - target, axis=1) < 0.025
    ltop, lbot = top[near], bot[near]
    out["layer_pairs_near_grasp"] = int(near.sum())
    out["n_pinched"] = int(len(pinched))
    out["jaw_deg_closed"] = round(math.degrees(b.jaw_angle()), 2)
    out["squeeze_nm"] = round(b.squeeze_nm(gc), 3)
    out["arm_give_mm_closed"], out["press_n_closed"] = round(b.arm_give_mm(), 2), round(b.press_n(), 2)
    if len(pinched) == 0:
        out["error"] = "nothing pinched"
        return out

    def measure(tag):
        v = b.cloth()
        if not np.isfinite(v).all():
            out[tag] = "nan"
            return False
        t = b.tip()
        d = np.linalg.norm(v[pinched] - t, axis=1)
        out[tag] = dict(held=round(float((d < 0.03).mean()), 3),
                        height_cm=round(float((v[pinched, 2].mean() - table_z) * 100), 1),
                        jaw_deg=round(math.degrees(b.jaw_angle()), 2), squeeze_nm=round(b.squeeze_nm(gc), 3),
                        cloth_up_pct=round(float((v[:, 2] > table_z + 0.02).mean() * 100), 1))
        return True

    p0 = m.tip
    m.move((p0[0], p0[1], 0.60 - 0.52), 0.0007 / TICK)
    if not measure("after_lift"):
        return out
    if len(ltop):
        vz = b.cloth()[:, 2]
        out["lifted_top"] = round(float((vz[ltop] > table_z + 0.02).mean()), 3)
        out["lifted_bottom"] = round(float((vz[lbot] > table_z + 0.02).mean()), 3)
    m.hold(135 * TICK)
    if not measure("after_hold"):
        return out
    p1 = m.tip
    dx = 0.15 if side == "L" else -0.15
    m.move((p1[0] + dx, p1[1], p1[2]), a.speed / TICK)
    m.hold(30 * TICK)
    if not measure("after_carry"):
        return out
    m.hold(15 * TICK, jaw=GRIP_OPEN, jaw_ramp=TICK)
    p2 = m.tip
    m.move((p2[0], p2[1], 0.30), ds)
    v = b.cloth()
    out["stuck_after_release"] = int((np.linalg.norm(v - b.tip(), axis=1) < 0.03).sum())
    m.hold(90 * TICK)
    v = b.cloth()
    out["hanging_after_release"] = int((v[:, 2] > table_z + 0.05).sum())
    if len(top):
        out["layer_swaps_pct"] = round(float((v[top, 2] < v[bot, 2] - 0.0005).mean() * 100), 2)
    out["sim_s"], out["wall_s"] = round(b.t, 2), round(time.time() - t0, 1)
    out["realtime_x"] = round(b.t / (time.time() - t0), 3)
    return out


def snapshots(path, rec, title):
    """Side and top views of the recorded cloth at a few moments (quick visual check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = rec.rows
    if not rows:
        return
    picks = sorted(set(np.linspace(0, len(rows) - 1, 6).round().astype(int)))
    fig, ax = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.4))
    for c, i in enumerate(picks):
        t, v, w, q = rows[i]
        for r, (ix, iy, lab) in enumerate(((0, 2, "x-z (side)"), (0, 1, "x-y (top)"))):
            a = ax[r, c]
            a.scatter(v[::3, ix], v[::3, iy], s=0.3, c=v[::3, 2], cmap="viridis")
            a.plot([w[ix]], [w[iy]], "r+", ms=8)
            a.set_aspect("equal")
            a.set_title(f"t={t:.1f}s jaw={math.degrees(q):.0f}deg {lab}", fontsize=7)
            a.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["settle", "grip", "prep", "jaws"], default="grip")
    p.add_argument("--no-cloth", action="store_true", help="the gripper alone (jaws test)")
    p.add_argument("--subdivide", type=int, default=0, help="midpoint subdivisions of the shirt mesh (4x triangles each)")
    p.add_argument("--rest-from", default=None, help="settled shirt (prep output): rest shape and start")
    p.add_argument("--rest-out", default="~/wf/assets/shirt_flat.npy", help="prep: where to save the settled shirt")
    p.add_argument("--friction-eps", type=float, default=1e-2,
                   help="VBD friction smoothing (m/s): below this slip speed friction fades; smaller = truer sticking")
    p.add_argument("--jaw-contact", action="store_true", help="jaws collide and close until they touch")
    p.add_argument("--jaw-lower", type=float, default=-0.35, help="jaw joint limit with --jaw-contact (rad)")
    p.add_argument("--close-hold", type=float, default=25 * TICK, help="seconds to wait after the close ramp")
    p.add_argument("--out", default="results/newton")
    p.add_argument("--label", default=None)
    p.add_argument("--garment", default=os.path.join(HERE, "..", "oracle", "garments", "Top_Long_Seen_0.npz"))
    # cloth: 160 g/m^2 jersey; membrane and bending stiffness from jersey ranges (step 2 calibrates them on RGBench)
    p.add_argument("--density", type=float, default=0.16, help="kg/m^2")
    p.add_argument("--tri-ke", type=float, default=400.0)
    p.add_argument("--tri-ka", type=float, default=400.0)
    p.add_argument("--tri-kd", type=float, default=1e-3)
    p.add_argument("--edge-ke", type=float, default=2e-3, help="N; per-hinge stiffness = edge_ke * edge length")
    p.add_argument("--edge-kd", type=float, default=5e-6)
    p.add_argument("--radius", type=float, default=0.00075, help="cloth half-thickness for body contact (m)")
    p.add_argument("--self-margin", type=float, default=0.001)
    p.add_argument("--self-gap", type=float, default=0.001)
    p.add_argument("--rest-exclusion", type=float, default=0.003)
    p.add_argument("--contact-ke", type=float, default=1e4, help="cloth side of every contact (N/m)")
    p.add_argument("--shape-ke", type=float, default=1e5, help="rigid side: jaws and table (N/m)")
    p.add_argument("--arm-kz", type=float, default=1.4e4, help="vertical give of the arm (N/m)")
    p.add_argument("--arm-fz", type=float, default=15.0, help="most the arm pushes down or up (N)")
    p.add_argument("--arm-kdz", type=float, default=40.0, help="N*s/m")
    p.add_argument("--contact-kd", type=float, default=1e-2)
    p.add_argument("--contact-gap", type=float, default=0.002)
    p.add_argument("--mu-cloth", type=float, default=0.5, help="cotton on cotton")
    p.add_argument("--mu-jaw", type=float, default=0.4, help="cotton on the printed jaw")
    p.add_argument("--mu-table", type=float, default=0.4, help="cotton on the table")
    p.add_argument("--squeeze", type=float, default=SERVO_MAX * 0.5, help="servo torque cap (N*m); LeRobot: 50 %%")
    p.add_argument("--close", type=float, default=None)
    p.add_argument("--servo-ke", type=float, default=20.0,
                   help="jaw drive stiffness (N*m/rad). The real servo is stiff (998) but saturated whenever it squeezes "
                        "or swings; with the clamped target VBD moves the jaw at most squeeze/ke per substep, so 998 "
                        "caps the jaw at ~0.9 rad/s. 20 keeps the same saturated squeeze without that limit.")
    p.add_argument("--servo-kd", type=float, default=SERVO_DAMPING,
                   help="jaw drive damping. MuJoCo applies kv=2.731 inside the torque limit; VBD applies kd outside it, "
                        "where 2.731 slows the jaw ~10x. The joint's passive 0.6 gives ~2.4 rad/s at 1.47 N*m.")
    p.add_argument("--particle-contacts-only", action="store_true")
    p.add_argument("--sdf-voxel", type=float, default=0.0005)
    p.add_argument("--body-contact-buffer", type=int, default=4096)
    p.add_argument("--substeps", type=int, default=10)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--self-detect", choices=["default", "pre_init"], default="default")
    p.add_argument("--no-graph", dest="graph", action="store_false")
    p.add_argument("--drop-gap", type=float, default=0.005)
    p.add_argument("--settle-s", type=float, default=10.0)
    p.add_argument("--settle-first", type=float, default=1.5)
    p.add_argument("--side", default="L")
    p.add_argument("--grasp", choices=["slide_in", "edge_scoop", "pinch"], default="slide_in",
                   help="slide_in = the Isaac test's move; edge_scoop = moving finger sweeps the cuff edge in; "
                        "pinch = straight down, press, close, lift")
    p.add_argument("--inset", type=float, default=0.015, help="edge_scoop: fixed finger this far inside the cuff (m)")
    p.add_argument("--speed", type=float, default=0.0015, help="carry speed per Isaac tick (m)")
    p.add_argument("--land-mm", type=float, default=0.7, help="fixed finger above the table at landing")
    p.add_argument("--press-mm", type=float, default=-3.0,
                   help="fixed finger height the arm aims for at the end of the slide; below 0 = pressing on the table")
    p.add_argument("--layer-dz-min", type=float, default=0.0004)
    p.add_argument("--record-every", type=float, default=0.1)
    p.add_argument("--video", action="store_true", help="close-up mp4 of the grasp (Newton's renderer, headless)")
    p.add_argument("--video-s", type=float, default=3.5)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    label = a.label or f"{a.test}_sq{a.squeeze:g}_mu{a.mu_jaw:g}"
    t0 = time.time()
    b = Bench(a)
    log(f"built in {time.time() - t0:.1f} s; finger_drop {b.finger_drop * 1000:.1f} mm")
    rec = Recorder(a.record_every)
    vid = Video(b, os.path.join(a.out, f"{label}.mp4"), seconds=a.video_s) if a.video else None
    both = (lambda bb: (rec(bb), vid(bb))) if vid else rec
    row = {"settle": settle_test, "prep": prep_test, "jaws": jaws_test, "grip": grip_test}[a.test](b, a, both)
    if vid:
        vid.close()
    row = dict(label=label, test=a.test, **row, cloth_g=round(b.cloth_mass() * 1000, 1) if b.n_cloth else 0,
               params={k: v for k, v in vars(a).items() if k not in ("out", "garment")})
    log("ROW " + json.dumps(row))
    with open(os.path.join(a.out, "rows.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    rec.save(os.path.join(a.out, f"{label}.npz"))
    snapshots(os.path.join(a.out, f"{label}.png"), rec, label)
    log(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

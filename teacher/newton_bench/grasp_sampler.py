#!/usr/bin/env python3
"""Batched grasp sampling on the honest SO-101 + LeHome shirt bench (the OmniReset recipe for cloth).

N copies of the grip_bench.py scene (shirt, table, kinematic arm, gripper on its capped vertical give, moving jaw on
its capped servo) in one Newton model, 1.2 m apart (VBD cloth self-contact does not separate worlds). Each copy
draws its own physics, randomized the way OmniReset randomizes (uwlab omnireset rl_state_cfg.py BaseEventCfg), and its
own grasp on the left sleeve. All copies run one timeline:
  settle 2 s -> hover over the grasp spot -> down to the press height -> slide -> close -> lift after a delay ->
  lift to 8 cm -> hold 1 s -> measure
Physics per copy: jaw-cloth friction U(0.3, 1.2) (OmniReset's gripper range), table-cloth U(0.3, 0.6) (its table
range), squeeze U(0.74, 1.47) N*m (LeRobot's gripper: 25 % overload torque .. 50 % torque cap of the 2.94 N*m servo).
Grasp per copy: spot along the sleeve, jaw direction, tilt, opening, press, slide, lift delay (see sample()).
Success: cloth held in the jaws and lifted after the hold. Also reports throughput (step 3 of the plan).

  python grasp_sampler.py --worlds 32 --batches 4 --out results/newton/grasps
"""
import argparse
import json
import math
import os
import time

import numpy as np
import warp as wp

import newton
import grip_bench as gb

SPACING = 1.2


@wp.kernel
def kinematic_poses(a: wp.array[wp.transform], b: wp.array[wp.transform], v: wp.array[wp.spatial_vector],
                    sub: wp.array[int], n_sub: int, q0: wp.array[int], qd0: wp.array[int],
                    joint_q: wp.array[float], joint_qd: wp.array[float]):
    w = wp.tid()
    t = float(sub[0] + 1) / float(n_sub)
    p = wp.lerp(wp.transform_get_translation(a[w]), wp.transform_get_translation(b[w]), t)
    r = wp.quat_slerp(wp.transform_get_rotation(a[w]), wp.transform_get_rotation(b[w]), t)
    for i in range(3):
        joint_q[q0[w] + i] = p[i]
    for i in range(4):
        joint_q[q0[w] + 3 + i] = r[i]
    for i in range(6):
        joint_qd[qd0[w] + i] = v[w][i]


@wp.kernel
def bump(sub: wp.array[int]):
    sub[0] = sub[0] + 1


@wp.kernel
def drives(body_q: wp.array[wp.transform], arm: wp.array[int], g: wp.array[int], j: wp.array[int],
           x_pj: wp.transform, cmd: wp.array[float], cap: wp.array[float], tgt: wp.array[int],
           cap_z: wp.array[float], tgt_z: wp.array[int], joint_target_q: wp.array[float], q_out: wp.array[float]):
    """grip_bench.drives for every copy (saturating jaw servo, capped give along the arm's axis)."""
    w = wp.tid()
    anchor = wp.transform_multiply(body_q[g[w]], x_pj)
    rel = wp.quat_inverse(wp.transform_get_rotation(anchor)) * wp.transform_get_rotation(body_q[j[w]])
    q = 2.0 * wp.atan2(rel[2], rel[3])
    joint_target_q[tgt[w]] = q + wp.clamp(cmd[w] - q, -cap[w], cap[w])
    xa = body_q[arm[w]]
    d = wp.dot(wp.transform_get_translation(body_q[g[w]]) - wp.transform_get_translation(xa),
               wp.quat_rotate(wp.transform_get_rotation(xa), wp.vec3(0.0, 0.0, 1.0)))
    joint_target_q[tgt_z[w]] = d + wp.clamp(-d, -cap_z[w], cap_z[w])
    q_out[2 * w] = q
    q_out[2 * w + 1] = d


def rot(yaw, tilt):
    """Gripper orientation: tilt about its own y axis (the jaw hinge direction), then yaw about the vertical."""
    cy, sy, ct, st = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    return rz @ ry


def quat_of(r):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return (x, y, z, w)


def moving_jaw_g(moving_v_j, q):
    """Moving jaw vertices in the gripper frame at jaw angle q."""
    rq = np.array([[math.cos(q), -math.sin(q), 0], [math.sin(q), math.cos(q), 0], [0, 0, 1]])
    rx90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    return (moving_v_j @ rq.T) @ rx90.T + np.array(gb.JAW_JOINT[0])


def sample(rng, n):
    """One grasp + physics draw per copy. Angles in rad, lengths in m, torque in N*m."""
    return dict(
        mu_jaw=rng.uniform(0.3, 1.2, n), mu_table=rng.uniform(0.3, 0.6, n), squeeze=rng.uniform(0.74, 1.47, n),
        along=rng.uniform(0.0, 0.06, n),            # grasp spot: this far in from the cuff along the sleeve
        across=rng.uniform(-0.012, 0.012, n),       # and this far across it
        yaw_rel=rng.uniform(-math.pi, math.pi, n),  # jaw direction (+x = moving finger) relative to cuff->armpit
        tilt=np.radians(rng.uniform(-25.0, 25.0, n)),
        open=rng.uniform(0.3, 1.0, n),              # jaw opening before the close
        press_mm=rng.uniform(-3.0, 3.0, n),         # lowest finger point vs the table; < 0 = pressing on it
        slide_x=rng.uniform(-0.03, 0.03, n),        # slide while low, in the gripper's own x / y
        slide_y=rng.uniform(-0.015, 0.015, n),
        lift_delay=rng.uniform(0.0, 0.5, n))


class Batch:
    def __init__(self, a, n):
        self.a, self.n = a, n
        self.fps, self.n_sub = 60, a.substeps
        self.frame_dt = 1.0 / self.fps
        self.dt = self.frame_dt / self.n_sub
        g = np.load(a.garment)
        rest = g["points"].astype(np.float64)
        self.keys = gb.gk.top_keys(rest, g["check_point"])
        verts = rest * gb.SCALE
        verts[:, 2] += a.drop_gap - verts[:, 2].min()
        self.n_cloth = len(verts)

        sub = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        cfg = lambda mu, ke: newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu, ke=ke, kd=a.contact_kd)
        sub.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.02), wp.quat_identity()), hx=0.5, hy=0.5,
                          hz=0.02, cfg=cfg(0.3, a.shape_ke), label="table")
        sub.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                           vertices=[wp.vec3(*p) for p in verts], indices=g["faces"].astype(np.int32).reshape(-1).tolist(),
                           density=a.density, tri_ke=a.tri_ke, tri_ka=a.tri_ka, tri_kd=a.tri_kd,
                           edge_ke=a.edge_ke, edge_kd=a.edge_kd, particle_radius=a.radius)
        fixed_mesh, fixed_v = gb.load_mesh("wrist_roll_follower_so101_v1", a.sdf_voxel)
        moving_mesh, moving_v = gb.load_mesh("moving_jaw_so101_v1", a.sdf_voxel)
        arm = sub.add_link(xform=wp.transform_identity(), inertia=gb.inertia(1e-4, 1e-4, 1e-4, 0, 0, 0), mass=0.1,
                           is_kinematic=True, label="arm")
        grip = sub.add_link(xform=wp.transform_identity(), com=wp.vec3(*gb.GRIPPER_COM), inertia=gb.inertia(*gb.GRIPPER_I),
                            mass=gb.GRIPPER_MASS, label="gripper")
        sub.add_shape_mesh(grip, xform=gb.xf(gb.FIXED_JAW_GEOM), mesh=fixed_mesh, cfg=cfg(0.3, a.shape_ke), label="fixed_jaw")
        jaw = sub.add_link(xform=wp.transform_identity(), com=wp.vec3(*gb.JAW_COM), inertia=gb.inertia(*gb.JAW_I),
                           mass=gb.JAW_MASS, label="moving_jaw")
        sub.add_shape_mesh(jaw, xform=gb.xf(gb.MOVING_JAW_GEOM), mesh=moving_mesh, cfg=cfg(0.3, a.shape_ke), label="moving_jaw")
        j_free = sub.add_joint_free(arm, label="arm")
        j_z = sub.add_joint_prismatic(parent=arm, child=grip, axis=(0.0, 0.0, 1.0), target_pos=0.0, target_ke=a.arm_kz,
                                      target_kd=a.arm_kdz, limit_lower=-0.03, limit_upper=0.03, label="arm_give")
        j_jaw = sub.add_joint_revolute(parent=grip, child=jaw, parent_xform=gb.xf(gb.JAW_JOINT), axis=(0.0, 0.0, 1.0),
                                       target_pos=gb.GRIP_OPEN, target_ke=a.servo_ke, target_kd=a.servo_kd,
                                       limit_lower=gb.JAW_RANGE[0], limit_upper=gb.JAW_RANGE[1], label="gripper")
        sub.add_articulation([j_free, j_z, j_jaw], label="so101_gripper")
        sub.joint_q[sub.joint_q_start[j_jaw]] = gb.GRIP_OPEN
        park = np.array([0.0, 0.0, 0.40]) - np.asarray(gb.TCP)
        sub.joint_q[sub.joint_q_start[j_free]:sub.joint_q_start[j_free] + 7] = list(park) + [0.0, 0.0, 0.0, 1.0]
        loc = dict(bodies=sub.body_count, shapes=sub.shape_count, joints=sub.joint_count, arm=arm, grip=grip, jaw=jaw,
                   j_free=j_free, j_z=j_z, j_jaw=j_jaw)

        side = int(math.ceil(math.sqrt(n)))
        self.centers = np.array([[(i % side) * SPACING, (i // side) * SPACING, 0.0] for i in range(n)])
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        scene.replicate(sub, n, xforms=[wp.transform(wp.vec3(*c), wp.quat_identity()) for c in self.centers])
        t0 = time.time()
        scene.color()
        gb.log(f"colored {n} copies in {time.time() - t0:.1f} s")
        self.model = scene.finalize()
        m = self.model
        m.edge_rest_angle.zero_()
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = a.contact_ke, a.contact_kd, a.mu_cloth

        bws, sws = m.body_world_start.numpy(), m.shape_world_start.numpy()
        jws, pws = m.joint_world_start.numpy(), m.particle_world_start.numpy()
        self.arm = np.array([bws[w] + loc["arm"] for w in range(n)])
        self.grip = np.array([bws[w] + loc["grip"] for w in range(n)])
        self.jaw = np.array([bws[w] + loc["jaw"] for w in range(n)])
        self.pstart = np.array([pws[w] for w in range(n)])
        assert all(pws[w + 1] - pws[w] == self.n_cloth for w in range(n - 1)), "particles per world"
        jq, jqd, jt = m.joint_q_start.numpy(), m.joint_qd_start.numpy(), m.joint_target_q_start.numpy()
        self.q0 = np.array([jq[jws[w] + loc["j_free"]] for w in range(n)])
        self.qd0 = np.array([jqd[jws[w] + loc["j_free"]] for w in range(n)])
        self.tgt = np.array([jt[jws[w] + loc["j_jaw"]] for w in range(n)])
        self.jq_jaw = np.array([jq[jws[w] + loc["j_jaw"]] for w in range(n)])
        self.jq_z = np.array([jq[jws[w] + loc["j_z"]] for w in range(n)])
        self.tgt_z = np.array([jt[jws[w] + loc["j_z"]] for w in range(n)])
        self.shape_table = np.array([sws[w] + 0 for w in range(n)])
        self.shape_jaws = np.array([[sws[w] + 1, sws[w] + 2] for w in range(n)])

        # geometry for heights: gripper-frame vertices of both fingers (moving one at its opening angle)
        rx180 = np.diag([1.0, -1.0, -1.0])
        self.fixed_g = fixed_v @ rx180.T + np.array(gb.FIXED_JAW_GEOM[0])
        self.moving_v_j = moving_v + np.array(gb.MOVING_JAW_GEOM[0])
        self.loc = loc

    def set_physics(self, s):
        """Per-copy friction (VBD mixes cloth and shape friction as sqrt(mu_cloth * mu_shape))."""
        a, m = self.a, self.model
        mu = m.shape_material_mu.numpy()
        mu[self.shape_table] = s["mu_table"] ** 2 / a.mu_cloth
        mu[self.shape_jaws[:, 0]] = s["mu_jaw"] ** 2 / a.mu_cloth
        mu[self.shape_jaws[:, 1]] = s["mu_jaw"] ** 2 / a.mu_cloth
        m.shape_material_mu = wp.array(mu, dtype=m.shape_material_mu.dtype, device=m.device)

    def make_solver(self, s, cloth0=None, arm_poses=None):
        """Fresh solver and state. cloth0: a settled shirt (local coords) to start every copy from; arm_poses:
        (n, 7) world arm poses to start from (default: parked 40 cm up); the jaw starts at each copy's opening."""
        a, m, n, dev = self.a, self.model, self.n, self.model.device
        self.set_physics(s)
        jq = m.joint_q.numpy()
        for w in range(n):
            if arm_poses is None:                               # parked 40 cm over its own copy
                jq[self.q0[w]:self.q0[w] + 7] = list(self.centers[w] + np.array([0.0, 0.0, 0.40]) - np.asarray(gb.TCP)) + \
                    [0.0, 0.0, 0.0, 1.0]
            else:
                jq[self.q0[w]:self.q0[w] + 7] = arm_poses[w]
            jq[self.jq_jaw[w]] = s["open"][w]
            jq[self.jq_z[w]] = 0.0
        m.joint_q.assign(jq)
        newton.eval_fk(m, m.joint_q, m.joint_qd, m)
        self.pipe = newton.CollisionPipeline(m, soft_contact_gap=a.contact_gap, include_static_kinematic_pairs=False,
                                             enable_rigid_soft_full_surface_contact=True)
        self.contacts = self.pipe.contacts()
        kw = {}
        if getattr(a, "self_detect", "default") == "pre_init":     # self-contact detection once per substep
            S = newton.solvers.SolverBase                              # (as Newton's cloth_franka example)
            kw["collision_frequency_type"] = {S.CollisionSlot.SOFT_SELF_CONTACT: S.CollisionFrequencyType.PRE_INIT}
        self.solver = newton.solvers.SolverVBD(
            m, iterations=a.iterations, rigid_compliant_alm=True, particle_enable_self_contact=True,
            particle_self_contact_margin=a.self_margin, particle_self_contact_gap=a.self_gap,
            particle_rest_shape_contact_exclusion_radius=a.rest_exclusion,
            rigid_body_particle_contact_buffer_size=a.body_contact_buffer, **kw)
        self.s0, self.s1 = m.state(), m.state()
        self.control = m.control()
        newton.eval_fk(m, m.joint_q, m.joint_qd, self.s0)
        newton.eval_fk(m, m.joint_q, m.joint_qd, self.s1)
        if cloth0 is not None:                                  # the model's particle_q stays the rest shape
            q = self.s0.particle_q.numpy()
            for w in range(n):
                q[self.pstart[w]:self.pstart[w] + self.n_cloth] = cloth0 + self.centers[w]
            for st in (self.s0, self.s1):
                st.particle_q.assign(q)
                st.particle_qd.zero_()
        self.poses = np.array([jq[self.q0[w]:self.q0[w] + 7] for w in range(n)], dtype=np.float64)
        arr = lambda x, dt=int: wp.array(np.asarray(x), dtype=dt, device=dev)
        self.w_arm, self.w_grip, self.w_jaw = arr(self.arm), arr(self.grip), arr(self.jaw)
        self.w_q0, self.w_qd0, self.w_tgt, self.w_tgt_z = arr(self.q0), arr(self.qd0), arr(self.tgt), arr(self.tgt_z)
        self.cap = arr(s["squeeze"] / a.servo_ke, float)
        self.cap_z = arr(np.full(n, a.arm_fz / a.arm_kz), float)
        self.cmd = arr(np.full(n, gb.GRIP_OPEN), float)
        self.qout = wp.zeros(2 * n, dtype=float, device=dev)
        self.pose_a = wp.zeros(n, dtype=wp.transform, device=dev)
        self.pose_b = wp.zeros(n, dtype=wp.transform, device=dev)
        self.vel = wp.zeros(n, dtype=wp.spatial_vector, device=dev)
        self.sub = wp.zeros(1, dtype=int, device=dev)
        self.x_pj = gb.xf(gb.JAW_JOINT)
        self.graph = None
        if a.graph and dev.is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def simulate(self):
        wp.launch(gb.reset_counter, dim=1, inputs=[self.sub])
        self.solver.rebuild_bvh(self.s0)
        for _ in range(self.n_sub):
            self.s0.clear_forces()
            wp.launch(kinematic_poses, dim=self.n, inputs=[self.pose_a, self.pose_b, self.vel, self.sub, self.n_sub,
                                                           self.w_q0, self.w_qd0, self.s0.joint_q, self.s0.joint_qd])
            wp.launch(bump, dim=1, inputs=[self.sub])
            newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0,
                           body_flag_filter=int(newton.BodyFlags.KINEMATIC))
            wp.launch(drives, dim=self.n, inputs=[self.s0.body_q, self.w_arm, self.w_grip, self.w_jaw, self.x_pj,
                                                  self.cmd, self.cap, self.w_tgt, self.cap_z, self.w_tgt_z,
                                                  self.control.joint_target_q, self.qout])
            self.pipe.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
            self.s0, self.s1 = self.s1, self.s0

    def frame(self, poses_next, jaw_cmd):
        """poses_next: (n, 7) arm poses [p, q] at the end of this frame; jaw_cmd: (n,) servo commands."""
        poses_now = self.poses
        self.pose_a.assign(poses_now.astype(np.float32))
        self.pose_b.assign(poses_next.astype(np.float32))
        v = np.zeros((self.n, 6), np.float32)
        v[:, :3] = (poses_next[:, :3] - poses_now[:, :3]) / self.frame_dt
        self.vel.assign(v)
        self.cmd.assign(np.asarray(jaw_cmd, np.float32))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.poses = poses_next

    def settle(self, s, seconds):
        """Drop the 3D shirt and let it flatten in every copy (arms parked high); returns copy 0's shirt."""
        self.make_solver(s)
        jaw = np.asarray(s["open"])
        for _ in range(int(round(seconds * self.fps))):
            self.frame(self.poses.copy(), jaw)
        return self.cloth()[0]

    def cloth(self):
        q = self.s0.particle_q.numpy()
        return np.stack([q[p:p + self.n_cloth] for p in self.pstart]).astype(np.float64) - self.centers[:, None, :]

    def tips(self):
        """Tip frames (the Isaac tests' tcp) of every copy, local to each copy."""
        bq = self.s0.body_q.numpy()[self.grip]
        out = np.zeros((self.n, 3))
        for w in range(self.n):
            r = quat_to_mat(bq[w, 3:7])
            out[w] = bq[w, :3] + r @ np.asarray(gb.TCP) - self.centers[w]
        return out

    def jaws(self):
        q = self.qout.numpy()
        return q[0::2], q[1::2]


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def run_batch(B, s, v1, rng_seed, out_dir, tag, record=4):
    """One synchronized episode in every copy, from the settled shirt v1 (local coords) with each arm already over
    its grasp spot (OmniReset-style reset). Returns per-copy rows, wall seconds, sim seconds."""
    a, n = B.a, B.n
    tcp = np.asarray(gb.TCP)
    jaw = np.array(s["open"])
    fr = 0

    def step(tips_local, rots, jaw_cmd):
        nonlocal fr
        nxt = np.zeros((n, 7))
        for w in range(n):
            nxt[w, :3] = B.centers[w] + tips_local[w] - rots[w] @ tcp
            nxt[w, 3:] = quat_of(rots[w])
        B.frame(nxt, jaw_cmd)
        fr += 1
        if record and fr % 6 == 0:
            rec.append((fr / B.fps, B.cloth()[:record].astype(np.float32), B.tips()[:record], B.jaws()[0][:record]))

    rec = []
    table_z = 0.0
    # each copy's grasp on the settled shirt
    plan = []
    for w in range(n):
        k = B.keys["L"]
        cuff, pit = v1[k["cuff_mid"], :2], v1[k["armpit"], :2]
        ax = (pit - cuff) / np.linalg.norm(pit - cuff)
        perp = np.array([-ax[1], ax[0]])
        spot = cuff + ax * s["along"][w] + perp * s["across"][w]
        yaw = math.atan2(ax[1], ax[0]) + s["yaw_rel"][w]
        r = rot(yaw, s["tilt"][w])
        sl = r @ np.array([s["slide_x"][w], s["slide_y"][w], 0.0])
        sl[2] = 0.0
        # lowest point of either finger (moving one at its opening) below the tip frame, in the tilted pose
        pts = np.vstack([B.fixed_g, moving_jaw_g(B.moving_v_j, s["open"][w])]) @ r.T
        drop = (r @ tcp)[2] - pts[:, 2].min()
        z_press = table_z + drop + s["press_mm"][w] / 1000
        p0 = np.r_[spot - sl[:2] / 2, z_press]
        p1 = np.r_[spot + sl[:2] / 2, z_press]
        plan.append(dict(r=r, p0=p0, p1=p1, hover=np.r_[p0[:2], 0.0725]))
    rots = [p["r"] for p in plan]
    start = np.zeros((n, 7))
    for w in range(n):
        start[w, :3] = B.centers[w] + plan[w]["hover"] - rots[w] @ tcp
        start[w, 3:] = quat_of(rots[w])
    B.make_solver(s, cloth0=v1, arm_poses=start)
    t0 = time.time()
    # each phase: interpolate tips over the given seconds
    def phase(get_tip, seconds, jaw_fn):
        nf = max(1, int(round(seconds * B.fps)))
        for i in range(1, nf + 1):
            u = i / nf
            step(np.array([get_tip(w, u) for w in range(n)]), rots, jaw_fn(u))

    lerp = lambda x, y, u: x + (y - x) * u
    phase(lambda w, u: plan[w]["hover"], 0.1, lambda u: jaw)                            # let the reset settle
    phase(lambda w, u: lerp(plan[w]["hover"], plan[w]["p0"], u), 0.6, lambda u: jaw)    # down to the press height
    phase(lambda w, u: lerp(plan[w]["p0"], plan[w]["p1"], u), 0.3, lambda u: jaw)       # slide
    closed = np.full(n, gb.GRIP_CLOSED)
    top = np.array([p["p1"] for p in plan])
    lift_to = top.copy()
    lift_to[:, 2] = 0.08
    t_close = 0.0
    lift_s = (0.08 - top[:, 2]) / 0.06                       # 6 cm/s, the Isaac test's lift speed
    total = 0.5 + float(lift_s.max())
    nf = int(round(total * B.fps))
    for i in range(1, nf + 1):
        tt = i / B.fps
        jc = np.where(tt < 0.15, jaw + (closed - jaw) * tt / 0.15, closed)
        u = np.clip((tt - s["lift_delay"]) / lift_s, 0.0, 1.0)
        step(top + (lift_to - top) * u[:, None], rots, jc)
    after_lift = B.cloth()
    for _ in range(int(1.0 * B.fps)):                        # hold 1 s at the top
        step(lift_to, rots, closed)
    v = B.cloth()
    tips = B.tips()
    qj, give = B.jaws()
    wall = time.time() - t0
    sim_s = fr / B.fps
    rows = []
    top_i, bot_i = gb.stacked_pairs(v1, dz_min=0.0004)
    for w in range(n):
        d = np.linalg.norm(v[w] - tips[w], axis=1)
        up = v[w, :, 2] > table_z + 0.03
        near = np.linalg.norm(v1[top_i, :2] - (plan[w]["p0"][:2] + plan[w]["p1"][:2]) / 2, axis=1) < 0.025
        held = int((d < 0.03).sum())
        lifted = int(up.sum())
        row = {k_: (float(s[k_][w]) if k_ not in ("tilt", "yaw_rel") else round(math.degrees(float(s[k_][w])), 1))
               for k_ in s}
        row.update(world=w, seed=rng_seed, batch=tag, held=held, lifted=lifted,
                   lifted_after_lift=int((after_lift[w, :, 2] > table_z + 0.03).sum()),
                   success=bool(held >= a.min_held and lifted >= a.min_lifted),
                   top_up=round(float((v[w, top_i[near], 2] > 0.02).mean()), 3) if near.any() else None,
                   bottom_up=round(float((v[w, bot_i[near], 2] > 0.02).mean()), 3) if near.any() else None,
                   jaw_deg_end=round(math.degrees(float(qj[w])), 2), give_mm_end=round(float(give[w]) * 1000, 2),
                   nan=bool(not np.isfinite(v[w]).all()))
        rows.append(row)
    if record:
        np.savez_compressed(os.path.join(out_dir, f"{tag}_rec.npz"), t=np.array([r[0] for r in rec]),
                            cloth=np.stack([r[1] for r in rec]), tips=np.stack([r[2] for r in rec]),
                            jaw=np.stack([r[3] for r in rec]))
    return rows, wall, sim_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--batches", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.expanduser("~/wf/results/newton/grasps"))
    p.add_argument("--garment", default=os.path.join(gb.HERE, "..", "oracle", "garments", "Top_Long_Seen_0.npz"))
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument("--min-held", type=int, default=10, help="cloth points within 3 cm of the tip after the hold")
    p.add_argument("--min-lifted", type=int, default=50, help="cloth points above 3 cm after the hold")
    p.add_argument("--no-graph", dest="graph", action="store_false")
    p.add_argument("--throughput-only", type=float, default=0.0, help="just time this many sim seconds and exit")
    p.add_argument("--self-detect", choices=["default", "pre_init"], default="pre_init")
    # the honest bench settings (grip_bench.py defaults)
    for k, v in (("density", 0.16), ("tri_ke", 400.0), ("tri_ka", 400.0), ("tri_kd", 1e-3), ("edge_ke", 2e-3),
                 ("edge_kd", 5e-6), ("radius", 0.00075), ("self_margin", 0.001), ("self_gap", 0.001),
                 ("rest_exclusion", 0.003), ("contact_ke", 1e4), ("shape_ke", 1e5), ("arm_kz", 1.4e4),
                 ("arm_fz", 15.0), ("arm_kdz", 40.0), ("contact_kd", 1e-2), ("contact_gap", 0.002),
                 ("mu_cloth", 0.5), ("servo_ke", 20.0), ("servo_kd", gb.SERVO_DAMPING), ("sdf_voxel", 0.0005),
                 ("body_contact_buffer", 4096), ("substeps", 6), ("iterations", 10), ("drop_gap", 0.005)):
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = p.parse_args()
    assert a.substeps % 2 == 0 or not a.graph, "graph capture swaps the two states each substep: use an even count"
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    B = Batch(a, a.worlds)
    gb.log(f"built {a.worlds} copies ({a.worlds * B.n_cloth} cloth points) in {time.time() - t0:.1f} s")
    rng = np.random.default_rng(a.seed)
    if a.throughput_only:
        s = sample(rng, a.worlds)
        B.make_solver(s)
        B.poses = np.zeros((a.worlds, 7))
        B.poses[:, :3] = B.centers + np.array([0.0, 0.0, 0.40]) - np.asarray(gb.TCP)
        B.poses[:, 6] = 1.0
        nf = int(a.throughput_only * B.fps)
        B.frame(B.poses.copy(), np.full(a.worlds, gb.GRIP_OPEN))      # warm up
        wp.synchronize()
        t1 = time.time()
        for _ in range(nf):
            B.frame(B.poses.copy(), np.full(a.worlds, gb.GRIP_OPEN))
        wp.synchronize()
        wall = time.time() - t1
        free, total = wp.get_device("cuda:0").free_memory, wp.get_device("cuda:0").total_memory
        row = dict(worlds=a.worlds, sim_s=a.throughput_only, wall_s=round(wall, 2),
                   realtime_x_per_copy=round(a.throughput_only / wall, 4),
                   copy_sim_s_per_wall_s=round(a.worlds * a.throughput_only / wall, 3),
                   gpu_mem_used_gb=round((total - free) / 1e9, 2))
        gb.log("THROUGHPUT " + json.dumps(row))
        with open(os.path.join(a.out, "throughput.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        return
    t1 = time.time()
    v1 = B.settle(sample(rng, a.worlds), a.settle_s)          # one shared flat shirt for every try
    gb.log(f"settled the shirt ({a.settle_s} s sim) in {time.time() - t1:.0f} s")
    np.save(os.path.join(a.out, "settled_shirt.npy"), v1)
    allrows = []
    for bi in range(a.batches):
        s = sample(rng, a.worlds)
        rows, wall, sim_s = run_batch(B, s, v1, a.seed, a.out, f"b{bi}", record=4 if bi == 0 else 0)
        allrows += rows
        ok = sum(r["success"] for r in rows)
        gb.log(f"BATCH {bi}: {ok}/{len(rows)} grasps held, sim {sim_s:.1f} s in {wall:.0f} s wall "
               f"({a.worlds * sim_s / wall:.2f} copy-sim-s per wall-s)")
        with open(os.path.join(a.out, "grasps.jsonl"), "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    ok = [r for r in allrows if r["success"]]
    gb.log(f"DONE {len(ok)}/{len(allrows)} held in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

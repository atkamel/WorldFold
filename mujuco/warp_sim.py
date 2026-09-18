"""ClothFoldEnv's physics as a batch of worlds on MuJoCo Warp (docs/warp_port.md, phase 1).

Same model as ClothFoldEnv (compile_model is reused unchanged), stepped for
nworld worlds at once. Only joint_delta action mode is implemented, which is the
mode both task wrappers use; the per-substep IK of ee_delta mode is not ported.
All per-world state lives in torch tensors that alias the Warp arrays, so
writing ctrl or reading xpos never copies.

    sim = WarpClothSim(nworld=1024)
    sim.reset(randomize=True, rng=np.random.default_rng(0))
    sim.step(actions)          # float[nworld, 14] in [-1, 1], 100 substeps
    sim.cloth_vertices()       # float[nworld, 121, 3]
"""
from __future__ import annotations

import os
import sys

import mujoco
import warp as wp

# WORLDFOLD_WARP_MEMPOOL=0 makes Warp use plain cudaMalloc instead of its stream-ordered pool.
# On the WATcloud 3090s a 1.26 GB pool allocation failed with 22 GiB reported free while
# cudaMalloc of 18 GiB succeeded in the same job type (scripts/warp_gpu_diag.py).
if os.environ.get("WORLDFOLD_WARP_MEMPOOL", "1") == "0":
    wp.config.enable_mempools_at_init = False
import mujoco_warp as mjw
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from sim_main import (ARM_JOINTS, ARM_TIMESTEP, CLOTH_COUNT, GRASP_CORNERS, GRASP_RADIUS, GRIPPER_CLOSED,  # noqa: E402
                      GRIPPER_OPEN, JOINT_DELTA_SCALE, QACC_LIMIT, SETTLE_STEPS, WORKSPACE_XY, compile_model)

# mujoco-warp 3.13.0 + warp-lang 1.17.0 on CUDA: the CCD kernel's occupancy query compiles a
# second module variant and then fails to find the kernel symbol in it ("CUDA error 500:
# named symbol not found" in _ccd_grid_size). Upstream fix: mujoco_warp PR #1675 (2026-09-17,
# unreleased). Until it ships, size the CCD grid the way the CPU branch does; the kernel
# strides over the real candidate count, so this only costs idle threads.
import mujoco_warp._src.collision_convex as _collision_convex  # noqa: E402
_collision_convex._ccd_grid_size = lambda kernel, naconmax, device: naconmax

BATCHED_FIELDS = ("eq_data", "body_mass", "geom_friction", "dof_damping")
# Warp's defaults (512 contacts and 512 constraint rows in total) overflow once
# the cloth lies on the table: 398 element contacts and 1,924 rows per world.
NCONMAX = 1024
NJMAX = 4096
# Convex-convex (CCD) contacts are only the arm geoms against the table and each other; the
# cloth's ~400 contacts per world take the flex path. mujoco_warp sizes eleven CCD work arrays
# by nccdmax x mesh degree (tens of KB per slot) and defaults nccdmax to nconmax, which at 1,024
# worlds x 2,048 contacts filled a 24 GB card before the first step.
NCCDMAX = 64


class WarpClothSim:
    PREFIXES = ("left_", "right_")

    def __init__(self, nworld, device=None, control_dt=0.05, grasp_corners=None, grasp_radius=GRASP_RADIUS,
                 nconmax=NCONMAX, njmax=NJMAX, nccdmax=NCCDMAX, spec_hook=None, timestep=ARM_TIMESTEP, solver=None,
                 use_graph=True):
        """timestep and solver ("newton" | "cg") override the model's options; the defaults
        are ClothFoldEnv's. Both exist for the phase 2 throughput sweep."""
        wp.init()
        self.device = wp.get_device(device) if device is not None else wp.get_device()
        self.nworld = nworld
        self.n_substeps = int(round(control_dt / timestep))
        self.grasp_corners = dict(GRASP_CORNERS if grasp_corners is None else grasp_corners)
        self.grasp_radius = grasp_radius

        self.mjm = compile_model(timestep, self.grasp_corners, spec_hook=spec_hook)
        if solver is not None:
            self.mjm.opt.solver = {"newton": mujoco.mjtSolver.mjSOL_NEWTON, "cg": mujoco.mjtSolver.mjSOL_CG}[solver]
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(self.mjm, batch_sizes={k: nworld for k in BATCHED_FIELDS})
            self.m.opt.warn_overflow = True
            self.d = mjw.put_data(self.mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax, nccdmax=nccdmax)

        d, m = self.d, self.m
        self.qpos, self.qvel, self.ctrl, self.qacc = map(wp.to_torch, (d.qpos, d.qvel, d.ctrl, d.qacc))
        self.xpos, self.xmat, self.site_xpos, self.site_xmat = map(wp.to_torch, (d.xpos, d.xmat, d.site_xpos, d.site_xmat))
        self.eq_active, self.overflow = wp.to_torch(d.eq_active), wp.to_torch(d.overflow)
        self.eq_data, self.body_mass, self.geom_friction, self.dof_damping = map(
            wp.to_torch, (m.eq_data, m.body_mass, m.geom_friction, m.dof_damping))
        self.tdev = self.qpos.device
        self._base = {name: t[0].clone() for name, t in
                      (("body_mass", self.body_mass), ("geom_friction", self.geom_friction), ("dof_damping", self.dof_damping))}

        mjm = self.mjm
        idx = lambda xs: torch.as_tensor(list(xs), dtype=torch.long, device=self.tdev)
        self._site_id, self._arm_qpos_adr, self._arm_dof_adr, self._arm_act_id = {}, {}, {}, {}
        self._weld_ids, self._corner_body, self._gripper_act, self._ctrl_low, self._ctrl_high = {}, {}, {}, {}, {}
        for p in self.PREFIXES:
            self._site_id[p] = mjm.site(f"{p}gripperframe").id
            joints = [mjm.joint(f"{p}{n}") for n in ARM_JOINTS]
            acts = [mjm.actuator(f"{p}{n}").id for n in ARM_JOINTS]
            self._arm_qpos_adr[p] = idx(j.qposadr[0] for j in joints)
            self._arm_dof_adr[p] = idx(j.dofadr[0] for j in joints)
            self._arm_act_id[p] = idx(acts)
            rng_ = torch.as_tensor(mjm.actuator_ctrlrange[acts], dtype=torch.float32, device=self.tdev)
            self._ctrl_low[p], self._ctrl_high[p] = rng_[:, 0], rng_[:, 1]
            self._weld_ids[p] = [mjm.equality(f"{p}weld_{v}").id for v in self.grasp_corners[p]]
            self._corner_body[p] = [mjm.body(f"cloth_{v}").id for v in self.grasp_corners[p]]
            self._gripper_act[p] = mjm.actuator(f"{p}gripper").id
        self.weld_mask = {p: None for p in self.PREFIXES}   # per-prefix allow-list of vertices, as in ClothFoldEnv
        self._gripper_closed = torch.zeros(nworld, len(self.PREFIXES), dtype=torch.bool, device=self.tdev)

        n_vert = CLOTH_COUNT * CLOTH_COUNT
        self._cloth_body_ids = idx(mjm.body(f"cloth_{i}").id for i in range(n_vert))
        self._corner_ids = self._cloth_body_ids[[0, CLOTH_COUNT - 1, (CLOTH_COUNT - 1) * CLOTH_COUNT, n_vert - 1]]
        self._table_geom_id = mjm.geom("table").id
        cloth_qpos, cloth_dof = [], []
        for i in range(mjm.njnt):
            if mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i) is None:   # cloth joints are the unnamed ones
                cloth_qpos.append(mjm.jnt_qposadr[i])
                cloth_dof.append(mjm.jnt_dofadr[i])
        self._cloth_qpos_x, self._cloth_qpos_y = idx(cloth_qpos[0::3]), idx(cloth_qpos[1::3])
        self._cloth_dof_adr = idx(cloth_dof)
        self.step_count = torch.zeros(nworld, dtype=torch.long, device=self.tdev)
        self.use_graph = use_graph
        self._graphs = {}

    # ---- physics ----------------------------------------------------------

    def _substeps(self, n):
        """n physics substeps. On CUDA the loop replays a captured graph of at most one control
        step: eager stepping runs the solver's convergence loop on the host and measured half
        the throughput (11.8k vs 22.7k world-substeps/s at 1,024 worlds). A graph of the whole
        2,000-substep settle is too large to instantiate, hence the per-control-step chunks.
        Any graph failure falls back to eager stepping."""
        if not (self.device.is_cuda and self.use_graph):
            for _ in range(n):
                mjw.step(self.m, self.d)
            return
        chunk = min(n, self.n_substeps)
        for size in (chunk, n % chunk) if n % chunk else (chunk,):
            if size not in self._graphs:
                try:
                    with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                        for _ in range(size):
                            mjw.step(self.m, self.d)
                    self._graphs[size] = capture.graph
                except RuntimeError as e:
                    return self._eager_fallback(n, e)
        try:   # Warp instantiates the graph executable on the first launch, so failures surface here
            for _ in range(n // chunk):
                wp.capture_launch(self._graphs[chunk])
            if n % chunk:
                wp.capture_launch(self._graphs[n % chunk])
        except RuntimeError as e:
            return self._eager_fallback(n, e)

    def _eager_fallback(self, n, err):
        print(f"warp_sim: CUDA graph failed ({err}); free {self.device.free_memory / 2**30:.1f} GiB; stepping eagerly", flush=True)
        self.use_graph, self._graphs = False, {}
        return self._substeps(n)

    def reset(self, cloth_pose=None, randomize=False, rng=None, domain=None):
        """Reset every world. cloth_pose: float[nworld, 2] xy offsets, or None.
        randomize samples per-world physics like ClothFoldEnv; domain fixes the
        scales instead ({"cloth_mass_scale": [...], "table_friction_scale": [...],
        "cloth_damping_scale": [...]}, each length nworld). Returns the scales used."""
        rng = np.random.default_rng() if rng is None else rng
        for name, base in self._base.items():
            getattr(self, name)[:] = base
        params = {}
        if randomize or domain is not None:
            if domain is None:
                domain = {k: rng.uniform(0.7, 1.3, self.nworld) for k in
                          ("cloth_mass_scale", "table_friction_scale", "cloth_damping_scale")}
            scale = lambda k: torch.as_tensor(np.asarray(domain[k], dtype=np.float32), device=self.tdev)
            self.body_mass[:, self._cloth_body_ids] *= scale("cloth_mass_scale")[:, None]
            self.geom_friction[:, self._table_geom_id] *= scale("table_friction_scale")[:, None]
            self.dof_damping[:, self._cloth_dof_adr] *= scale("cloth_damping_scale")[:, None]
            params = {k: np.asarray(domain[k], dtype=np.float32) for k in domain}
            if cloth_pose is None and randomize:
                cloth_pose = rng.uniform(-0.03, 0.03, size=(self.nworld, 2))
                params["cloth_offset_xy"] = cloth_pose
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.m, self.d)
        for p in self.PREFIXES:
            self.ctrl[:, self._gripper_act[p]] = GRIPPER_OPEN
            self.eq_active[:, self._weld_ids[p]] = False
        self._gripper_closed[:] = False
        if cloth_pose is not None:
            pose = torch.as_tensor(np.asarray(cloth_pose, dtype=np.float32), device=self.tdev)
            self.qpos[:, self._cloth_qpos_x] += pose[:, 0:1]
            self.qpos[:, self._cloth_qpos_y] += pose[:, 1:2]
        self._substeps(int(round(SETTLE_STEPS * ARM_TIMESTEP / self.mjm.opt.timestep)))   # 1.0 s regardless of timestep
        self.step_count[:] = 0
        return params

    def grasp_active(self, prefix):
        return self.eq_active[:, self._weld_ids[prefix]].any(dim=1)

    def set_gripper(self, prefix, command):
        """command: float[nworld]. Hysteresis and weld engagement as ClothFoldEnv.set_gripper."""
        col = self.PREFIXES.index(prefix)
        closed = self._gripper_closed[:, col]
        closed[command < -0.3] = True
        closed[command > 0.3] = False
        act = self._gripper_act[prefix]
        self.ctrl[:, act] = torch.where(closed, GRIPPER_CLOSED, GRIPPER_OPEN)
        site = self.site_xpos[:, self._site_id[prefix]]
        allowed = self.weld_mask.get(prefix)
        for vtx, eqid, body in zip(self.grasp_corners[prefix], self._weld_ids[prefix], self._corner_body[prefix]):
            if allowed is not None and vtx not in allowed:
                continue
            gap = torch.linalg.norm(site - self.xpos[:, body], dim=1)
            engage = closed & ~self.eq_active[:, eqid] & (gap < self.grasp_radius)
            if engage.any():
                b1, b2 = int(self.mjm.eq_obj1id[eqid]), int(self.mjm.eq_obj2id[eqid])
                offset_world = self.xpos[engage, b2] - self.xpos[engage, b1]
                offset_local = torch.einsum("nji,nj->ni", self.xmat[engage, b1], offset_world)   # R1^T @ offset
                self.eq_data[engage, eqid, 3:6] = offset_local
                self.eq_active[engage, eqid] = True
            self.eq_active[~closed, eqid] = False

    def apply_joint_delta(self, prefix, deltas):
        """deltas: float[nworld, 5] in [-1, 1]."""
        target = self.qpos[:, self._arm_qpos_adr[prefix]] + deltas * JOINT_DELTA_SCALE
        self.ctrl[:, self._arm_act_id[prefix]] = torch.clamp(target, self._ctrl_low[prefix], self._ctrl_high[prefix])

    def step(self, action):
        """action: float[nworld, 14] as ClothFoldEnv.step in joint_delta mode."""
        a = torch.as_tensor(action, dtype=torch.float32, device=self.tdev).clamp(-1.0, 1.0)
        self.set_gripper("left_", a[:, 6])
        self.set_gripper("right_", a[:, 13])
        self.apply_joint_delta("left_", a[:, 0:5])
        self.apply_joint_delta("right_", a[:, 7:12])
        self._substeps(self.n_substeps)
        self.step_count += 1

    # ---- readouts ---------------------------------------------------------

    def cloth_vertices(self):
        return self.xpos[:, self._cloth_body_ids]

    def corners(self):
        return self.xpos[:, self._corner_ids]

    def robot_state(self, prefix):
        """float[nworld, 15]: the collector's robot block (cloth_angles/data/fold_observation.py)."""
        return torch.cat([self.qpos[:, self._arm_qpos_adr[prefix]], self.qvel[:, self._arm_dof_adr[prefix]],
                          self.ctrl[:, self._gripper_act[prefix]][:, None], self.site_xpos[:, self._site_id[prefix]],
                          self.grasp_active(prefix)[:, None].float()], dim=1)

    def failed(self):
        """ClothFoldEnv._failed per world: cloth out of the workspace or the solver blowing up."""
        out = (self.cloth_vertices()[:, :, :2].abs() > WORKSPACE_XY).any(dim=(1, 2))
        return out | (self.qacc.abs().amax(dim=1) > QACC_LIMIT)

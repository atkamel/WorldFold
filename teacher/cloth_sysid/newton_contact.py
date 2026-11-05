"""Contact bench for the RL cloth: does Newton's contact give the forces Coulomb friction says it should, at the
settings the RL env will run?

The SO-101 jaw is two pads squeezing cloth, so every test is pads, cloth and a table:
  pinch    two 9 x 8 mm pads squeeze 1 or 2 cloth layers with a set force N; the free end is pulled with a slowly
           rising force. Coulomb says the cloth slips at F = 2*mu*N (two faces). Also gives the pad gap = how far
           cloth holds the jaws apart (compared with the jaw stop angle in public SO-101 logs).
  incline  a cloth patch on a pad with gravity tilted by theta: it must stick below tan(theta) = mu, slide above.
  drag     a 30 x 30 cm sheet on a table, one corner dragged 20 cm (same scene as zozo/ for the solver comparison).
  layers   two stacked 30 x 15 cm sheets, the top one pulled 5 cm (cloth on cloth; same scene as zozo/).

  python newton_contact.py --stage pinch|incline|drag|layers|all [--device cuda:0]
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
RES = os.path.join(HERE, "results", "contact")
HZ = 100.0
SETTINGS = {"full": (10, 10), "sub5_it2": (5, 2), "sub2_it5": (2, 5), "it50": (10, 50), "sub5_it5": (5, 5),
            "sub10_it5": (10, 5), "sub5_it10": (5, 10), "sub2_it10": (2, 10)}  # (substeps, iterations) per 1/100 s frame
EPS = (1e-2, 1e-3, 1e-4)                    # VBD friction smoothing (m/s): below this slip speed friction fades
MU_CLOTH = 0.6                              # cloth on cloth (model-wide soft_contact_mu)
G = 9.8
# cotton, fitted on Coltraro mocap (results/newton/cloth_fitted_rl.json, fabrics_full), bending B = 4.1e-6 N*m
CLOTH = {"density": 0.17, "tri_ke": 14792.0, "tri_ka": 9520.0, "tri_kd": 0.496, "B": 4.1e-6, "tau": 5e-4}
RADIUS = 0.00075                            # cloth half-thickness for body contact (m)
# Contact setup chosen from the 2026-10-17 diagnosis (diag*.py on the L4):
#   shape_ke = soft_ke: with a stiffer shape side (1e5 vs 1e4) friction came out ~2.1x the set mu, with equal ke ~1.2x
#   self_margin 3 mm: with 1 mm, the penetration-free truncation froze in-plane sliding (a frictionless patch stuck)
#   rest_excl 1 mm: the self-contact rest-shape exclusion must stay below the layer spacing; at 3 mm it dropped contact
#   between two sheets 2.5 mm apart, they merged and the truncation locked them together ("glued layers")
CONTACT = {"soft_ke": 1e4, "shape_ke": 1e4, "kd": 1e-2, "gap": 0.002, "self_margin": 0.003, "self_gap": 0.001,
           "rest_excl": 0.001}
TAG = [""]
MU_SCALE = {}                               # setting -> 1/kappa (measured by stage kappa; mu set = target / kappa)
CLOTH_SCALE = {}                            # setting -> 1/kappa_cloth (stage kappa_cloth; self-contact friction)
PATCH = [0.0]                               # drag: half-size of the pinned patch at the corner (m); 0 = one vertex
PAD = (0.0045, 0.004, 0.0025)               # pad half-extents (m): 9 x 8 mm face, like the SO-101 jaw tip
FULL_SURFACE = [True]                       # rigid-soft contact on edges/faces too (else particles only)
SELF = [True]                               # cloth self-contact (needed for layers; distorts friction, see notes)


def log(msg):
    print(f"[contact {time.strftime('%H:%M:%S')}] {msg}", flush=True)


SETTING = [None]


def cloth_mu():
    """soft_contact_mu actually set: cloth-on-cloth target divided by the self-contact excess (CLOTH_SCALE)."""
    return MU_CLOTH * CLOTH_SCALE.get(SETTING[0], 1.0)


def shape_mu(mu_pair):
    """VBD mixes cloth/shape friction as sqrt(mu_cloth * mu_shape); pick mu_shape so the pair gets mu_pair, after
    dividing by the measured excess kappa of the current setting (MU_SCALE, empty = uncorrected)."""
    m = mu_pair * MU_SCALE.get(SETTING[0], 1.0)
    return m ** 2 / cloth_mu()


def grid(nx, ny, h, x0, y0, z):
    """Flat sheet: nx x ny vertices (x fastest), spacing h, alternating diagonals."""
    xs, ys = np.meshgrid(x0 + h * np.arange(nx), y0 + h * np.arange(ny))
    v = np.stack([xs.ravel(), ys.ravel(), np.full(nx * ny, z)], 1)
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i
            tris += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return v, np.array(tris, np.int32)


def add_sheet(b, v, tris, h):
    """Cotton sheet; hinge stiffness from B (edge_ke = 3*B*L/A_pair, L ~ 1.14 h and A_pair = h^2 on this grid)."""
    p0 = len(b.particle_q)
    edge_ke = 3.0 * CLOTH["B"] * 1.14 * h / (h * h)
    b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                     vertices=[wp.vec3(*p) for p in v], indices=tris.reshape(-1).tolist(), density=CLOTH["density"],
                     tri_ke=CLOTH["tri_ke"], tri_ka=CLOTH["tri_ka"], tri_kd=CLOTH["tri_kd"], edge_ke=edge_ke,
                     edge_kd=CLOTH["tau"] * edge_ke, particle_radius=RADIUS)
    return np.arange(p0, p0 + len(v))


def static_box(b, center, half, mu_pair, shapes=True):
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=shape_mu(mu_pair), ke=CONTACT["shape_ke"], kd=CONTACT["kd"],
                                          has_shape_collision=shapes)
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(*center), wp.quat_identity()), hx=half[0], hy=half[1], hz=half[2],
                    cfg=cfg)


# ------------------------------------------------------------------------------------------------ kernels
@wp.kernel
def k_tick(t: wp.array[float], dt: float):
    t[0] = t[0] + dt


@wp.kernel
def k_pull(t: wp.array[float], t0: float, idx: wp.array[int], rate: wp.array[float], f: wp.array[wp.vec3]):
    """Pull force on the strip's end particles: rate * (t - t0) in +x (rate is per particle)."""
    j = wp.tid()
    s = wp.max(t[0] - t0, 0.0)
    f[idx[j]] = f[idx[j]] + wp.vec3(rate[j] * s, 0.0, 0.0)


@wp.kernel
def k_move(t: wp.array[float], t0: float, t1: float, idx: wp.array[int], p0: wp.array[wp.vec3], vel: wp.vec3,
           dt: float, q: wp.array[wp.vec3], prev: wp.array[wp.vec3]):
    """Pinned particles (mass 0) moved at vel from t0 to t1: end-of-substep target in q, start-of-substep in prev."""
    j = wp.tid()
    a = wp.clamp(t[0] - dt - t0, 0.0, t1 - t0)
    e = wp.clamp(t[0] - t0, 0.0, t1 - t0)
    prev[j] = p0[j] + vel * a
    q[idx[j]] = p0[j] + vel * e


@wp.kernel
def k_fix_prev(idx: wp.array[int], prev: wp.array[wp.vec3], q_prev: wp.array[wp.vec3]):
    j = wp.tid()
    q_prev[idx[j]] = prev[j]


@wp.kernel
def k_fix_prev_n(n: wp.array[int], idx: wp.array[int], prev: wp.array[wp.vec3], q_prev: wp.array[wp.vec3]):
    j = wp.tid()
    if j < n[0]:
        q_prev[idx[j]] = prev[j]


class Solver(newton.solvers.SolverVBD):
    """SolverVBD whose previous-position buffer holds the moved particles' start-of-substep positions."""
    drv = None

    def _initialize_particles(self, *args):          # (state_in, state_out, [contacts,] dt): 1.6 / 1.7
        super()._initialize_particles(*args)
        if self.drv is not None and len(self.drv) == 3:          # (idx, prev, active count): contact_grasp
            wp.launch(k_fix_prev_n, dim=len(self.drv[0]), inputs=[self.drv[2], self.drv[0], self.drv[1],
                                                                  self.particle_q_prev], device=self.device)
        elif self.drv is not None:
            wp.launch(k_fix_prev, dim=len(self.drv[0]), inputs=[self.drv[0], self.drv[1], self.particle_q_prev],
                      device=self.device)


class Scene:
    """One model holding many copies of a test. Subclasses build copies in `build`, then `run` steps it."""

    def __init__(self, setting, eps, device, gravity=(0.0, 0.0, -G)):
        self.n_sub, self.iters = SETTINGS[setting]
        SETTING[0] = setting
        self.eps, self.device = eps, wp.get_device(device)
        self.dt = 1.0 / HZ / self.n_sub
        b = newton.ModelBuilder(gravity=gravity)
        self.build(b)
        b.color(include_bending=True)
        self.model = b.finalize(device=self.device)
        self.model.edge_rest_angle.zero_()
        self.model.soft_contact_ke, self.model.soft_contact_kd = CONTACT["soft_ke"], CONTACT["kd"]
        self.model.soft_contact_mu = cloth_mu()
        if self.model.body_count:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.pipe = newton.CollisionPipeline(self.model, soft_contact_gap=CONTACT["gap"],
                                             include_static_kinematic_pairs=False,
                                             enable_rigid_soft_full_surface_contact=FULL_SURFACE[0])
        self.contacts = self.pipe.contacts()
        self.solver = Solver(self.model, iterations=self.iters, rigid_compliant_alm=True, friction_epsilon=eps,
                             particle_enable_self_contact=SELF[0], particle_self_contact_margin=CONTACT["self_margin"],
                             particle_self_contact_gap=CONTACT["self_gap"],
                             particle_rest_shape_contact_exclusion_radius=CONTACT["rest_excl"],
                             rigid_body_particle_contact_buffer_size=4096)
        self.control = self.model.control()
        self.s0, self.s1 = self.model.state(), self.model.state()
        if self.model.body_count:
            for s in (self.s0, self.s1):
                newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, s)
        self.t = wp.zeros(1, dtype=float, device=self.device)
        self.after_build()

    def build(self, b):
        raise NotImplementedError

    def after_build(self):
        pass

    def forces(self):
        pass

    def substep(self):
        self.s0.clear_forces()
        wp.launch(k_tick, dim=1, inputs=[self.t, self.dt], device=self.device)
        self.forces()
        self.pipe.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
        for name in ("particle_q", "particle_qd", "body_q", "body_qd"):      # s1 -> s0 (fixed buffers for the graph)
            dst = getattr(self.s0, name)
            if dst is not None and len(dst):
                wp.copy(dst, getattr(self.s1, name))

    def frame(self):
        self.solver.rebuild_bvh(self.s0)            # self-contact BVH once per frame (as grip_bench / cloth_franka)
        for _ in range(self.n_sub):
            self.substep()

    def run(self, seconds, record):
        """Advance `seconds`; calls record(frame) after every 1/100 s frame. One CUDA graph per frame."""
        n = int(round(seconds * HZ))
        graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture(device=self.device) as cap:
                self.frame()
            graph = cap.graph
        tic = time.perf_counter()
        for k in range(n):
            if graph is not None:
                wp.capture_launch(graph)
            else:
                self.frame()
            record(k)
        wp.synchronize_device(self.device)
        return time.perf_counter() - tic


# ------------------------------------------------------------------------------------------------ pinch
PINCH_MU = (0.10, 0.17, 0.25)               # PLA jaw on cotton (contact_params.json bracket)
PINCH_N = (1.0, 3.0, 6.0, 12.0, 18.0)       # squeeze (N): the SO-101 range from public logs + the BAM servo model
PINCH_LAYERS = [(0, 1, 2)]                  # layer counts per copy (2 needs self-contact)
PAD_MASS = 0.2                              # kg; the real jaw is heavier still through its gearbox (~5 kg at the tip)


class Pinch(Scene):
    """Copies: (mu, N, layers 1|2) plus empty pads (layers 0). Pads: bottom static, top on a vertical slider; its
    weight plus a drive spring of stiffness |N - m g| per metre aimed 1 m away (force within 0.1 % of constant, since
    q ~ mm) press with N in total. Newton clamps a drive target to the joint limits, so the limits are +-2 m."""
    H = 0.0015                                  # strip vertex spacing (m)
    SETTLE, RAMP = 0.3, 2.0                     # s

    def build(self, b):
        self.copies, self.pads = [], []
        self.under, self.ends, rates = [], [], []
        k = 0
        for layers in PINCH_LAYERS[0]:
            for mu in (PINCH_MU if layers else (PINCH_MU[1],)):
                for N in PINCH_N:
                    ox, oy = 0.15 * k, 0.0
                    k += 1
                    static_box(b, (ox, oy, -PAD[2]), PAD, mu, shapes=layers == 0)   # with cloth: only cloth holds them apart
                    z_top = 2 * RADIUS * layers + (CONTACT["self_gap"] if layers == 2 else 0.0) + 0.00005
                    body = b.add_link(xform=wp.transform(wp.vec3(ox, oy, z_top + PAD[2]), wp.quat_identity()),
                                      mass=PAD_MASS, inertia=wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 1e-8))
                    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=shape_mu(mu), ke=CONTACT["shape_ke"],
                                                          kd=CONTACT["kd"], has_shape_collision=layers == 0)
                    b.add_shape_box(body, hx=PAD[0], hy=PAD[1], hz=PAD[2], cfg=cfg)
                    j = b.add_joint_prismatic(parent=-1, child=body, axis=(0.0, 0.0, 1.0),
                                              parent_xform=wp.transform(wp.vec3(ox, oy, z_top + PAD[2]),
                                                                        wp.quat_identity()),
                                              target_pos=-math.copysign(1.0, N - PAD_MASS * G),
                                              target_ke=abs(N - PAD_MASS * G), target_kd=0.5,
                                              limit_lower=-2.0, limit_upper=2.0)
                    b.add_articulation([j])
                    under, ends = [], []
                    for L in range(layers):
                        z = RADIUS + L * (2 * RADIUS + CONTACT["self_gap"])
                        v, tris = grid(39, 9, self.H, ox - 0.006, oy - 0.006, z)
                        ids = add_sheet(b, v, tris, self.H)
                        under += list(ids[np.abs(v[:, 0] - ox) < PAD[0] - 0.0005])
                        ends += list(ids[v[:, 0] > v[:, 0].max() - 1e-6])
                    n_end = max(len(ends), 1)
                    # pull ramps to 2x the Coulomb slip force over RAMP seconds (slip expected half-way)
                    rate = 2.0 * (2.0 * mu * N) / self.RAMP / n_end
                    self.copies.append({"layers": layers, "mu": mu, "N": N, "body": body, "ox": ox,
                                        "F_coulomb": 2.0 * mu * N, "rate_total": rate * n_end})
                    self.under.append(np.array(under, int))
                    self.ends += ends
                    rates += [rate] * len(ends)
        self.end_idx, self.end_rate = np.array(self.ends, np.int32), np.array(rates, np.float32)

    def after_build(self):
        d = self.device
        self.w_idx = wp.array(self.end_idx, dtype=int, device=d)
        self.w_rate = wp.array(self.end_rate, dtype=float, device=d)

    def forces(self):
        if len(self.end_idx):
            wp.launch(k_pull, dim=len(self.end_idx), inputs=[self.t, self.SETTLE, self.w_idx, self.w_rate,
                                                             self.s0.particle_f], device=self.device)

    def measure(self):
        T = int(round((self.SETTLE + self.RAMP) * HZ))
        n_c = len(self.copies)
        slip = np.full((T, n_c), np.nan)
        gap = np.full(n_c, np.nan)
        x_start = {}

        def rec(k):
            q = self.s0.particle_q.numpy()
            bq = self.s0.body_q.numpy()
            t = (k + 1) / HZ
            for c, cp in enumerate(self.copies):
                if k + 1 == int(round(self.SETTLE * HZ)):
                    gap[c] = bq[cp["body"], 2] - PAD[2]                     # top pad face above the bottom pad face
                    if cp["layers"]:
                        x_start[c] = q[self.under[c], 0].copy()
                if cp["layers"] and c in x_start:
                    slip[k, c] = float(np.mean(q[self.under[c], 0] - x_start[c]))
        wall = self.run(self.SETTLE + self.RAMP, rec)
        rows = []
        for c, cp in enumerate(self.copies):
            r = {k: cp[k] for k in ("layers", "mu", "N", "F_coulomb")}
            r["gap_mm"] = round(float(gap[c]) * 1000, 4)
            if cp["layers"]:
                s = slip[:, c]
                hit = np.where(s > 0.001)[0]
                k = hit[0] if len(hit) else None
                F = cp["rate_total"] * ((k + 1) / HZ - self.SETTLE) if k is not None else float("nan")
                creep = s[int(round(self.SETTLE * HZ)) + int(0.25 * self.RAMP * HZ)]    # at F = 0.5 F_coulomb
                r.update({"F_slip": round(F, 4), "ratio": round(F / cp["F_coulomb"], 4),
                          "creep_mm_at_half_load": round(float(creep) * 1000, 4),
                          "finite": bool(np.isfinite(self.s0.particle_q.numpy()).all())})
            rows.append(r)
        return rows, wall


# ------------------------------------------------------------------------------------------------ incline
INCLINE_THETA = (8.0, 20.0, 35.0)           # deg: jaw-like, table-like and cloth-on-cloth-like friction
INCLINE_R = (0.8, 0.9, 0.95, 1.05, 1.1, 1.2)    # tan(theta) / mu


class Incline(Scene):
    """Gravity tilted by theta about y (down-slope = +x); copies differ in mu so tan(theta)/mu spans INCLINE_R."""
    H = 0.002

    def __init__(self, setting, eps, device, theta):
        self.theta = theta
        th = math.radians(theta)
        super().__init__(setting, eps, device, gravity=(G * math.sin(th), 0.0, -G * math.cos(th)))

    def build(self, b):
        self.copies, self.ids = [], []
        for k, r in enumerate(INCLINE_R):
            mu = math.tan(math.radians(self.theta)) / r
            oy = 0.12 * k
            static_box(b, (0.80, oy, -0.005), (1.00, 0.04, 0.005), mu)      # 2 m long: a frictionless patch stays on
            v, tris = grid(21, 21, self.H, -0.02, oy - 0.02, RADIUS)
            self.ids.append(add_sheet(b, v, tris, self.H))
            self.copies.append({"theta": self.theta, "r": r, "mu": round(mu, 4)})

    def measure(self, seconds=1.5):
        disp = np.zeros((int(round(seconds * HZ)), len(self.copies)))
        x0 = [self.model.particle_q.numpy()[i, 0].mean() for i in self.ids]

        def rec(k):
            q = self.s0.particle_q.numpy()
            for c, i in enumerate(self.ids):
                disp[k, c] = q[i, 0].mean() - x0[c]
        wall = self.run(seconds, rec)
        rows = []
        for c, cp in enumerate(self.copies):
            d = disp[:, c]
            late = d[-1] - d[int(0.5 * HZ)]                        # creep over the last second
            r = dict(cp, disp_mm=round(float(d[-1]) * 1000, 3), creep_mm_per_s=round(float(late) * 1000, 4))
            r["slides"] = bool(d[-1] > 0.002)
            r["expected"] = "slide" if cp["r"] > 1.0 else "stick"
            r["ok"] = r["slides"] == (cp["r"] > 1.0) and (r["slides"] or r["creep_mm_per_s"] < 0.1)
            rows.append(r)
        return rows, wall


def measure_kappa(setting, eps, device, theta=30.0, mus=(0.1, 0.15, 0.2, 0.25, 0.3)):
    """Effective friction of a sliding patch: from its acceleration a = g (sin - mu_eff cos) on a steep incline."""
    th = math.radians(theta)
    old = INCLINE_R
    globals()["INCLINE_R"] = tuple(math.tan(th) / m for m in mus)
    sc = Incline(setting, eps, device, theta)
    globals()["INCLINE_R"] = old
    T = 60
    xs = np.zeros((T, len(sc.ids)))

    def rec(k):
        q = sc.s0.particle_q.numpy()
        for c, i in enumerate(sc.ids):
            xs[k, c] = q[i, 0].mean()
    sc.run(T / HZ, rec)
    t = (np.arange(T) + 1) / HZ
    rows = []
    for c, m in enumerate(mus):
        a = 2 * np.polyfit(t[20:], xs[20:, c], 2)[0]
        mu_eff = (G * math.sin(th) - a) / (G * math.cos(th))
        rows.append({"mu_set": m, "mu_eff": round(float(mu_eff), 4), "kappa": round(float(mu_eff / m), 4)})
    return rows


class ClothIncline(Scene):
    """Self-contact friction: a 4 x 4 cm patch on a fixed sheet (all its vertices mass 0), gravity tilted by theta."""
    H = 0.002

    def __init__(self, setting, eps, device, theta):
        th = math.radians(theta)
        super().__init__(setting, eps, device, gravity=(G * math.sin(th), 0.0, -G * math.cos(th)))

    def build(self, b):
        vb, tb = grid(151, 26, self.H, -0.05, -0.025, 0.0)          # 30 x 5 cm fixed sheet
        bot = add_sheet(b, vb, tb, self.H)
        for i in bot:
            b.particle_mass[int(i)] = 0.0
        vt, tt = grid(21, 21, self.H, -0.02, -0.02, CONTACT["self_gap"] + 0.0002)
        self.ids = [add_sheet(b, vt, tt, self.H)]


def measure_kappa_cloth(setting, eps, device, theta=35.0, mus=(0.1, 0.2, 0.3)):
    """Effective cloth-on-cloth friction per set value (one model each: soft_contact_mu is model-wide). The fit
    window stays below the self-contact speed cap (0.425 (margin + gap) / dt)."""
    th = math.radians(theta)
    cap = 0.425 * (CONTACT["self_margin"] + CONTACT["self_gap"]) * HZ * SETTINGS[setting][0]
    rows = []
    for m in mus:
        old = (MU_CLOTH, dict(CLOTH_SCALE))
        globals()["MU_CLOTH"] = m
        CLOTH_SCALE.clear()
        sc = ClothIncline(setting, eps, device, theta)
        globals()["MU_CLOTH"] = old[0]
        CLOTH_SCALE.update(old[1])
        T = 30
        xs = np.zeros(T)

        def rec(k):
            xs[k] = sc.s0.particle_q.numpy()[sc.ids[0], 0].mean()
        sc.run(T / HZ, rec)
        t = (np.arange(T) + 1) / HZ
        v = np.gradient(xs, t)
        ok = (t > 0.04) & (v < 0.8 * cap)
        a = 2 * np.polyfit(t[ok], xs[ok], 2)[0] if ok.sum() > 4 else float("nan")
        mu_eff = (G * math.sin(th) - a) / (G * math.cos(th))
        rows.append({"mu_set": m, "mu_eff": round(float(mu_eff), 4), "kappa": round(float(mu_eff / m), 4),
                     "frames_fit": int(ok.sum()), "v_end": round(float(v[-1]), 3), "cap": round(cap, 3)})
    return rows


# ------------------------------------------------------------------------------------------------ drag, layers
class Sheets(Scene):
    """Shared by drag and layers: sheets on a table box, a few pinned particles moved at constant speed."""
    H = 0.01

    def after_build(self):
        d = self.device
        self.m_idx = wp.array(np.array(self.pins, np.int32), dtype=int, device=d)
        self.m_p0 = wp.array(self.model.particle_q.numpy()[self.pins], dtype=wp.vec3, device=d)
        self.m_prev = wp.zeros(len(self.pins), dtype=wp.vec3, device=d)
        self.solver.drv = (self.m_idx, self.m_prev)

    def forces(self):
        wp.launch(k_move, dim=len(self.pins), inputs=[self.t, self.t0, self.t1, self.m_idx, self.m_p0,
                                                      wp.vec3(*self.vel), self.dt, self.s0.particle_q, self.m_prev],
                  device=self.device)

    def trajectories(self, seconds):
        T = int(round(seconds * HZ))
        out = np.zeros((T + 1, self.model.particle_count, 3), np.float32)
        out[0] = self.model.particle_q.numpy()

        def rec(k):
            out[k + 1] = self.s0.particle_q.numpy()
        wall = self.run(seconds, rec)
        return out, wall


class Drag(Sheets):
    """ZOZO scene D: 0.30 m sheet (31 x 31) on the table, corner (0,0) moved +x 0.20 m at 0.10 m/s after 0.5 s."""
    MU_TABLE = (0.3, 0.5, 0.7)
    t0, t1, vel = 0.5, 2.5, (0.10, 0.0, 0.0)

    def build(self, b):
        self.pins, self.spans = [], []
        for k, mu in enumerate(self.MU_TABLE):
            oy = 0.6 * k
            static_box(b, (0.15, oy + 0.15, -0.005), (0.5, 0.25, 0.005), mu)
            v, tris = grid(31, 31, self.H, 0.0, oy, RADIUS)
            ids = add_sheet(b, v, tris, self.H)
            local = v - v[0]
            grip = ids[(np.abs(local[:, 0]) <= PATCH[0] + 1e-6) & (np.abs(local[:, 1]) <= PATCH[0] + 1e-6)]
            for i in grip:                          # PATCH 0: the corner vertex only (the zozo/ scene)
                b.particle_mass[int(i)] = 0.0
            self.pins += [int(i) for i in grip]
            self.spans.append((int(ids[0]), int(ids[-1]) + 1, mu))


class Layers(Sheets):
    """ZOZO scene F: two 0.30 x 0.15 m sheets stacked; the top sheet's min-x edge middle moved -x 0.05 m at
    0.05 m/s after 0.5 s; cloth-cloth 0.6, table 0.5."""
    t0, t1, vel = 0.5, 1.5, (-0.05, 0.0, 0.0)

    def build(self, b):
        static_box(b, (0.15, 0.075, -0.005), (0.5, 0.25, 0.005), 0.5)
        vb, tb = grid(31, 16, self.H, 0.0, 0.0, RADIUS)
        bot = add_sheet(b, vb, tb, self.H)
        vt, tt = grid(31, 16, self.H, 0.0, 0.0, 3 * RADIUS + CONTACT["self_gap"])
        top = add_sheet(b, vt, tt, self.H)
        mid = int(top[8 * 31])                     # row 8 (y = 0.08 m), column 0 (min x)
        b.particle_mass[mid] = 0.0
        self.pins = [mid]
        self.spans = [(int(bot[0]), int(bot[-1]) + 1, "bottom"), (int(top[0]), int(top[-1]) + 1, "top")]


# ------------------------------------------------------------------------------------------------ stages
def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def stage_pinch(device, settings, eps_list):
    out = []
    for s in settings:
        for e in eps_list:
            sc = Pinch(s, e, device)
            rows, wall = sc.measure()
            ok = [r for r in rows if r["layers"]]
            ratios = np.array([r["ratio"] for r in ok], float)
            worst = float(np.nanmax(np.abs(ratios - 1.0))) if np.isfinite(ratios).any() else float("nan")
            log(f"pinch {s} eps {e:g}: F_slip/2muN median {np.nanmedian(ratios):.3f}, worst |err| {worst:.3f}, "
                f"no-slip copies {int(np.isnan(ratios).sum())}, wall {wall:.1f} s")
            out.append({"setting": s, "eps": e, "wall_s": round(wall, 2), "worst_abs_err": round(worst, 4),
                        "rows": rows})
    jdump(out, os.path.join(RES, f"pinch{TAG[0]}.json"))
    return out


def stage_incline(device, settings, eps_list):
    out = []
    for s in settings:
        for e in eps_list:
            rows, wall = [], 0.0
            for th in INCLINE_THETA:
                r, w = Incline(s, e, device, th).measure()
                rows += r
                wall += w
            n_ok = sum(r["ok"] for r in rows)
            log(f"incline {s} eps {e:g}: {n_ok}/{len(rows)} right (stick below, slide above), wall {wall:.1f} s")
            out.append({"setting": s, "eps": e, "wall_s": round(wall, 2), "ok": n_ok, "n": len(rows), "rows": rows})
    jdump(out, os.path.join(RES, f"incline{TAG[0]}.json"))
    return out


ZOZO_MEMBRANE = {"tri_ke": 4000.0, "tri_ka": 2667.0}   # the zozo/ runs' plane-stress Lame constants (E h = 1e4 N/m)


def stage_sheets(device, settings, eps_list, cls, name, seconds, membrane="zozo"):
    """Same scenes as zozo/: membrane matched to ZOZO's (E h = 1e4 N/m, nu 0.25), or the fitted cotton."""
    if membrane == "zozo":
        CLOTH.update(ZOZO_MEMBRANE)
    out = []
    for s in settings:
        for e in eps_list:
            sc = cls(s, e, device)
            traj, wall = sc.trajectories(seconds)
            np.savez_compressed(os.path.join(RES, f"{name}{TAG[0]}_{s}_eps{e:g}.npz"), x=traj,
                                spans=np.array([sp[:2] for sp in sc.spans]), dt=1.0 / HZ)
            summ = []
            for p0, p1, tag in sc.spans:
                d = traj[-1, p0:p1] - traj[0, p0:p1]
                summ.append({"tag": tag, "centroid_dx_mm": round(float(d[:, 0].mean()) * 1000, 2),
                             "far_corner_dx_mm": round(float(d[-1, 0]) * 1000, 2)})
            log(f"{name} {s} eps {e:g}: {summ}, wall {wall:.1f} s")
            out.append({"setting": s, "eps": e, "wall_s": round(wall, 2), "summary": summ})
    jdump(out, os.path.join(RES, f"{name}{TAG[0]}.json"))
    return out


def stage_kappa(device, settings, eps_list):
    """Measure the friction excess kappa per setting (uncorrected), save MU_SCALE = 1/median kappa."""
    out, scale = [], {}
    MU_SCALE.clear()
    for s in settings:
        for e in eps_list:
            rows = measure_kappa(s, e, device)
            k = float(np.median([r["kappa"] for r in rows]))
            log(f"kappa {s} eps {e:g}: {[(r['mu_set'], r['mu_eff']) for r in rows]} -> median kappa {k:.3f}")
            out.append({"setting": s, "eps": e, "rows": rows, "kappa_median": round(k, 4)})
            scale[f"{s}|{e:g}"] = 1.0 / k
    jdump({"runs": out, "mu_scale": scale}, os.path.join(RES, f"kappa{TAG[0]}.json"))
    return scale


def stage_kappa_cloth(device, settings, eps_list):
    out, scale = [], {}
    for s_ in settings:
        for e in eps_list:
            rows = measure_kappa_cloth(s_, e, device)
            lim = 0.95 * math.tan(math.radians(35.0))
            good = [r["kappa"] for r in rows if np.isfinite(r["kappa"]) and r["mu_eff"] < lim]
            k = float(np.median(good)) if good else float("nan")
            log(f"kappa_cloth {s_} eps {e:g}: {[(r['mu_set'], r['mu_eff'], r['frames_fit']) for r in rows]} "
                f"-> median kappa {k:.3f}")
            out.append({"setting": s_, "eps": e, "rows": rows, "kappa_median": round(k, 4)})
            scale[f"{s_}|{e:g}"] = 1.0 / k if np.isfinite(k) else 1.0
    jdump({"runs": out, "cloth_scale": scale}, os.path.join(RES, f"kappa_cloth{TAG[0]}.json"))
    return scale


def load_cloth_scale(eps, tag):
    p = os.path.join(RES, f"kappa_cloth{tag}.json")
    if os.path.exists(p):
        sc = json.load(open(p))["cloth_scale"]
        CLOTH_SCALE.clear()
        CLOTH_SCALE.update({k.split("|")[0]: v for k, v in sc.items() if k.split("|")[1] == f"{eps:g}"})


def load_scale(eps, tag=None):
    p = os.path.join(RES, f"kappa{TAG[0] if tag is None else tag}.json")
    if os.path.exists(p):
        sc = json.load(open(p))["mu_scale"]
        MU_SCALE.clear()
        MU_SCALE.update({k.split("|")[0]: v for k, v in sc.items() if k.split("|")[1] == f"{eps:g}"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["kappa", "kappa_cloth", "pinch", "incline", "drag", "layers", "grasp", "all"],
                   default="all")
    p.add_argument("--kappa-tag", default=None, help="which kappa*.json corrects table friction (default: --tag)")
    p.add_argument("--cloth-kappa-tag", default=None, help="which kappa_cloth*.json corrects cloth-on-cloth friction")
    p.add_argument("--patch", type=float, default=0.0, help="drag: half-size of the pinned corner patch (m)")
    p.add_argument("--membrane", choices=["zozo", "cotton"], default="zozo", help="drag/layers sheet membrane")
    p.add_argument("--uncorrected", action="store_true", help="ignore kappa.json (mu set as is)")
    p.add_argument("--device", default=None)
    p.add_argument("--settings", default="full,sub5_it2,sub2_it5")
    p.add_argument("--eps", default=",".join(f"{e:g}" for e in EPS))
    p.add_argument("--particles-only", action="store_true", help="rigid-soft contact on particles only")
    p.add_argument("--no-self", action="store_true", help="cloth self-contact off (single-layer tests)")
    p.add_argument("--tag", default="", help="suffix for the result files")
    a = p.parse_args()
    settings = a.settings.split(",")
    eps_list = [float(e) for e in a.eps.split(",")]
    wp.init()
    FULL_SURFACE[0] = not a.particles_only
    SELF[0] = not a.no_self
    if a.no_self:
        PINCH_LAYERS[0] = (0, 1)
    TAG[0] = a.tag
    PATCH[0] = a.patch
    t0 = time.time()
    if a.stage in ("kappa", "all"):
        stage_kappa(a.device, settings, eps_list)
    if a.stage == "kappa_cloth":
        stage_kappa_cloth(a.device, settings, eps_list)
    if not a.uncorrected and a.stage not in ("kappa", "kappa_cloth"):
        load_scale(eps_list[0], a.kappa_tag)
        if a.cloth_kappa_tag is not None:
            load_cloth_scale(eps_list[0], a.cloth_kappa_tag)
        log(f"mu correction 1/kappa: table {MU_SCALE}, cloth {CLOTH_SCALE}")
    if a.stage in ("pinch", "all"):
        stage_pinch(a.device, settings, eps_list)
    if a.stage in ("incline", "all"):
        stage_incline(a.device, settings, eps_list)
    if a.stage in ("drag", "all"):
        stage_sheets(a.device, settings, eps_list, Drag, "drag", 3.5, a.membrane)
    if a.stage in ("layers", "all"):
        stage_sheets(a.device, settings, eps_list, Layers, "layers", 2.5, a.membrane)
    if a.stage == "grasp":
        log("the grasp tests run from contact_grasp.py (it sets this module's globals itself)")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

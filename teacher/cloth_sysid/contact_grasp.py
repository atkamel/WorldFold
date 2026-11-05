"""Attach grasp for the RL cloth (the model ~80% of cloth sim-to-real work uses), with failures from geometry and load.

  close    every cloth vertex inside the jaw box (pad face 9 x 8 mm x the jaw gap) is attached: mass 0, moved with the
           jaw. Nothing inside -> miss; vertices from two separate layers inside -> double-layer grasp.
  slip     each frame the load on the grasp is estimated (weight of the cloth off the table, plus table friction on the
           cloth that is dragged); above 2 * mu_jaw * N (two jaw faces) the cloth is released. N and mu_jaw come from
           results/contact/contact_params.json (public SO-101 logs + BAM servo model; pressure-law friction bracket).
The jaws themselves can still collide with cloth and table, so pushing the cloth away is simulated by contact.

  python contact_grasp.py --settings full --eps 1e-3 --kappa-tag _t2
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newton_contact as C  # noqa: E402

PAD_HALF = (0.0045, 0.004)                  # jaw pad face half-extents (m)
LAYER_SPLIT = 0.0006                        # z gap between captured vertices that starts a new layer (m)
MAX_ATTACH = 512


@wp.kernel
def k_carry(n: wp.array[int], t: wp.array[float], t0: float, t1: float, idx: wp.array[int], p0: wp.array[wp.vec3],
            vel: wp.array[wp.vec3], dt: float, q: wp.array[wp.vec3], prev: wp.array[wp.vec3]):
    """Attached vertices move with their jaw at constant velocity between t0 and t1 (one velocity per vertex)."""
    j = wp.tid()
    if j >= n[0]:
        return
    a = wp.clamp(t[0] - dt - t0, 0.0, t1 - t0)
    e = wp.clamp(t[0] - t0, 0.0, t1 - t0)
    prev[j] = p0[j] + vel[j] * a
    q[idx[j]] = p0[j] + vel[j] * e


class Grasp:
    """Attach grasps for one Scene (several jaws). Decisions happen between frames on the CPU; the carry kernel runs
    inside the scene's per-frame graph (fixed-size buffers, active count in n)."""

    def __init__(self, scene, t0, t1):
        self.sc, self.t0, self.t1 = scene, t0, t1
        d = scene.device
        self.mass0 = scene.model.particle_mass.numpy().copy()
        self.inv0 = scene.model.particle_inv_mass.numpy().copy()
        self.n = wp.zeros(1, dtype=int, device=d)
        self.idx = wp.zeros(MAX_ATTACH, dtype=int, device=d)
        self.p0 = wp.zeros(MAX_ATTACH, dtype=wp.vec3, device=d)
        self.vel = wp.zeros(MAX_ATTACH, dtype=wp.vec3, device=d)
        self.prev = wp.zeros(MAX_ATTACH, dtype=wp.vec3, device=d)
        self.jaws = []                          # dicts: idx (np), vel (3,), capacity_N, outcome

    def kernel(self):
        wp.launch(k_carry, dim=MAX_ATTACH, inputs=[self.n, self.sc.t, self.t0, self.t1, self.idx, self.p0, self.vel,
                                                   self.sc.dt, self.sc.s0.particle_q, self.prev], device=self.sc.device)

    def close(self, center, gap, vel, mu_jaw, squeeze_N, cloth_ids, yaw=0.0):
        """Close a jaw box at `center` (box from center z - gap/2 to + gap/2). Returns the outcome dict."""
        q = self.sc.s0.particle_q.numpy()
        c, s = math.cos(yaw), math.sin(yaw)
        rel = q[cloth_ids] - np.asarray(center)
        lx, ly = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
        inside = (np.abs(lx) <= PAD_HALF[0]) & (np.abs(ly) <= PAD_HALF[1]) & (np.abs(rel[:, 2]) <= gap / 2)
        idx = np.asarray(cloth_ids)[inside]
        z = np.sort(q[idx, 2]) if len(idx) else np.zeros(0)
        layers = int(1 + np.sum(np.diff(z) > LAYER_SPLIT)) if len(idx) else 0
        outcome = {"vertices": int(len(idx)), "layers": layers,
                   "result": "miss" if not len(idx) else ("single" if layers == 1 else "multi"),
                   "capacity_N": 2.0 * mu_jaw * squeeze_N}
        self.jaws.append({"idx": idx, "vel": np.asarray(vel, float), "cloth": np.asarray(cloth_ids), **outcome,
                          "released_at": None})
        self._sync()
        return outcome

    def _sync(self):
        """Write the attached set to the solver: masses, carry buffers, the driven-particle hook."""
        mass, inv = self.mass0.copy(), self.inv0.copy()
        idx, vel = [], []
        for j in self.jaws:
            if j["released_at"] is None and len(j["idx"]):
                idx += list(j["idx"])
                vel += [j["vel"]] * len(j["idx"])
        mass[idx], inv[idx] = 0.0, 0.0
        m = self.sc.model
        m.particle_mass.assign(mass)
        m.particle_inv_mass.assign(inv)
        n = len(idx)
        assert n <= MAX_ATTACH
        q = self.sc.s0.particle_q.numpy()
        pad = lambda a, shape: np.concatenate([np.asarray(a, np.float32).reshape(-1, *shape),
                                               np.zeros((MAX_ATTACH - n, *shape), np.float32)])
        t = float(self.sc.t.numpy()[0])
        moved = max(0.0, min(t, self.t1) - self.t0)                 # the jaw has already moved this far
        p0 = q[idx] - np.asarray(vel).reshape(-1, 3) * moved if n else np.zeros((0, 3))
        self.idx.assign(np.concatenate([np.asarray(idx, np.int32), np.zeros(MAX_ATTACH - n, np.int32)]))
        self.p0.assign(pad(p0, (3,)))
        self.vel.assign(pad(vel, (3,)))
        self.n.assign(np.array([n], np.int32))
        self.sc.solver.drv = (self.idx, self.prev, self.n)

    def load_N(self, j, mu_table, table_z=0.0, off=0.003):
        """Estimated pull on the grasp: weight of the cloth hanging off the table + table friction on the cloth still
        on it while the jaw moves horizontally (conservative: all of it slides)."""
        q = self.sc.s0.particle_q.numpy()
        ids = np.setdiff1d(j["cloth"], j["idx"])
        m = self.mass0[ids]
        off_table = q[ids, 2] > table_z + off
        hanging = m[off_table].sum() * C.G
        dragging = np.linalg.norm(j["vel"][:2]) > 0
        friction = mu_table * m[~off_table].sum() * C.G if dragging else 0.0
        return float(math.hypot(hanging, friction))

    def check(self, frame_t, mu_table):
        """Release every jaw whose load exceeds its capacity. Returns the jaws released now."""
        out = []
        for j in self.jaws:
            if j["released_at"] is None and len(j["idx"]):
                j["last_load"] = self.load_N(j, mu_table)
                if j["last_load"] > j["capacity_N"]:
                    j["released_at"] = frame_t
                    out.append(j)
        if out:
            self._sync()
        return out


class GraspScene(C.Scene):
    """Copies of a strip on a table with a weight on its far end; each copy gets a grasp with a different capacity."""
    H = 0.005
    LIFT = (0.0, 0.0, 0.10)                     # m/s for 2.5 s after 0.3 s

    def __init__(self, setting, eps, device, ratios, weight_g):
        self.ratios, self.weight_g = ratios, weight_g
        super().__init__(setting, eps, device)

    def build(self, b):
        self.strips = []
        for k, r in enumerate(self.ratios):
            ox = 0.3 * k
            C.static_box(b, (ox + 0.1, 0.0, -0.005), (0.2, 0.1, 0.005), 0.5)
            v, tris = C.grid(5, 41, self.H, ox - 0.01, -0.10, C.RADIUS)          # 2 x 20 cm strip along y
            ids = C.add_sheet(b, v, tris, self.H)
            far = ids[v[:, 1] < -0.10 + 1e-6]
            for i in far:                                                       # the weight, on the far edge
                b.particle_mass[int(i)] += self.weight_g / 1000.0 / len(far)
            self.strips.append({"ids": ids, "near": (ox, 0.10 - 0.002, C.RADIUS)})

    def after_build(self):
        self.grasp = Grasp(self, 0.3, 2.8)

    def forces(self):
        self.grasp.kernel()


def hold_test(device, setting, eps, ratios=(0.5, 0.9, 1.1, 2.0), weight_g=50.0, mu_table=0.5):
    """Lift a strip with a weight by one end; capacity = ratio x the expected hanging load. Expect release for
    ratio < 1 (once the strip is off the table), hold for ratio > 1."""
    sc = GraspScene(setting, eps, device, ratios, weight_g)
    strip_g = C.CLOTH["density"] * 0.02 * 0.20 * 1000
    expect = (weight_g + strip_g) / 1000 * C.G
    res = []
    for st, r in zip(sc.strips, ratios):
        mu, N = 0.17, r * expect / (2 * 0.17)
        res.append(sc.grasp.close(st["near"], 0.006, GraspScene.LIFT, mu, N, st["ids"]))
    events = []

    def rec(k):
        t = (k + 1) / C.HZ
        for j in sc.grasp.check(t, mu_table):
            events.append((t, int(np.flatnonzero([jj is j for jj in sc.grasp.jaws])[0])))
    sc.run(2.8, rec)
    rows = []
    q = sc.s0.particle_q.numpy()
    for c, (st, r, j) in enumerate(zip(sc.strips, ratios, sc.grasp.jaws)):
        far_z = float(q[st["ids"], 2].min())
        released = j["released_at"] is not None
        rows.append({"ratio": r, "vertices": j["vertices"], "released": released, "released_at": j["released_at"],
                     "last_load_N": round(j.get("last_load", float("nan")), 4), "expected_N": round(expect, 4),
                     "capacity_N": round(j["capacity_N"], 4), "lowest_z_mm": round(far_z * 1000, 1),
                     "ok": released == (r < 1.0)})
    return rows


class SweepScene(C.Scene):
    """A 20 x 20 cm towel with its near half folded over (two layers near the fold edge), for jaw-geometry outcomes."""
    H = 0.005

    def build(self, b):
        C.static_box(b, (0.1, 0.1, -0.005), (0.3, 0.3, 0.005), 0.5)
        vb, tb = C.grid(41, 41, self.H, 0.0, 0.0, C.RADIUS)
        self.bottom = C.add_sheet(b, vb, tb, self.H)
        vt, tt = C.grid(41, 21, self.H, 0.0, 0.0, 3 * C.RADIUS + C.CONTACT["self_gap"])   # top layer over y 0..0.1
        self.top = C.add_sheet(b, vt, tt, self.H)
        self.ids = np.concatenate([self.bottom, self.top])

    def after_build(self):
        self.grasp = Grasp(self, 0.0, 0.0)

    def forces(self):
        self.grasp.kernel()


def geometry_sweep(device, setting, eps):
    """Close the jaw box at the top layer's free edge (y = 0.1) and on single cloth, over jaw heights and insets."""
    sc = SweepScene(setting, eps, device)
    sc.run(0.3, lambda k: None)                     # settle
    rows = []
    for spot, y_edge in (("fold edge (2 layers)", 0.10), ("single-layer edge", 0.20)):
        for inset in (-0.004, 0.0, 0.004, 0.008):
            for z_lo in (0.0, 0.001, 0.002, 0.003, 0.004):
                gap = 0.006
                center = (0.1, y_edge - inset, z_lo + gap / 2)
                g = Grasp(sc, 0.0, 0.0)
                o = g.close(center, gap, (0, 0, 0), 0.17, 5.5, sc.ids)
                g.jaws.clear()
                g._sync()                           # probe only: leave the cloth free
                rows.append({"spot": spot, "inset_mm": inset * 1000, "jaw_bottom_mm": z_lo * 1000, **o})
    return rows


def stage_grasp(device, settings, eps):
    out = {}
    for s in settings:
        hold = hold_test(device, s, eps)
        n_ok = sum(r["ok"] for r in hold)
        C.log(f"grasp hold {s}: {n_ok}/{len(hold)} right (release below 2 mu N, hold above); "
              f"{[(r['ratio'], r['released'], r['last_load_N'], r['capacity_N']) for r in hold]}")
        sweep = geometry_sweep(device, s, eps)
        summary = {}
        for r in sweep:
            summary.setdefault(r["spot"], {}).setdefault(r["result"], 0)
            summary[r["spot"]][r["result"]] += 1
        C.log(f"grasp geometry {s}: {summary}")
        out[s] = {"hold": hold, "sweep": sweep, "sweep_summary": summary}
    C.jdump(out, os.path.join(C.RES, f"grasp{C.TAG[0]}.json"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--settings", default="full")
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--kappa-tag", default="_t2", help="table friction correction (kappa*.json)")
    p.add_argument("--tag", default="")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    wp.init()
    C.TAG[0] = a.tag
    C.load_scale(a.eps, a.kappa_tag)
    C.log(f"table friction correction 1/kappa: {C.MU_SCALE}")
    stage_grasp(a.device, a.settings.split(","), a.eps)


if __name__ == "__main__":
    main()

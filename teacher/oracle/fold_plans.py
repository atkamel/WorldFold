"""Scripted (privileged) folding primitives for the two SO-101 arms: pure numpy, testable offline.

A primitive turns a few Cartesian keyframes per arm into joint-space ticks (one tick = one env.step,
1/90 s). The grasp copies what the teacher does in its recorded episodes: land the open jaws on the
table just outside the grasp point, slide ~4 cm along the jaw axis so cloth bunches between the jaws,
then close.
"""
import numpy as np

from so101_kin import SO101, HOME, GRIP_OPEN, GRIP_CLOSED

K = SO101()
Z_TABLE = 0.531         # lowest height the tip frame reaches: the jaws are then resting on the table (surface ~0.52)
PRESS = 0.013           # the teacher COMMANDS the tip 1-1.5 cm below that while it slides and closes, so the jaws press
                        # on the table and the cloth cannot slip underneath (measured from its recorded actions)
Z_HOVER = 0.5925
SEED = {"left": np.array([0.86, 0.15, 0.42, 0.83, -0.81]), "right": np.array([-0.63, -0.23, 0.86, 0.86, 0.68])}
ROLL_NEAR = {"left": -0.8, "right": 0.7}
MAX_DQ = 0.03           # rad per tick while moving freely


def _solve(side, pos, q_prev, roll=None, direction=(0, 0, -1)):
    q0 = np.array(q_prev, float)
    if roll is not None:
        q0[4] = roll
    return K.ik(pos, side, q0, direction=direction)


def _path(side, pts, q_start, roll, ds=0.003, ds_ik=0.006):
    """Cartesian polyline -> dense list of joint configs (ds metres apart). IK is solved every ds_ik metres and
    joint angles are interpolated linearly in between. Returns (qs, max_err)."""
    qs, q, worst = [], np.array(q_start, float), 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        dist = np.linalg.norm(b - a)
        n_ik = max(1, int(np.ceil(dist / ds_ik)))
        sub = max(1, int(np.ceil(dist / n_ik / ds)))
        for i in range(1, n_ik + 1):
            q_new, e = _solve(side, a + (b - a) * i / n_ik, q, roll)
            worst = max(worst, e)
            for j in range(1, sub + 1):
                qs.append(q + (q_new - q) * j / sub)
            q = q_new
    return qs, worst


def _joint_move(q_a, q_b, max_dq=MAX_DQ):
    q_a, q_b = np.asarray(q_a, float), np.asarray(q_b, float)
    n = max(1, int(np.ceil(np.abs(q_b - q_a).max() / max_dq)))
    return [q_a + (q_b - q_a) * i / n for i in range(1, n + 1)]


def _arc(g, p, z0, z1, h, n=40):
    """Fold path from g to p (xy), rising to h above the straight line around the midpoint."""
    g, p = np.asarray(g, float), np.asarray(p, float)
    out = []
    for s in np.linspace(0, 1, n + 1)[1:]:
        xy = g + (p - g) * (1 - np.cos(np.pi * s)) / 2
        out.append(np.array([xy[0], xy[1], z0 + (z1 - z0) * s + h * np.sin(np.pi * s)]))
    return out


class Plan:
    """Accumulates ticks for both arms. Tick = dict(q={'left': q5, 'right': q5}, g={'left': g, 'right': g}, tag)."""

    def __init__(self, q_now, g_now):
        self.q = {s: np.array(q_now[s], float) for s in ("left", "right")}
        self.g = dict(g_now)
        self.ticks, self.log = [], []

    def _emit(self, seqs, tag, g=None, hold=0):
        """seqs: {side: [q, ...]} (a missing side holds). The shorter sequence holds its last pose."""
        if g:
            self.g.update(g)
        n = max([len(v) for v in seqs.values()] + [hold])
        for i in range(n):
            for s, v in seqs.items():
                if v:
                    self.q[s] = np.array(v[min(i, len(v) - 1)], float)
            self.ticks.append(dict(q={s: self.q[s].copy() for s in self.q}, g=dict(self.g), tag=tag))

    def hold(self, n, tag, g=None):
        self._emit({}, tag, g=g, hold=n)

    def home(self, tag="home", sides=("left", "right")):
        self._emit({s: _joint_move(self.q[s], HOME[s]) for s in sides}, tag)

    def pick_place(self, spec, tag, carry_ds=0.0015, only_grasp=0.0):
        """spec: {side: dict(grasp=(x,y), slide=(dx,dy) direction of the final slide, place=(x,y),
                              lift=arc height in m (0 = drag just above the table), z_place, slide_len, drag_z)}.
        All arms in spec go through the same phases together."""
        sides = list(spec)
        roll, start, grasp = {}, {}, {}
        for s in sides:
            sp = spec[s]
            d = np.asarray(sp["slide"], float); d = d / np.linalg.norm(d)
            g = np.asarray(sp["grasp"], float)
            st = g - d * sp.get("slide_len", 0.04)
            q_g, _ = _solve(s, (st[0], st[1], Z_HOVER), SEED[s])
            roll[s], align = K.roll_for_slide(q_g, s, (d[0], d[1], 0), near=ROLL_NEAR[s])   # moving finger leads
            start[s], grasp[s] = st, g
            self.log.append(dict(tag=tag, side=s, roll=round(float(roll[s]), 3), jaw_align=round(float(align), 3)))
        errs = {}
        # 1. open, go above the slide start
        seq = {}
        for s in sides:
            q_h, e = _solve(s, (start[s][0], start[s][1], Z_HOVER), SEED[s], roll[s]); errs[(s, "hover")] = e
            seq[s] = _joint_move(self.q[s], q_h)
        self._emit(seq, tag + ":approach", g={s: spec[s].get("grip_open", GRIP_OPEN) for s in sides})
        # 2. descend to the table, 3. slide into the cloth, pressing down
        seq = {}
        zg = {s: Z_TABLE - spec[s].get("press", PRESS) for s in sides}
        for s in sides:
            qs, e = _path(s, [(start[s][0], start[s][1], Z_HOVER), (start[s][0], start[s][1], zg[s]),
                              (grasp[s][0], grasp[s][1], zg[s])], self.q[s], roll[s], ds=spec[s].get("slide_ds", 0.002))
            errs[(s, "slide")] = e; seq[s] = qs
        self._emit(seq, tag + ":slide")
        # 4. close (ramped), still pressing
        g0 = {s: self.g[s] for s in sides}
        for i in range(1, 9):
            self.hold(1, tag + ":close", g={s: g0[s] + (GRIP_CLOSED - g0[s]) * i / 8 for s in sides})
        self.hold(spec[sides[0]].get("close_hold", 14), tag + ":close")
        if only_grasp:
            seq = {}
            for s in sides:
                qs, e = _path(s, [(grasp[s][0], grasp[s][1], zg[s]), (grasp[s][0], grasp[s][1], Z_TABLE + only_grasp)],
                              self.q[s], roll[s], ds=carry_ds)
                errs[(s, "lift")] = e; seq[s] = qs
            self._emit(seq, tag + ":lift")
            self.hold(30, tag + ":liftheld")
            return errs
        # 5. carry
        seq = {}
        for s in sides:
            sp = spec[s]; p = np.asarray(sp["place"], float); zp = sp.get("z_place", Z_TABLE + 0.02)
            g0 = np.array([grasp[s][0], grasp[s][1], zg[s]])
            if sp.get("lift", 0.08) > 0:
                pts = [g0, np.array([grasp[s][0], grasp[s][1], Z_TABLE])] + _arc(grasp[s], p, Z_TABLE, zp, sp.get("lift", 0.08))
            else:
                zc = Z_TABLE + sp.get("drag_z", 0.012)
                pts = [g0, np.array([grasp[s][0], grasp[s][1], zc]), np.array([p[0], p[1], zc]), np.array([p[0], p[1], zp])]
            qs, e = _path(s, pts, self.q[s], roll[s], ds=carry_ds); errs[(s, "carry")] = e; seq[s] = qs
        self._emit(seq, tag + ":carry")
        self.hold(15, tag + ":settle")
        # 6. release, 7. retreat straight up
        self.hold(20, tag + ":open", g={s: spec[s].get("grip_open", GRIP_OPEN) for s in sides})
        seq = {}
        for s in sides:
            p, _ = K.tip(self.q[s], s)
            qs, e = _path(s, [p, (p[0], p[1], max(p[2], Z_HOVER) + 0.02)], self.q[s], roll[s], ds=0.003)
            errs[(s, "retreat")] = e; seq[s] = qs
        self._emit(seq, tag + ":retreat")
        self.log.append(dict(tag=tag, max_ik_err_mm={f"{s}:{ph}": round(e * 1000, 1) for (s, ph), e in errs.items()}))
        return errs


# ---- towel ------------------------------------------------------------------------------------------
def towel_corners(verts, N=72):
    """Grid towel from make_towel.py (row 0 and row N-1 are opposite edges). Returns the four corner xy,
    labelled by where they are now: near = closer to the arms (smaller world y), left = smaller x."""
    idx = lambda r, c: r * N + c
    c = {(0, 0): verts[idx(0, 0), :2], (0, 1): verts[idx(0, N - 1), :2],
         (1, 0): verts[idx(N - 1, 0), :2], (1, 1): verts[idx(N - 1, N - 1), :2]}
    near_row = 0 if (c[(0, 0)][1] + c[(0, 1)][1]) < (c[(1, 0)][1] + c[(1, 1)][1]) else 1
    near = sorted([c[(near_row, 0)], c[(near_row, 1)]], key=lambda p: p[0])
    far = sorted([c[(1 - near_row, 0)], c[(1 - near_row, 1)]], key=lambda p: p[0])
    return dict(near_left=np.array(near[0]), near_right=np.array(near[1]), far_left=np.array(far[0]), far_right=np.array(far[1]))


def towel_grasps(c, inset_x=0.027, inset_y=0.02):
    """Grasp points just inside the two near corners (the teacher grasps ~2.7 cm in along x, ~2 cm in along y)."""
    gl = c["near_left"] + np.array([inset_x, inset_y])
    gr = c["near_right"] + np.array([-inset_x, inset_y])
    return gl, gr


def towel_drag(plan, verts, near_edge_y=-0.185, params=None):
    """Pull the towel toward the arms so that its far edge comes within reach."""
    p = params or {}
    c = towel_corners(verts); gl, gr = towel_grasps(c, p.get("inset_x", 0.027), p.get("inset_y", 0.02))
    dy = near_edge_y - (c["near_left"][1] + c["near_right"][1]) / 2
    common = dict(lift=0, drag_z=p.get("drag_z", 0.012), z_place=Z_TABLE + 0.004)
    spec = {"left": dict(grasp=gl, slide=(1, 0), place=gl + [0, dy], **common),
            "right": dict(grasp=gr, slide=(-1, 0), place=gr + [0, dy], **common)}
    errs = plan.pick_place(spec, "drag", carry_ds=p.get("carry_ds", 0.0012))
    return dict(dy=float(dy), corners={k: v.round(4).tolist() for k, v in c.items()}), errs


def towel_fold(plan, verts, params=None):
    """Half fold: carry the near edge over onto the far edge."""
    p = params or {}
    ix, iy = p.get("inset_x", 0.027), p.get("inset_y", 0.02)
    c = towel_corners(verts); gl, gr = towel_grasps(c, ix, iy)
    over = np.array([0, p.get("overshoot", 0.0)])
    # each grasp point should land on its mirror image across the fold line: same inset, from the far corners
    pl = c["far_left"] + np.array([ix, -iy]) + over
    pr = c["far_right"] + np.array([-ix, -iy]) + over
    common = dict(lift=p.get("lift", 0.10), z_place=Z_TABLE + p.get("z_place", 0.012))
    spec = {"left": dict(grasp=gl, slide=(1, 0), place=pl, **common),
            "right": dict(grasp=gr, slide=(-1, 0), place=pr, **common)}
    errs = plan.pick_place(spec, "fold", carry_ds=p.get("carry_ds", 0.0012))
    mid = (c["near_left"] + c["near_right"] + c["far_left"] + c["far_right"]) / 4
    line = dict(p=mid.round(4).tolist(), d=(c["near_right"] - c["near_left"]).round(4).tolist())
    return dict(corners={k: v.round(4).tolist() for k, v in c.items()}, place_left=pl.round(4).tolist(),
                place_right=pr.round(4).tolist(), fold_line=line), errs


# ---- shirts (tops) ----------------------------------------------------------------------------------
def reflect(pt, fold):
    p = np.asarray(fold["p"], float); d = np.asarray(fold["d"], float); d = d / np.linalg.norm(d)
    rel = np.asarray(pt, float) - p
    along = rel @ d
    return p + along * d - (rel - along * d)


def shirt_arms(v, keys):
    """Which robot arm serves each mesh side: the side whose cuff has the smaller world x gets the left arm."""
    left_side = "L" if v[keys["L"]["cuff_mid"], 0] < v[keys["R"]["cuff_mid"], 0] else "R"
    return {left_side: "left", ("R" if left_side == "L" else "L"): "right"}


def shirt_spec(v, keys, fold, arm_of, params):
    """Pick-and-place spec for one fold of a top, from the CURRENT vertex positions v (N,3)."""
    p = params or {}
    inset, lift = p.get("inset", 0.02), p.get("lift", 0.09)
    zp = Z_TABLE + p.get("z_place", 0.015)
    name, spec = fold["name"], {}
    if name.startswith("sleeve_"):
        s = name[-1]
        cuff, armpit = v[keys[s]["cuff_mid"], :2], v[keys[s]["armpit"], :2]
        axis = (armpit - cuff) / np.linalg.norm(armpit - cuff)
        grasp = cuff + axis * inset
        spec[arm_of[s]] = dict(grasp=grasp, slide=axis, place=reflect(grasp, fold) + np.asarray(p.get("sleeve_place_offset", [0, 0]), float),
                               lift=lift, z_place=zp, slide_len=p.get("slide_len", 0.04))
    else:
        hl, hr = v[keys["L"]["hem_out"], :2], v[keys["R"]["hem_out"], :2]
        across = (hr - hl) / np.linalg.norm(hr - hl)
        up = np.asarray(fold["p"], float) - (hl + hr) / 2; up = up / np.linalg.norm(up)
        for s, h, inward in (("L", hl, across), ("R", hr, -across)):
            grasp = h + inward * p.get("hem_inset_x", 0.025) + up * p.get("hem_inset_y", 0.02)
            spec[arm_of[s]] = dict(grasp=grasp, slide=inward, place=reflect(grasp, fold) + up * p.get("overshoot", 0.0),
                                   lift=p.get("hem_lift", lift), z_place=zp, slide_len=p.get("slide_len", 0.04))
    for sp in spec.values():
        for k in ("press", "grip_open", "slide_ds", "close_hold"):
            if k in p:
                sp[k] = p[k]
    return spec

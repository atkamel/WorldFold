"""Self-collision test: fold the shirt bottom-up (hem carried over the body), then count how much of the
folded layer has sunk INTO the layer below it instead of lying on top.
Usage: python foldtest.py [variant ...]"""
import sys, time, numpy as np, mujoco
import server

BASE = [("<option ", '<option solver="CG" '), ('radius="0.002"', 'radius="0.004"'), ('mass="0.05"', 'mass="0.15"'),
        ('timestep="0.002"', 'timestep="0.004"')]
VARIANTS = {
    "current": [],
    "radius6": [('radius="0.004"', 'radius="0.006"')],
    "radius6+dt2": [('radius="0.004"', 'radius="0.006"'), ('timestep="0.004"', 'timestep="0.002"')],
    "narrow": [('selfcollide="auto"', 'selfcollide="narrow"')],
    "radius6+stiff": [('radius="0.004"', 'radius="0.006"'), ('young="3e4"', 'young="1e5"'), ('solref="0.004 1"', 'solref="0.008 1"')],
}


def make(reps, H=None):
    import scene
    if H:
        scene.H = H
    def bm():
        txt = open(scene.__file__, encoding="utf-8-sig").read()
        for a, b in BASE + reps:
            assert a in txt, a
            txt = txt.replace(a, b)
        ns = {"__file__": scene.__file__, "__name__": "v"}
        exec(compile(txt, "v", "exec"), ns)
        if H:
            ns["H"] = H
        return ns["build"]()
    orig = server.build_model
    server.build_model = bm
    s = server.Sim()
    server.build_model = orig
    return s


def fold(s):
    m, d = s.m, s.d
    v = s.verts()
    hem = np.where(v[:, 1] < 0.09)[0]           # hem row
    rest = s.pts.copy()
    off = v[hem].copy()
    t0 = time.time()
    n = int(2.0 / m.opt.timestep)
    for i in range(n):                            # carry hem along an arc to y=0.27 (body is 22 cm long)
        u = (i + 1) / n
        ang = np.pi * u
        for j, o in zip(hem, off):
            r = (0.27 - 0.08) / 2
            y = 0.175 - r * np.cos(ang)
            z = 0.004 + r * np.sin(ang) * 0.5
            tgt = np.array([o[0], y, z])
            d.qpos[s.vqadr[j]:s.vqadr[j] + 3] = tgt - s.v0[j]
            d.qvel[s.vdadr[j]:s.vdadr[j] + 3] = 0
        mujoco.mj_step(m, d)
    for _ in range(int(1.5 / m.opt.timestep)):    # let go, settle
        mujoco.mj_step(m, d)
    wall = time.time() - t0
    v = s.verts()
    # bottom-layer vertices: rest y in [0.18, 0.27]; top-layer: rest y < 0.17 (folded over)
    top = np.where(rest[:, 1] < 0.165)[0]
    bot = np.where((rest[:, 1] > 0.185) & (np.abs(rest[:, 0]) <= 0.10))[0]
    sunk = 0; pairs = 0; gaps = []
    for i in top:
        dxy = np.linalg.norm(v[bot, :2] - v[i, :2], axis=1)
        k = np.argmin(dxy)
        if dxy[k] < 0.015:
            pairs += 1
            gap = v[i, 2] - v[bot[k], 2]
            gaps.append(gap)
            sunk += gap < 0.002   # top layer not clearly above the bottom one
    gaps = np.array(gaps) * 1000
    return dict(overlapping_points=pairs, sunk_through=int(sunk), pct_sunk=round(100 * sunk / max(pairs, 1)),
                median_gap_mm=round(float(np.median(gaps)), 1) if len(gaps) else None, wall_s=round(wall, 1),
                nan=bool(np.isnan(v).any()), nvert=len(v))


if __name__ == "__main__":
    for name in (sys.argv[1:] or VARIANTS):
        H = None
        reps = VARIANTS.get(name, [])
        if name.startswith("grid1cm"):
            H = 0.01
        s = make(reps, H)
        for _ in range(int(0.5 / s.m.opt.timestep)):
            mujoco.mj_step(s.m, s.d)
        print(f"{name:16s}", fold(s), flush=True)

"""Drag test: pin left cuff, carry it over the body like the sleeve fold, measure how far the body slides."""
import sys, numpy as np, mujoco
import server

def make(reps):
    orig = server.build_model
    def bm():
        txt = open(server.scene.__file__, encoding="utf-8-sig").read()
        txt = txt.replace("<option ", '<option solver="CG" ').replace('radius="0.002"', 'radius="0.004"')
        for a, b in reps:
            assert a in txt, a
            txt = txt.replace(a, b)
        ns = {"__file__": server.scene.__file__, "__name__": "v"}
        exec(compile(txt, "v", "exec"), ns)
        return ns["build"]()
    server.build_model = bm
    s = server.Sim()
    server.build_model = orig
    return s

def drag(s):
    m, d = s.m, s.d
    for _ in range(500): mujoco.mj_step(m, d)
    v = s.verts(); k = s.keys
    hem0, arm0 = v[k["hem_L"]].copy(), v[k["armpit_L"]].copy()
    tip = np.array([-0.202, 0.257, 0.008])
    ids = np.where(np.linalg.norm(v[:, :2] - tip[:2], axis=1) < 0.045)[0]
    off = v[ids] - tip
    path = [(-0.12, 0.257, 0.04), (-0.03, 0.257, 0.012)]
    p = tip.copy()
    for goal in path:
        goal = np.array(goal); n = int(np.linalg.norm(goal - p) / 0.03 / m.opt.timestep)
        for i in range(n):
            q = p + (goal - p) * (i + 1) / n
            for j, o in zip(ids, off):
                d.qpos[s.vqadr[j]:s.vqadr[j]+3] = q + o - s.v0[j]; d.qvel[s.vdadr[j]:s.vdadr[j]+3] = 0
            mujoco.mj_step(m, d)
        p = goal
    for _ in range(500): mujoco.mj_step(m, d)
    v = s.verts()
    return np.linalg.norm(v[k["hem_L"]][:2] - hem0[:2]) * 100, np.linalg.norm(v[k["armpit_L"]][:2] - arm0[:2]) * 100, v[k["sleeve_L_lo"]][:2] * 100, s.score()["area_cm2"]

V = {
    "current": [],
    "mass0.15": [('mass="0.05"', 'mass="0.15"')],
    "young3e3": [('young="3e4"', 'young="3e3"')],
    "mass0.15+young3e3": [('mass="0.05"', 'mass="0.15"'), ('young="3e4"', 'young="3e3"')],
    "mass0.15+young3e3+fric1.2": [('mass="0.05"', 'mass="0.15"'), ('young="3e4"', 'young="3e3"'), ('friction="0.6 0.01 0.001"', 'friction="1.2 0.01 0.001"'), ('friction="0.8 0.01 0.001"', 'friction="1.2 0.01 0.001"')],
}
if __name__ == "__main__":
  for name in (sys.argv[1:] or V):
      s = make(V[name])
      hem, arm, sl, area = drag(s)
      print(f"{name:28s} hem_L slid {hem:4.1f} cm, armpit_L slid {arm:4.1f} cm, cuff-lo ends at {sl.round(1)}, area {area}", flush=True)

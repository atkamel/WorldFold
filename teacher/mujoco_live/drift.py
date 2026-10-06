"""Idle drift: let the flat shirt sit for 60 s sim time with nobody touching it; how far does it move?"""
import sys, time, numpy as np, mujoco
import server
def run(label, reps):
    orig = server.build_model
    def bm():
        txt = open(server.scene.__file__, encoding="utf-8-sig").read()
        txt = txt.replace("<option ", '<option solver="CG" ').replace('radius="0.002"', 'radius="0.004"').replace('mass="0.05"', 'mass="0.15"').replace('timestep="0.002"', 'timestep="0.004"')
        txt = txt.replace('<geom name="table" type="plane" size="0.6 0.6 0.01" material="table"', '<geom name="table" type="plane" size="0.6 0.6 0.01" rgba="0 0 0 0"')
        for a, b in reps:
            assert a in txt, a; txt = txt.replace(a, b)
        ns = {"__file__": server.scene.__file__, "__name__": "v"}; exec(compile(txt, "v", "exec"), ns); return ns["build"]()
    server.build_model = bm; s = server.Sim(); server.build_model = orig
    v0 = s.verts().copy(); out = []
    for k in range(6):
        for _ in range(int(10 / s.m.opt.timestep)): mujoco.mj_step(s.m, s.d)
        out.append(np.round((s.verts() - v0)[:, :2].mean(0) * 100, 1).tolist())
    print(f"{label:18s} mean shift (cm) every 10 s: {out}", flush=True)
V = {"game now": [],
     "fric 1.5": [('friction="0.6 0.01 0.001"', 'friction="1.5 0.01 0.001"'), ('friction="0.8 0.01 0.001"', 'friction="1.5 0.01 0.001"')],
     "dt 0.002": [('timestep="0.004"', 'timestep="0.002"')],
     "no bend": [('elastic2d="bend"', 'elastic2d="none"')]}
if __name__ == "__main__":
    for k in (sys.argv[1:] or V): run(k, V[k])

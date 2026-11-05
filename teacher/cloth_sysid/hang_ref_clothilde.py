"""clothilde (paper2022 solver) still hang: denim A2 shake hanger, corners held at frame 0 for 6 s -> markers."""
import sys, os, numpy as np
sys.path.insert(0, r"C:\Users\valen\Documents\vla-distill\WorldFold\teacher\cloth_sysid")
import replay, coltraro_data as CD
from paper_solver import InextensibleCloth
r = [x for x in CD.free_hanging() if x["mode"] == "hanger" and x["fabric"] == "denim" and x["size"] == "A2" and x["motion"] == "shake"][0]
t, P = replay.load(r["path"]); f0 = CD.first_full_frame(P); P0 = P[f0]
N = P.shape[1]; R, C = CD.grid_shape(N)
Tc, Tf, S, X0 = replay.build(P0, R, C, 2)
ps = [v for k, v in CD.PAPER["II"].items() if k.startswith("denim_")]
a, d = np.mean([p["alpha"] for p in ps]), np.mean([p["delta"] for p in ps])
cl = InextensibleCloth(X0, Tf)
o = replay.PAPER2022
cl.setSimulatorParameters(dt=0.01, sub_steps=1, tol=o["tol"], rho=CD.RHO["denim"], delta=d, alpha=a, kappa=0.0, kappa_bnd=0.0, str=0.0, shr=0.0, mu_f=0.2, mu_s=0.35, thck=0.95)
cl.implicitEuler = True
cl.floorCollisions = lambda phi: phi
ctrl = CD.corner_ids(N); u = P0[ctrl].copy()
out = []
for n in range(600):
    cl.simulate(u=u.copy(), control=list(ctrl))
    if n + 1 in (50, 100, 300, 600):
        out.append(np.asarray(cl.positions[:N]).copy())
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "newton", "hang_ref_clothilde.npy"),
        np.stack(out))
print("saved; marker19 disp at 6 s mm", np.round((out[-1][19] - P0[19]) * 1000, 1), "max", round(np.linalg.norm(out[-1] - P0, axis=1).max() * 1000, 1))

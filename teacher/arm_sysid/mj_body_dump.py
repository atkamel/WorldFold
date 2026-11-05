"""Dump the fitted model's (Menagerie so101.xml) body frames and centres of mass at qpos = 0.

    python mj_body_dump.py <so101.xml> <out json>
"""
import json
import sys

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path(sys.argv[1])
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
out = {}
for b in range(1, m.nbody):
    out[m.body(b).name] = {"xpos": d.xpos[b].tolist(), "xmat": d.xmat[b].reshape(3, 3).tolist(),
                           "xipos": d.xipos[b].tolist(), "ipos": m.body_ipos[b].tolist(), "mass": float(m.body_mass[b])}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("MJ_DUMP", json.dumps({k: [round(x, 4) for x in v["xpos"]] for k, v in out.items()}))

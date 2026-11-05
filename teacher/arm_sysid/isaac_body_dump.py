"""Dump LeHome's SO-101 link frames and centres of mass at the zero pose (for porting the fitted model's COMs).

    .venv/bin/python isaac_body_dump.py --out <json>
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG  # noqa: E402

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 90, device="cpu"))
cfg = SO101_FOLLOWER_CFG.replace(prim_path="/World/R")
cfg.init_state.pos = (0.0, 0.0, 0.0)
cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
r = Articulation(cfg)
sim.reset()
z = torch.zeros(1, r.num_joints)
r.write_joint_state_to_sim(z, z)
sim.step(render=False)
r.update(1 / 90)
v = r.root_physx_view
links = v.get_link_transforms()[0].cpu().numpy().tolist()      # per link: pos xyz + quat xyzw (world)
coms = v.get_coms()[0].cpu().numpy().tolist()                   # per link: COM pos + quat in the link frame
masses = v.get_masses()[0].cpu().numpy().tolist()
out = {"bodies": r.body_names, "joints": r.joint_names, "link_tf_w": links, "com_local": coms, "mass": masses}
json.dump(out, open(args.out, "w"), indent=1)
print("BODY_DUMP", json.dumps(out), flush=True)
sys.stdout.flush()
os._exit(0)

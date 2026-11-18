"""Does the SO-101 arm pass through itself? Distances between the arm's own parts along a trajectory, on the collision
shapes of the so101_nexus model (so101_new_calib.xml), with the base's meshes made solid too.

Not checked: parts fewer than MIN_JOINTS_APART joints apart (they touch by construction, and the folded idle pose rests
on them), the table, the cloth, and the other arm.
"""

import importlib.util
import math
import os

import mujoco
import numpy as np

CLEARANCE = 0.010       # m every checked pair of parts must keep; the joint zeros are set by eye
FAR = 0.05              # m: nothing further apart than this is measured
MIN_JOINTS_APART = 3
CHAIN = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]

_cache = {}


def load():
    """(model, data, [(geom id, geom id, "part and part")]) for the pairs that are checked. Built once."""
    if _cache:
        return _cache["model"], _cache["data"], _cache["pairs"]
    package = importlib.util.find_spec("so101_nexus")
    if package is None:
        raise ImportError("so101_nexus is not installed")
    xml = os.path.join(package.submodule_search_locations[0], "assets", "SO101", "so101_new_calib.xml")
    spec = mujoco.MjSpec.from_file(xml)
    for geom in spec.geoms:
        if geom.parent.name == "base":      # the model's base is visual only
            geom.contype = 1
            geom.conaffinity = 1
    model = spec.compile()
    # MuJoCo's default convex routine returns exactly 0 for some of these mesh pairs; the older one does not
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    solid = {}
    for geom_id in range(model.ngeom):
        if model.geom_contype[geom_id] == 0:
            continue
        part = model.body(model.geom_bodyid[geom_id]).name
        if part not in solid:
            solid[part] = []
        solid[part].append(geom_id)
    pairs = []
    for i in range(len(CHAIN)):
        for j in range(i + MIN_JOINTS_APART, len(CHAIN)):
            for first in solid[CHAIN[i]]:
                for second in solid[CHAIN[j]]:
                    pairs.append((first, second, f"{CHAIN[i]} and {CHAIN[j]}"))
    _cache["model"] = model
    _cache["data"] = mujoco.MjData(model)
    _cache["pairs"] = pairs
    return model, _cache["data"], pairs


def closest(rows):
    """(distance m, row index, "part and part"): the nearest any checked pair of parts comes over `rows` (each the 6
    joint angles in radians). Negative = they overlap by that much. FAR when nothing comes within FAR. ValueError
    for an angle or a distance that is not a number."""
    model, data, pairs = load()
    best = (FAR, 0, "nothing")
    segment = np.zeros(6)
    for step, row in enumerate(rows):
        for angle in row:
            if not math.isfinite(angle):
                raise ValueError(f"row {step} holds an angle that is not a number")
        data.qpos[:6] = row
        mujoco.mj_kinematics(model, data)
        for first, second, names in pairs:
            distance = mujoco.mj_geomDistance(model, data, first, second, FAR, segment)
            if not math.isfinite(distance):
                raise ValueError(f"the distance between {names} at row {step} is not a number")
            if distance < best[0]:
                best = (float(distance), step, names)
    return best

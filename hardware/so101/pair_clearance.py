"""Collision checks for the two side-by-side SO-101 followers and their table."""

from itertools import product

import mujoco
import numpy as np

import collision

TABLE_CLEARANCE = 0.002  # m above the nominal table plane in the base frame


class PairClearance:
    def __init__(self, spacing):
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("Gamepad mode needs the measured spacing between the two bases")
        spec = mujoco.MjSpec()
        spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
        for side, y in (("left", spacing / 2), ("right", -spacing / 2)):
            frame = spec.worldbody.add_frame(pos=[0, y, 0])
            spec.attach(collision.arm_spec(), prefix=f"{side}_", frame=frame)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        table = self.model.geom("table").id
        solids = [collision.solid_geoms(self.model, f"{side}_") for side in ("left", "right")]
        moving = [g for arm in solids for part in collision.CHAIN[1:] for g in arm[part]]
        left, right = [[g for part in collision.CHAIN for g in arm[part]] for arm in solids]
        self.pairs = [(geom, table, TABLE_CLEARANCE, "table") for geom in moving]
        self.pairs.extend((a, b, collision.CLEARANCE, "between arms") for a, b in product(left, right))
        for arm in solids:
            self.pairs.extend((a, b, collision.CLEARANCE, "within an arm")
                              for a, b, _ in collision.self_pairs(arm))

    def obstruction(self, left, right):
        """Test each arm's own progress samples against all progress samples of the other."""
        for lq, rq in product(left, right):
            self.data.qpos[:] = (*lq, *rq)
            if not np.isfinite(self.data.qpos).all():
                return "invalid joint position"
            mujoco.mj_kinematics(self.model, self.data)
            for a, b, clearance, region in self.pairs:
                distance = mujoco.mj_geomDistance(self.model, self.data, a, b, clearance, None)
                if not np.isfinite(distance) or distance < clearance:
                    return region
        return None

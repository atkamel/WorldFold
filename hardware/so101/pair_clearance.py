"""Collision checks for the two side-by-side SO-101 followers and their table."""

from itertools import product

import mujoco
import numpy as np

import collision

TABLE_CLEARANCE = 0.002  # m above the nominal table plane in the base frame
# m between parts two joints apart (forearm and shoulder housing, wrist and upper arm). The folded idle pose
# rests inside it, so a step is refused only when it also brings the parts closer than they are.
NEAR_CLEARANCE = 0.005
NEAR_JOINTS_APART = 2
CLOSER = 1e-6            # m a pair inside its clearance may close in one step: rounding, not motion


def near_pairs(solid):
    """Pairs of parts two joints apart. The moving jaw is left out: it sits beside the wrist by construction."""
    parts = collision.CHAIN[:-1]
    pairs = []
    for i in range(len(parts) - NEAR_JOINTS_APART):
        for first in solid[parts[i]]:
            for second in solid[parts[i + NEAR_JOINTS_APART]]:
                pairs.append((first, second))
    return pairs


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
            for a, b in near_pairs(arm):
                self.pairs.append((a, b, NEAR_CLEARANCE, "within an arm"))

    def distances(self, lq, rq):
        """The distance of every checked pair with the arms at `lq` and `rq`, each capped at the pair's clearance.
        None for a pose or a distance that is not a number."""
        self.data.qpos[:] = (*lq, *rq)
        if not np.isfinite(self.data.qpos).all():
            return None
        mujoco.mj_kinematics(self.model, self.data)
        found = []
        for a, b, clearance, _ in self.pairs:
            distance = mujoco.mj_geomDistance(self.model, self.data, a, b, clearance, None)
            if not np.isfinite(distance):
                return None
            found.append(distance)
        return found

    def floors(self, left, right):
        """How close every pair already is over the poses the arms hold now. None for an invalid pose."""
        lowest = None
        for lq, rq in product(left, right):
            found = self.distances(lq, rq)
            if found is None:
                return None
            if lowest is None:
                lowest = found
            else:
                for i in range(len(found)):
                    lowest[i] = min(lowest[i], found[i])
        return lowest

    def obstruction(self, left, right, floors=None):
        """Test each arm's own progress samples against all progress samples of the other. With `floors` (from
        the poses held now), a pair already inside its clearance may stay as close as it is, so an arm that
        starts too close can back away, and may not come closer."""
        for lq, rq in product(left, right):
            found = self.distances(lq, rq)
            if found is None:
                return "invalid joint position"
            for i in range(len(found)):
                clearance = self.pairs[i][2]
                if found[i] >= clearance:
                    continue
                if floors is not None and found[i] >= floors[i] - CLOSER:
                    continue
                return self.pairs[i][3]
        return None

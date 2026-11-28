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
# m parts two joints apart may come closer than they are at the rest pose given to `rest`. The model draws the
# folded arm up to 12 mm inside itself where the real one rests clear, and every stick direction from there brings
# such a pair 1 to 3 mm closer in the model (results.md). The forearm deep in the shoulder housing is 20 mm and more.
NEAR_SLACK = 0.005
REST_RANGE = 0.05        # m: how far apart a pair is still measured at the rest pose
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
        # A pair whose bounding spheres are further apart than its clearance is not measured. The table plane
        # has no sphere (radius 0), so its pairs are always measured.
        self.skip_far = True
        self.first = np.array([pair[0] for pair in self.pairs])
        self.second = np.array([pair[1] for pair in self.pairs])
        self.clearances = np.array([pair[2] for pair in self.pairs])
        radius = self.model.geom_rbound
        self.spheres = radius[self.first] + radius[self.second]
        self.has_spheres = (radius[self.first] > 0) & (radius[self.second] > 0)

    def rest(self, lq, rq):
        """Take `lq` and `rq` as a pose the real arms rest in untouched. A pair of parts two joints apart is then
        held to NEAR_CLEARANCE or to NEAR_SLACK closer than it is in this pose, whichever is closer."""
        self.data.qpos[:] = (*lq, *rq)
        mujoco.mj_kinematics(self.model, self.data)
        for i in range(len(self.pairs)):
            a, b, clearance, region = self.pairs[i]
            if clearance != NEAR_CLEARANCE:
                continue
            at_rest = mujoco.mj_geomDistance(self.model, self.data, a, b, REST_RANGE, None)
            if not np.isfinite(at_rest):
                continue
            allowed = min(NEAR_CLEARANCE, at_rest - NEAR_SLACK)
            self.pairs[i] = (a, b, allowed, region)
            self.clearances[i] = allowed

    def distances(self, lq, rq):
        """The distance of every checked pair with the arms at `lq` and `rq`, each capped at the pair's clearance.
        None for a pose or a distance that is not a number."""
        self.data.qpos[:] = (*lq, *rq)
        if not np.isfinite(self.data.qpos).all():
            return None
        mujoco.mj_kinematics(self.model, self.data)
        centres = self.data.geom_xpos
        apart = np.linalg.norm(centres[self.first] - centres[self.second], axis=1) - self.spheres
        far = self.has_spheres & (apart > np.maximum(self.clearances, 0.001))
        found = []
        for i in range(len(self.pairs)):
            a, b, clearance, _ = self.pairs[i]
            if self.skip_far and far[i]:
                found.append(clearance)
                continue
            # A pair allowed to overlap in the model is still measured from just apart.
            reach = max(clearance, 0.001)
            distance = mujoco.mj_geomDistance(self.model, self.data, a, b, reach, None)
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

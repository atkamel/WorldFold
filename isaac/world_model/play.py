"""Both arms play with the cloth on Isaac's friction profile: bulk data for the full-state world model, not a task.

Each arm runs its own cycles until the episode's step cap, so sometimes both hold the cloth at once. A cycle picks a
random cloth vertex on the arm's own side of the cloth (corners, edges, interior, vertices lying on other layers) that
the arm can pinch, pinches it with IsaacArmExpert's top-down pinch, carries it to a random point the arm can reach on
its side of the cloth's start footprint, and lets go. Per cycle the carry is an arc 4 to 20 cm high or a 1 cm drag
along the table, and the jaw opens at the target ("target"), at the top of the arc ("drop") or at a random point of
the carry ("forced"). Some cycles shake the held cloth at the top of the arc, some add Gaussian noise to every joint
action. The sides are the halves either side of the start footprint's x midline, SIDE_GAP clear of it, so the arms
never reach across each other.

Reused from IsaacArmExpert: planning (pinch, arc, place, retreat IK), the phase machine and its arrival tests, and
release / retreat. Added here: the cycle draw with its reach check, the forced release, the shake and the noise.
"""

from __future__ import annotations

import sys

import numpy as np

from cloth_angles.data.state_episode import GRASP_INDEX, ROBOT_DIM
from cloth_angles.model.grasp_events import grasp_events
from cloth_angles.tasks import HALF
from isaac.experts.fold_expert import IsaacArmExpert
from isaac.experts.half_fold_demo import SEED_Q
from isaac.sim.isaac_env import CLOTH_JITTER
from mujoco_sim.cloth_params import CLOTH_COUNT, TABLE_TOP_Z

ARMS = HALF.arms               # left then right: the half fold's 393-value state and 12-dim action
SIDE_GAP = 0.02                # m, each arm keeps this far to its own side of the footprint's x midline
PINCH_TOL = 0.005              # m, the planned pinch pose's horizontal miss (IsaacArmExpert.REACH_TOL)
PLACE_TOL = 0.02               # m, the planned place pose's miss; play needs no precise placement
MIN_JAW = 0.02                 # m, a vertex this close to the cloth centroid has no inward jaw direction
MAX_TRIES = 30                 # (vertex, target) draws per cycle before the episode ends as "unreachable"
ARC_HEIGHTS = (0.04, 0.20)     # m, an arc carry's height, uniform
DRAG_HEIGHT = 0.01             # m, a drag along the table (isaac.tools.quarter_batch.FLAT_ARC)
P_DRAG = 0.25
RELEASES = ("target", "drop", "forced")
P_SHAKE = 0.3
SHAKE_STEPS = 12               # joints reverse every 2 steps: 3 swings of +-0.1 rad
P_NOISE = 0.5
NOISE_STD = 0.3                # the MuJoCo collector's expert_noisy std
CYCLE_STEPS = 250              # a cycle still running after this many steps is abandoned for a new one


class PlayExpert(IsaacArmExpert):
    """IsaacArmExpert seeded by its pan sweep on every layout. From its one far-corner seed (SEED_Q) on the friction
    layout the IK misses 35 of a flat cloth's 120 off-centre vertices for the left arm, near the base and toward the
    far side; the sweep reaches all but 9 of them."""
    PINCH_FOLLOW_Z = 1                       # a vertex lying on another layer is pinched at its own height
    _jw = IsaacArmExpert.JAW_WEIGHT          # sweep seeds can land the jaw half a turn round (as on quarter2)

    def _home_seed(self, tip, jaw):
        """IsaacArmExpert._home_seed's sweep: moderate fold first, then closest, then fingers most nearly down."""
        best = None
        for pan in np.linspace(-1.8, 1.8, 7):
            for rest in (SEED_Q["left_"][1:], SEED_Q["right_"][1:]):
                q, e = self._ik.solve(self.prefix, tip, jaw, [pan] + list(rest), iters=150,
                                      orientation_weight=self.PINCH_ORIENTATION_WEIGHT, jaw_weight=self._jw)
                key = (max(q[2], q[3]) > 1.3, round(e, 4), self._ik.site_pose(self.prefix, q)[1][2, 2])
                if best is None or key < best[0]:
                    best = (key, q)
        return list(best[1])


class PlayArm:
    """One arm's cycles. side(x) says whether a point at x is on the arm's side; box bounds its targets."""

    def __init__(self, base, arm, rng, side, box, cycles):
        self.base, self.arm, self.rng, self.side, self.box = base, arm, rng, side, box
        self.cycles = cycles                 # shared with the other arm: every cycle in the order drawn
        self.target = None
        self.expert = PlayExpert(base, prefix=arm.prefix, corner=(0,), goal=lambda: self.target, release=True)
        self.t = 0
        self.cycle = None

    def _reaches(self, plan):
        e = self.expert
        pinch = plan["corner"][:2] - e.PINCH_INSET * plan["jaw"][:2]
        site = e._ik.site_pose(self.arm.prefix, plan["pinch"])[0]
        place = e._ik.site_pose(self.arm.prefix, plan["arc"][-1])[0]
        return (np.linalg.norm(site[:2] - pinch) <= PINCH_TOL
                and np.linalg.norm(place - plan["arc_tips"][-1]) <= PLACE_TOL)

    def _new_cycle(self):
        e, rng = self.expert, self.rng
        cloth = np.asarray(self.base.cloth_positions())
        center = cloth.mean(axis=0)
        e.ARC_HEIGHT = DRAG_HEIGHT if rng.random() < P_DRAG else float(rng.uniform(*ARC_HEIGHTS))
        for tries in range(1, MAX_TRIES + 1):
            vertex = int(rng.integers(len(cloth)))
            if not self.side(cloth[vertex, 0]) or np.linalg.norm(cloth[vertex, :2] - center[:2]) < MIN_JAW:
                continue
            self.target = np.r_[rng.uniform(*self.box), TABLE_TOP_Z]
            e.corners = (vertex,)
            e.reset()
            e.plan = e._make_plan()
            if self._reaches(e.plan):
                break
        else:
            return False
        self.base.weld_mask[self.arm.prefix] = {vertex}      # the grasp flag: this arm holds the vertex it plays with
        release = RELEASES[rng.integers(len(RELEASES))]
        last = len(e.plan["arc"]) - 1
        self.open_wp = {"target": None, "drop": last // 2, "forced": int(rng.integers(1, last))}[release]
        self.shake_left = SHAKE_STEPS if rng.random() < P_SHAKE else 0
        self.shake = rng.uniform(0.5, 1.0, 5) * rng.choice((-1.0, 1.0), 5)
        self.noise = NOISE_STD if rng.random() < P_NOISE else 0.0
        self.cycle = {"arm": self.arm.prefix, "start": self.t, "vertex": vertex,
                      "target": self.target.round(4).tolist(), "arc_height": round(e.ARC_HEIGHT, 4),
                      "release": release, "shake": bool(self.shake_left), "noise": self.noise, "tries": tries,
                      "steps": 0, "done": False}
        self.cycles.append(self.cycle)
        return True

    def act(self):
        """This arm's 6-dim action (joint deltas, gripper), or None when no reachable cycle is left."""
        e = self.expert
        if self.cycle is None or e.PHASES[e.phase] == "done" or self.cycle["steps"] >= CYCLE_STEPS:
            if self.cycle is not None:
                self.cycle["done"] = e.PHASES[e.phase] == "done"
            if not self._new_cycle():
                return None
        name = e.PHASES[e.phase]
        carrying = name in ("lift", "carry")
        if carrying and self.shake_left and e.wp >= (len(e.plan["arc"]) - 1) // 2:
            k = SHAKE_STEPS - self.shake_left
            self.shake_left -= 1
            action = np.zeros(6, dtype=np.float32)
            action[0:5] = self.shake * (1.0 if (k // 2) % 2 == 0 else -1.0)
            action[5] = e._close_command()
        else:
            if carrying and self.open_wp is not None and e.wp >= self.open_wp:
                e.phase, e.phase_steps = e.PHASES.index("release"), 0      # let go here: release, retreat, done
            action = e.act()
        if self.noise:
            action[0:5] = np.clip(action[0:5] + self.rng.normal(0.0, self.noise, 5), -1.0, 1.0)
        self.cycle["steps"] += 1
        self.t += 1
        return action


class PlayPolicy:
    """Both arms' play on a reset env. act() -> the 12-dim action, or None once an arm has no reachable cycle left;
    vertices -> each arm's current cycle vertex; cycles -> every cycle drawn (arm, first state index, vertex, target,
    carry height, release mode, shake, noise, whether it finished)."""

    def __init__(self, base, rng):
        start = np.asarray(base.cloth_positions())
        low, high = start[:, :2].min(axis=0), start[:, :2].max(axis=0)
        mid = float(start[:, 0].mean())
        self.cycles = []
        self.arms = (PlayArm(base, ARMS[0], rng, lambda x: x < mid - SIDE_GAP,
                             (low, np.r_[mid - SIDE_GAP, high[1]]), self.cycles),
                     PlayArm(base, ARMS[1], rng, lambda x: x > mid + SIDE_GAP,
                             (np.r_[mid + SIDE_GAP, low[1]], high), self.cycles))

    @property
    def vertices(self):
        return [a.cycle["vertex"] for a in self.arms]

    def act(self):
        parts = [a.act() for a in self.arms]
        return None if any(p is None for p in parts) else np.concatenate(parts)


def expand_action(action):
    """IsaacClothFoldEnv's 14-dim action from the 12-dim one (per arm: 5 joint deltas, then its gripper at 6 / 13)."""
    full = np.zeros(14, dtype=np.float32)
    for i, arm in enumerate(ARMS):
        full[7 * i:7 * i + 5] = action[arm.joints]
        full[7 * i + 6] = action[arm.gripper]
    return full


def make_base(max_steps, cameras=None):
    """IsaacClothFoldEnv on the friction profile with its dynamics DR, every cloth vertex allowed in either arm's
    grasp flag (the jaw shut and settled with the vertex within HOLD_DIST): play's weld_mask narrows it to each arm's
    cycle vertex, the half fold's to its corners."""
    sys.argv = sys.argv[:1]                  # AppLauncher reads sys.argv
    from cloth_fold_rl.quarter_fold_env import GRASP_RADIUS
    from isaac.sim.isaac_env import IsaacClothFoldEnv
    every = tuple(range(CLOTH_COUNT ** 2))
    base = IsaacClothFoldEnv(observation_mode="state", max_episode_steps=max_steps, profile="friction",
                             cameras=cameras, grasp_corners={"left_": every, "right_": every},
                             grasp_radius=GRASP_RADIUS)
    base.domain_randomization = True
    return base


def reset_play(base, seed):
    """Reset with a cloth offset drawn from the seed (the half fold's +-1 cm) -> (rng for the play policy, offset,
    reset info)."""
    rng = np.random.default_rng(seed)
    pose = rng.uniform(-CLOTH_JITTER, CLOTH_JITTER, 2)
    _, info = base.reset(seed=seed, options={"cloth_pose": pose})
    return rng, pose, info


def disturbance_stats(vertices, robot, actions, cycles):
    """How much the cloth moved: vertex displacement from the first state (mm, mean and max over steps and vertices),
    the highest vertex above the table, and over both arms: grasped states, grasps gained, grasps lost with the jaw
    commanded open (releases) and closed (slips), cycles started and finished."""
    disp = np.linalg.norm(vertices - vertices[:1], axis=-1)
    stats = {"mean_disp_mm": round(1000 * float(disp.mean()), 1), "max_disp_mm": round(1000 * float(disp.max()), 1),
             "max_height_mm": round(1000 * float(vertices[..., 2].max() - TABLE_TOP_Z), 1),
             "grasped_steps": 0, "grasp_events": 0, "release_events": 0, "slip_events": 0}
    for i, arm in enumerate(ARMS):
        grasp = robot[:, ROBOT_DIM * i + GRASP_INDEX]
        events = grasp_events(grasp[:-1], grasp[1:], actions[:, arm.gripper])
        stats["grasped_steps"] += int((grasp > 0.5).sum())
        stats["grasp_events"] += int(events["onset"].sum())
        stats["release_events"] += int(events["release"].sum())
        stats["slip_events"] += int(events["slip"].sum())
    stats.update(cycles=len(cycles), cycles_done=sum(c.get("done", False) for c in cycles))
    return stats

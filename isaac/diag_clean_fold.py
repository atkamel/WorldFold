"""R0 diagnostic (temporary, 2026-10-15): start the quarter fold at its second fold from a geometrically perfect half fold.

Opt-in only, never on the default path: WORLDFOLD_QF_START=clean on the isaac_friction quarter fold. After the usual
reset and settle, the flap half of the cloth (the side vertex 10 lies on) is folded onto the other half by writing the
particles directly: a length-preserving fold with a semicircular crease of radius CREASE_RADIUS, the flap lying flat
2 * CREASE_RADIUS above the bottom layer, velocities zeroed. The cloth then settles SETTLE_STEPS control steps with the
arms at home, and the episode begins at stage 1 (the second fold). No weld anywhere: the write happens once, before
any arm moves.

The question it answers: does the fold-over second fold work when fold 1 is accurate (plan R0)? Delete it, or keep it
only as a test fixture, once fold 1 itself is accurate (R1).
"""

import os

import numpy as np

CREASE_RADIUS = 0.003     # m; the flap lies 2r = 6 mm above the bottom layer (particle contact offset 5 mm)
SETTLE_STEPS = 20         # control steps (1 s) after the write, arms at home
FLAG = "WORLDFOLD_QF_START"


def enabled():
    return os.environ.get(FLAG, "") == "clean"


def fold_half(particles, flap_corner, fixed_corner, radius=CREASE_RADIUS):
    """A perfect half fold of a flat cloth: the half on flap_corner's side folded onto the other.

    particles: [P, 3]; flap_corner / fixed_corner: [3] points (e.g. vertex 10 and vertex 0) whose xy offset gives the
    fold direction. The fold line is placed so that the flap's edge lands exactly on the fixed edge: it sits pi*r/2
    short of the cloth's mid-line, the semicircular crease uses pi*r of length and the rest lies flat at 2r. Each
    particle keeps its own z offset; arc length along the fold direction is preserved.
    """
    p = np.asarray(particles, dtype=np.float64).copy()
    w = np.asarray(flap_corner, dtype=np.float64)[:2] - np.asarray(fixed_corner, dtype=np.float64)[:2]
    w /= np.linalg.norm(w)                         # unit vector toward the flap
    s = p[:, :2] @ w
    line = 0.5 * (s.min() + s.max()) - 0.5 * np.pi * radius
    d = s - line                                   # arc length past the fold line (flap where > 0)
    flap = d > 0
    on_arc = flap & (d < np.pi * radius)
    flat = flap & ~on_arc
    s_new, dz = s.copy(), np.zeros(len(p))
    theta = d[on_arc] / radius
    s_new[on_arc] = line + radius * np.sin(theta)
    dz[on_arc] = radius * (1.0 - np.cos(theta))
    s_new[flat] = line - (d[flat] - np.pi * radius)
    dz[flat] = 2.0 * radius
    p[:, :2] += np.outer(s_new - s, w)
    p[:, 2] += dz
    return p


def clean_start(env):
    """QuarterFoldEnv.start_hook: write the perfect half fold, settle, and begin stage 1."""
    base = env.unwrapped
    cloth = base.cloth_positions()
    from cloth_fold_rl.quarter_fold_env import CLOTH_0, CLOTH_10
    folded = fold_half(base._particles, cloth[CLOTH_10], cloth[CLOTH_0])
    base._scene_call(base.lab.set_particles, folded, base._copy)
    base._particles = folded
    for _ in range(SETTLE_STEPS):
        base._advance()
    begin_stage(env, 1)


def begin_stage(env, stage):
    """Make `stage` the env's current stage from the cloth as it lies now (what QuarterFoldEnv.step does when a stage
    settles); _start stays the reset pose, so fold_score counts the earlier stages as done."""
    base = env.unwrapped
    env.stage = stage
    env._stage_start = base.cloth_positions()
    env._begin_stage()
    env._apply_weld_mask()
    env._set_goals()
    env._settle_steps = 0
    env._prev_potential = env._potential()


def maybe_install(env, task, backend):
    """Attach clean_start to a quarter-fold env on isaac_friction when the flag is set; otherwise nothing."""
    if enabled() and task == "quarter_fold" and backend == "isaac_friction":
        env.start_hook = clean_start
    return env

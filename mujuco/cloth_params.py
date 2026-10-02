"""Scene constants and sensor models shared by the MuJoCo and Isaac Sim envs.

numpy only: the Isaac Sim container cannot install mujoco or so101-nexus, so
anything both simulators must agree on lives here.
"""

import gymnasium as gym
import numpy as np

TABLE_TOP_Z           = 0.42
CLOTH_COUNT           = 11      # grid resolution; higher = finer/drapier mesh
CLOTH_SPACING         = 0.03    # vertex gap; chosen with COUNT to keep span = (COUNT-1)*spacing = 0.30m
CLOTH_RADIUS          = 0.01    # collision thickness (physics only); MUST be >0 or cloth sits IN the table
VISUAL_THICKNESS      = 0.003   # how thick the cloth LOOKS (render only), decoupled from collision radius
CLOTH_MASS            = 0.05    # level3 value; stability comes from soft finger pads + heavy damping
CLOTH_HALF            = (CLOTH_COUNT-1) * CLOTH_SPACING / 2   # cloth spans +-0.09m

# Side-by-side south of the cloth, both facing +y (north, into the cloth),
# instead of the old diagonal-corner placement. Bases are 0.44m apart in x
# (0.22m either side of center), sit 0.15m south of the cloth's south edge, and
# are raised 0.06m above the table top. The extra south set-back and the raised
# mount matter: with bases at table height right at the edge, reaching the near
# south corners (the stage-1 grasp points, only ~9cm away) drove the wrist down
# into the table and jammed the arm; from further back and higher the arm comes
# down onto every corner from above. All four fold corners plus the stage-1
# centre goals stay in reach from here (see cloth_fold_rl/README.md's
# reachability table, re-measured via the same 40k-sample FK sweep method).
ARM_BASE_LEFT         = (-0.22, -(CLOTH_HALF + 0.15), TABLE_TOP_Z + 0.06)
ARM_BASE_RIGHT        = ( 0.22, -(CLOTH_HALF + 0.15), TABLE_TOP_Z + 0.06)
# both arms rotated 90 deg about z from their zero pose, so "local +x" (the
# old left arm's stock forward direction) points world +y, i.e. north into
# the cloth from the south edge. Both bases use this SAME quat now -- there is
# no more "left faces one way, right faces the opposite way" split.
ARM_BASE_QUAT         = [0.70710678, 0.0, 0.0, 0.70710678]
ARM_JOINTS            = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_CLOSED        = -0.1
ARM_TIMESTEP         = 0.0005     # contact-heavy grasp needs 0.5ms; 1ms explodes on contact
MAX_STEP_MOVE        = 0.0003    # max amount can move per step
MAX_JOINT_STEP       = 0.005
CLOTH_DAMPING        = 0.3        # viscous damping per cloth vertex DOF; calms jitter/explosions (level3 value)

# Cloth Fold params
GRIPPER_OPEN         = 1.0      # gripper ctrlrange is [-0.175, 1.745]
ROT_WEIGHT           = 0.2
MAX_STEP_ROT         = 0.002    # rad per substep
GRASP_RADIUS         = 0.03     # weld engages when gripper is this close to its corner
JOINT_DELTA_SCALE    = 0.05     # rad per control step at full action, joint_delta mode
# cloth vertices each gripper may weld to: every listed vertex within GRASP_RADIUS
# is welded when the gripper closes, so a gripper can pick up a stack of corners
GRASP_CORNERS        = {"left_": (CLOTH_COUNT - 1,), "right_": ((CLOTH_COUNT - 1) * CLOTH_COUNT,)}
HOLD_STEPS           = 10       # consecutive success steps (0.5s) before terminating
SETTLE_STEPS         = 2000     # 1.0s hands-off settle at reset, mirrors the demo above
WORKSPACE_XY         = 0.45
WORKSPACE_Z_LOW      = TABLE_TOP_Z + 0.003
WORKSPACE_Z_HIGH     = TABLE_TOP_Z + 0.35
SUCCESS_FOLD_SCORE   = 0.85
CORNER_PLACED_DIST   = 0.03
QACC_LIMIT           = 1e5
TASK_NAMES           = ["fold", "drop", "push", "drag"]   # index = task id (one-hot slot)

# for repositioning the cam
DEPTH_MIN            = 0.2      # m
DEPTH_MAX            = 3.0      # m
DEPTH_NOISE_STD_1M   = 0.002    # m
DEPTH_QUANT_1M       = 0.001    # m
DEPTH_GRAZING_DEG    = 80.0     # surfaces tilted past this from the view ray become holes

CAMERA_POS              = (0.274, 0.0, 1.172)
CAMERA_FOVY_DEG         = 45.0   # vertical field of view, both sims
CAMERA_TARGET           = (0.0, 0.0, TABLE_TOP_Z)  # aim point


def camera_axes(pos, target):
    # look-at: returns (right, up) unit vectors for a camera at pos aimed at target
    forward = np.array(target, dtype=float) - np.array(pos, dtype=float)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        raise ValueError("camera position and target are the same point")
    forward = forward / forward_norm
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm
    up = np.cross(right, forward)
    return right, up


def sensor_depth(raw, pixel_angle, rng):
    # turns a clean (H, W) depth render into a noisy sensor read; 0 = invalid pixel
    depth = raw.astype(np.float32)
    scale = depth * depth
    noise = rng.normal(0.0, DEPTH_NOISE_STD_1M, depth.shape).astype(np.float32)
    depth = depth + noise * scale
    step = DEPTH_QUANT_1M * scale
    depth = np.round(depth / step) * step

    # depth change per pixel / (depth * pixel angle) ~ tan(angle between surface and view ray)
    dy, dx = np.gradient(raw)
    slope = np.hypot(dx, dy) / (raw * pixel_angle)
    grazing = slope > np.tan(np.radians(DEPTH_GRAZING_DEG))
    invalid = (depth < DEPTH_MIN) | (depth > DEPTH_MAX) | grazing
    depth[invalid] = 0.0
    return depth


def check_contract(env, n_episodes=4, n_steps=5):
    # verification before running: observation shapes/dtypes must not drift across resets and steps
    ref = None
    for ep in range(n_episodes):
        obs, info = env.reset(options={"task": ep % env.n_tasks})
        for _ in range(n_steps):
            sig = {}
            for k in sorted(obs.keys()):
                sig[k] = (obs[k].shape, str(obs[k].dtype))
            if ref is None:
                ref = sig
            assert sig == ref, f"contract drift: {sig} != {ref}"
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("contract ok:", ref)
    return ref


class StateOnlyWrapper(gym.ObservationWrapper):

    KEYS = ("proprio", "cloth_state", "task") # flat state vector for sb3, fixed key order is needed

    def __init__(self, env):
        super().__init__(env)
        dim = 0
        for k in self.KEYS:
            dim += env.observation_space[k].shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

    def observation(self, obs):
        parts = []
        for k in self.KEYS:
            parts.append(obs[k])
        return np.concatenate(parts).astype(np.float32)

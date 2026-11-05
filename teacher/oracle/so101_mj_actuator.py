"""Our fitted SO-101 arm (arm_sysid/so101_fitted.json, MuJoCo, held-out 0.284 deg) as Isaac Lab actuators.

MuJoCo model per joint:  tau = clip(kp*(q_cmd - q) - kv*qd, +-forcerange) - damping*qd - frictionloss*sgn(qd), plus
armature on the joint. Isaac Lab equivalents (Isaac Sim 5.1):
  kp, kv, forcerange -> implicit PhysX drive stiffness, damping, effort limit (same law, integrated implicitly)
  damping            -> PhysX joint viscous friction (velocity-proportional, implicit)
  armature           -> joint armature
  frictionloss       -> NOT PhysX joint friction: Isaac's static/dynamic joint friction is a unitless coefficient times
                        the force transmitted through the joint (load-dependent; IsaacLab actuator_base_cfg docs), while
                        ours is a constant torque. So it is applied as a feed-forward torque
                        -frictionloss * clip(qd / v_friction, -1, 1); PhysX joint friction is set to 0.
                        v_friction stands in for MuJoCo's friction softness (solref 0.008), which Isaac has no knob for.
Link masses are ported separately (isaac_arm_replay.port_masses): Menagerie CAD masses + the 23.1 g wrist camera.
"""
import json
import os

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FITTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arm_sysid", "so101_fitted.json")


class MjFrictionActuator(ImplicitActuator):
    """Implicit PD drive (as ImplicitActuator) plus a constant-torque Coulomb friction as feed-forward effort."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.frictionloss = self._parse_joint_parameter(cfg.frictionloss, 0.0)

    def compute(self, control_action, joint_pos, joint_vel):
        ff = -self.frictionloss * torch.clamp(joint_vel / self.cfg.v_friction, -1.0, 1.0)
        control_action.joint_efforts = ff if control_action.joint_efforts is None else control_action.joint_efforts + ff
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class MjFrictionActuatorCfg(ImplicitActuatorCfg):
    class_type: type = MjFrictionActuator
    frictionloss: dict | float = 0.0
    v_friction: float = 0.15       # rad/s; explicit friction slope stays < ~1.2 * armature / dt at dt = 1/90


def fitted_actuators(path=FITTED, v_friction=0.15):
    """{group name: MjFrictionActuatorCfg} for SO101_FOLLOWER_CFG.actuators, one group per joint."""
    J = json.load(open(path))["joints"]
    out = {}
    for j in JOINTS:
        p = J[j]
        out[f"fit-{j}"] = MjFrictionActuatorCfg(
            joint_names_expr=[j], stiffness=p["kp"], damping=p["kv"], effort_limit_sim=p["forcerange"],
            velocity_limit_sim=100.0, armature=p["armature"], friction=0.0, dynamic_friction=0.0,
            viscous_friction=p["damping"], frictionloss=p["frictionloss"], v_friction=v_friction)
    return out

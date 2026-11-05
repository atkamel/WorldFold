"""Put our fitted SO-101 arm (oracle/so101_mj_actuator.py) into LeHome's env, gated by TEACHER_ARM=fitted
(anything else = LeHome's original arm):
  lerobot.py        SO101_FOLLOWER_CFG actuators -> one fitted actuator per joint; arm gravity ON (LeHome disables it)
  garment_bi_v2.py  + 23.1 g wrist camera on each gripper link, after the env is built
Link masses and centres of mass need no port: LeHome's USD already matches the fitted model's (2026-10-18 dump:
masses identical, COMs within 0.1 mm, jaw 1.4 mm). Held-out replay: 10.0 % of real motion at dt 1/90 (stock 18 %).

    python arm_port_patch.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-arm-port-patch]"
CH = ROOT / "lehome-challenge" / "source" / "lehome" / "lehome"

p = CH / "assets" / "robots" / "lerobot.py"
s = p.read_text()
if MARK not in s:
    s += '''
''' + MARK + '''
import os as _os_arm
if _os_arm.environ.get("TEACHER_ARM") == "fitted":
    import sys as _sys_arm
    _od = _os_arm.environ.get("ORACLE_DIR", "/opt/teacher/oracle")
    if _od not in _sys_arm.path:
        _sys_arm.path.insert(0, _od)
    import so101_mj_actuator as _mj_arm
    SO101_FOLLOWER_CFG.actuators = _mj_arm.fitted_actuators()
    SO101_FOLLOWER_CFG.spawn.rigid_props.disable_gravity = False
    print("[arm] fitted SO-101 actuators, arm gravity on", flush=True)
'''
    p.write_text(s)
    print("patched:", p.name)
else:
    print("already patched:", p.name)

p = CH / "tasks" / "bedroom" / "garment_bi_v2.py"
s = p.read_text()
old = "        self.right_joint_pos = self.right_arm.data.joint_pos\n"
new = old + '''        try:  ''' + MARK + '''
            import os as _os_arm
            if _os_arm.environ.get("TEACHER_ARM") == "fitted":
                import torch as _t_arm
                for _arm in (self.left_arm, self.right_arm):
                    _v = _arm.root_physx_view
                    _m = _v.get_masses().clone()
                    _m[:, _arm.body_names.index("gripper")] += 0.0231      # wrist camera (fit: 23.1 g)
                    _v.set_masses(_m, _t_arm.arange(_m.shape[0]))
                print("[arm] +23.1 g wrist camera on both grippers", flush=True)
        except Exception as _e_arm:
            print(f"[arm] camera mass NOT set: {_e_arm!r}", flush=True)
'''
if MARK not in s:
    assert s.count(old) == 1, "anchor line not found in garment_bi_v2.py"
    p.write_text(s.replace(old, new))
    print("patched:", p.name)
else:
    print("already patched:", p.name)
print("ARM_PORT_PATCH_OK")

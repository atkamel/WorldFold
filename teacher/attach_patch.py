"""Attach + slip grasp model (oracle/attach_grasp.py) in the teacher's LeHome env: appended to GarmentEnv.__init__
in garment_bi_v2.py, active only when TEACHER_ATTACH=1. ORACLE_DIR points at teacher/oracle.

    python attach_patch.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-attach-patch]"
p = ROOT / "lehome-challenge/source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
s = p.read_text()
if MARK in s:
    print("already patched: attach")
    sys.exit(0)
anchor = "        self.right_joint_pos = self.right_arm.data.joint_pos\n"
assert s.count(anchor) == 1, "garment_bi_v2.py: __init__ anchor not found"
s = s.replace(anchor, anchor + f'''        try:  {MARK}
            import os as _os_att
            if _os_att.environ.get("TEACHER_ATTACH") == "1":
                import sys as _sys_att
                _od = _os_att.environ.get("ORACLE_DIR", "/opt/teacher/oracle")
                if _od not in _sys_att.path:
                    _sys_att.path.insert(0, _od)
                import attach_grasp as _att
                _att.install(self, log=lambda m: print(m, flush=True))
        except Exception as _e_att:
            print(f"ATTACH NOT installed: {{_e_att!r}}", flush=True)
''')
p.write_text(MARK + "\n" + s)
print("ATTACH_PATCH_OK")

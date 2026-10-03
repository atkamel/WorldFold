"""Save the cloth's vertices at the end of every teacher episode (for the neatness metric).

Runtime patch of lehome-challenge/scripts/utils/remote_loop.py, active only when TEACHER_VERTS_DIR is set.
At the end of each episode it saves the vertices as they are when the episode stops ("at_stop", grippers usually
still closed), then opens both grippers in place, lets the cloth settle 60 steps and saves them again ("released").

    python verts_dump_patch.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-verts-dump-patch]"
p = ROOT / "lehome-challenge/scripts/utils/remote_loop.py"
s = p.read_text()
if MARK in s:
    print("already patched: verts dump")
    sys.exit(0)
old = '    return {"return": episode_return, "length": length, "success": success}\n'
new = '''    try:
        import os as _os, time as _tm
        _d = _os.environ.get("TEACHER_VERTS_DIR")
        if _d:
            _os.makedirs(_d, exist_ok=True)
            _v = lambda: np.asarray(env.object.get_current_mesh_points()[0], dtype=np.float32)
            _g = getattr(policy, "_garment_name", None) or env.cfg.garment_name
            _stop = _v()
            _act = env.actions.clone()
            _act[:, 5] = 0.55; _act[:, 11] = 0.55
            for _ in range(60):
                env.step(_act)
            np.savez_compressed(f"{_d}/{_g}_{int(_tm.time() * 1000)}.npz", at_stop=_stop, released=_v(),
                                success=bool(success), length=int(length))
            print(f"[{label}] VERTS saved {_g} success={bool(success)} length={length}", flush=True)
    except Exception as _e:
        print(f"[{label}] VERTS failed: {_e!r}", flush=True)
''' + old
assert s.count(old) == 1, "return line not found"
p.write_text(MARK + "\n" + s.replace(old, new))
print("VERTS_PATCH_OK")

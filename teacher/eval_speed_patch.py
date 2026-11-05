"""Our measured speed fixes, applied to his teacher eval loop (they change no physics and no model input):

  1. cloth read via the buffer protocol (84c71f3 audit): Isaac Sim 5.1's SingleClothPrim._get_points_pose builds a
     torch tensor element by element (~160 ms per read); LeHome's success checks read the whole cloth. Same USD
     attribute, same values, ~0 ms. Installed when remote_loop.py is imported (the Isaac worker process).
  2. client-side 224 resize (795a21a): the worker resizes the three camera images with HIS function
     (openpi_client.image_tools.resize_with_pad) before sending; the server's ResizeImages short-circuits on 224x224,
     so the model sees the same pixels. Requests 3.7 MB -> 0.6 MB. Skipped (sent full size) if openpi_client is not
     importable from the openpi checkout.
Each is gated (TEACHER_FAST_CLOTH_READ=1, TEACHER_RESIZE224=1); OFF = original code.

    python eval_speed_patch.py <lehome_solution dir>
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-eval-speed-patch]"


def patch(rel, edits, prepend=""):
    p = ROOT / rel
    s = p.read_text()
    if MARK in s:
        print(f"already patched: {rel}")
        return
    for old, new in edits:
        n = s.count(old)
        assert n == 1, f"{rel}: expected 1 match, found {n} for:\n{old[:160]}"
        s = s.replace(old, new)
    p.write_text(MARK + "\n" + prepend + s)
    print(f"patched: {rel}")


# ---- 1. fast cloth read, installed at import of the worker's loop module ----
patch("lehome-challenge/scripts/utils/remote_loop.py", [], prepend='''import os as _os_esp
if _os_esp.environ.get("TEACHER_FAST_CLOTH_READ") == "1":
    try:
        import numpy as _np_esp
        import torch as _torch_esp
        from isaacsim.core.prims import SingleClothPrim as _SCP_esp

        def _fast_points_esp(self):
            return _torch_esp.from_numpy(_np_esp.array(self._prim.GetAttribute("points").Get(), dtype=_np_esp.float32))
        _SCP_esp._get_points_pose = _fast_points_esp
        print("[speed] fast cloth read installed", flush=True)
    except Exception as _e_esp:
        print(f"[speed] fast cloth read NOT installed: {_e_esp!r}", flush=True)
''')

# ---- 2. 224 resize in the worker's request serializer ----
RESIZE = '''
def _esp_resizer():
    """HIS resize_with_pad (openpi_client), or None when unavailable / switched off."""
    import os
    if os.environ.get("TEACHER_RESIZE224") != "1":
        return None
    try:
        import sys as _s
        p = os.environ.get("OPENPI_CLIENT_SRC", "/opt/lehome_solution/openpi/packages/openpi-client/src")
        if p not in _s.path:
            _s.path.insert(0, p)
        from openpi_client import image_tools
        print("[speed] client-side 224 resize on", flush=True)
        return image_tools.resize_with_pad
    except Exception as e:
        print(f"[speed] client-side 224 resize OFF: {e!r}", flush=True)
        return None


_ESP_RESIZE = _esp_resizer()

'''
patch("lehome-challenge/scripts/eval_policy/remote_policy.py", [
    ("                if \"images\" in key:\n                    arr = np.ascontiguousarray(value)\n",
     "                if \"images\" in key:\n"
     "                    if _ESP_RESIZE is not None and value.ndim == 3 and value.shape[-1] == 3:\n"
     "                        value = _ESP_RESIZE(value, 224, 224)\n"
     "                    arr = np.ascontiguousarray(value)\n"),
    # above the decorator: between it and the class, the decorator registered the helper instead (2026-10-19)
    ('\n@PolicyRegistry.register("remote")\n', RESIZE + '\n@PolicyRegistry.register("remote")\n'),
])
print("EVAL_SPEED_PATCH_OK")

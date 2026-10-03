"""Compact wrapper: send one or more live commands (JSON list or object), print a short summary after each."""
import json, subprocess, sys
cmds = json.loads(sys.argv[1]); cmds = cmds if isinstance(cmds, list) else [cmds]
for c in cmds:
    r = subprocess.run([sys.executable, "-W", "ignore", "live_client.py", json.dumps(c)], capture_output=True, text=True).stdout
    try:
        j = json.loads(r)
    except Exception:
        print("RAW", r[:400]); continue
    o = j.get("obs") or {}; i = j.get("info") or {}
    kp = o.get("keypoints_cm", {})
    short = {k: v for k, v in i.items() if k != "track_err_rad"}
    print(f"n={j.get('n')} {c.get('op')} {json.dumps(short)[:160]}")
    if "score" in i:
        s = i["score"]; print("  SCORE", {k: s[k] for k in ("neat", "vertex_err_cm", "mirror_err_cm", "h95_cm", "area_ratio", "rectangularity", "size_cm", "target_size_cm")})
    print("  tips", {a: v["pos"] for a, v in o.get("tips", {}).items()}, "held", {a: (h["verts_within_3cm"], h["their_height_cm"]) for a, h in o.get("held", {}).items()}, "hmax", o.get("cloth", {}).get("hmax_cm"))
    if kp and c.get("op") in ("home", "wait", "score", "reset"):
        for s in ("L", "R"):
            print(f"  {s}", {k: kp[s][k][:2] for k in ("cuff_mid", "armpit", "shoulder", "hem_out")})

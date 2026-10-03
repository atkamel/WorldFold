"""Agent wrapper: send commands, print compact fold/score results."""
import json, subprocess, sys
cmds = json.loads(sys.argv[1]); cmds = cmds if isinstance(cmds, list) else [cmds]
for c in cmds:
    r = subprocess.run([sys.executable, "-W", "ignore", "live_client.py", json.dumps(c)], capture_output=True, text=True).stdout
    try:
        j = json.loads(r)
    except Exception:
        print("RAW", r[:300]); continue
    i = j.get("info") or {}; o = j.get("obs") or {}
    sc = i.get("score_step") or i.get("score") or o.get("score")
    if isinstance(sc, dict) and "score" in sc: sc = sc["score"]
    arms = {a: {k: v for k, v in d.items() if k in ("align", "held_closed", "held_at_place", "stuck")} for a, d in (i.get("arms") or {}).items()}
    extra = {k: v for k, v in i.items() if k not in ("arms", "score_step", "score", "track_err_rad")}
    print(f"n={j.get('n')} {c.get('op')} {c.get('step','')} {json.dumps(arms)} {json.dumps(extra)[:200]}")
    if isinstance(sc, dict):
        print("   SCORE", {k: sc.get(k) for k in ("neat", "neat_area", "s_area", "s_rect", "vertex_err_cm", "mirror_err_cm", "h95_cm", "area_ratio", "rectangularity", "size_cm", "target_size_cm")})
    print("   done", o.get("done"))

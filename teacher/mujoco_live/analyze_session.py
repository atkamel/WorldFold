"""Summarise a game session log: what the player did, and where the arms lagged or jammed.
Usage: python analyze_session.py demos/<session>.jsonl"""
import sys, json
import numpy as np

recs = [json.loads(l) for l in open(sys.argv[1])]
frames = [r for r in recs if r.get("frame")]
print(f"records {len(recs)}, frames {len(frames)} ({len(frames) / 10 / 60:.1f} min of play)")

# split into episodes at resets
eps, cur = [], []
for r in recs:
    if r.get("reset"):
        eps.append(cur); cur = []
    else:
        cur.append(r)
eps.append(cur)
print(f"episodes (separated by R resets): {len(eps)}")

def summarize(ep, label):
    ev = [r for r in ep if r.get("event") in ("grasp", "release") or r.get("undo")]
    fr = [r for r in ep if r.get("frame")]
    if not fr:
        return
    t0 = fr[0]["t_sim"]
    print(f"\n=== {label}: {len(fr) / 10:.0f} s of play, {sum(1 for r in ev if r.get('event') == 'grasp')} grabs, "
          f"{sum(1 for r in ev if r.get('undo'))} undos ===")
    for r in ev:
        if r.get("undo"):
            print(f"  t={r.get('t_sim', 0) - t0:6.1f}s  UNDO"); continue
        sc = r.get("score", {})
        extra = f"grabbed {r['grabbed']:2d} pts" if r["event"] == "grasp" else f"area {sc.get('area_cm2')} cm2 ({sc.get('area_vs_flat')}), box {sc.get('rect_cm')}"
        print(f"  t={r['t_sim'] - t0:6.1f}s  {r['arm']} {r['event']:7s} at ({r['tip_cm'][0]:6.1f}, {r['tip_cm'][1]:5.1f}, {r['tip_cm'][2]:4.1f}) cm  {extra}")
    # tandem: grabs by both arms within 1 s of each other
    g = [(r["t_sim"], r["arm"]) for r in ev if r.get("event") == "grasp"]
    tandem = sum(1 for i in range(len(g) - 1) if g[i][1] != g[i + 1][1] and g[i + 1][0] - g[i][0] < 1.0)
    print(f"  two-hand grabs (both arms within 1 s): {tandem}")
    # tracking: how far each arm's tip lagged its target; long lags = jams
    for a in ("L", "R"):
        tgt = np.array([r["target_cm"][a] for r in fr]); tip = np.array([r["tip_cm"][a] for r in fr])
        err = np.linalg.norm(tgt - tip, axis=1)
        moving = np.r_[0, np.linalg.norm(np.diff(tgt, axis=0), axis=1)] > 0.05
        stuck = np.r_[0, np.linalg.norm(np.diff(tgt[:, :2], axis=0), axis=1)] < 1e-6
        # target frozen while the arm is mid-air = probably held at the reach limit
        print(f"  {a}: tip-target error median {np.median(err):.2f} cm, p95 {np.percentile(err, 95):.2f} cm, max {err.max():.1f} cm; "
              f"frames with error > 1 cm: {(err > 1).sum()}; target moving {moving.mean() * 100:.0f}% of frames; "
              f"x range {tgt[:, 0].min():.1f}..{tgt[:, 0].max():.1f}, y range {tgt[:, 1].min():.1f}..{tgt[:, 1].max():.1f}, z range {tgt[:, 2].min():.1f}..{tgt[:, 2].max():.1f} cm")
        # carry height changes (wheel) while carrying
        ph = [r.get("phase", {}).get(a) for r in fr]
        zc = [tgt[i, 2] for i in range(len(fr)) if ph[i] == "carry"]
        print(f"     carrying {sum(p == 'carry' for p in ph) / 10:.0f} s; carry-height range {min(zc) if zc else 0:.1f}..{max(zc) if zc else 0:.1f} cm")

for i, ep in enumerate(eps):
    if any(r.get("frame") for r in ep):
        summarize(ep, f"episode {i + 1}" + (" (last)" if i == len(eps) - 1 else ""))

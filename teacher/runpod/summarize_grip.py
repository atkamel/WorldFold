"""Summarise the RunPod grip test (adhesion A/B): one line per setup.
    python summarize_grip.py <results dir>
grip_assay rows (assay_<pads>.json): 'held' fraction after lift / hold / carry.
mechanics rows (mech_release_<garment>.json): cloth stuck to the gripper after release, drag of the released cloth."""
import glob, json, os, sys
import numpy as np

out = sys.argv[1]
mean = lambda xs: round(float(np.mean(xs)), 3) if xs else None

for f in sorted(glob.glob(os.path.join(out, "assay_*.json"))):
    rows = json.load(open(f))
    for cloth in sorted({r.get("cloth") for r in rows}):
        for speed in sorted({r.get("speed_mm_step") for r in rows if r.get("cloth") == cloth}, key=lambda s: s or 0):
            rs = [r for r in rows if r.get("cloth") == cloth and r.get("speed_mm_step") == speed]
            held = {ph: [r[ph]["held"] for r in rs if isinstance(r.get(ph), dict)] for ph in ("after_lift", "after_hold", "after_carry")}
            print(f"GRIP  {os.path.basename(f)[6:-5]:7s} cloth={cloth:11s} speed={speed} mm/step  trials={len(rs)} "
                  f"errors={sum('error' in r for r in rs)}  held: lift {mean(held['after_lift'])}  hold {mean(held['after_hold'])}  "
                  f"carry {mean(held['after_carry'])}")

for f in sorted(glob.glob(os.path.join(out, "mech_release_*.json"))):
    rows = json.load(open(f))
    for cloth in sorted({r.get("cloth") for r in rows}):
        rs = [r for r in rows if r.get("cloth") == cloth]
        ok = [r for r in rs if "error" not in r]
        print(f"RELEASE cloth={cloth:11s} trials={len(rs)} errors={len(rs) - len(ok)}  "
              f"stuck to gripper: {sum(bool(r.get('stuck')) for r in ok)}/{len(ok)}  "
              f"drag {mean([r['drag_cm'] for r in ok if 'drag_cm' in r])} cm  "
              f"cloth lifted max {mean([r['lifted_cloth_max_cm'] for r in ok if 'lifted_cloth_max_cm' in r])} cm")
print("(real = LeHome adhesion 0.1, real_noadh = adhesion 0 as in Adam's env; both 50 g cloth, gravity x1, flat pads)")

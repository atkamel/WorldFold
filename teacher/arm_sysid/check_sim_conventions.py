"""Checks the zero pose, joint signs and jaw geometry of the SO-101 MJCF models.

Run:  python -I check_sim_conventions.py [path/to/episode.parquet]

What it checks (all numbers printed come from MuJoCo forward kinematics):
  1. TRS so101_new_calib.xml and Menagerie robotstudio_so101/so101.xml have identical
     kinematics (body poses compared at random joint configurations).
  2. Shape of the sim zero pose (upper arm / forearm / gripper directions in the world),
     to compare with LeRobot's calibration diagram (so_follower.py _HOMING_POSITION_DIAGRAM).
  3. What a positive rotation of each joint does to the jaw tip.
  4. Rest-pose sign test: LeRobot's own "comfortable rest" pose (examples/isaac_teleop_to_so101/
     common.py RESET_ORIGIN_DEG) fed with identity signs vs flipped signs.
  5. Jaw gap vs gripper angle using Menagerie's primitive jaw pads/tips (meshes are not present
     locally, so mesh geoms are stripped; primitives remain).
  6. Optional: gravity sign test on a real LeRobot episode (sign of action-state at rest vs
     sign of the sim gravity torque at that pose, under identity mapping).

Mesh files are not available locally, so every geom that references a mesh is removed before
loading. Inertials are explicit in both files, so masses/inertias are unaffected (except the
Menagerie camera-mount mesh geom which carries mass 0.012 kg; it is dropped here).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

SCRATCH = Path(
    "C:/Users/valen/AppData/Local/Temp/claude/C--Users-valen-Documents-vla-distill/"
    "c1e14d5c-c49d-4f52-b957-d99b0501209c/scratchpad/so101"
)
TRS_XML = SCRATCH / "t_SO101_so101_new_calib.xml"
MEN_XML = SCRATCH / "m_robotstudio_so101_so101.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_without_meshes(path: Path) -> mujoco.MjModel:
    tree = ET.parse(path)
    root = tree.getroot()
    # drop mesh assets
    for asset in root.iter("asset"):
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
    # find classes whose default geom type is mesh (e.g. Menagerie "visual" class)
    mesh_classes = set()
    for d in root.iter("default"):
        g = d.find("geom")
        if g is not None and g.get("type") == "mesh" and d.get("class"):
            mesh_classes.add(d.get("class"))

    def strip(parent):
        for child in list(parent):
            if child.tag == "geom" and (
                child.get("mesh") is not None
                or child.get("type") == "mesh"
                or (child.get("type") is None and child.get("class") in mesh_classes)
            ):
                parent.remove(child)
            else:
                strip(child)

    for wb in root.iter("worldbody"):
        strip(wb)
    compiler = root.find("compiler")
    if compiler is not None and "meshdir" in compiler.attrib:
        del compiler.attrib["meshdir"]
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)


def qpos_for(m, deg: dict[str, float]) -> np.ndarray:
    q = np.zeros(m.nq)
    for j, v in deg.items():
        q[m.joint(j).qposadr[0]] = np.deg2rad(v)
    return q


def fk(m, d, q):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)


def joint_anchor(m, d, name):
    return d.xanchor[m.joint(name).id].copy()


def main(parquet: str | None):
    np.set_printoptions(precision=4, suppress=True)
    mt = load_without_meshes(TRS_XML)
    mm = load_without_meshes(MEN_XML)
    dt, dm = mujoco.MjData(mt), mujoco.MjData(mm)

    # ---- 1. kinematic identity --------------------------------------------------------------
    rng = np.random.default_rng(0)
    worst_p, worst_r = 0.0, 0.0
    bodies = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
    for _ in range(200):
        deg = {j: rng.uniform(np.rad2deg(mt.joint(j).range[0]), np.rad2deg(mt.joint(j).range[1])) for j in JOINTS}
        fk(mt, dt, qpos_for(mt, deg))
        fk(mm, dm, qpos_for(mm, deg))
        for b in bodies:
            worst_p = max(worst_p, np.abs(dt.body(b).xpos - dm.body(b).xpos).max())
            worst_r = max(worst_r, np.abs(dt.body(b).xmat - dm.body(b).xmat).max())
    print(f"[1] TRS vs Menagerie body poses over 200 random configs: max |dpos| = {worst_p*1e3:.4f} mm, "
          f"max |dR| = {worst_r:.2e}")
    for j in JOINTS:
        rt, rm = np.rad2deg(mt.joint(j).range), np.rad2deg(mm.joint(j).range)
        print(f"    {j:13s} TRS range deg {rt.round(3)}  mid {rt.mean():+.3f} | Menagerie {rm.round(3)} mid {rm.mean():+.3f}")

    # ---- 2. zero pose shape -------------------------------------------------------------------
    m, d = mt, dt
    fk(m, d, np.zeros(m.nq))
    a_lift = joint_anchor(m, d, "shoulder_lift")
    a_elb = joint_anchor(m, d, "elbow_flex")
    a_wf = joint_anchor(m, d, "wrist_flex")
    tip = d.site("gripperframe").xpos.copy()

    def unit(v):
        return v / np.linalg.norm(v)

    print("[2] Sim zero pose (TRS new_calib), world frame: +z up; base at origin")
    print(f"    shoulder_lift anchor {a_lift}, elbow anchor {a_elb}, wrist_flex anchor {a_wf}, gripperframe site {tip}")
    up = unit(a_elb - a_lift)
    fo = unit(a_wf - a_elb)
    gr = unit(tip - a_wf)
    ang = lambda v: np.rad2deg(np.arctan2(v[2], np.hypot(v[0], v[1])))
    print(f"    upper arm dir {up}  elevation {ang(up):+.1f} deg")
    print(f"    forearm   dir {fo}  elevation {ang(fo):+.1f} deg")
    print(f"    wrist->gripperframe dir {gr}  elevation {ang(gr):+.1f} deg")
    print("    joint axes (world) at zero:")
    for j in JOINTS:
        print(f"      {j:13s} axis {d.xaxis[m.joint(j).id]}")

    # ---- 3. positive rotation semantics --------------------------------------------------------
    print("[3] Effect of +10 deg on each joint from zero pose (gripperframe site displacement, m):")
    for j in JOINTS[:5]:
        fk(m, d, qpos_for(m, {j: 10.0}))
        print(f"    +{j:13s} dtip = {d.site('gripperframe').xpos - tip}")

    # ---- 4. rest-pose sign test ---------------------------------------------------------------
    rest = {"shoulder_pan": -4.0, "shoulder_lift": -103.0, "elbow_flex": 97.0, "wrist_flex": 78.0,
            "wrist_roll": -65.0, "gripper": 0.0}

    def clip(deg):
        out = {}
        for j, v in deg.items():
            lo, hi = np.rad2deg(m.joint(j).range)
            out[j] = float(np.clip(v, lo, hi))
        return out

    def report(label, deg):
        deg = clip(deg)
        fk(m, d, qpos_for(m, deg))
        zs = {b: d.body(b).xpos[2] for b in ["lower_arm", "wrist", "gripper"]}
        print(f"    {label:38s} wrist z={zs['wrist']*100:6.1f} cm, gripper body z={zs['gripper']*100:6.1f} cm, "
              f"jaw site z={d.site('gripperframe').xpos[2]*100:6.1f} cm, horiz reach={np.hypot(*d.site('gripperframe').xpos[:2])*100:5.1f} cm")

    print("[4] LeRobot RESET_ORIGIN_DEG rest pose (clipped to sim limits); base bottom is z=0 (table):")
    report("identity signs", rest)
    flipped = dict(rest)
    for j in ["shoulder_lift", "elbow_flex", "wrist_flex"]:
        flipped[j] = -rest[j]
    report("lift/elbow/wrist_flex sign-flipped", flipped)
    rest2 = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 90.0, "wrist_flex": 50.0,
             "wrist_roll": 0.0, "gripper": -10.0}
    report("LeIsaac rest pose (identity)", rest2)

    # ---- 5. jaw gap vs gripper angle (Menagerie primitives) ----------------------------------
    mm_, dm_ = mm, dm
    fixed = [mm_.geom(n).id for n in ["fixed_jaw_box4", "fixed_jaw_box5", "fixed_jaw_box6", "fixed_jaw_box7",
                                       "fixed_jaw_sph_tip1", "fixed_jaw_sph_tip2", "fixed_jaw_sph_tip3", "fixed_jaw_box3"]]
    moving = [mm_.geom(n).id for n in ["moving_jaw_box2", "moving_jaw_box3", "moving_jaw_sph_tip1",
                                        "moving_jaw_sph_tip2", "moving_jaw_sph_tip3"]]
    print("[5] Menagerie jaw primitives: min signed distance (mm) between moving and fixed jaw pads/tips vs gripper angle")
    fromto = np.zeros(6)
    rows = []
    for gdeg in [-10, -5, -2, 0, 2, 5, 10, 20, 30, 45, 60, 80, 100]:
        fk(mm_, dm_, qpos_for(mm_, {"gripper": gdeg}))
        dmin = min(mujoco.mj_geomDistance(mm_, dm_, a, b, 0.2, fromto) for a in moving for b in fixed)
        tipsep = np.linalg.norm(dm_.geom("moving_jaw_sph_tip1").xpos - dm_.geom("fixed_jaw_sph_tip1").xpos)
        rows.append((gdeg, dmin * 1e3, tipsep * 1e3))
        print(f"    gripper {gdeg:+5.0f} deg: min pad/tip distance {dmin*1e3:7.2f} mm, tip-sphere centre separation {tipsep*1e3:6.1f} mm")
    # find the contact angle by bisection
    lo, hi = -10.0, 30.0

    def gap(gdeg):
        fk(mm_, dm_, qpos_for(mm_, {"gripper": gdeg}))
        return min(mujoco.mj_geomDistance(mm_, dm_, a, b, 0.2, fromto) for a in moving for b in fixed)

    if gap(lo) < 0 < gap(hi):
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
        print(f"    -> jaw primitives first touch at gripper = {0.5*(lo+hi):+.2f} deg (closing direction = decreasing angle)")
    else:
        print(f"    -> no sign change of the gap between {lo} and {hi} deg (gap({lo})={gap(lo)*1e3:.2f} mm, gap({hi})={gap(hi)*1e3:.2f} mm)")

    # ---- 6. gravity sign test on a real episode -------------------------------------------------
    if parquet:
        import pandas as pd

        df = pd.read_parquet(parquet)
        s = np.stack(df["observation.state"].values)
        a = np.stack(df["action"].values)
        arms = [(slice(0, 6), slice(0, 6))]
        if s.shape[1] >= 24 and a.shape[1] >= 18:  # bimanual custom layout seen in neuroto_ep0
            arms = [(slice(0, 6), slice(0, 6)), (slice(18, 24), slice(12, 18))]
        print(f"[6] Gravity sign test on {parquet} (state {s.shape}, action {a.shape})")
        from lerobot_units import infer_units, to_radians

        for ai, (ss, as_) in enumerate(arms):
            S, A = s[:, ss], a[:, as_]
            inf = infer_units(S)
            print(f"    arm {ai}: units inferred from quantisation = {inf['units']}; converting with follower spans")
            spans = inf["spans_deg"] if inf["units"] == "m100" else None
            S = np.rad2deg(to_radians(S, inf["units"], spans_deg=spans))
            A = np.rad2deg(to_radians(A, inf["units"], spans_deg=spans))
            # quasi-static frames: target and state both nearly constant over +-3 frames
            k = 3
            ok = np.zeros(len(S), bool)
            for t in range(k, len(S) - k):
                if (np.abs(A[t - k:t + k + 1] - A[t]).max() < 0.5) and (np.abs(S[t - k:t + k + 1] - S[t]).max() < 0.5):
                    ok[t] = True
            idx = np.where(ok)[0]
            tested = ["shoulder_lift", "elbow_flex", "wrist_flex"]
            print(f"    arm {ai}: {len(idx)} quasi-static frames. % frames where sign(action-state) == sign(sim gravity "
                  f"torque) [n frames], per sign hypothesis (|tau|>0.05 N.m, |err|>0.3 deg):")
            for hyp in ["identity"] + [f"flip {j}" for j in tested]:
                Sh, Ah = S.copy(), A.copy()
                if hyp != "identity":
                    jf = JOINTS.index(hyp.split()[1])
                    Sh[:, jf] *= -1
                    Ah[:, jf] *= -1
                res = {j: [] for j in tested}
                for t in idx:
                    fk(m, d, qpos_for(m, {j: Sh[t, i] for i, j in enumerate(JOINTS[:5])}))
                    bias = d.qfrc_bias.copy()
                    for j in tested:
                        i = JOINTS.index(j)
                        tau = bias[m.joint(j).dofadr[0]]
                        err = Ah[t, i] - Sh[t, i]
                        if abs(tau) > 0.05 and abs(err) > 0.3:
                            res[j].append(np.sign(tau) == np.sign(err))
                cells = "  ".join(f"{j}: {np.mean(v)*100:5.1f}% [{len(v)}]" if v else f"{j}:   n/a  [0]" for j, v in res.items())
                print(f"      {hyp:20s} {cells}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

"""Full 3D replay of teacher folds: both SO-101 arms (real meshes, so101_new_calib.urdf + Menagerie STLs) and the cloth,
rebuilt from the harvest run's snapshots (cloth points, garment pose and both arms' joint angles every 5 policy steps),
so Roy can watch what the teacher does, not just where the cloth ends.

    python fold_videos_3d.py <harvest run dir> <garments_flat dir> <out.mp4> GARMENT [GARMENT ...]

Camera: three-quarter view from the far side of the table, looking back at the robots. Painter's-algorithm render with
Lambert shading (PIL); meshes decimated to ~1.5k triangles per part.
"""
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TEACHER = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from fold_videos import world  # noqa: E402

URDF = os.path.join(TEACHER, "oracle", "so101_new_calib.urdf")
MESHES = os.path.join(TEACHER, "arm_sysid", "assets", "menagerie_so101")
# URDF base_link in the world (oracle/so101_kin.py: Rz(+90 deg), measured USD offset)
_OFF = np.array([-0.0208, -0.0157, 0.0325])
BASES = {"left": np.array([-0.23, -0.25, 0.5]) + _OFF, "right": np.array([0.23, -0.25, 0.5]) + _OFF}
BASE_ROT = np.array([[0.0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])
W, Hh = 960, 720
LIGHT = np.array([0.4, 0.5, 0.75]); LIGHT /= np.linalg.norm(LIGHT)
ARM_COLOR = np.array([235, 205, 60.0])
CLOTH_COLOR = np.array([70, 130, 200.0])
TABLE_Z = 0.5


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def origin(el):
    o = el.find("origin")
    if o is None:
        return np.eye(4)
    return T(rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()]), [float(v) for v in o.get("xyz", "0 0 0").split()])


class Arm:
    def __init__(self):
        import fast_simplification
        import trimesh
        root = ET.parse(URDF).getroot()
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.child_of = {}
        for j in self.joints.values():
            self.child_of.setdefault(j.find("parent").get("link"), []).append(j)
        self.visuals = {}
        cache = {}
        for link in root.findall("link"):
            vs = []
            for v in link.findall("visual"):
                m = v.find("geometry/mesh")
                if m is None:
                    continue
                f = os.path.join(MESHES, m.get("filename"))
                if f not in cache:
                    tm = trimesh.load(f)
                    keep = min(1.0, 1500 / max(len(tm.faces), 1))
                    if keep < 1.0:
                        pv, fv = fast_simplification.simplify(np.asarray(tm.vertices, np.float32),
                                                              np.asarray(tm.faces, np.int32), 1 - keep)
                    else:
                        pv, fv = tm.vertices, tm.faces
                    cache[f] = (np.asarray(pv, float), np.asarray(fv, np.int64))
                vs.append((origin(v), *cache[f]))
            self.visuals[link.get("name")] = vs

    def triangles(self, q6, base_T):
        """World triangles (K,3,3) of the arm at joints q6 (5 arm joints + gripper)."""
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        qd = dict(zip(names, q6))
        out = []

        def walk(link, M):
            for (O, pv, fv) in self.visuals.get(link, []):
                P = (M @ O)
                out.append((pv @ P[:3, :3].T + P[:3, 3])[fv])
            for j in self.child_of.get(link, []):
                A = M @ origin(j)
                if j.get("type") == "revolute":
                    ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                    a = qd.get(j.get("name"), 0.0)
                    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                    A = A @ T(np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K, [0, 0, 0])
                walk(j.find("child").get("link"), A)
        walk("base_link", base_T)
        return np.concatenate(out) if out else np.zeros((0, 3, 3))


class Camera:
    def __init__(self, eye, target, fov_deg=45):
        f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.R, self.eye = np.stack([r, u, f]), np.asarray(eye, float)
        self.fpx = (W / 2) / np.tan(np.radians(fov_deg / 2))

    def project(self, p):
        c = (p - self.eye) @ self.R.T
        z = np.maximum(c[..., 2], 1e-3)
        return np.stack([W / 2 + self.fpx * c[..., 0] / z, Hh / 2 - self.fpx * c[..., 1] / z], -1), c[..., 2]


def draw(d, cam, tris, color):
    if len(tris) == 0:
        return []
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.3 + 0.7 * np.abs(n @ LIGHT)
    px, depth = cam.project(tris)
    return [(depth[i].mean(), px[i], tuple(int(c) for c in color * shade[i])) for i in range(len(tris))]


def render(cam, items, text):
    img = Image.new("RGB", (W, Hh), (238, 236, 230))
    d = ImageDraw.Draw(img)
    # table top
    tbl = np.array([[-0.45, -0.32, TABLE_Z], [0.45, -0.32, TABLE_Z], [0.45, 0.38, TABLE_Z], [-0.45, 0.38, TABLE_Z]])
    tp, _ = cam.project(tbl)
    d.polygon([tuple(p) for p in tp], fill=(250, 250, 248), outline=(200, 200, 200))
    for _, p, c in sorted(items, key=lambda t: -t[0]):
        d.polygon([tuple(x) for x in p], fill=c)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, W, 34], fill=(30, 30, 30))
    d.text((10, 6), text, fill=(255, 255, 255), font=font)
    return img


def main():
    import cv2
    run, flat_dir, out = sys.argv[1:4]
    garments = sys.argv[4:]
    arm = Arm()
    cam = Camera(eye=[0.55, 0.75, 1.15], target=[0.0, -0.05, 0.52], fov_deg=50)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 6, (W, Hh))
    bases = {s: T(BASE_ROT, BASES[s]) for s in BASES}
    for rank, g in enumerate(garments, 1):
        fl = np.load(os.path.join(flat_dir, f"flat_{g}.npz"))
        L, Wp = fl["local_points"], fl["world_points"]
        scale = np.linalg.norm(Wp - Wp.mean(0), axis=1).mean() / np.linalg.norm(L - L.mean(0), axis=1).mean()
        faces = fl["faces"].astype(np.int64)
        sub = faces
        fs = sorted(glob.glob(os.path.join(run, "snapshots", g, "*.npz")), key=lambda f: int(f.rsplit("_t", 1)[1][:4]))
        for k, f in enumerate(fs):
            d = np.load(f)
            x = world(d, scale)
            items = draw(None, cam, x[sub], CLOTH_COLOR)
            items += draw(None, cam, arm.triangles(d["left_joint_pos"], bases["left"]), ARM_COLOR)
            items += draw(None, cam, arm.triangles(d["right_joint_pos"], bases["right"]), ARM_COLOR)
            t = int(d["t"])
            img = render(cam, items, f"Teacher fold #{rank}: {g}    step {t} ({t / 30:.1f} s)")
            frame = np.asarray(img)[..., ::-1].copy()
            reps = 12 if k == len(fs) - 1 else 1                   # hold the final shape 2 s
            for _ in range(reps):
                vw.write(frame)
        print(f"{g}: {len(fs)} frames", flush=True)
    vw.release()
    print("wrote", out)


if __name__ == "__main__":
    main()

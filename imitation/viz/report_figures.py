"""Figures and data for the report's "The data" section (reads frozen data only).

    python -m imitation.viz.report_figures --out docs/reports/media

Writes:
  obs_layout.json     the 139-D vector as blocks: index range, name, units, sensor/privileged
  episode_trace.json  one clean expert episode and one knocked-off-course student episode,
                      over time: fold score, corner-to-goal distance of the carried corners,
                      grasp flags, gripper commands, settle counter, the 12 action channels,
                      who acted (teacher / student / perturbation)
  obs_heatmap.png     the whole 139-D vector over the clean episode, z-scored per dim with
                      the v1 normalizer (rows = dims, columns = steps)
  cams_panel.png      main + both wrist cameras at 4 moments (start, grasp, carry, settled),
                      upscaled nearest-neighbour so the real pixel grid stays visible
  dr_grid.png         the main camera at t = 0 for 6 seeds: the visual randomization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from imitation.data.collect import DEFAULT_ROOT
from imitation.data.dataset import Normalizer
from imitation.data.schema import ACTOR_PERTURB, ACTOR_STUDENT, load_dataset, load_manifest
from imitation.spec import OBS_SUBSETS

CORNER_TO_GOAL = slice(107, 119)

# (start, end_inclusive, block, name, units, sensor-available)
LAYOUT = [
    (0, 4, "proprio L", "joint positions", "rad", True),
    (5, 9, "proprio L", "joint velocities", "rad/s", True),
    (10, 10, "proprio L", "gripper command", "ctrl", True),
    (11, 13, "proprio L", "end-effector position", "m", True),
    (14, 17, "proprio L", "end-effector quaternion", "", True),
    (18, 23, "proprio L", "end-effector velocity (ang, lin)", "rad/s, m/s", True),
    (24, 24, "proprio L", "grasp flag (sim weld)", "0/1", False),
    (25, 29, "proprio R", "joint positions", "rad", True),
    (30, 34, "proprio R", "joint velocities", "rad/s", True),
    (35, 35, "proprio R", "gripper command", "ctrl", True),
    (36, 38, "proprio R", "end-effector position", "m", True),
    (39, 42, "proprio R", "end-effector quaternion", "", True),
    (43, 48, "proprio R", "end-effector velocity (ang, lin)", "rad/s, m/s", True),
    (49, 49, "proprio R", "grasp flag (sim weld)", "0/1", False),
    (50, 61, "cloth", "4 corner positions", "m", False),
    (62, 73, "cloth", "4 corner velocities", "m/s", False),
    (74, 100, "cloth", "9 sampled vertices", "m", False),
    (101, 103, "cloth", "centre of mass", "m", False),
    (104, 106, "cloth", "height min / max / mean", "m", False),
    (107, 118, "cloth", "corner-to-goal vectors", "m", False),
    (119, 119, "task", "mean corner progress", "0-1", False),
    (120, 131, "task", "goal keypoints", "m", True),
    (132, 135, "task", "per-corner progress", "0-1", False),
    (136, 136, "task", "time left", "0-1", True),
    (137, 137, "wrapper", "stage", "0-1", True),
    (138, 138, "wrapper", "settle counter", "0-1", False),
]


def layout_json():
    sensor = set(OBS_SUBSETS["proprio"])
    blocks = [{"start": a, "end": b, "dims": b - a + 1, "block": blk, "name": n, "units": u,
               "policy_sensor": all(i in sensor for i in range(a, b + 1)), "robot_available": avail}
              for a, b, blk, n, u, avail in LAYOUT]
    assert sum(x["dims"] for x in blocks) == 139
    return {"obs_dim": 139, "base_env_dim": 141, "blocks": blocks,
            "note": "base env emits 141 (4-dim task one-hot); M1.1 drops it and appends stage + settle"}


def trace(ep, label):
    d = np.linalg.norm(ep.obs[:, CORNER_TO_GOAL].reshape(-1, 4, 3), axis=2)
    carried = np.argsort(d[0])[-2:]                    # the two corners that start far from goal
    r = lambda x: [round(float(v), 4) for v in x]      # noqa: E731
    return {"label": label, "seed": ep.meta["seed"], "success": bool(ep.meta["success"]),
            "source": ep.meta["source"], "steps": ep.steps, "dt": 0.05,
            "fold_score": r(ep.fold_score), "corner_to_goal": [r(d[:, c]) for c in carried],
            "grasp_left": ep.grasped[:, 0].astype(int).tolist(), "grasp_right": ep.grasped[:, 1].astype(int).tolist(),
            "gripper_left": r(ep.actions[:, 5]), "gripper_right": r(ep.actions[:, 11]),
            "settle": r(ep.obs[:, 138]), "actions": [r(ep.actions[:, k]) for k in range(12)],
            "actor": ep.actor.astype(int).tolist()}


def heatmap(ep, norm, path, cell=(3, 3)):
    z = np.clip((ep.obs - norm.mean) / norm.std, -3, 3).T          # [139, T]
    h, w = z.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    pos, neg = np.clip(z / 3, 0, 1), np.clip(-z / 3, 0, 1)         # diverging: red above mean, blue below
    base = 246
    rgb[..., 0] = (base - neg * 190).astype(np.uint8)
    rgb[..., 1] = (base - (pos + neg) * 150).astype(np.uint8)
    rgb[..., 2] = (base - pos * 190).astype(np.uint8)
    img = Image.fromarray(rgb).resize((w * cell[0], h * cell[1]), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for a, *_ in LAYOUT[1:]:
        if a in (25, 50, 119, 137):                                 # block boundaries
            draw.line([(0, a * cell[1]), (img.width, a * cell[1])], fill=(40, 40, 40), width=1)
    img.save(path)
    return {"rows": h, "cols": w, "cell": cell, "boundaries": [25, 50, 119, 137]}


def cams_panel(ep, path, size=192):
    g = ep.grasped.any(1)
    first_grasp = int(np.argmax(g)) if g.any() else ep.steps // 3
    held = np.flatnonzero(g)
    carry = int(held[len(held) // 2]) if len(held) else ep.steps // 2
    moments = [("start", 0), ("grasp", first_grasp), ("carry", carry), ("settled", ep.steps - 1)]
    cams = [c for c in ("main", "left_wrist_cam", "right_wrist_cam") if c in ep.images]
    pad = 6
    panel = Image.new("RGB", (len(cams) * (size + pad) - pad, len(moments) * (size + pad) - pad), (255, 255, 255))
    for i, (_, t) in enumerate(moments):
        for j, c in enumerate(cams):
            frame = Image.fromarray(ep.images[c][t].transpose(1, 2, 0)).resize((size, size), Image.NEAREST)
            panel.paste(frame, (j * (size + pad), i * (size + pad)))
    panel.save(path)
    return {"moments": [{"name": n, "step": t} for n, t in moments], "cameras": cams,
            "native": {c: int(ep.images[c].shape[-1]) for c in cams}, "tile": size}


def dr_grid(eps, path, size=160, n=6):
    pad = 6
    pick = eps[:n]
    grid = Image.new("RGB", (3 * (size + pad) - pad, 2 * (size + pad) - pad), (255, 255, 255))
    for k, ep in enumerate(pick):
        frame = Image.fromarray(ep.images["main"][0].transpose(1, 2, 0)).resize((size, size), Image.NEAREST)
        grid.paste(frame, ((k % 3) * (size + pad), (k // 3) * (size + pad)))
    grid.save(path)
    return {"seeds": [int(e.meta["seed"]) for e in pick]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--demos", default="v1")
    ap.add_argument("--images", default="v1_img128")
    ap.add_argument("--student", default=None, help="a version with knocked student episodes (e.g. harvest_v1)")
    ap.add_argument("--out", default="docs/reports/media")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _, demos = load_dataset(args.root, args.demos)
    norm = Normalizer.fit(demos)
    clean = next(e for e in demos if not e.meta.get("perturb"))
    info = {"layout": "obs_layout.json"}
    (out / "obs_layout.json").write_text(json.dumps(layout_json(), indent=1))

    traces = [trace(clean, "clean expert demo")]
    if args.student:
        _, stud = load_dataset(args.root, args.student)
        knocked = next((e for e in stud if e.meta.get("perturb") and (e.actor == ACTOR_PERTURB).any()
                        and (e.actor == ACTOR_STUDENT).any()), None)
        if knocked is not None:
            traces.append(trace(knocked, "student, knocked off course"))
    (out / "episode_trace.json").write_text(json.dumps(traces))
    info["heatmap"] = heatmap(clean, norm, out / "obs_heatmap.png") | {"seed": clean.meta["seed"]}

    m = load_manifest(args.root, args.images)
    _, img_eps = load_dataset(args.root, args.images, images=True)
    by_seed = {e.meta["seed"]: e for e in img_eps}
    ep = by_seed.get(clean.meta["seed"], img_eps[0])
    info["cams"] = cams_panel(ep, out / "cams_panel.png") | {"seed": ep.meta["seed"], "version": args.images}
    info["dr"] = dr_grid(img_eps, out / "dr_grid.png") | {"version": args.images, "hash": m["content_hash"][:12]}
    (out / "figures.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

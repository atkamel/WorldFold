"""Roy's hypothesis: a crumpled pile has more 'depth valleys' (dips in its top surface) than a neat fold.
Measure: top-surface height map -> black top-hat (morphological closing minus the surface) = how deep each
spot sits below the bumps around it ('putty needed to fill the dips'). Draws the valley maps."""
import json, os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as win
from PIL import Image, ImageDraw, ImageFont
from shirt import shirt_mesh
from neat_metric import _height_map

pts, tris = shirt_mesh(0.01)
H, RES = 0.004, 0.002


def hmap(x):
    lo = x[:, :2].min(0) - 0.03
    hi = x[:, :2].max(0) + 0.03
    nx, ny = (np.ceil((hi - lo) / RES)).astype(int)
    hm = _height_map(np.asarray(x, float), tris, lo[0], lo[1], RES, nx, ny)
    occ = hm > -0.5
    top = np.where(occ, np.maximum(hm, 0) + H / 2, 0.0)
    return top, occ


def _filt(a, k, fn):
    p = k // 2
    b = np.pad(a, p, mode="edge")
    b = fn(win(b, k, axis=0), axis=-1)
    return fn(win(b, k, axis=1), axis=-1)


def valleys(x, width_cm=3.0):
    top, occ = hmap(x)
    k = int(width_cm / 100 / RES) | 1
    closed = _filt(_filt(top, k, np.max), k, np.min)          # fill dips narrower than ~width
    depth = np.where(occ, np.maximum(closed - top, 0), 0) * 1000  # mm
    d = depth[occ]
    return dict(mean_mm=float(d.mean()), deep_share=float((d > 3).mean()), vol_cm3=float(d.sum() * (RES * 100) ** 2 / 10),
                rough_mm=float(np.abs(np.diff(top, axis=0))[occ[1:] & occ[:-1]].mean() * 1000)), depth, occ, top


def picture(depth, occ, top, title, sub):
    vmax = 15.0
    t = np.clip(top / 0.06, 0, 1)
    img = np.zeros(depth.shape + (3,))
    img[..., 0] = img[..., 1] = img[..., 2] = 0.12
    base = np.stack([0.25 + 0.5 * t, 0.35 + 0.5 * t, 0.55 + 0.4 * t], -1)      # pile: blue-ish, lighter = higher
    v = np.clip(depth / vmax, 0, 1)[..., None]
    col = base * (1 - v) + np.array([1.0, 0.25, 0.1]) * v                        # valleys: red
    img[occ] = col[occ]
    im = Image.fromarray((img[::-1] * 255).astype(np.uint8)).resize((depth.shape[1] * 3, depth.shape[0] * 3), Image.NEAREST)
    canvas = Image.new("RGB", (420, 380), (20, 20, 20))
    canvas.paste(im, ((420 - im.width) // 2, 60 + (300 - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 16); g = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 14)
    d.text((10, 8), title, font=f, fill=(255, 255, 255)); d.text((10, 32), sub, font=g, fill=(255, 220, 120))
    return canvas


recs = [json.loads(l) for l in open(r"demos\20261003_213517.jsonl")]
drops = [r for r in recs if r.get("episode") == 3 and r.get("event") == "drop"]
s26 = min((r for r in drops if r["tick"] - drops[0]["tick"] < 3600), key=lambda r: r["score"]["area_cm2"])
s22 = drops[-1]
panels = []
for name, st in (("26% - after the hem fold", s26), ("22% - after the tidy-ups", s22)):
    x = np.array(st["cloth_cm"]) / 100
    v, depth, occ, top = valleys(x)
    print(f"{name:26s} valley depth: mean {v['mean_mm']:.2f} mm | share of the top deeper than 3 mm: {100 * v['deep_share']:.0f}% "
          f"| valley volume {v['vol_cm3']:.1f} cm3 | roughness {v['rough_mm']:.2f} mm/2mm")
    panels.append(picture(depth, occ, top, name, f"red = valley depth (up to 15 mm). mean {v['mean_mm']:.1f} mm, deep {100 * v['deep_share']:.0f}%"))
sheet = Image.new("RGB", (840, 380))
sheet.paste(panels[0], (0, 0)); sheet.paste(panels[1], (420, 0))
sheet.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "valleys_26_vs_22.png"))

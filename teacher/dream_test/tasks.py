"""The Dream-1 grasp battery (agreed with Roy 2026-10-19): every try starts with the garment flat and the arms at rest.

kind: block = the sanity pick of the red block; grasp = one arm, one named part; both = two-arm lift; fold = Dream's
own showcase prompt. target names the part for scoring (score.py): sleeve_L/R, hem_L, collar for tops; leg_L/R,
waist_L, waist for pants. Left/right are the robot's (world -x is the left arm's side).
"""
GARMENTS = {"top_short": "Top_Short_Seen_0", "top_long": "Top_Long_Seen_0",
            "pant_long": "Pant_Long_Seen_0", "pant_short": "Pant_Short_Seen_0"}

TOP = [("grasp", "pick up the left sleeve", "sleeve_L"),
       ("grasp", "pick up the right sleeve", "sleeve_R"),
       ("grasp", "pick up the bottom left corner of the shirt", "hem_L"),
       ("grasp", "pick up the shirt by the collar", "collar")]
PANT = [("grasp", "pick up the bottom of the left leg", "leg_L"),
        ("grasp", "pick up the bottom of the right leg", "leg_R"),
        ("grasp", "pick up the left corner of the waistband", "waist_L"),
        ("grasp", "pick up the pants by the waistband", "waist")]
SECONDS = {"block": 15.0, "grasp": 15.0, "both": 20.0, "fold": 60.0}


def battery(name="full"):
    """List of episode dicts, in run order (block first: the block is parked off the table afterwards)."""
    eps = []
    tries = {"smoke": (1, 1, 1, 0), "full": (6, 3, 2, 1)}[name]      # block, per grasp prompt, both, fold
    for k in range(tries[0]):
        eps.append(dict(kind="block", garment=GARMENTS["top_short"], prompt="pick up the red block", target="block"))
    for gtype, g in GARMENTS.items():
        top = gtype.startswith("top")
        for kind, prompt, target in (TOP if top else PANT):
            for k in range(tries[1]):
                eps.append(dict(kind=kind, garment=g, prompt=prompt, target=target))
        both = ("pick up the shirt by both bottom corners and lift it" if top
                else "pick up the pants by both corners of the waistband and lift them")
        for k in range(tries[2]):
            eps.append(dict(kind="both", garment=g, prompt=both, target="hem_LR" if top else "waist_LR"))
        for k in range(tries[3]):
            eps.append(dict(kind="fold", garment=g, prompt="fold the shirt" if top else "fold the pants", target="fold"))
    if name == "smoke":       # smoke: one try of each kind, on two garment types
        keep, seen = [], set()
        for e in eps:
            key = (e["kind"], e["garment"].split("_")[0])
            if key not in seen and (e["kind"] != "fold"):
                seen.add(key)
                keep.append(e)
        eps = keep + [dict(kind="fold", garment=GARMENTS["top_short"], prompt="fold the shirt", target="fold")]
    for i, e in enumerate(eps):
        e["seconds"] = SECONDS[e["kind"]]
        e["seed"] = 1000 + i
        e["idx"] = i
    return eps


if __name__ == "__main__":
    import collections
    for name in ("smoke", "full"):
        b = battery(name)
        motion = sum(e["seconds"] for e in b)
        print(name, len(b), "tries,", collections.Counter(e["kind"] for e in b), f"{motion / 60:.1f} min of motion,"
              f" ${motion / 3600 * 3:.2f} at $3/h")
        if name == "smoke":
            for e in b:
                print("  ", e["kind"], e["garment"], repr(e["prompt"]))

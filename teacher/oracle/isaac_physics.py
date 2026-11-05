"""Cloth/contact settings for the LeHome Isaac env, one place for the grip assay (oracle_fold.CLOTH_VARIANTS) and the
teacher runs (written into particle_garment_cfg.yaml). Each value says where it comes from.

    python isaac_physics.py apply <variant> <lehome-challenge dir>    # rewrite the yaml (teacher eval)
    python isaac_physics.py show <variant>

Variants:
  orig      LeHome's own settings: ~114-147 kg shirts, gravity x2, stickiness 0.1, round jaws.
  fixed     Oct 2 "fixed physics": 50 g total on the 14,746-point shirt, gravity x1 (springs NOT rescaled, sticky).
  v2        most accurate we can set without new fits:
            - mass from area x knit density 0.19 kg/m2 (cloth_fitted_rl.json knit_tshirt_prior): the two LeHome
              meshes give 1.69e-5 / 1.52e-5 m2 of cloth per point -> 3.1e-6 kg per point (Top_Long 47 g, Top_Short 33 g)
            - gravity x1, stickiness (adhesion) 0: real cloth does not stick to PLA jaws
            - stretch/bend/shear stiffness and spring damping scaled by the mass ratio: their PhysX units carry mass,
              so cutting the mass 3,200x without them makes the shirt 3,200x stiffer for its weight
            - cloth-on-cloth friction 0.6 (measured nominal) = friction x particle_friction_scale
            - particle-rigid friction 0.5 = cloth-table nominal (measured 0.25-0.8); the jaw (PLA-cotton 0.07-0.25)
              shares this value in PhysX's particle material -> see v2_lowmu
  v2_lowmu  v2 with particle-rigid friction 0.25 (the jaw's measured upper bound; the table is then slippery)
"""
import json
import re
import sys

LEHOME = {"particle_mass": 1e-2, "stretch_stiffness": 1e8, "bend_stiffness": 5000.0, "shear_stiffness": 5000.0,
          "spring_damping": 10.0, "gravity_scale": 2.0, "adhesion": 0.1, "friction": 0.5, "particle_friction_scale": 0.6,
          "solver_position_iteration_count": 32}
MASS_PER_POINT = 3.1e-6
CLOTH_CLOTH_MU = 0.6


def values(variant):
    """Full set of the yaml keys this module controls, for one variant."""
    v = dict(LEHOME)
    if variant == "orig":
        return v
    if variant == "fixed":
        v.update(particle_mass=3.4e-6, gravity_scale=1.0)
        return v
    if variant in ("v2", "v2_lowmu", "v2_prelock", "v2_prelock_lowmu"):
        r = MASS_PER_POINT / LEHOME["particle_mass"]
        mu = 0.25 if variant.endswith("lowmu") else 0.5
        v.update(particle_mass=MASS_PER_POINT, gravity_scale=1.0, adhesion=0.0, friction=mu,
                 particle_friction_scale=round(CLOTH_CLOTH_MU / mu, 4),
                 **{k: LEHOME[k] * r for k in ("stretch_stiffness", "bend_stiffness", "shear_stiffness", "spring_damping")})
        if variant.startswith("v2_prelock"):
            # 2026-10-19 on the table: the locked bend/shear x100 + 64 iterations below push LeHome's shirt back into
            # its worn build shape (it tumbles into a 10 cm heap); without them it lies flat (oracle cloth_test).
            # Fixed the other way too: run_isaac.sh flatten + FLAT=1 use a flattened copy of the shirt model, on which
            # v2 lies flat (footprint -0.3 %, 14 mm tall, 0.03 mm/s). v2_prelock stays for settling the worn model.
            return v
        # 2026-10-19 replay vs real hanging cloth (isaac_cloth_replay.py, 4 Coltraro held-out recordings): bend/shear
        # x100 cut the error 113 % -> 43 % (more had no effect, stretch none); 64 solver iterations 43 % -> 28 %
        v["bend_stiffness"] *= 100.0
        v["shear_stiffness"] *= 100.0
        v["solver_position_iteration_count"] = 64
        return v
    raise SystemExit(f"unknown variant {variant}")


SECTION = {"particle_mass": "garment_config", "stretch_stiffness": "garment_config", "bend_stiffness": "garment_config",
           "shear_stiffness": "garment_config", "spring_damping": "garment_config", "gravity_scale": "particle_material",
           "adhesion": "particle_material", "friction": "particle_material", "particle_friction_scale": "particle_material",
           "solver_position_iteration_count": "particle_system"}


def overrides(variant):
    """{"section.key": value} for oracle_fold.set_cloth (absolute values, so they don't depend on the yaml on disk)."""
    return {f"{SECTION[k]}.{k}": val for k, val in values(variant).items()}


def apply(variant, ch):
    """Rewrite lehome-challenge's particle_garment_cfg.yaml in place with the variant's values."""
    p = f"{ch}/source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml"
    y = open(p).read()
    for key, val in values(variant).items():
        y, n = re.subn(r"(?m)^(\s+)" + key + r":\s*[0-9.eE+-]+", r"\g<1>" + f"{key}: {val:.6g}", y)
        assert n == 1, f"{key}: expected 1 match in {p}, got {n}"
    open(p, "w").write(y)
    return {k: v for k, v in values(variant).items()}


if __name__ == "__main__":
    cmd, variant = sys.argv[1], sys.argv[2]
    if cmd == "show":
        print(json.dumps(values(variant), indent=1))
    elif cmd == "apply":
        print("PHYSICS " + variant + " " + json.dumps(apply(variant, sys.argv[3])))

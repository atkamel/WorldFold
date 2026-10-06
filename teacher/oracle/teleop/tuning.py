"""Live tuning: every setting that shapes how the game feels, adjustable from the laptop while playing.

Three kinds, by when a change takes effect:
  control  next tick           arm speeds, wheel step, reach tilt, heights (controller.py)
  grip     next grab/release   the scripted Isaac grasp and release moves (isaac_world.py)
  cloth    on "apply cloth"    PhysX cloth physics; the shirt is rebuilt with the new values (no Isaac restart)
The server owns the values; the client sends ops and draws the panel from the HUD. Every change goes into the
demo log, so each demo records the settings it was played with.
"""
import numpy as np

# key: (default, lo, hi, step, kind, label, unit, scale)   step < 0 means multiply/divide by |step| per press
SPECS = {
    # ---- control (controller.py)
    "arm_speed":    (0.40, 0.10, 1.00, 0.05, "control", "arm speed (sideways)", "cm/s", 100),
    "vz_fast":      (0.25, 0.05, 0.60, 0.05, "control", "speed down / place / rise", "cm/s", 100),
    "vz_carry":     (0.15, 0.05, 0.50, 0.05, "control", "speed up to carry height", "cm/s", 100),
    "wheel_deg":    (15.0, 5.0, 45.0, 5.0, "control", "jaw turn per wheel notch", "deg", 1),
    "max_tilt":     (45.0, 10.0, 70.0, 5.0, "control", "max gripper tilt (reach)", "deg", 1),
    "down_offset":  (0.006, 0.0, 0.02, 0.001, "control", "grab height over cloth", "mm", 1000),
    "place_offset": (0.010, 0.0, 0.03, 0.002, "control", "drop height over cloth", "mm", 1000),
    "slide_h":      (0.007, 0.002, 0.02, 0.001, "control", "drag height (right button)", "mm", 1000),
    # ---- grip (isaac_world.py; the tested values are the defaults)
    "grab_slide":   (0.030, 0.0, 0.06, 0.005, "grip", "grab: slide length", "mm", 1000),
    "grab_press":   (0.012, 0.0, 0.025, 0.002, "grip", "grab: press into table", "mm", 1000),
    "grab_close":   (16.0, 2.0, 40.0, 2.0, "grip", "grab: closing ticks", "ticks", 1),
    "grab_lift":    (0.015, 0.0, 0.04, 0.005, "grip", "grab: lift after closing", "mm", 1000),
    "grab_ds":      (0.0015, 0.0005, 0.005, 0.0005, "grip", "grab: slide speed per tick", "mm", 1000),
    "rel_backoff":  (0.020, 0.0, 0.04, 0.005, "grip", "release: back off", "mm", 1000),
    "rel_lift":     (0.0007, 0.0002, 0.003, 0.0001, "grip", "release: lift per tick", "mm", 1000),
    "rel_open":     (8.0, 1.0, 20.0, 1.0, "grip", "release: opening ticks", "ticks", 1),
    "rel_hold":     (15.0, 0.0, 30.0, 1.0, "grip", "release: wait after opening", "ticks", 1),
    "rel_up":       (0.03, 0.0, 0.05, 0.005, "grip", "release: scripted lift", "mm", 1000),
    # ---- cloth (PhysX particle cloth; defaults = the "real" variant: 50 g shirt, gravity x1)
    "mass_g":       (50.0, 5.0, 500.0, -1.25, "cloth", "shirt mass", "g", 1),
    "gravity":      (1.0, 0.25, 3.0, 0.25, "cloth", "gravity scale", "x", 1),
    "adhesion":     (0.1, 0.0, 1.0, 0.05, "cloth", "stickiness (adhesion)", "", 1),
    "friction":     (0.5, 0.0, 2.0, 0.1, "cloth", "friction", "", 1),
    "damping":      (0.05, 0.0, 2.0, 0.05, "cloth", "damping (floatiness)", "", 1),
    "bend":         (5000.0, 10.0, 1e6, -2.0, "cloth", "bend stiffness", "", 1),
    "shear":        (5000.0, 10.0, 1e6, -2.0, "cloth", "shear stiffness", "", 1),
    "spring_damp":  (10.0, 0.1, 1000.0, -2.0, "cloth", "spring damping", "", 1),
}
KINDS = ("control", "grip", "cloth")

# Two-mouse play in Isaac: feel like the MuJoCo game (hold = grab, let go = drop, no scripted slide or slow lift).
# The jaws still squeeze the cloth for real (MuJoCo pins it instead), so grip may slip more than with the slide.
MUJOCO_FEEL = {"grab_slide": 0.0, "grab_close": 4.0, "grab_lift": 0.0,
               "rel_backoff": 0.0, "rel_open": 2.0, "rel_hold": 0.0, "rel_up": 0.0}

# Grab assist (shared autonomy, like lane-keeping assist): the player aims and holds; the tested gather grasp runs
# by itself in a fraction of a second (moving finger leads a 3 cm slide pressed into the table, close, small lift), the
# jaws then stay shut until the player lets go. Real physics throughout: demos replay as recorded.
ASSIST = {"grab_slide": 0.030, "grab_press": 0.012, "grab_ds": 0.003, "grab_close": 6.0, "grab_lift": 0.010,
          "rel_backoff": 0.010, "rel_open": 4.0, "rel_hold": 4.0, "rel_up": 0.02, "rel_lift": 0.002}

# cloth key -> LeHome particle_config entry (oracle_fold.set_cloth overrides)
CLOTH_KEYS = {"gravity": "particle_material.gravity_scale", "adhesion": "particle_material.adhesion",
              "friction": "particle_material.friction", "damping": "particle_material.damping",
              "bend": "garment_config.bend_stiffness", "shear": "garment_config.shear_stiffness",
              "spring_damp": "garment_config.spring_damping"}


class Tuning:
    def __init__(self, kinds=KINDS, overrides=None):
        self.keys = [k for k, s in SPECS.items() if s[4] in kinds]
        self.v = {k: float(SPECS[k][0]) for k in self.keys}
        self.applied_cloth = {k: self.v[k] for k in self.keys if SPECS[k][4] == "cloth"}
        for k, x in (overrides or {}).items():
            self.set(k, x)
        self.applied_cloth = {k: self.v[k] for k in self.applied_cloth}   # overrides at start count as applied

    def __getitem__(self, k):
        return self.v.get(k, SPECS[k][0])

    def set(self, k, x):
        if k not in self.v:
            return False
        lo, hi = SPECS[k][1], SPECS[k][2]
        self.v[k] = float(np.clip(float(x), lo, hi))
        return True

    def adjust(self, k, direction):
        if k not in self.v:
            return False
        step = SPECS[k][3]
        x = self.v[k] * (abs(step) ** direction) if step < 0 else self.v[k] + direction * step
        return self.set(k, round(x, 6))

    def reset(self, k):
        return self.set(k, SPECS[k][0]) if k in self.v else False

    def cloth_pending(self):
        return any(abs(self.v[k] - x) > 1e-12 for k, x in self.applied_cloth.items())

    def cloth_overrides(self, n_points):
        """particle_config overrides for oracle_fold.set_cloth (mass is per particle there)."""
        o = {CLOTH_KEYS[k]: self.v[k] for k in self.applied_cloth if k in CLOTH_KEYS}
        if "mass_g" in self.v:
            o["garment_config.particle_mass"] = self.v["mass_g"] / 1000 / max(1, n_points)
        return o

    def mark_cloth_applied(self):
        self.applied_cloth = {k: self.v[k] for k in self.applied_cloth}

    def spec(self):
        """Sent once in the hello: what the client's panel lists."""
        return [dict(key=k, label=SPECS[k][5], kind=SPECS[k][4], unit=SPECS[k][6], scale=SPECS[k][7],
                     default=SPECS[k][0]) for k in self.keys]

    def values(self):
        return dict(self.v)

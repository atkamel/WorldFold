"""Run the REAL-robot checkpoint (lehome_real, config pi_modified_real_bc) inside the Isaac sim eval.

Runtime patch of his eval_wrapper.py, gated by TEACHER_REAL_IN_SIM=1 (OFF = original behaviour):
  inputs : sim state (USD radians) -> real units (degrees, gripper 0-100) with HIS sim_radians_to_real_units;
           top image rotated 180 deg (real-canonical orientation; the real config has rotate_top_image=False);
           inpainting initial_actions radians -> real units.
  outputs: action chunk + next_initial_actions real units -> sim radians with HIS real_units_to_sim_radians.
Run the sim with the real-aligned top camera (1280x720 + his top_camera offsets) -- see isaac_eval(real_in_sim=True).

    python real_in_sim_patch.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-real-in-sim-patch]"
p = ROOT / "src/lehome_solution/shared/eval_wrapper.py"
s = p.read_text()
if MARK in s:
    print("already patched: real-in-sim")
    sys.exit(0)

HELPERS = '''
import os as _os_ris


def _ris_on():
    return _os_ris.environ.get("TEACHER_REAL_IN_SIM") == "1"


def _ris_inputs(obs, initial_actions):
    from lehome_solution.training.real_data_transforms import sim_radians_to_real_units
    obs = dict(obs)
    obs["observation.state"] = sim_radians_to_real_units(np.asarray(obs["observation.state"], dtype=np.float32))
    k = "observation.images.top_rgb"
    if k in obs:
        obs[k] = np.ascontiguousarray(np.asarray(obs[k])[::-1, ::-1])
    if initial_actions is not None:
        initial_actions = sim_radians_to_real_units(np.asarray(initial_actions, dtype=np.float32))
    return obs, initial_actions


def _ris_outputs(chunk, next_initial):
    from lehome_solution.training.real_data_transforms import real_units_to_sim_radians
    chunk = real_units_to_sim_radians(np.asarray(chunk, dtype=np.float32))
    if next_initial is not None:
        next_initial = real_units_to_sim_radians(np.asarray(next_initial, dtype=np.float32))
    return chunk, next_initial


class LeHomePolicyWrapper'''

edits = [
    ("\nclass LeHomePolicyWrapper", HELPERS),
    ("        for obs, initial_actions, inference_config in requests:\n",
     "        for obs, initial_actions, inference_config in requests:\n"
     "            if _ris_on():\n"
     "                obs, initial_actions = _ris_inputs(obs, initial_actions)\n"),
    ("            results.append((\n                chunk,\n                next_initial,\n",
     "            if _ris_on():\n"
     "                chunk, next_initial = _ris_outputs(chunk, next_initial)\n"
     "            results.append((\n                chunk,\n                next_initial,\n"),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected 1 match, found {n} for:\n{old[:120]}"
    s = s.replace(old, new)
p.write_text(MARK + "\n" + s)
print("REAL_IN_SIM_PATCH_OK")

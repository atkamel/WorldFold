"""Build data/manifest.json from download_log.json, meta/info.json and sanity.json of each dataset."""
import json, os
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
log = json.load(open(os.path.join(ROOT, "download_log.json")))

LATTICE = ("All 5 arm joints of observation.state AND action lie exactly on the LeRobot DEGREES lattice "
           "(ticks-mid)*360/4095: circular concentration R=1.000 at step 0.08791 deg, smallest distinct step 0.0879 deg "
           "(scripts/sanity_check.py). Values beyond +-100 occur (RANGE_M100_100 mode clamps to +-100, motors_bus.py). "
           "Dataset created after lerobot PR #2968 (2026-02-13) switched use_degrees default False->True. "
           "Gripper is RANGE_0_100 (0..100), on its own tick lattice.")
REAL_COMMON = ("state AND action are quantized to STS3215 encoder ticks (follower and leader), state lags action by "
               "~4-5 frames (130-170 ms) with elbow_flex sitting ~1-3 deg above command (gravity sag under P-only servo), "
               "gripper state never reaches action 0 (jaw stall on object/closed stop); no sim words in card; real camera names.")
PER = {
    "dobri420/pick-cube-so101": dict(
        units_extra="README states: 'State / action: 6-DoF joint positions (shoulder_pan ... gripper), in degrees.'",
        units_confidence="high",
        real=("README: 'teleoperated pick-and-place dataset on the SO-101 arm', 'SO-101 leader->follower pair, "
              "teleoperated at 30 fps'; a separate MuJoCo re-render lives in a DIFFERENT repo "
              "(dobri420/pick-cube-so101-sim), not this one. "),
        so101="README names SO-101 and links TheRobotStudio SO-ARM100.",
        notes=("540 episodes total in 7 structured tranches (discrete angles at 25/15 cm, jitter, wrist twist CCW/CW, "
               "distractors). Downloaded episodes 0-39 = tranche 'discrete @ 25 cm' (5 target angles 0, +-45, +-90 deg) "
               "-> large shoulder_pan sweep (-99..98 deg). Wrist_roll barely moves in eps 0-39 (-9..24 deg); the twist "
               "tranches (eps 240-439) would excite wrist_roll if needed. Cameras camera1..3.")),
    "HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1": dict(
        units_extra=("README: 'serialized position units emitted by the SO-101/LeRobot recording stack' "
                     "(does not say degrees explicitly)."),
        units_confidence="high",
        real=("README: '195 physical SO-101 demonstrations', 'A human operator controlled an SO-101 follower through a "
              "leader arm'; README mentions a 'simulation configuration' only for the GitHub project, not this data. "),
        so101="README: 'Robot | SO-101 follower', 'SO-101 leader-to-follower teleoperation'.",
        notes=("Tic-tac-toe piece placement over a 3x3 board; 15 game groups x 13 demos. Card has its own action-vs-state "
               "section and warns those are not lag-compensated tracking errors. Extra meta files: "
               "hashtag_episode_plan.jsonl, hub_upload_provenance.json. No license tag on the Hub.")),
    "DecisionFacts/Physical_AI_SO101_Cup_Nesting_Task": dict(
        units_extra="README does not state units.",
        units_confidence="high",
        real=("README: 'curated collection of real-world teleoperation data captured on the SO-101 robotic arm "
              "(so_follower)', 'not scripted or simulated motion'. The words Isaac/simulated in the card are only the "
              "tag 'Isaac-compatible' and that disclaimer. "),
        so101="README: 'SO-101 robotic arm (so_follower)'.",
        notes=("Cup nesting; operators hold the final pose at the end of each episode. Largest wrist motion of the set: "
               "wrist_flex -103..102 and wrist_roll -76..169 deg (action). License cc-by-nc-4.0 (non-commercial). "
               "Extra meta file 'quality metrics.ods' (not opened).")),
    "Chaenn/so101_cube_pnp_real_0908": dict(
        units_extra="README is the LeRobot template (no units stated).",
        units_confidence="high",
        real=("Repo name says 'real'; the same author publishes separately named sim / simreal / drsim datasets "
              "(e.g. so101_cube_sim_place_0901_1000), so 'real' is a deliberate label. "),
        so101=("Only the repo name says so101 (robot_type so_follower is shared by SO-100 and SO-101 in current "
               "lerobot) -> SO-101 vs SO-100 is UNVERIFIED beyond the name."),
        notes=("CAUTION: the one downloaded parquet file holds episodes 0-106 (not 40). In episodes 87-106, 10-45% of "
               "action frames are OFF the encoder lattice (smooth, not a leader-arm reading) -> likely policy or "
               "HIL-intervention rollouts; use only episodes 0-86 for tracking checks unless the action source is "
               "confirmed. Episode lengths 17..2032 frames (some very short). Cameras wrist, side.")),
    "dharak22/so101-red-cube_20260908_113622": dict(
        units_extra="README is the LeRobot template (no units stated).",
        units_confidence="high",
        real="",
        so101=("Only the repo name says so101 (robot_type so_follower is shared by SO-100 and SO-101) -> UNVERIFIED "
               "beyond the name."),
        notes=("Red cube pick-and-place; 43 episodes downloaded (file boundaries). Large shoulder_lift/elbow motion; "
               "gripper action/state differ most here (gripper RMS 8.7 at lag 0), consistent with squeezing a cube. "
               "Cameras main, side.")),
    "vasco281204/so101_green_block": dict(
        units_extra="README is the LeRobot template (no units stated).",
        units_confidence="high",
        real=("Author also publishes sim / sim2real-mix datasets under different names (e.g. "
              "so101_sim_real_mix_50_50_200); this repo's data is lattice-quantized, so treated as the real-arm "
              "recording. "),
        so101=("Only the repo name says so101 (robot_type so_follower is shared by SO-100 and SO-101) -> UNVERIFIED "
               "beyond the name."),
        notes=("Green block pick; 65 episodes downloaded (file boundaries). Widest shoulder_pan (-96..100) and "
               "wrist_roll (-4..170) excursions. Cameras top, front.")),
}
GRIP = {"Chaenn/so101_cube_pnp_real_0908": 1286, "DecisionFacts/Physical_AI_SO101_Cup_Nesting_Task": 1392,
        "HashtagRobotics/tic-tac-toe-so101-block-a-clean-v1": 1481, "dharak22/so101-red-cube_20260908_113622": 1525,
        "dobri420/pick-cube-so101": 1435, "vasco281204/so101_green_block": 1481}

man = []
for rid, l in log.items():
    d = l["local_dir"]
    info = json.load(open(os.path.join(d, "meta", "info.json")))
    s = json.load(open(os.path.join(d, "sanity.json")))
    p = PER[rid]
    man.append(dict(
        repo_id=rid, revision=l["revision"], local_dir=d.replace("\\", "/"),
        robot_type=info.get("robot_type"), fps=info["fps"], codebase_version=info["codebase_version"],
        total_episodes_on_hub=info.get("total_episodes"),
        units_guess=("arm joints: DEGREES (LeRobot use_degrees=True, zero = midpoint of calibrated range); "
                     "gripper: RANGE_0_100"),
        units_evidence=LATTICE + " " + p["units_extra"],
        units_confidence=p["units_confidence"],
        is_real_evidence=p["real"] + REAL_COMMON + " SO-101 check: " + p["so101"],
        n_episodes_downloaded=s["n_episodes"], episodes_downloaded=s["episodes"], frames_downloaded=s["rows"],
        data_files=l["files"],
        action_names=info["features"]["action"]["names"],
        state_names=info["features"]["observation.state"]["names"],
        cameras=[k for k in info["features"] if k.startswith("observation.images")],
        gripper_tick_span_state=GRIP[rid],
        raw_tracking_lag0_rms_deg=s["rms_state_minus_action_lag0"],
        raw_best_lag_frames=s["median_best_lag_frames"],
        median_state_minus_action=s["median_state_minus_action"],
        action_min=s["action_min"], action_max=s["action_max"], state_min=s["state_min"], state_max=s["state_max"],
        notes=p["notes"]))

# NeuroTo local bimanual episode
man.append(dict(
    repo_id="NeuroTo/bimanual-so101-fold-tshirt (local copy of episode 0 only)",
    revision=("1733c8ca5b13 (Hub sha at check time; local ep0.parquet values identical to Hub "
              "data/chunk-000/episode_000000.parquet)"),
    local_dir="C:/Users/valen/Documents/vla-distill/teacher/results/real_so101/neuroto_ep0",
    robot_type="so101_two_followers", fps=30, codebase_version="v2.1", total_episodes_on_hub=7,
    units_guess=("arm joints: RANGE_M100_100 (normalized, clamped to +-100); gripper: RANGE_0_100; "
                 ".vel and .load look like raw register values"),
    units_evidence=(
        "Positions are NOT on the 360/4095 degree lattice (R 0.01-0.17). They ARE exactly on a RANGE_M100_100 lattice "
        "(v+100)*N/200 = integer with R=1.000 for every joint, N = calibrated tick span. Follower (state) spans N: "
        "right arm pan 2501, lift 2368, elbow 2207, wrist_flex 2361, wrist_roll 3853, gripper(0-100) 1430; "
        "left arm pan 2726, lift 2550, elbow 2203, wrist_flex 2395, wrist_roll 3833, gripper 1448. "
        "Leader (action) spans differ (right: 2289, 2375, 2202, 2308, 3823). Values never exceed +-100 and hit exactly "
        "100.0 (clamp). Recorded 2025-07-30, before lerobot switched use_degrees default to True (2026-02-13). "
        "Conversion to LeRobot degrees (same zero = range midpoint): deg = v * N_follower * 360 / (200*4095); "
        "convert the ACTION with the FOLLOWER's N because the follower unnormalizes goals with its own calibration "
        "(stock lerobot behaviour; UNVERIFIED for this custom robot class). Beyond-range motion is lost to clamping. "
        ".vel values are multiples of 50 (raw Present_Velocity-like); .load values 0..1576 look like raw "
        "sign-magnitude Present_Load (bit 10 = direction) - UNVERIFIED."),
    units_confidence="high for RANGE_M100_100 positions; low for vel/load semantics",
    is_real_evidence=("Feature names (two follower arms, wrist + global cameras), positions quantized to encoder "
                      "ticks, already used as the real reference in this project. Hub meta: robot_type "
                      "so101_two_followers (custom, not stock lerobot)."),
    n_episodes_downloaded=1, episodes_downloaded=[0, 0], frames_downloaded=1796,
    data_files=["ep0.parquet"],
    action_names=["right_arm.{shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper}.pos (idx 0-5)",
                  "right_arm.*.vel (6-11)", "left_arm.*.pos (12-17)", "left_arm.*.vel (18-23)"],
    state_names=["right_arm.*.pos (idx 0-5)", "right_arm.*.vel (6-11)", "right_arm.*.load (12-17)",
                 "left_arm.*.pos (18-23)", "left_arm.*.vel (24-29)", "left_arm.*.load (30-35)"],
    cameras=["observation.images.global1", "observation.images.global2",
             "observation.images.right_arm.wrist", "observation.images.left_arm.wrist"],
    notes=("Bimanual t-shirt fold, 1796 frames. Feature names taken from the Hub meta/info.json (the local parquet has "
           "no names). Motion is task-driven (cloth), smaller excursions than the pick-place sets (right pan -54..36). "
           "Usable as 2 more arms but needs the RANGE->degree conversion above first.")))

json.dump(man, open(os.path.join(ROOT, "manifest.json"), "w"), indent=1)
print("wrote", len(man), "entries")

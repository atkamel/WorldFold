#!/usr/bin/env bash
# Run one Isaac session on the RunPod pod, then STOP THE POD (so nothing keeps billing).
#
#   bash teacher/runpod/run_isaac.sh grip              # adhesion A/B: grip held + release stuck, flat pads
#   bash teacher/runpod/run_isaac.sh teleop            # two-mouse play: laptop runs mujoco_live/teleop_client.py (no cameras)
#
# Options (env vars): DEVICE=cpu|cuda:0 (sim device)  GARMENT=Top_Long_Seen_0  PADS=flat  MAX_S=1800 (hard cap)  KEEP_POD=1 (don't stop the pod)
#                     FRICTION/ADHESION (-1 = keep the challenge's values; the A/B uses cloth variants instead)
set -uo pipefail
MODE=${1:?usage: run_isaac.sh grip|contact|flatten|teleop}
WS=${WS:-/workspace}
WF_DIR=${WF_DIR:-$WS/WorldFold}
# the prebuilt image has the install in /opt (its ENV CH is NOT seen by ssh sessions); setup.sh installs in /workspace
[ -d /opt/lehome_solution/lehome-challenge/.venv ] && CH_DEFAULT=/opt/lehome_solution/lehome-challenge || CH_DEFAULT=$WS/lehome_solution/lehome-challenge
CH=${CH:-$CH_DEFAULT}
export CH
GARMENT=${GARMENT:-Top_Long_Seen_0}
PADS=${PADS:-flat}
MAX_S=${MAX_S:-1800}
FRICTION=${FRICTION:--1}
ADHESION=${ADHESION:--1}
STAMP=$(date +%Y%m%d-%H%M%S)-$MODE
OUT=$WS/results/isaac/$STAMP
mkdir -p "$OUT"
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y XDG_RUNTIME_DIR=/tmp
export __GLX_VENDOR_LIBRARY_NAME=nvidia VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
T0=$(date +%s)

stop_pod() {
  echo "[run] session took $(( $(date +%s) - T0 )) s; results in $OUT"
  if [ "${KEEP_POD:-0}" != 1 ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    echo "[run] stopping pod $RUNPOD_POD_ID (KEEP_POD=1 to keep it)"; sync; runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap stop_pod EXIT

bash "$WF_DIR/teacher/runpod/link_code.sh"

# Isaac Sim 5.1 is certified for NVIDIA driver 580; 610 segfaults its renderer at startup (isaac-sim/IsaacSim #650, #651)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "[run] GPU driver $DRV"
case "$DRV" in 580.*) ;; *) echo "[run] FATAL: driver $DRV is not 580.x - Isaac 5.1 would crash at startup"; exit 3 ;; esac

# ---- the same scene edits as the Modal functions: cloth physics values, every top starts flat at the centre ----
"$CH/.venv/bin/python" - "$CH" "$FRICTION" "$ADHESION" <<'PY'
import glob, json, re, sys
ch, fr, ad = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
p = f"{ch}/source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml"
y = open(p).read()
for key, val in (("friction", fr), ("adhesion", ad)):
    if val >= 0:
        y, n = re.subn(r"(?m)^(\s+)" + key + r":\s*[0-9.eE+-]+", r"\g<1>" + f"{key}: {val}", y)
        assert n == 1, f"{key}: expected 1 match, got {n}"
open(p, "w").write(y)
for jp in glob.glob(f"{ch}/Assets/objects/Challenge_Garment/Release/Top_*/*/*.json"):
    c = json.load(open(jp))
    c["initial_pos_range"] = c["soft_reset_pos_range"] = [0.0, 0.0, 0.63, 0.0, 0.0, 0.63]
    c["initial_rot_range"] = c["soft_reset_rot_range"] = [0, 0, 0, 0, 0, 0]
    json.dump(c, open(jp, "w"), indent=4)
print("[run] scene edits applied")
PY

GTYPE=$(echo "$GARMENT" | cut -d_ -f1-2 | tr 'A-Z' 'a-z')
export PYTHONPATH=/opt/teacher/oracle ORACLE_DIR=/opt/teacher/oracle ORACLE_OUT=$OUT ORACLE_PADS=$PADS
export LEHOME_NO_DEPTH=1 LEHOME_DISABLE_KEYBOARD=1

case "$MODE" in
  grip)
    # cloth settings on flat pads (oracle/isaac_physics.py): Oct 2 "real" (sticky, springs not rescaled) vs v2 vs v2_lowmu
    #  grip_assay: does the cloth stay held through lift / hold / carry?   mechanics: does it stick on release?
    GV=${GRIP_VARIANTS:-'["real", "v2", "v2_lowmu"]'}
    export ORACLE_JOBS=$(cat <<J
[{"garment": "$GARMENT", "mode": "grip_assay", "reps": 2, "budget_s": ${GRIP_BUDGET_S:-600}, "cloth": $GV, "speeds": [0.0015, 0.0035]},
 {"garment": "$GARMENT", "mode": "mechanics", "reps": 2, "budget_s": 300, "cloth": $GV,
  "opens": [0.55], "aways": [0.02], "lift_dss": [0.0007]}]
J
)
    # NOCAM=1: no cameras (the grip test reads cloth vertices only; Isaac starts without compiling render shaders)
    if [ "${NOCAM:-0}" = 1 ]; then
      "$CH/.venv/bin/python" "$WF_DIR/teacher/no_cameras_patch.py" "$(dirname "$CH")"
      export LEHOME_NO_CAMERAS=1
    fi
    ;;
  flatten)
    # flatten step 1 (oracle_fold.flatten_dump): the shirt as built from CT_PHYSICS (default v2_prelock: lies flat)
    # settles, its flat shape is saved; after Isaac exits flatten_garment.py writes it into <asset>_flat.usd
    CT_PHYSICS=${CT_PHYSICS:-v2_prelock}
    # GARMENTS="A B C": all in one Isaac session (switch_garment in place), the first one is the session's garment
    GARMENTS=${GARMENTS:-$GARMENT}; GARMENT=$(echo $GARMENTS | cut -d" " -f1); GTYPE=$(echo "$GARMENT" | cut -d_ -f1-2 | tr 'A-Z' 'a-z')
    export ORACLE_JOBS=$(python3 - "$GARMENTS" "${FLAT_SETTLE_S:-6}" <<'PY'
import json, sys
gs, st = sys.argv[1].split(), float(sys.argv[2])
print(json.dumps([{"garment": g, "mode": "flatten", "settle_s": st, "last": i == len(gs) - 1} for i, g in enumerate(gs)]))
PY
)
    export ORACLE_HARD_EXIT=1 FLAT_KEEP="$WF_DIR/teacher/oracle/assets/garments_flat"   # the USD copy is written in-process (pxr)
    mkdir -p "$FLAT_KEEP"
    "$CH/.venv/bin/python" "$WF_DIR/teacher/oracle/isaac_physics.py" apply "$CT_PHYSICS" "$CH"
    for g in $GARMENTS; do "$CH/.venv/bin/python" "$WF_DIR/teacher/oracle/flatten_garment.py" restore "$CH" "$g"; done   # settle the modelled shapes
    ;;
  contact)
    # contact check (oracle_fold.contact_check): Isaac's own jaw-cloth contact on flat pads vs real targets
    # (pass-through, jaw stop angle, hold, slip load, layers); cameras on, same env as the teacher runs
    export ORACLE_JOBS=${CT_JOBS:-"[{\"garment\": \"$GARMENT\", \"mode\": \"contact_check\", \"reps\": ${CT_REPS:-2}, \"variants\": ${CT_VARIANTS:-[\"v2\", \"v2_lowmu\"]}}]"}   # CT_JOBS: any other job list, same env
    export ORACLE_HARD_EXIT=1
    [ "${ATTACH:-0}" = 1 ] && export ORACLE_ATTACH=1 && echo "[run] attach grasp model on (ATTACH_MU=${ATTACH_MU:-0.15} ATTACH_N=${ATTACH_N:-9})"
    if [ -n "${CT_PHYSICS:-}" ]; then              # cloth settings written into the yaml, used as Isaac builds the shirt
      "$CH/.venv/bin/python" "$WF_DIR/teacher/oracle/isaac_physics.py" apply "$CT_PHYSICS" "$CH"
    fi
    # FLAT=1: the garment JSON points at the flattened asset copy (flatten mode wrote it); else the modelled one
    "$CH/.venv/bin/python" "$WF_DIR/teacher/oracle/flatten_garment.py" $([ "${FLAT:-0}" = 1 ] && echo use || echo restore) "$CH" "$GARMENT"
    ;;
  assist)
    # grab assist test (oracle_fold.assist_test): the live-play world, N random grabs per grab preset, no client
    export ORACLE_JOBS="[{\"garment\": \"$GARMENT\", \"mode\": \"assist_test\", \"n\": ${N_GRABS:-20}, \"presets\": ${PRESETS:-[\"mujoco_feel\", \"tested\", \"assist\"]}}]" ORACLE_CLOTH=${CLOTH:-real}
    export LEHOME_CHECK_INTERVAL=1000000 ORACLE_STEPS_PER_TICK=1 LEHOME_NO_CAMERAS=1
    "$CH/.venv/bin/python" "$WF_DIR/teacher/no_cameras_patch.py" "$(dirname "$CH")"
    ;;
  clothtest)
    # roadmap 7a: cloth physics tests (oracle_fold.cloth_test): per cloth setting, the shirt left alone + the sleeve
    # grip test, Isaac up throughout, no client. CT_VARIANTS: JSON list of CLOTH_VARIANTS names or override dicts
    export ORACLE_JOBS="[{\"garment\": \"$GARMENT\", \"mode\": \"cloth_test\", \"variants\": ${CT_VARIANTS:-[\"asis\", \"real_noadh\", \"scaled\"]}, \"reps\": ${CT_REPS:-2}, \"settle_s\": ${CT_SETTLE_S:-10}}]"
    export LEHOME_CHECK_INTERVAL=1000000 ORACLE_STEPS_PER_TICK=1 LEHOME_NO_CAMERAS=1
    "$CH/.venv/bin/python" "$WF_DIR/teacher/no_cameras_patch.py" "$(dirname "$CH")"
    ;;
  teleop)
    export ORACLE_JOBS="[{\"garment\": \"$GARMENT\"}]" ORACLE_TELEOP=1 ORACLE_TELEOP_PORT=7777
    export ORACLE_TELEOP_IDLE=${IDLE_S:-300} ORACLE_TELEOP_MAX=$MAX_S ORACLE_CLOTH=${CLOTH:-real}
    # LeHome re-reads the whole cloth for its success check every 30 observations: rare during live play only
    export LEHOME_CHECK_INTERVAL=${LEHOME_CHECK_INTERVAL:-1000000}
    export ORACLE_STEPS_PER_TICK=${STEPS_PER_TICK:-1}   # physics steps per control tick (Isaac barely renders in the state view)
    # live play without LeHome's 3 cameras (the laptop draws; demos get images when replayed): NOCAM=0 keeps them
    if [ "${NOCAM:-1}" = 1 ]; then
      "$CH/.venv/bin/python" "$WF_DIR/teacher/no_cameras_patch.py" "$(dirname "$CH")"
      export LEHOME_NO_CAMERAS=1
    fi
    # Blosc2 for the compressed state stream (teleop/protocol.py; without it the stream is sent uncompressed)
    NPV=$("$CH/.venv/bin/python" -c 'import numpy; print(numpy.__version__)')
    (uv pip install --python "$CH/.venv/bin/python" blosc2 "numpy==$NPV" -q 2>/dev/null \
       || "$CH/.venv/bin/python" -m pip install -q blosc2 "numpy==$NPV") \
      && "$CH/.venv/bin/python" -c 'import blosc2; print("[run] blosc2", blosc2.__version__)' || echo "[run] blosc2 NOT installed (uncompressed stream)"
    echo "[run] TELEOP: connect from the laptop to this pod's public TCP port for 7777"
    echo "      (RunPod console -> Connect -> 'TCP port mappings'):  python teleop_client.py <IP> <PORT>"
    ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac

cd "$CH"
timeout --kill-after=60 "$MAX_S" .venv/bin/python -u -m scripts.oracle_fold --garment_type "$GTYPE" --garment_name "$GARMENT" \
  --headless $([ "${LEHOME_NO_CAMERAS:-0}" = 1 ] || echo --enable_cameras) --device "${DEVICE:-cpu}" --seed 42 2>&1 | tee "$OUT/run.log" \
  | grep --line-buffered -E "\[oracle\]|TELEOP|Traceback|Error:" | grep --line-buffered -v carb.launcher
echo "[run] isaac exit: ${PIPESTATUS[0]}"

if [ "$MODE" = flatten ]; then
  cp "$OUT"/flat_*.npz "$FLAT_KEEP/" 2>/dev/null && echo "[run] flat shapes kept in $FLAT_KEEP: $(ls "$OUT" | grep -c '^flat_.*npz$')"
fi
if [ "$MODE" = grip ]; then
  "$CH/.venv/bin/python" "$WF_DIR/teacher/runpod/summarize_grip.py" "$OUT" | tee "$OUT/summary.txt"
fi

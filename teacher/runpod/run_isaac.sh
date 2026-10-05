#!/usr/bin/env bash
# Run one Isaac session on the RunPod pod, then STOP THE POD (so nothing keeps billing).
#
#   bash teacher/runpod/run_isaac.sh grip              # adhesion A/B: grip held + release stuck, flat pads
#   bash teacher/runpod/run_isaac.sh teleop            # two-mouse play: laptop runs mujoco_live/teleop_client.py
#
# Options (env vars): DEVICE=cpu|cuda:0 (sim device)  GARMENT=Top_Long_Seen_0  PADS=flat  MAX_S=1800 (hard cap)  KEEP_POD=1 (don't stop the pod)
#                     FRICTION/ADHESION (-1 = keep the challenge's values; the A/B uses cloth variants instead)
set -uo pipefail
MODE=${1:?usage: run_isaac.sh grip|teleop}
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
    # adhesion A/B on flat pads: LeHome's 0.1 ("real") vs 0 ("real_noadh", Adam's choice)
    #  grip_assay: does the cloth stay held through lift / hold / carry?   mechanics: does it stick on release?
    export ORACLE_JOBS=$(cat <<J
[{"garment": "$GARMENT", "mode": "grip_assay", "reps": 2, "budget_s": 360, "cloth": ["real", "real_noadh"], "speeds": [0.0015, 0.0035]},
 {"garment": "$GARMENT", "mode": "mechanics", "reps": 2, "budget_s": 300, "cloth": ["real", "real_noadh"],
  "opens": [0.55], "aways": [0.02], "lift_dss": [0.0007]}]
J
)
    ;;
  teleop)
    export ORACLE_JOBS="[{\"garment\": \"$GARMENT\"}]" ORACLE_TELEOP=1 ORACLE_TELEOP_PORT=7777
    export ORACLE_TELEOP_IDLE=${IDLE_S:-300} ORACLE_TELEOP_MAX=$MAX_S ORACLE_CLOTH=${CLOTH:-real}
    # LeHome re-reads the whole cloth for its success check every 30 observations: rare during live play only
    export LEHOME_CHECK_INTERVAL=${LEHOME_CHECK_INTERVAL:-1000000}
    export ORACLE_STEPS_PER_TICK=${STEPS_PER_TICK:-1}   # physics steps per control tick (Isaac barely renders in the state view)
    echo "[run] TELEOP: connect from the laptop to this pod's public TCP port for 7777"
    echo "      (RunPod console -> Connect -> 'TCP port mappings'):  python teleop_client.py <IP> <PORT>"
    ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac

cd "$CH"
timeout --kill-after=60 "$MAX_S" .venv/bin/python -u -m scripts.oracle_fold --garment_type "$GTYPE" --garment_name "$GARMENT" \
  --headless --enable_cameras --device "${DEVICE:-cpu}" --seed 42 2>&1 | tee "$OUT/run.log" \
  | grep --line-buffered -E "\[oracle\]|TELEOP|Traceback|Error:" | grep --line-buffered -v carb.launcher
echo "[run] isaac exit: ${PIPESTATUS[0]}"

if [ "$MODE" = grip ]; then
  "$CH/.venv/bin/python" "$WF_DIR/teacher/runpod/summarize_grip.py" "$OUT" | tee "$OUT/summary.txt"
fi

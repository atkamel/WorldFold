#!/usr/bin/env bash
# Dream-1 grasp test on the GCP L4 VM, inside the wfisaac container (docker_run.sh dream): our realistic Isaac scene
# (Run B settings: fitted arm, v2 cloth, flattened garments, flat pads, attach or native jaw contact) driven by Dream-1
# through dream_test/server.py, one sim per worker. The key comes in as DREAM_API_KEY (never written to disk here).
#   BATTERY=smoke|full  WORKERS=1|2  FAKE_DREAM=1 (plumbing test, no API calls)  MAX_S=...
set -uo pipefail
WS=${WS:-/workspace}
WF=$WS/WorldFold/teacher
SRC=/opt/lehome_solution
CH=$SRC/lehome-challenge
BATTERY=${BATTERY:-smoke}
WORKERS=${WORKERS:-1}
OUT=$WS/results/dream/$(date +%Y%m%d-%H%M%S)-$BATTERY
DV=$WS/results/dream_env                # Dream's library (Python 3.11, jax CPU); kept on the VM disk
mkdir -p "$OUT"
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y XDG_RUNTIME_DIR=/tmp
export __GLX_VENDOR_LIBRARY_NAME=nvidia VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
T0=$(date +%s)
[ "${FAKE_DREAM:-0}" = 1 ] || [ -n "${DREAM_API_KEY:-}" ] || { echo "[dr] FATAL no DREAM_API_KEY"; exit 3; }

bash "$WF/runpod/link_code.sh"
for p in grasp_log_patch.py arm_port_patch.py attach_patch.py dream_test/dream_isaac_patch.py; do
  "$SRC/.venv/bin/python" "$WF/$p" "$SRC" || { echo "[dr] FATAL patch $p"; exit 3; }
done
"$CH/.venv/bin/python" "$WF/oracle/isaac_physics.py" apply v2 "$CH" | tee "$OUT/physics.txt"
export TEACHER_ARM=fitted
CONTACT=${CONTACT:-$(cat "$WS/results/runB_contact.txt" 2>/dev/null || echo attach)}
[ "$CONTACT" = attach ] && export TEACHER_ATTACH=1 || export TEACHER_ATTACH=0
echo "[dr] arm fitted, contact $CONTACT (runB_contact.txt: $(cat "$WS/results/runB_contact.txt" 2>/dev/null || echo none))" | tee -a "$OUT/physics.txt"
ROBOT="$CH/Assets/robots/lerobot/so101_follower_good.usd"
[ -f "$ROBOT.orig" ] || cp "$ROBOT" "$ROBOT.orig"
cp "$WF/oracle/assets/so101_flatpads.usd" "$ROBOT" && echo "[dr] flat pads"

GL=$("$DV/bin/python" -c "import sys; sys.path.insert(0, '$WF/dream_test'); import tasks; print(' '.join(sorted({e['garment'] for e in tasks.battery('$BATTERY')})))" 2>/dev/null)
if [ ! -x "$DV/bin/python" ]; then
  echo "[dr] installing Dream's library into $DV"
  if command -v uv >/dev/null; then
    uv venv -q --python "$CH/.venv/bin/python" "$DV" && uv pip install -q --python "$DV/bin/python" dreamlabs==1.0.1
  else
    "$CH/.venv/bin/python" -m venv "$DV" && "$DV/bin/pip" install -q dreamlabs==1.0.1
  fi
  "$DV/bin/python" -c "import dream, dream_ik; print('[dr] dream', dream.__version__)" || { echo "[dr] FATAL dream install"; exit 3; }
  GL=$("$DV/bin/python" -c "import sys; sys.path.insert(0, '$WF/dream_test'); import tasks; print(' '.join(sorted({e['garment'] for e in tasks.battery('$BATTERY')})))")
fi
echo "[dr] battery $BATTERY, $WORKERS workers, garments: $GL" | tee "$OUT/garments.txt"
for g in $GL; do
  "$CH/.venv/bin/python" "$WF/oracle/flatten_garment.py" use "$CH" "$g" | grep -q "^FLAT_USE $g" || {
    echo "[dr] flattening $g"; GARMENT=$g KEEP_POD=1 MAX_S=600 bash "$WF/runpod/run_isaac.sh" flatten >"$OUT/flatten_$g.log" 2>&1
    "$CH/.venv/bin/python" "$WF/oracle/flatten_garment.py" use "$CH" "$g" || { echo "[dr] FATAL no flat model for $g"; exit 3; }; }
done

export TEACHER_GRASP_LOG=$OUT/grasps ORACLE_DIR=$WF/oracle
FAKE=""; [ "${FAKE_DREAM:-0}" = 1 ] && FAKE="--fake-dream"
for i in $(seq 0 $((WORKERS - 1))); do touch "$OUT/server$i.log"; done
tail -n +1 -F "$OUT"/server*.log 2>/dev/null | grep --line-buffered -E "DREAMRESULT|DREAMDONE|\[dream\]|\[sim|FATAL|Traceback" &
PIDS=()
for i in $(seq 0 $((WORKERS - 1))); do
  [ "$i" -gt 0 ] && sleep 60                      # staggered starts (PhysX start hang with simultaneous boots)
  timeout --kill-after=60 "${MAX_S:-10800}" "$DV/bin/python" -u "$WF/dream_test/server.py" --battery "$BATTERY" \
    --shard "$i" --shards "$WORKERS" --out "$OUT" --ch "$CH" --python "$CH/.venv/bin/python" $FAKE \
    > "$OUT/server$i.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
sleep 2; pkill -f "tail -n +1 -F $OUT" 2>/dev/null
echo "[dr] DONE $OUT after $(( $(date +%s) - T0 )) s"

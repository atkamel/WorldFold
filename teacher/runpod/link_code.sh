#!/usr/bin/env bash
# Put our code at the paths the Modal image used (/opt/teacher/..., lehome-challenge/scripts/oracle_fold.py).
# The container filesystem resets with a new pod, so run_isaac.sh calls this every time (cheap).
set -euo pipefail
WS=${WS:-/workspace}
WF_DIR=${WF_DIR:-$WS/WorldFold}
[ -d /opt/lehome_solution/lehome-challenge/.venv ] && CH_DEFAULT=/opt/lehome_solution/lehome-challenge || CH_DEFAULT=$WS/lehome_solution/lehome-challenge
CH=${CH:-$CH_DEFAULT}
mkdir -p /opt
ln -sfn "$WF_DIR/teacher" /opt/teacher
cp "$WF_DIR/teacher/oracle/oracle_fold.py" "$CH/scripts/oracle_fold.py"
TOWEL=Top_Short_Unseen_9   # flat towel disguised as a short-sleeve top (teacher/towel/)
if [ -d "$WF_DIR/teacher/towel/$TOWEL" ]; then
  mkdir -p "$CH/Assets/objects/Challenge_Garment/Release/Top_Short"
  cp -r "$WF_DIR/teacher/towel/$TOWEL" "$CH/Assets/objects/Challenge_Garment/Release/Top_Short/"
fi
echo "[link] /opt/teacher -> $WF_DIR/teacher; oracle_fold.py copied"

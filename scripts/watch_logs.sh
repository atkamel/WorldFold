#!/usr/bin/env bash
# Poll logs without holding them open (a `tail -F` keeps a Windows file handle, which blocks renames and the
# launcher's own final append). Prints each new line matching PATTERN, prefixed by its log's directory name.
#   bash scripts/watch_logs.sh 'pattern' log1 [log2 ...]
pat="$1"; shift
declare -A seen
while true; do
  for f in "$@"; do
    [ -f "$f" ] || continue
    n=$(wc -l < "$f")
    s=${seen[$f]:-$n}
    if [ "$n" -gt "$s" ]; then
      sed -n "$((s + 1)),${n}p" "$f" | grep -E "$pat" | sed "s|^|$(basename "$(dirname "$f")"): |"
    fi
    seen[$f]=$n
  done
  sleep 30
done

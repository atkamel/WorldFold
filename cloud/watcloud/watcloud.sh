#!/usr/bin/env bash
# Mac-side driver for one WATcloud job (docs/warp_port.md phase 2; trimmed from the
# minecraft-rl kit). Ships the repo at git HEAD plus any --in paths, runs one command
# in a fresh venv on the node's tmpdisk, tars the --out paths back. Nothing stays on
# the cluster after pull.
#
#   cloud/watcloud/watcloud.sh run [--time 0:30:00] [--gpu] [--cpus 8] [--mem 32G] [--partition compute]
#                                  [--req requirements-warp.txt] [--in PATH]... [--out PATH]... -- <command>
#     = push + submit + wait + pull
#   cloud/watcloud/watcloud.sh push ... | submit | wait | pull | status | clean
#
# Needs the "watcloud" ssh alias from the mcrl kit's README (override with WATCLOUD_HOST).
# Commands run from the repo root with the venv on PATH, e.g.
#   cloud/watcloud/watcloud.sh run --gpu --time 0:30:00 -- python scripts/warp_throughput.py --nworld 256 1024 4096
# The job log (job-<id>.out) is always pulled, into outputs/watcloud/<tag>/.
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST="${WATCLOUD_HOST:-watcloud}"
STATE="$HOME/.worldfold-watcloud"     # the one job in flight
TIME="0:30:00"; GPU=0; CPUS=8; MEM="32G"; PARTITION="compute"; REQ="requirements-warp.txt"
INS=(); OUTS=()

die() { echo "watcloud: $*" >&2; exit 1; }
rsh() { ssh "$HOST" "bash -lc $(printf %q "$*")"; }     # SLURM is on PATH only in a login shell
load() { [ -f "$STATE" ] || die "no job in flight (run 'push' first)"; . "$STATE"; }

push() {
    while [ $# -gt 0 ]; do case "$1" in
        --time) TIME="$2"; shift 2;; --gpu) GPU=1; shift;; --cpus) CPUS="$2"; shift 2;; --mem) MEM="$2"; shift 2;;
        --partition) PARTITION="$2"; shift 2;; --req) REQ="$2"; shift 2;;
        --in) INS+=("$2"); shift 2;; --out) OUTS+=("$2"); shift 2;;
        --) shift; break;; *) die "unknown arg $1";; esac; done
    [ $# -gt 0 ] || die "no command after --"
    [ -f "$REQ" ] || die "no such requirements file $REQ"
    local cmd; cmd=$(printf '%q ' "$@")
    local tag; tag=$(date +%Y%m%d-%H%M%S)
    local tmp; tmp=$(mktemp -d)
    echo "== packing repo (git HEAD)"
    git archive --format=tar --prefix=repo/ -o "$tmp/repo.tar" HEAD
    if [ ${#INS[@]} -gt 0 ]; then
        for p in "${INS[@]}"; do [ -e "$p" ] || die "no such --in path $p"; done
        echo "== packing inputs: ${INS[*]}"
        COPYFILE_DISABLE=1 tar cf "$tmp/inputs.tar" --no-xattrs "${INS[@]}"
    fi
    # %q on the whole line: job.sbatch sources this file, so CMD must be one quoted word.
    printf 'CMD=%q\nREQ=%q\nOUTS=%q\nGPU=%s\n' "$cmd" "$REQ" "${OUTS[*]:-}" "$GPU" > "$tmp/job.env"
    du -sh "$tmp"/*.tar
    echo "== rsync to $HOST:worldfold-stage/$tag"
    rsh "mkdir -p worldfold-stage/$tag"
    rsync -az --progress "$tmp"/*.tar "$tmp/job.env" cloud/watcloud/job.sbatch "$HOST:worldfold-stage/$tag/"
    rm -rf "$tmp"
    printf 'TAG=%s\nTIME=%s\nGPU=%s\nCPUS=%s\nMEM=%s\nPARTITION=%s\nJOBID=\n' "$tag" "$TIME" "$GPU" "$CPUS" "$MEM" "$PARTITION" > "$STATE"
    echo "staged as $tag"
}

submit() {
    load
    [ -z "${JOBID:-}" ] || die "job $JOBID already submitted for $TAG"
    local gres="tmpdisk:20480"; [ "$GPU" = 1 ] && gres="shard:rtx_3090:8192,$gres"
    local out
    out=$(rsh "cd worldfold-stage/$TAG && sbatch --parsable --time=$TIME --partition=$PARTITION --cpus-per-task=$CPUS --mem=$MEM \
        --gres=$gres --job-name=worldfold-$TAG --output=\$HOME/worldfold-stage/$TAG/job-%j.out \
        --export=ALL,STAGE=\$HOME/worldfold-stage/$TAG job.sbatch")
    JOBID="${out%%;*}"
    [[ "$JOBID" =~ ^[0-9]+$ ]] || die "sbatch said: $out"
    sed -i '' "s/^JOBID=.*/JOBID=$JOBID/" "$STATE"
    echo "submitted job $JOBID ($PARTITION, $TIME, $CPUS cpus, $MEM, gres $gres)"
}

status() {
    load
    rsh "squeue -u \$(whoami) -o '%.10i %.24j %.8T %.11M %.11l %R'; tail -n 8 worldfold-stage/$TAG/job-${JOBID:-x}.out 2>/dev/null"
}

wait_() {
    load
    [ -n "${JOBID:-}" ] || die "not submitted"
    while :; do
        rc=0; out=$(rsh "squeue -h -j $JOBID" 2>&1) || rc=$?
        if [ $rc -eq 255 ]; then echo "ssh unreachable, retrying: $out"; sleep 60; continue; fi
        [ -z "$out" ] && break
        case "$out" in *"Invalid job id"*) break;; esac
        rsh "tail -n 1 worldfold-stage/$TAG/job-$JOBID.out 2>/dev/null" || true
        sleep 60
    done
    echo "job $JOBID left the queue"
}

pull() {
    load
    local dest="outputs/watcloud/$TAG"
    mkdir -p "$dest"
    rsync -az "$HOST:worldfold-stage/$TAG/job-*.out" "$dest/" 2>/dev/null || true
    if rsh "test -f worldfold-stage/$TAG/out.tar"; then
        rsync -az --progress "$HOST:worldfold-stage/$TAG/out.tar" "$dest/"
        tar xf "$dest/out.tar" && rm "$dest/out.tar"      # --out paths land back in place, merged (no delete)
        echo "== outputs restored in place"
    fi
    rsh "test -f worldfold-stage/$TAG/DONE" || echo "WARNING: no DONE marker; the job did not finish cleanly (see $dest/job-*.out)"
    echo "== job log in $dest"; tail -n 20 "$dest"/job-*.out 2>/dev/null || true
    rsh "rm -rf worldfold-stage/$TAG"
    rm -f "$STATE"
    echo "staging dir removed from $HOST"
}

clean() { rsh "rm -rf worldfold-stage"; rm -f "$STATE"; echo "worldfold-stage removed from $HOST"; }

case "${1:-}" in
    run)    shift; push "$@"; submit; wait_; pull;;
    push)   shift; push "$@";;
    submit) submit;;
    wait)   wait_;;
    pull)   pull;;
    status) status;;
    clean)  clean;;
    *) sed -n '2,15p' "$0"; exit 1;;
esac

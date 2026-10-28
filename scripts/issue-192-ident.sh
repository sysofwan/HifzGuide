#!/usr/bin/env bash
# issue-192-ident.sh — tlog S7 confirm decodes, identification and tranche manifests on cuda-dev.
#
# Mac side, from the worktree root:
#   scripts/issue-192-ident.sh decode launch     # timed 500-row slice, then all four tranches (GPU lock)
#   scripts/issue-192-ident.sh decode check
#   scripts/issue-192-ident.sh decode fetch DIR  # pull the decode trees + slice_timing.json back
#   scripts/issue-192-ident.sh identify launch   # after the decodes are committed
#   scripts/issue-192-ident.sh identify check
#   scripts/issue-192-ident.sh identify fetch DIR
#   scripts/issue-192-ident.sh manifests launch  # after the ident trees are committed
#   scripts/issue-192-ident.sh manifests check
#   scripts/issue-192-ident.sh manifests fetch DIR
#
# Every job stages the current branch as a bundle on the box, writes one log, and writes its
# exit file last, so `check` never reads a log tail. Jobs run under setsid with stdin closed,
# so they survive the ssh session. Only `decode` holds /root/scratch/gpu.lock: identify and
# manifests are CPU work. Logs carry aggregate counts only: the confirm tranches are sealed,
# so no label or decode may appear in them.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog/s7}
LOGS=$ROOT/logs
REPO=$ROOT/repo
CACHE=$ROOT/cache
OUT=tools/tadabur/tlog/ident/decodes
IDENT=tools/tadabur/tlog/ident
TRANCHES=tools/tadabur/tlog/tranches
EXPOSURE=tools/tadabur/exposure
# The PR that keeps the cited code SHA resolvable after the squash merge (refs/pull/<N>/head).
PR=${PR:-211}

job=${1:-}
action=${2:-}
case "$job" in
  decode)    NAME=decode ;;
  identify)  NAME=identify ;;
  manifests) NAME=manifests ;;
  *) echo "usage: $0 decode|identify|manifests [launch|check|fetch DIR]" >&2
     exit 2 ;;
esac

remote() { ssh -n "$HOST" "$@"; }

stage_repo() {
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT' '$LOGS' '$CACHE' && rm -rf '$REPO' '$ROOT/repo.bundle'"
  git -C "$REPO_DIR" bundle create - "$branch" | ssh "$HOST" "cat > '$ROOT/repo.bundle'"
  if [ "$branch" = HEAD ]; then
    remote "git clone -q '$ROOT/repo.bundle' '$REPO'"
  else
    remote "git clone -q -b '$branch' '$ROOT/repo.bundle' '$REPO'"
  fi
  remote "rm -f '$ROOT/repo.bundle' && test \"\$(git -C '$REPO' rev-parse HEAD)\" = '$sha'" \
    || { echo "the box's clone is not $sha" >&2; return 1; }
}

command_for() {
  case "$job" in
    decode)
      echo "flock /root/scratch/gpu.lock python -u -m tadabur.tlog_tranches decode --cache-dir '$CACHE' --out-dir '$REPO/$OUT' --pr $PR" ;;
    identify)
      echo "python -u -m tadabur.tlog_tranches identify --decodes-dir '$REPO/$OUT' --out-dir '$REPO/$IDENT' --pr $PR" ;;
    manifests)
      echo "python -u -m tadabur.tlog_tranches manifests --ident-dir '$REPO/$IDENT' --tranches-dir '$REPO/$TRANCHES' --exposure-dir '$REPO/$EXPOSURE' --slice-timing '$REPO/$OUT/slice_timing.json' --pr $PR" ;;
  esac
}

write_runner() {
  remote "cat > '$ROOT/run-$NAME.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
source '$CONDA'
conda activate $ENV_NAME
cd '$REPO/tools' || exit 1
$(command_for)
code=\$?
echo \"\$code\" > '$LOGS/$NAME.exit'
exit \$code
SH
chmod +x '$ROOT/run-$NAME.sh'"
}

launch() {
  stage_repo
  write_runner
  remote "rm -f '$LOGS/$NAME.exit' '$LOGS/$NAME.pid'
          setsid bash '$ROOT/run-$NAME.sh' </dev/null >'$LOGS/$NAME.log' 2>&1 &
          echo \$! > '$LOGS/$NAME.pid'"
  echo "job started on $HOST"
  echo "  pid:  $(remote "cat '$LOGS/$NAME.pid'")"
  echo "  log:  $HOST:$LOGS/$NAME.log"
  echo "  exit: $HOST:$LOGS/$NAME.exit (0 = clean; absent while running)"
  echo "  check with: scripts/issue-192-ident.sh $job check"
}

check() {
  if remote "test -f '$LOGS/$NAME.exit'"; then
    echo "exit $(remote "cat '$LOGS/$NAME.exit'")"
  else
    echo "running (pid $(remote "cat '$LOGS/$NAME.pid' 2>/dev/null || echo unknown"))"
    remote "ps -p \"\$(cat '$LOGS/$NAME.pid')\" -o pid=,etime=,stat= 2>/dev/null || true"
  fi
}

fetch() {
  local destination=${1:?fetch needs a destination directory}
  mkdir -p "$destination"
  case "$job" in
    decode) rsync -a "$HOST:$REPO/$OUT/" "$destination/" ;;
    identify)
      for role in confirm.1 confirm.2 confirm.3 confirm.4; do
        mkdir -p "$destination/$role"
        rsync -a "$HOST:$REPO/$IDENT/$role/" "$destination/$role/"
      done ;;
    manifests)
      mkdir -p "$destination/tranches" "$destination/exposure"
      rsync -a "$HOST:$REPO/$TRANCHES/" "$destination/tranches/"
      rsync -a "$HOST:$REPO/$EXPOSURE/tlog.*.clips.jsonl" "$destination/exposure/" ;;
    *) echo "fetch: $job has no fetched tree" >&2; exit 2 ;;
  esac
  echo "fetched $job into $destination"
}

case "$action" in
  launch) launch ;;
  check) check ;;
  fetch) fetch "${3:-}" ;;
  "") launch ;;
  *) echo "usage: $0 $job [launch|check|fetch DIR]" >&2; exit 2 ;;
esac

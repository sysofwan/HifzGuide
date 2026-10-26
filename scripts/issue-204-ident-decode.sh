#!/usr/bin/env bash
# issue-204-ident-decode.sh — S5b's confirmation decodes on cuda-dev (#204).
#
# Mac side, from the worktree root, in this order:
#   scripts/issue-204-ident-decode.sh launch          # stage HEAD on the box, start detached
#   scripts/issue-204-ident-decode.sh check           # print the exit code, or "running"
#   scripts/issue-204-ident-decode.sh fetch DIR       # pull the three committed stores back
#
# The job decodes the 1,000 fresh Freesound recordings (stream_b0, the mode the 445 pool was
# decoded in) and the 1,000 Common Voice clips (whole-spans, the S6/S7 mode) with the base
# teacher at 01a1ef9. The drawn CV clips are expected at CV_CLIPS (rsynced from the Mac); the
# Freesound shards are streamed into CACHE and deleted after reading. Every store cites the
# pre-registration and its manifest (path, SHA-256, commit).
#
# The job holds /root/scratch/gpu.lock, writes one log, and writes logs/ident-confirm.exit
# last; `check` reads that file, never a log tail. Nothing is waited for here: the job runs
# under setsid with stdin closed, so it survives the ssh session.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog/s5b}
LOGS=$ROOT/logs
REPO=$ROOT/repo
CACHE=$ROOT/freesound
CV_CLIPS=${CV_CLIPS:-/root/scratch/tlog/s5b/cv-clips}
CONFIRM_ARMS=${CONFIRM_ARMS:-freesound cv_ar cv_other}
OUT=tools/tadabur/tlog/ident/decodes
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

remote() { ssh -n "$HOST" "$@"; }

stage_repo() {
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT' '$LOGS' && rm -rf '$REPO' '$ROOT/repo.bundle'"
  # stdin is the bundle itself, so this one ssh must not use -n.
  git -C "$REPO_DIR" bundle create - "$branch" | ssh "$HOST" "cat > '$ROOT/repo.bundle'"
  if [ "$branch" = HEAD ]; then
    remote "git clone -q '$ROOT/repo.bundle' '$REPO'"
  else
    remote "git clone -q -b '$branch' '$ROOT/repo.bundle' '$REPO'"
  fi
  remote "rm -f '$ROOT/repo.bundle' && test \"\$(git -C '$REPO' rev-parse HEAD)\" = '$sha'" \
    || { echo "the box's clone is not $sha" >&2; return 1; }
}

write_runner() {
  remote "cat > '$ROOT/run-ident-confirm.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
source '$CONDA'
conda activate $ENV_NAME
cd '$REPO/tools' || exit 1
flock /root/scratch/gpu.lock python -u -m tadabur.tlog_ident_decode \\
  --only confirm --confirm-arms $CONFIRM_ARMS \\
  --cache-dir '$CACHE' --cv-clips-dir '$CV_CLIPS' \\
  --out-dir '$REPO/$OUT'
code=\$?
echo \"\$code\" > '$LOGS/ident-confirm.exit'
exit \$code
SH
chmod +x '$ROOT/run-ident-confirm.sh'"
}

launch() {
  stage_repo
  write_runner
  remote "rm -f '$LOGS/ident-confirm.exit' '$LOGS/ident-confirm.pid'
          setsid bash '$ROOT/run-ident-confirm.sh' </dev/null >'$LOGS/ident-confirm.log' 2>&1 &
          echo \$! > '$LOGS/ident-confirm.pid'"
  echo "job started on $HOST"
  echo "  pid:  $(remote "cat '$LOGS/ident-confirm.pid'")"
  echo "  log:  $HOST:$LOGS/ident-confirm.log"
  echo "  exit: $HOST:$LOGS/ident-confirm.exit (0 = clean; absent while running)"
  echo "  check with: scripts/issue-204-ident-decode.sh check"
}

check() {
  if remote "test -f '$LOGS/ident-confirm.exit'"; then
    echo "exit $(remote "cat '$LOGS/ident-confirm.exit'")"
  else
    echo "running (pid $(remote "cat '$LOGS/ident-confirm.pid' 2>/dev/null || echo unknown"))"
    remote "ps -p \"\$(cat '$LOGS/ident-confirm.pid')\" -o pid=,etime=,stat= 2>/dev/null || true"
  fi
}

fetch() {
  local destination=${1:?fetch needs a destination directory}
  mkdir -p "$destination"
  rsync -a "$HOST:$REPO/$OUT/" "$destination/"
  echo "fetched $HOST:$REPO/$OUT/ into $destination"
}

case "${1:-}" in
  launch) launch ;;
  check) check ;;
  fetch) fetch "${2:-}" ;;
  *) echo "usage: $0 launch|check|fetch DIR" >&2; exit 2 ;;
esac

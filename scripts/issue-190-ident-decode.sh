#!/usr/bin/env bash
# issue-190-ident-decode.sh — S5's whole-clip base-teacher decodes on cuda-dev (#190).
#
# Mac side, from the worktree root, in this order:
#   scripts/issue-190-ident-decode.sh launch          # stage HEAD on the box, start detached
#   scripts/issue-190-ident-decode.sh check           # print the exit code, or "running"
#   scripts/issue-190-ident-decode.sh fetch DIR       # pull the two committed stores back
#
# The job decodes the 52 real-mistake clips and the Freesound pool whole with the base teacher at
# 01a1ef9 (tadabur.tlog_ident_decode). It holds /root/scratch/gpu.lock, writes one log, and
# writes logs/ident.exit last; `check` reads that file, never a log tail. Nothing is waited for
# here: the job runs under setsid with stdin closed, so it survives the ssh session.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog/s5}
LOGS=$ROOT/logs
REPO=$ROOT/repo
OUT=tools/tadabur/tlog/ident/decodes
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

remote() { ssh "$HOST" "$@"; }

stage_repo() {
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT' '$LOGS' && rm -rf '$REPO' '$ROOT/repo.bundle'"
  git -C "$REPO_DIR" bundle create - "$branch" | remote "cat > '$ROOT/repo.bundle'"
  if [ "$branch" = HEAD ]; then
    remote "git clone -q '$ROOT/repo.bundle' '$REPO'"
  else
    remote "git clone -q -b '$branch' '$ROOT/repo.bundle' '$REPO'"
  fi
  remote "rm -f '$ROOT/repo.bundle' && test \"\$(git -C '$REPO' rev-parse HEAD)\" = '$sha'" \
    || { echo "the box's clone is not $sha" >&2; return 1; }
}

write_runner() {
  remote "cat > '$ROOT/run-ident.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
source '$CONDA'
conda activate $ENV_NAME
cd '$REPO/tools' || exit 1
flock /root/scratch/gpu.lock python -u -m tadabur.tlog_ident_decode \\
  --clips-dir /root/scratch/issue-83/stage/clips \\
  --clips-dir /root/scratch/issue-126/stage/clips \\
  --cache-dir '$ROOT/freesound' --pool '$ROOT/noise_pool.npz' \\
  --out-dir '$REPO/$OUT'
code=\$?
echo \"\$code\" > '$LOGS/ident.exit'
exit \$code
SH
chmod +x '$ROOT/run-ident.sh'"
}

launch() {
  stage_repo
  write_runner
  remote "rm -f '$LOGS/ident.exit' '$LOGS/ident.pid'
          setsid bash '$ROOT/run-ident.sh' </dev/null >'$LOGS/ident.log' 2>&1 &
          echo \$! > '$LOGS/ident.pid'"
  echo "job started on $HOST"
  echo "  pid:  $(remote "cat '$LOGS/ident.pid'")"
  echo "  log:  $HOST:$LOGS/ident.log"
  echo "  exit: $HOST:$LOGS/ident.exit (0 = clean; absent while running)"
  echo "  check with: scripts/issue-190-ident-decode.sh check"
}

check() {
  if remote "test -f '$LOGS/ident.exit'"; then
    echo "exit $(remote "cat '$LOGS/ident.exit'")"
  else
    echo "running (pid $(remote "cat '$LOGS/ident.pid' 2>/dev/null || echo unknown"))"
    remote "ps -p \"\$(cat '$LOGS/ident.pid')\" -o pid=,etime=,stat= 2>/dev/null || true"
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

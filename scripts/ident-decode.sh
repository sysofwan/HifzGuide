#!/usr/bin/env bash
# ident-decode.sh — the base teacher's identification decodes on cuda-dev (#190, #204).
#
# Mac side, from the worktree root, in this order:
#   scripts/ident-decode.sh s5  launch          # S5: the 52 real-mistake clips and the noise pool
#   scripts/ident-decode.sh s5b launch          # S5b: 1,000 Freesound and 1,000 Common Voice
#   scripts/ident-decode.sh <job> check         # print the exit code, or "running"
#   scripts/ident-decode.sh <job> fetch DIR     # pull the committed stores back
#
# `s5` decodes the 52 real-mistake clips and the Freesound pool whole. `s5b` decodes the 1,000
# fresh Freesound recordings (stream_b0, the mode the 445 pool was decoded in) and the 1,000
# Common Voice clips (whole-spans, the S6/S7 mode); the drawn CV clips are expected at CV_CLIPS,
# and the Freesound shards stream into CACHE and are deleted after reading.
#
# Both jobs stage HEAD on the box as a bundle, hold /root/scratch/gpu.lock, write one log and
# write their exit file last; `check` reads that file, never a log tail. Nothing is waited for
# here: the job runs under setsid with stdin closed, so it survives the ssh session.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
OUT=tools/tadabur/tlog/ident/decodes
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIRM_ARMS=${CONFIRM_ARMS:-freesound cv_ar cv_other}

job=${1:-}
case "$job" in
  s5)  ROOT=${ISSUE_ROOT:-/root/scratch/tlog/s5}; NAME=ident ;;
  s5b) ROOT=${ISSUE_ROOT:-/root/scratch/tlog/s5b}; NAME=ident-confirm ;;
  *) echo "usage: $0 s5|s5b launch|check|fetch DIR" >&2; exit 2 ;;
esac
LOGS=$ROOT/logs
REPO=$ROOT/repo
CACHE=$ROOT/freesound
CV_CLIPS=${CV_CLIPS:-$ROOT/cv-clips}

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

# The one line that differs between the two jobs.
decoder_args() {
  if [ "$job" = s5 ]; then
    echo "--clips-dir /root/scratch/issue-83/stage/clips --clips-dir /root/scratch/issue-126/stage/clips --cache-dir '$CACHE' --pool '$ROOT/noise_pool.npz'"
  else
    echo "--only confirm --confirm-arms $CONFIRM_ARMS --cache-dir '$CACHE' --cv-clips-dir '$CV_CLIPS'"
  fi
}

write_runner() {
  remote "cat > '$ROOT/run-$NAME.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
source '$CONDA'
conda activate $ENV_NAME
cd '$REPO/tools' || exit 1
flock /root/scratch/gpu.lock python -u -m tadabur.tlog_ident_decode $(decoder_args) \\
  --out-dir '$REPO/$OUT'
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
  echo "  check with: scripts/ident-decode.sh $job check"
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
  rsync -a "$HOST:$REPO/$OUT/" "$destination/"
  echo "fetched $HOST:$REPO/$OUT/ into $destination"
}

case "${2:-}" in
  launch) launch ;;
  check) check ;;
  fetch) fetch "${3:-}" ;;
  *) echo "usage: $0 s5|s5b launch|check|fetch DIR" >&2; exit 2 ;;
esac

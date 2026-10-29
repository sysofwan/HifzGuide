#!/usr/bin/env bash
# issue-225-audit.sh — the round-2 catch-loss audit on cuda-dev (#225).
#
# Mac side, from the worktree root, after the pre-registration freeze and the live/EMA choice
# are committed (the draw refuses without them):
#   scripts/issue-225-audit.sh draw               # build the blind worklist on the box
#   scripts/issue-225-audit.sh draw fetch DIR     # pull worklist/key/summary back, then commit
#   scripts/issue-225-audit.sh ui launch          # serve the listening page on 0.0.0.0:8000
#   scripts/issue-225-audit.sh backup launch      # copy the verdicts every minute while listening
#   scripts/issue-225-audit.sh ui fetch DIR       # pull verdicts.jsonl after a sitting; commit it
#   scripts/issue-225-audit.sh backup fetch DIR   # pull the timestamped copies
#
# The worklist is committed before the UI starts: `ui launch` stages the current branch, so a
# worklist that is only in the box's clone (drawn above, not committed) is gone on relaunch.
# Restarting a running UI needs the owner's OK.
#
# Every job stages the current branch as a bundle on the box, writes one log and writes its
# exit file last, so `check` never reads a log tail. Jobs run under setsid with stdin closed,
# so they survive the ssh session. The UI reads the look clip cache #212 filled; no audio is
# copied off the box.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/catch_loss}
LOGS=$ROOT/logs
REPO=$ROOT/repo
# The owner's verdicts live in the scratch root, not the clone: a relaunch re-clones and would
# drop them. `ui fetch` pulls the file; commit it at tools/tadabur/consonant2_readout/
# catch_loss/verdicts.jsonl.
VERDICTS=$ROOT/verdicts.jsonl
BACKUP=$ROOT/backup
# #212's look clip cache: each clip's source bytes as <audio_sha256>.audio.
CLIPS=${CLIPS:-/root/scratch/tlog/r2/clips}
AUDIT=tools/tadabur/consonant2_readout/catch_loss
TRANCHE=tools/tadabur/tlog/tranches/look.jsonl

job=${1:-}
action=${2:-}
case "$job" in
  draw|ui|backup) NAME=$job ;;
  *) echo "usage: $0 draw|ui|backup [launch|check|fetch DIR]" >&2; exit 2 ;;
esac

remote() { ssh -n "$HOST" "$@"; }

stage_repo() {
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT' '$LOGS' && rm -rf '$REPO' '$ROOT/repo.bundle'"
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
    draw)
      echo "python -u -m tadabur.catch_loss_audit draw --tranche '$REPO/$TRANCHE' && cp '$REPO/$AUDIT/worklist.jsonl' '$REPO/$AUDIT/key.jsonl' '$REPO/$AUDIT/summary.json' '$ROOT/'" ;;
    ui)
      echo "python -u -m tadabur.tashkeel_audit_ui --catch-loss '$REPO/$AUDIT/worklist.jsonl' --clip-cache '$CLIPS' --verdicts '$VERDICTS' --port 8000 --host 0.0.0.0" ;;
    backup)
      echo "mkdir -p '$BACKUP'; while true; do
        cp '$VERDICTS' '$BACKUP/verdicts.\$(date +%s).jsonl' 2>/dev/null || true
        ls -1t '$BACKUP'/verdicts.*.jsonl 2>/dev/null | tail -n +101 | xargs -r rm -f
        sleep 60
      done" ;;
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
  echo "  check with: scripts/issue-225-audit.sh $job check"
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
    draw)
      rsync -a --include 'worklist.jsonl' --include 'key.jsonl' \
        --include 'summary.json' --exclude '*' "$HOST:$ROOT/" "$destination/" ;;
    ui)
      rsync -a "$HOST:$VERDICTS" "$destination/" ;;
    backup)
      rsync -a "$HOST:$BACKUP/" "$destination/" ;;
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

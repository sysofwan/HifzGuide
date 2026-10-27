#!/usr/bin/env bash
# issue-191-ident.sh — tlog S6 decodes, identification and the owner audit on cuda-dev.
#
# Mac side, from the worktree root:
#   scripts/issue-191-ident.sh limit launch     # measure D_max on synthetic audio (GPU lock)
#   scripts/issue-191-ident.sh limit check
#   scripts/issue-191-ident.sh limit fetch      # pull decode_limit.json back
#   scripts/issue-191-ident.sh decode launch    # stream+decode the selected rows (GPU lock)
#   scripts/issue-191-ident.sh decode check
#   scripts/issue-191-ident.sh decode fetch DIR
#   scripts/issue-191-ident.sh identify         # after the margin approval is on main
#   scripts/issue-191-ident.sh audit            # draw the stratified worklist
#   scripts/issue-191-ident.sh stage            # stage the drawn clips for listening
#   scripts/issue-191-ident.sh ui launch        # serve the listening page on 0.0.0.0:8000
#   scripts/issue-191-ident.sh report           # render docs/tlog-audit.md
#
# Every job stages the current branch as a bundle on the box, writes one log, and writes its
# exit file last, so `check` never reads a log tail. Jobs run under setsid with stdin closed,
# so they survive the ssh session. `limit` and `decode` hold /root/scratch/gpu.lock.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog/s6}
LOGS=$ROOT/logs
REPO=$ROOT/repo
CACHE=$ROOT/cache
AUDIO=$ROOT/audio
# The UI reads Uthmani ayah text from quran.db, a gitignored release asset: copy it once from
# any checkout that has one (`scp` it to this path, or set QURAN_DB).
QURAN_DB=${QURAN_DB:-$ROOT/data/quran.db}
OUT=tools/tadabur/tlog/ident/decodes
IDENT=tools/tadabur/tlog/ident

job=${1:-}
action=${2:-}
case "$job" in
  limit)     NAME=limit ;;
  decode)    NAME=decode ;;
  identify)  NAME=identify ;;
  audit)     NAME=audit ;;
  stage)     NAME=stage ;;
  ui)        NAME=ui ;;
  report)    NAME=report ;;
  *) echo "usage: $0 limit|decode|identify|audit|stage|ui|report [launch|check|fetch DIR]" >&2
     exit 2 ;;
esac

remote() { ssh -n "$HOST" "$@"; }

stage_repo() {
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT' '$LOGS' '$CACHE' '$AUDIO' && rm -rf '$REPO' '$ROOT/repo.bundle'"
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
    limit)
      echo "flock /root/scratch/gpu.lock python -u -m tadabur.tlog_ident_run limit --out '$REPO/$IDENT/decode_limit.json'" ;;
    decode)
      echo "flock /root/scratch/gpu.lock python -u -m tadabur.tlog_ident_run decode --cache-dir '$CACHE' --out-dir '$REPO/$OUT'" ;;
    identify)
      echo "python -u -m tadabur.tlog_ident_run identify --decodes-dir '$REPO/$OUT' --out-dir '$REPO/$IDENT'" ;;
    audit)
      echo "python -u -m tadabur.tlog_ident_run audit && cp '$REPO/tools/tadabur/tlog/audit/draw.jsonl' '$ROOT/draw.jsonl'" ;;
    stage)
      echo "python -u -m tadabur.tlog_ident_run stage --draw '$ROOT/draw.jsonl' --audio-dir '$AUDIO' --cache-dir '$CACHE'" ;;
    ui)
      echo "python -u -m tadabur.tlog_audit_ui --worklist '$REPO/tools/tadabur/tlog/audit/worklist.jsonl' --audio-dir '$AUDIO' --verdicts '$REPO/tools/tadabur/tlog/audit/verdicts.jsonl' --quran-db '$QURAN_DB' --port 8000 --host 0.0.0.0" ;;
    report)
      echo "python -u -m tadabur.tlog_ident_run report" ;;
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
  echo "  check with: scripts/issue-191-ident.sh $job check"
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
    limit) rsync -a "$HOST:$REPO/$IDENT/decode_limit.json" "$destination/" ;;
    decode) rsync -a "$HOST:$REPO/$OUT/" "$destination/" ;;
    identify) rsync -a "$HOST:$REPO/$IDENT/" "$destination/" ;;
    audit|stage) rsync -a "$HOST:$REPO/tools/tadabur/tlog/audit/" "$destination/" ;;
    report)
      rsync -a "$HOST:$REPO/docs/tlog-audit.md" "$destination/"
      rsync -a "$HOST:$REPO/tools/tadabur/tlog/audit/" "$destination/audit/" ;;
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

#!/usr/bin/env bash
# issue-225-audit.sh — the catch-loss audit on cuda-dev (#225, #257).
#
# Mac side, from the worktree root, after the pre-registration freeze and the live/EMA choice
# are committed (the draw refuses without them):
#   scripts/issue-225-audit.sh draw               # build the blind worklist on the box
#   scripts/issue-225-audit.sh draw fetch DIR     # pull worklist/key/summary back, then commit
#   scripts/issue-225-audit.sh ui launch          # serve the listening page (SPEC's port)
#   scripts/issue-225-audit.sh backup launch      # copy verdicts and review every minute
#   scripts/issue-225-audit.sh ui fetch DIR       # pull verdicts.jsonl and review.jsonl; commit
#   scripts/issue-225-audit.sh backup fetch DIR   # pull the timestamped copies
#
# SPEC names the round and nothing else changes with it. consonant2 is the default and keeps
# round 2's paths, port and draw (`/root/scratch/catch_loss`, :8000); consonant3 reads with
# `--spec consonant3` from `tools/tadabur/consonant3_readout/catch_loss` and keeps its own
# scratch root `/root/scratch/catch_loss_r3` and port :8001, so round 2's sitting keeps serving
# untouched. ISSUE_ROOT and PORT override either.
#
# The worklist is committed before the UI starts: `ui launch` stages the current branch, so a
# worklist that is only in the box's clone (drawn above, not committed) is gone on relaunch.
# Restarting a running UI needs the owner's OK.
#
# `draw` and `ui` stage the current branch as a bundle on the box; the backup loop only copies
# within the scratch root, so its launch never re-clones. Every job writes one log and writes its
# exit file last, so `check` never reads a log tail. Jobs run under setsid with stdin closed,
# so they survive the ssh session. The UI reads the look clip cache #212 filled; no audio is
# copied off the box.
set -euo pipefail

HOST_GIVEN=${HOST:+1}
HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SPEC_GIVEN=${SPEC:+1}
SPEC=${SPEC:-consonant2}
case "$SPEC" in
  consonant2) SPEC_ROOT=/root/scratch/catch_loss; SPEC_PORT=8000; DRAW_ARG= ;;
  consonant3) SPEC_ROOT=/root/scratch/catch_loss_r3; SPEC_PORT=8001; DRAW_ARG=" --spec $SPEC" ;;
  *) echo "unknown SPEC '$SPEC' (consonant2|consonant3)" >&2; exit 2 ;;
esac
ROOT=${ISSUE_ROOT:-$SPEC_ROOT}
PORT_GIVEN=${PORT:+1}
PORT=${PORT:-$SPEC_PORT}
LOGS=$ROOT/logs
REPO=$ROOT/repo
# The owner's verdicts and the sites queued for another listen live in the scratch root, not in
# the clone: a relaunch re-clones and would drop them. `ui fetch` pulls both; commit them at
# tools/tadabur/${SPEC}_readout/catch_loss/{verdicts,review}.jsonl.
VERDICTS=$ROOT/verdicts.jsonl
REVIEW=$ROOT/review.jsonl
BACKUP=$ROOT/backup
# #212's look clip cache: each clip's source bytes as <audio_sha256>.audio.
CLIPS_GIVEN=${CLIPS:+1}
CLIPS=${CLIPS:-/root/scratch/tlog/r2/clips}
AUDIT=tools/tadabur/${SPEC}_readout/catch_loss
TRANCHE=tools/tadabur/tlog/tranches/look.jsonl

job=${1:-}
action=${2:-}
case "$job" in
  draw|ui|backup) NAME=$job ;;
  *) echo "usage: $0 draw|ui|backup [launch|check|fetch DIR]" >&2; exit 2 ;;
esac

remote() { ssh -n "$HOST" "$@"; }

# A fetch treats the remote `test`'s 1 as "absent" (the review list before the owner queues
# anything, the backup directory before it runs) and every other nonzero as the transport
# failure it is: a failed ssh must fail the fetch, never print it as absent and succeed.
fetch_file() {  # remote path, destination
  local path=$1 destination=$2 status=0
  remote "test -f '$path'" || status=$?
  if [ "$status" -eq 0 ]; then
    rsync -a "$HOST:$path" "$destination/"
  elif [ "$status" -eq 1 ]; then
    echo "$HOST:$path is absent (nothing to fetch)"
  else
    echo "ssh $HOST failed (status $status) checking $path" >&2
    return "$status"
  fi
}

fetch_tree() {  # remote directory, destination
  local directory=$1 destination=$2 status=0
  remote "test -d '$directory'" || status=$?
  if [ "$status" -eq 0 ]; then
    rsync -a "$HOST:$directory/" "$destination/"
  elif [ "$status" -eq 1 ]; then
    echo "$HOST:$directory is absent (nothing to fetch)"
  else
    echo "ssh $HOST failed (status $status) checking $directory" >&2
    return "$status"
  fi
}

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
      echo "python -u -m tadabur.catch_loss_audit draw$DRAW_ARG --tranche '$REPO/$TRANCHE' && cp '$REPO/$AUDIT/worklist.jsonl' '$REPO/$AUDIT/key.jsonl' '$REPO/$AUDIT/summary.json' '$ROOT/'" ;;
    ui)
      echo "python -u -m tadabur.tashkeel_audit_ui --catch-loss '$REPO/$AUDIT/worklist.jsonl' --clip-cache '$CLIPS' --verdicts '$VERDICTS' --review '$REVIEW' --port $PORT --host 0.0.0.0" ;;
    backup)
      echo "mkdir -p '$BACKUP'; while true; do
        cp '$VERDICTS' '$BACKUP/verdicts.'\$(date +%s)'.jsonl' 2>/dev/null || true
        cp '$REVIEW' '$BACKUP/review.'\$(date +%s)'.jsonl' 2>/dev/null || true
        ls -1t '$BACKUP'/verdicts.*.jsonl 2>/dev/null | tail -n +101 | xargs -r rm -f
        ls -1t '$BACKUP'/review.*.jsonl 2>/dev/null | tail -n +101 | xargs -r rm -f
        sleep 60
      done" ;;
  esac
}

runner_body() {
  # `draw` and `ui` run from the staged clone; the backup loop only copies within the scratch
  # root, so it neither needs nor may re-clone (see `launch`).
  [ "$job" = backup ] || echo "cd '$REPO/tools' || exit 1"
  command_for
}

write_runner() {
  remote "cat > '$ROOT/run-$NAME.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
source '$CONDA'
conda activate $ENV_NAME
$(runner_body)
code=\$?
echo \"\$code\" > '$LOGS/$NAME.exit'
exit \$code
SH
chmod +x '$ROOT/run-$NAME.sh'"
}

# The page's first queue item, fetched the way the browser does it: `/api/sites`, then that
# item's clip. A queue that answers but cannot serve a clip is the failure this page has (a
# cache miss or a hash mismatch), so the probe reads the clip too and a failed launch says so.
audio_probe() {
  remote "python3 - <<'PY'
import json, time, urllib.request
base = 'http://127.0.0.1:$PORT'
for _ in range(20):
    try:
        sites = json.load(urllib.request.urlopen(base + '/api/sites', timeout=2))['sites']
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('the page never answered /api/sites')
clip = urllib.request.urlopen(
    base + '/api/audio?key=' + sites[0]['key'] + '&whole=0', timeout=10).read()
print(len(sites), 'items;', sites[0]['key'], 'audio', len(clip), 'bytes')
PY"
}

# The port's listener, as `ss` names it; empty when nothing is bound. A `ui` launch refuses an
# occupied port instead of reporting whatever already answers there as the new sitting.
listener_pid() {
  remote "ss -ltnp 2>/dev/null | grep -F ':$PORT ' | head -1"
}

# Whether the job just started owns the listener: its runner's child is the pid `ss` reports.
# The audio probe alone cannot tell this job from an earlier sitting that still answers.
serves_from() {  # runner pid
  remote "ss -ltnp 2>/dev/null | grep -F ':$PORT ' \
    | grep -q \"pid=\$(pgrep -P '$1' | head -1)\""
}

# A copyable check that selects this launch's root and port: a one-command environment
# assignment does not survive into the next command, so repeat every setting given explicitly.
launch_hint() {
  local hint="scripts/issue-225-audit.sh $job check"
  if [ -n "${SPEC_GIVEN:-}" ]; then hint="SPEC=$SPEC $hint"; fi
  if [ -n "${ISSUE_ROOT:-}" ]; then hint="ISSUE_ROOT='$ISSUE_ROOT' $hint"; fi
  if [ -n "${PORT_GIVEN:-}" ]; then hint="PORT=$PORT $hint"; fi
  if [ -n "${CLIPS_GIVEN:-}" ]; then hint="CLIPS='$CLIPS' $hint"; fi
  if [ -n "${HOST_GIVEN:-}" ]; then hint="HOST='$HOST' $hint"; fi
  echo "$hint"
}

launch() {
  # Relaunching `draw` or `ui` re-clones the branch under the scratch root, and the page's html
  # and js are read from disk, so a launch mid-sitting leaves a serving page unable to load its
  # assets; the backup loop stays safe because it never stages.
  local pid probe= status=0
  if [ "$job" = ui ]; then
    pid=$(listener_pid)
    if [ -n "$pid" ]; then
      echo "port $PORT on $HOST is already bound; nothing was staged or started:" >&2
      echo "  $pid" >&2
      return 1
    fi
  fi
  if [ "$job" = backup ]; then
    remote "mkdir -p '$ROOT' '$LOGS'"
  else
    stage_repo
  fi
  write_runner
  remote "rm -f '$LOGS/$NAME.exit' '$LOGS/$NAME.pid'
          setsid bash '$ROOT/run-$NAME.sh' </dev/null >'$LOGS/$NAME.log' 2>&1 &
          echo \$! > '$LOGS/$NAME.pid'"
  pid=$(remote "cat '$LOGS/$NAME.pid'")
  if [ "$job" = ui ]; then
    probe=$(audio_probe) || status=$?
    if [ "$status" -eq 0 ] && ! serves_from "$pid"; then
      echo "port $PORT on $HOST is served by a process other than the job just started" >&2
      status=1
    fi
    if [ "$status" -ne 0 ]; then
      echo "the ui did not come up: $HOST:$LOGS/$NAME.log" >&2
      if remote "test -f '$LOGS/$NAME.exit'"; then
        echo "  exit $(remote "cat '$LOGS/$NAME.exit'")" >&2
      fi
      return "$status"
    fi
  fi
  echo "job started on $HOST"
  echo "  pid:  $pid"
  echo "  log:  $HOST:$LOGS/$NAME.log"
  echo "  exit: $HOST:$LOGS/$NAME.exit (0 = clean; absent while running)"
  if [ "$job" = ui ]; then
    echo "  url:  http://${HOST#*@}:$PORT/"
    echo "  stop: kill -- -$pid   # the job's process group (runner and server); ask the owner first"
    echo "  audio: $probe"
  fi
  echo "  check with: $(launch_hint)"
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
      fetch_file "$VERDICTS" "$destination"
      fetch_file "$REVIEW" "$destination" ;;
    backup)
      fetch_tree "$BACKUP" "$destination" ;;
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

#!/usr/bin/env bash
# issue-311-audit.sh — the S5 follow-along listening UI on cuda-dev (#311).
#
# Mac side, from the worktree root, with the draw, its snips and its WAVs in place:
#   scripts/issue-311-audit.sh wavs push               # the 498 drawn clips' WAVs to the box
#   scripts/issue-311-audit.sh ui launch               # serve the pilot sitting on :8000
#   WITH_MAIN=1 scripts/issue-311-audit.sh ui launch   # ... and the main sitting after it
#   scripts/issue-311-audit.sh ui check                # running? its log and exit file
#   scripts/issue-311-audit.sh ui fetch DIR            # pull verdicts.jsonl, then commit it
#   scripts/issue-311-audit.sh backup launch           # copy the verdicts file every minute
#   scripts/issue-311-audit.sh backup fetch DIR        # pull the timestamped copies
#
# The pilot sitting is served alone: the pre-registration may resize the main sitting once the
# pilot is read (#311 PR3 Q2), so the main one starts only behind `WITH_MAIN=1`. Both come from
# `s5_draw/pilot.jsonl` and `s5_draw/main.jsonl` in the committed order, and the page is never
# told which sitting an item came from.
#
# The owner's verdicts live in the scratch root, not in the clone: a relaunch re-clones and would
# drop them. `ui fetch` pulls `verdicts.jsonl`; commit it at
# tools/tadabur/followalong_audit/s5_draw/verdicts.jsonl (the path the scorer reads, and the one
# `.gitignore` exempts). The file on the box is the only copy between sittings — fetch it.
#
# The audio is the Mac's own WAVs (`<audio_sha256>.wav`, the files the App arm graded);
# `wavs push` sends exactly the drawn clips' 498, about 140 MB. The UI verifies every one against
# the committed `app_arm/audio_manifest.jsonl.gz` before it serves a word, so a truncated or
# substituted clip never reaches the owner's ear.
#
# `ui` and `backup` launch stage the current branch as a bundle on the box; the backup loop only
# copies within the scratch root, so its launch never re-clones. Every job writes one log and its
# exit file last, so `check` never reads a log tail. Jobs run under setsid with stdin closed, so
# they survive the ssh session. Restarting a running UI needs the owner's OK.
set -euo pipefail

HOST_GIVEN=${HOST:+1}
HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/s5_audit}
PORT_GIVEN=${PORT:+1}
PORT=${PORT:-8000}
WITH_MAIN=${WITH_MAIN:-}
WAV_DIR=${WAV_DIR:-$HOME/hifzguide-scratch/fa-apparm/wav}
LOGS=$ROOT/logs
REPO=$ROOT/repo
# The owner's answers: one row per served id, the schema s5_score reads.
VERDICTS=$ROOT/verdicts.jsonl
BACKUP=$ROOT/backup
AUDIT=tools/tadabur/followalong_audit
WORKLIST=$AUDIT/s5_draw/worklist.jsonl

job=${1:-}
action=${2:-}
case "$job" in
  ui|backup|wavs) NAME=$job ;;
  *) echo "usage: $0 ui|backup [launch|check|fetch DIR] | wavs push [WAV_DIR]" >&2; exit 2 ;;
esac

remote() { ssh -n "$HOST" "$@"; }

# A fetch treats the remote `test`'s 1 as "absent" (the verdicts file before the first answer,
# the backup directory before the loop runs) and every other nonzero as the transport failure it
# is: a failed ssh must fail the fetch, never print it as absent and succeed.
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
    ui)
      local main=""
      if [ -n "$WITH_MAIN" ]; then main=" --with-main"; fi
      echo "python -u -m tadabur.followalong_audit.s5_ui --wav-dir '$ROOT/wav' --verdicts '$VERDICTS' --port $PORT --host 0.0.0.0$main" ;;
    backup)
      echo "mkdir -p '$BACKUP'; while true; do
        cp '$VERDICTS' '$BACKUP/verdicts.'\$(date +%s)'.jsonl' 2>/dev/null || true
        ls -1t '$BACKUP'/verdicts.*.jsonl 2>/dev/null | tail -n +101 | xargs -r rm -f
        sleep 60
      done" ;;
  esac
}

runner_body() {
  # `ui` runs from the staged clone; the backup loop only copies within the scratch root, so it
  # neither needs nor may re-clone (see `launch`).
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

# The page's first item, fetched the way the browser does it: `/api/sites`, then that item's
# audio. A page that answers but cannot serve a clip is this UI's failure mode (a missing WAV or
# a manifest mismatch), so the probe reads the clip too and a failed launch says so.
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
# The audio probe alone cannot tell this job from an earlier sitting that still answers, so an
# absent child is a mismatch (a dead runner must never match), and the pid matches whole — the
# field's delimiter included, or `pid=424` would claim a listener `pid=4242`.
serves_from() {  # runner pid
  local server holder
  server=$(remote "pgrep -P '$1' | head -1")
  case "$server" in
    ''|*[!0-9]*) return 1 ;;
  esac
  holder=$(remote "ss -ltnp 2>/dev/null | grep -F ':$PORT ' | head -1")
  case "$holder" in
    *"pid=$server,"*|*"pid=$server)"*) return 0 ;;
  esac
  return 1
}

launch_hint() {
  local hint="scripts/issue-311-audit.sh $job check"
  if [ -n "${ISSUE_ROOT:-}" ]; then hint="ISSUE_ROOT='$ISSUE_ROOT' $hint"; fi
  if [ -n "${PORT_GIVEN:-}" ]; then hint="PORT=$PORT $hint"; fi
  if [ -n "$WITH_MAIN" ]; then hint="WITH_MAIN=$WITH_MAIN $hint"; fi
  if [ -n "${HOST_GIVEN:-}" ]; then hint="HOST='$HOST' $hint"; fi
  echo "$hint"
}

launch() {
  # Relaunching `ui` re-clones the branch under the scratch root, and the page is read from disk,
  # so a launch mid-sitting leaves a serving page unable to load its page; the backup loop stays
  # safe because it never stages.
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
    if remote "test -f '$LOGS/$NAME.exit'"; then
      echo "the job exited $(remote "cat '$LOGS/$NAME.exit'") before serving" >&2
      return 1
    fi
    probe=$(audio_probe) || status=$?
    if [ "$status" -eq 0 ] && ! serves_from "$pid"; then
      echo "port $PORT on $HOST is not served by the job just started (pid $pid)" >&2
      status=1
    fi
    if [ "$status" -ne 0 ]; then
      echo "the ui did not come up: $HOST:$LOGS/$NAME.log" >&2
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
    ui) fetch_file "$VERDICTS" "$destination" ;;
    backup) fetch_tree "$BACKUP" "$destination" ;;
  esac
  echo "fetched $job into $destination"
}

# The drawn clips' WAVs, one name per line, sorted: the worklist's 498 clips over the Mac's
# 5,742. The names are the manifest's own keys, so nothing is sent that the UI would refuse.
wav_files() {
  python3 - "$REPO_DIR/$WORKLIST" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
        if line.strip()]
print("\n".join(sorted({row["audio_sha256"] + ".wav" for row in rows})))
PY
}

push_wavs() {
  local source=${1:-$WAV_DIR} list
  list=$(mktemp)
  trap 'rm -f "$list"' RETURN
  wav_files > "$list"
  remote "mkdir -p '$ROOT/wav'"
  rsync -a --files-from="$list" "$source/" "$HOST:$ROOT/wav/"
  echo "sent $(wc -l < "$list") wavs from $source to $HOST:$ROOT/wav"
}

case "$job" in
  wavs)
    if [ "${2:-}" != push ]; then
      echo "usage: $0 wavs push [WAV_DIR]" >&2
      exit 2
    fi
    push_wavs "${3:-$WAV_DIR}"
    ;;
  ui|backup)
    case "$action" in
      launch) launch ;;
      check) check ;;
      fetch) fetch "${3:-}" ;;
      "") launch ;;
      *) echo "usage: $0 $job [launch|check|fetch DIR]" >&2; exit 2 ;;
    esac
    ;;
esac

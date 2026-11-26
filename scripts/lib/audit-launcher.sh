#!/usr/bin/env bash
# audit-launcher.sh — the operational half of a cuda-dev listening-audit launcher.
#
# Sourced by scripts/issue-225-audit.sh (the catch-loss audit) and scripts/issue-311-audit.sh
# (the follow-along S5 audit). A round script sets its own config, defines three hooks, and
# dispatches; everything the two rounds share lives here once, because it is the subtle half: the
# bundle/clone/rev-parse staging, the setsid runner with its one log and exit file, the page
# probe, the port's owner check, and a fetch that tells "absent" from "the box is unreachable".
#
# What a round script provides before it uses anything here:
#   SCRIPT, job, NAME, HOST, ROOT, LOGS, REPO, REPO_DIR, CONDA, ENV_NAME, PORT
#   command_for()   what the job runs on the box; `$REPO` is the staged clone's root
#   fetch_into()    copy this job's artifacts from the box into a local directory ($1)
#   launch_hint()   the copyable `check` that selects this launch's settings
#   PORT_FALLBACK   optional: the port to retry once when PORT was not given and its port is
#                   bound; unset refuses the launch instead (the safe half)
#
# The job that serves a page is named `ui`. Any other job runs from the staged clone except
# `backup`, which only copies within the scratch root and so never re-clones.

remote() { ssh -n "$HOST" "$@"; }

# A fetch treats the remote `test`'s 1 as "absent" (the review list before the owner queues
# anything, the verdicts file before the first answer, the backup directory before it runs) and
# every other nonzero as the transport failure it is: a failed ssh must fail the fetch, never
# print it as absent and succeed.
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

# The current branch as a bundle on the box, cloned under the scratch root and checked against
# the local sha: a launch must serve the branch the owner is looking at, nothing older.
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

runner_body() {
  # The page and the draw run from the staged clone; the backup loop only copies within the
  # scratch root, so it neither needs nor may re-clone (see `launch`).
  [ "$job" = backup ] || echo "cd '$REPO/tools' || exit 1"
  command_for
}

write_runner() {
  remote "cat > '$ROOT/run-$NAME.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
$(runner_hold)
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
# item's clip. A page that answers but cannot serve a clip is the failure mode both rounds have
# (a cache miss, a missing WAV, a hash mismatch), so the probe reads the clip too and a failed
# launch says so.
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

# The port's listener, as `ss` names it; empty when nothing is bound.
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

# One writer per scratch root: the answers live in $ROOT/verdicts.jsonl, and a second ui would
# hold its own snapshot of that file and overwrite the first's saves, while staging would replace
# the clone the first one serves from. The reservation is `$ROOT/ui.lock`, and its creation is
# `mkdir` — one atomic syscall at the box, taken before anything is staged, so of two launches of
# one root exactly one gets in, whatever ports they were given. The winner holds it for the ui's
# lifetime: the runner records its pid there and removes the directory in its EXIT trap, and the
# launcher removes it when a start fails (`drop_reservation`). The backup job takes none: it only
# copies.
reserve_root_ui() {
  local pid args
  RESERVATION="$$-$RANDOM"
  if remote "mkdir -p '$ROOT' '$LOGS' && mkdir '$ROOT/ui.lock' 2>/dev/null \
      && printf '%s' '$RESERVATION' > '$ROOT/ui.lock/token'"; then
    return 0
  fi
  pid=$(remote "cat '$ROOT/ui.lock/pid' 2>/dev/null" || true)
  case "$pid" in
    ''|*[!0-9]*) pid= ;;
  esac
  if [ -n "$pid" ]; then
    args=$(remote "ps -p '$pid' -o args= 2>/dev/null" || true)
    case "$args" in
      *"$ROOT/run-ui.sh"*)
        echo "$ROOT on $HOST already serves a ui (pid $pid); nothing was staged or started:" >&2
        echo "  stop it first (ask the owner): kill -- -$pid" >&2
        return 1 ;;
    esac
    # A recorded runner that is gone holds nothing: move that reservation aside — only while it
    # still records the pid just read, so a reservation another launch has made is never taken —
    # and try again.
    remote "test \"\$(cat '$ROOT/ui.lock/pid' 2>/dev/null)\" = '$pid' \
      && mv '$ROOT/ui.lock' '$ROOT/ui.lock.stale.$$'" || true
  else
    # No pid yet: the holder is a launcher still staging. Only an old one is stale.
    remote "test -z \"\$(cat '$ROOT/ui.lock/pid' 2>/dev/null)\" \
      && find '$ROOT/ui.lock' -maxdepth 0 -mmin +15 | grep -q . \
      && mv '$ROOT/ui.lock' '$ROOT/ui.lock.stale.$$'" || true
  fi
  remote "rm -rf '$ROOT/ui.lock.stale.$$'"
  if remote "mkdir '$ROOT/ui.lock' 2>/dev/null \
      && printf '%s' '$RESERVATION' > '$ROOT/ui.lock/token'"; then
    return 0
  fi
  echo "$ROOT on $HOST is already being served or launched; nothing was staged or started:" >&2
  echo "  if nothing serves it, remove $ROOT/ui.lock and retry" >&2
  return 1
}

# A failed ui launch hands its root back — if the reservation is still the one it made and no
# runner has claimed it: a runner that did start releases it on the way out, and a reservation
# another launch took over is never removed by the one that lost it.
drop_reservation() {
  [ "$job" = ui ] || return 0
  remote "test \"\$(cat '$ROOT/ui.lock/token' 2>/dev/null)\" = '${RESERVATION:-}' \
    && test -z \"\$(cat '$ROOT/ui.lock/pid' 2>/dev/null)\" && rm -rf '$ROOT/ui.lock'"
  return 0
}

# What the runner adds to take the reservation over and release it, written into its script. Both
# steps check the launcher's token, so a runner whose reservation was taken over while it staged
# refuses to serve instead of claiming another launch's, and it never removes that one.
runner_hold() {
  [ "$job" = ui ] || return 0
  printf 'claim_reservation() { [ "$(cat "%s/token" 2>/dev/null)" = "%s" ] || { echo "the root reservation was taken over; not serving" >&2; exit 1; }; echo $$ > "%s/pid"; }\n' \
    "$ROOT/ui.lock" "${RESERVATION:-}" "$ROOT/ui.lock"
  printf 'release_reservation() { [ "$(cat "%s/token" 2>/dev/null)" = "%s" ] && rm -rf "%s"; }\n' \
    "$ROOT/ui.lock" "${RESERVATION:-}" "$ROOT/ui.lock"
  printf 'trap release_reservation EXIT\nclaim_reservation\n'
}

# Whether the page's port is free. An occupied port refuses the launch instead of reporting
# whatever already answers there as the new sitting; a round that names PORT_FALLBACK retries
# once on it, and only when PORT was not given explicitly.
refuse_occupied_port() {
  local pid
  pid=$(listener_pid)
  if [ -n "$pid" ] && [ -n "${PORT_FALLBACK:-}" ] && [ -z "${PORT_GIVEN:-}" ]; then
    echo "port $PORT on $HOST is already bound; trying $PORT_FALLBACK" >&2
    PORT=$PORT_FALLBACK
    # The port is now a choice, not the default, so the check hint must carry it.
    PORT_GIVEN=1
    pid=$(listener_pid)
  fi
  [ -n "$pid" ] || return 0
  echo "port $PORT on $HOST is already bound; nothing was staged or started:" >&2
  echo "  $pid" >&2
  return 1
}

# Relaunching `ui` re-clones the branch under the scratch root, and the page is read from disk,
# so a launch mid-sitting leaves a serving page unable to load its assets; the backup loop stays
# safe because it never stages. Restarting a running page needs the owner's OK.
launch() {
  local pid probe= status=0
  if [ "$job" = ui ]; then
    reserve_root_ui || return 1
    if ! refuse_occupied_port; then
      drop_reservation
      return 1
    fi
  fi
  if [ "$job" = backup ]; then
    remote "mkdir -p '$ROOT' '$LOGS'"
  elif ! stage_repo; then
    drop_reservation
    return 1
  fi
  if ! write_runner; then
    drop_reservation
    return 1
  fi
  if ! remote "rm -f '$LOGS/$NAME.exit' '$LOGS/$NAME.pid'
          setsid bash '$ROOT/run-$NAME.sh' </dev/null >'$LOGS/$NAME.log' 2>&1 &
          echo \$! > '$LOGS/$NAME.pid'"; then
    drop_reservation
    return 1
  fi
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

# The job's log and exit file, read whole — never a log tail, which would mix earlier runs.
check() {
  if remote "test -f '$LOGS/$NAME.exit'"; then
    echo "exit $(remote "cat '$LOGS/$NAME.exit'")"
  else
    echo "running (pid $(remote "cat '$LOGS/$NAME.pid' 2>/dev/null || echo unknown"))"
    remote "ps -p \"\$(cat '$LOGS/$NAME.pid')\" -o pid=,etime=,stat= 2>/dev/null || true"
  fi
}

fetch() {  # destination directory
  local destination=${1:?fetch needs a destination directory}
  mkdir -p "$destination"
  fetch_into "$destination"
  echo "fetched $job into $destination"
}

# `NAME launch|check|fetch DIR`, launch by default; the verbs every job has.
dispatch_action() {
  local action=${1:-}
  case "$action" in
    launch) launch ;;
    check) check ;;
    fetch) fetch "${2:-}" ;;
    "") launch ;;
    *) echo "usage: $SCRIPT $job [launch|check|fetch DIR]" >&2; exit 2 ;;
  esac
}

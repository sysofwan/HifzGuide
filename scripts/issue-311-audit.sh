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
# The operational half — bundle/clone staging, the detached runner, the page probe, the port's
# owner check, the fetch — is scripts/lib/audit-launcher.sh, shared with the catch-loss launcher.
# `ui` and `backup` launch stage the current branch as a bundle on the box; the backup loop only
# copies within the scratch root, so its launch never re-clones. Every job writes one log and its
# exit file last, so `check` never reads a log tail. Jobs run under setsid with stdin closed, so
# they survive the ssh session. Restarting a running UI needs the owner's OK.
set -euo pipefail

SCRIPT=issue-311-audit.sh
HOST_GIVEN=${HOST:+1}
HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/s5_audit}
PORT_GIVEN=${PORT:+1}
PORT=${PORT:-8000}
# The owner reads on 10.0.1.195:8000; when something else already answers there, the launch
# retries once here rather than making the human retry. `PORT=` picks a port, no fallback.
PORT_FALLBACK=8002
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

. "$(dirname "${BASH_SOURCE[0]}")/lib/audit-launcher.sh"

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

fetch_into() {  # destination directory: this job's artifacts
  case "$job" in
    ui) fetch_file "$VERDICTS" "$1" ;;
    backup) fetch_tree "$BACKUP" "$1" ;;
  esac
}

# A copyable check that selects this launch's root and port: a one-command environment
# assignment does not survive into the next command, so repeat every setting given explicitly.
launch_hint() {
  local hint="scripts/$SCRIPT $job check"
  if [ -n "${ISSUE_ROOT:-}" ]; then hint="ISSUE_ROOT='$ISSUE_ROOT' $hint"; fi
  if [ -n "${PORT_GIVEN:-}" ]; then hint="PORT=$PORT $hint"; fi
  if [ -n "$WITH_MAIN" ]; then hint="WITH_MAIN=$WITH_MAIN $hint"; fi
  if [ -n "${HOST_GIVEN:-}" ]; then hint="HOST='$HOST' $hint"; fi
  echo "$hint"
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
    dispatch_action "$action" "${3:-}"
    ;;
esac

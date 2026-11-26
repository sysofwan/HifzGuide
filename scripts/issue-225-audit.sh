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
# A `snips.jsonl` committed beside it (python -m tadabur.word_snips build) is passed as
# `--snips`, so each item plays its excerpt; without one the UI plays whole clips as before.
# Restarting a running UI needs the owner's OK.
#
# `draw` and `ui` stage the current branch as a bundle on the box; the backup loop only copies
# within the scratch root, so its launch never re-clones. Every job writes one log and writes its
# exit file last, so `check` never reads a log tail. Jobs run under setsid with stdin closed,
# so they survive the ssh session. The UI reads the look clip cache #212 filled; no audio is
# copied off the box. The operational half — staging, runner, probe, port check, fetch — is
# scripts/lib/audit-launcher.sh, shared with the follow-along launcher.
set -euo pipefail

SCRIPT=issue-225-audit.sh
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

. "$(dirname "${BASH_SOURCE[0]}")/lib/audit-launcher.sh"

command_for() {
  case "$job" in
    draw)
      echo "python -u -m tadabur.catch_loss_audit draw$DRAW_ARG --tranche '$REPO/$TRANCHE' && cp '$REPO/$AUDIT/worklist.jsonl' '$REPO/$AUDIT/key.jsonl' '$REPO/$AUDIT/summary.json' '$ROOT/'" ;;
    ui)
      snips=
      if [ -f "$REPO_DIR/$AUDIT/snips.jsonl" ]; then
        snips=" --snips '$REPO/$AUDIT/snips.jsonl'"
      fi
      echo "python -u -m tadabur.tashkeel_audit_ui --catch-loss '$REPO/$AUDIT/worklist.jsonl' --clip-cache '$CLIPS' --verdicts '$VERDICTS' --review '$REVIEW'$snips --port $PORT --host 0.0.0.0" ;;
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

# A copyable check that selects this launch's root and port: a one-command environment
# assignment does not survive into the next command, so repeat every setting given explicitly.
launch_hint() {
  local hint="scripts/$SCRIPT $job check"
  if [ -n "${SPEC_GIVEN:-}" ]; then hint="SPEC=$SPEC $hint"; fi
  if [ -n "${ISSUE_ROOT:-}" ]; then hint="ISSUE_ROOT='$ISSUE_ROOT' $hint"; fi
  if [ -n "${PORT_GIVEN:-}" ]; then hint="PORT=$PORT $hint"; fi
  if [ -n "${CLIPS_GIVEN:-}" ]; then hint="CLIPS='$CLIPS' $hint"; fi
  if [ -n "${HOST_GIVEN:-}" ]; then hint="HOST='$HOST' $hint"; fi
  echo "$hint"
}

fetch_into() {  # destination directory: this job's artifacts
  case "$job" in
    draw)
      rsync -a --include 'worklist.jsonl' --include 'key.jsonl' \
        --include 'summary.json' --exclude '*' "$HOST:$ROOT/" "$1" ;;
    ui)
      fetch_file "$VERDICTS" "$1"
      fetch_file "$REVIEW" "$1" ;;
    backup)
      fetch_tree "$BACKUP" "$1" ;;
  esac
}

dispatch_action "$action" "${3:-}"

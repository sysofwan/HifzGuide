#!/usr/bin/env bash
# mini-flags-decode.sh — the published mini's reporting decode of round 3's confirm tranches.
#
# The owner's measurement (2026-11-18): decode ``sysofwan/hifzguide-muaalem-mini`` at its pinned
# Hub revision on exactly the gating rows round 3 graded (confirm.3 + confirm.4), with the
# round's own protocol, and nothing else. It is a reporting decode: no gate reads it and the
# round's verdict, readout and pre-registration are untouched. :mod:`training.mini_flags` owns
# the pinned identity, the loader proof and the per-shard decode; this script stages the
# repository on the box and runs it detached under the GPU lock.
#
# Mac side, from the worktree root:
#   scripts/mini-flags-decode.sh plan      # the committed job: tranches, shards, identity
#   scripts/mini-flags-decode.sh launch    # stage, prove the loader, start detached
#   scripts/mini-flags-decode.sh check     # run.done / run.exit, then the log tails
#   scripts/mini-flags-decode.sh fetch     # pull the pooled decode and the verify record
#
# The tlog mining run stays paused: this holds /root/scratch/gpu.lock and decodes on the same
# box, and nothing here resumes that run.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${MINI_FLAGS_ROOT:-/root/scratch/mini-flags}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
#: Read-only: round 3's confirm run already cached these tranches' parquet shards.
CACHE=${MINI_FLAGS_CACHE:-/root/scratch/issue-256/confirm/cache}
PARALLEL=${PARALLEL:-4}
BUDGET=${BUDGET:-21600}
ROLES=(confirm.3 confirm.4)
#: Where ``fetch`` puts the artifacts, beside round 3's own decodes.
DECODES=${DECODES:-tools/tadabur/mini_flags_readout/decodes}


die() { echo "$*" >&2; exit 1; }

usage() {
  cat >&2 <<'EOF'
usage: mini-flags-decode.sh <plan|launch|check|fetch>
       (on the box: run)
EOF
  exit 2
}

# --- the Mac side -------------------------------------------------------------------------------


plan() {
  (cd "$(git rev-parse --show-toplevel)/tools" && python3 -m training.mini_flags plan)
}

bundle_ref() {  # <repo_root>
  # A branch, never a raw SHA: ``git bundle create`` with a SHA records no ref.
  local branch
  branch=$(git -C "$1" rev-parse --abbrev-ref HEAD)
  if [ "$branch" = HEAD ]; then echo HEAD; else echo "$branch"; fi
}

stage_repo() {
  local repo_root branch sha
  repo_root=$(git rev-parse --show-toplevel)
  branch=$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$repo_root" rev-parse HEAD)
  ssh "$HOST" "mkdir -p '$ROOT/logs'"
  ssh "$HOST" "rm -rf '$ROOT/repo' '$ROOT/repo.bundle'"
  git -C "$repo_root" bundle create - "$(bundle_ref "$repo_root")" \
    | ssh "$HOST" "cat > '$ROOT/repo.bundle'"
  if [ "$branch" = HEAD ]; then
    ssh "$HOST" "git clone -q '$ROOT/repo.bundle' '$ROOT/repo'"
  else
    ssh "$HOST" "git clone -q -b '$branch' '$ROOT/repo.bundle' '$ROOT/repo'"
  fi
  ssh "$HOST" "rm -f '$ROOT/repo.bundle' && \
    test \"\$(git -C '$ROOT/repo' rev-parse HEAD)\" = '$sha'" || {
    echo "the box's clone is not $sha" >&2
    return 1
  }
}

start_run() {
  ssh "$HOST" "MINI_FLAGS_ROOT='$ROOT' MINI_FLAGS_CACHE='$CACHE' BUDGET=$BUDGET \
    PARALLEL=$PARALLEL nohup flock /root/scratch/gpu.lock \
    '$ROOT/repo/scripts/mini-flags-decode.sh' run >> '$ROOT/logs/run.out' 2>&1 & echo \$!"
}

launch() {
  stage_repo
  start_run
}

check() {
  local status
  status=$(ssh "$HOST" "if [ ! -f '$ROOT/logs/run.done' ]; then echo pending; \
    elif [ ! -f '$ROOT/logs/run.exit' ]; then echo no-exit; \
    else cat '$ROOT/logs/run.exit' | tr -d '[:space:]'; fi")
  case "$status" in
    pending)
      echo "pending: $ROOT/logs/run.done is absent"
      ssh "$HOST" "tail -n 4 '$ROOT/logs/events' 2>/dev/null; \
        tail -n 3 '$ROOT/logs/run.out' 2>/dev/null" || true
      return 3
      ;;
    no-exit)
      echo "failed: $ROOT/logs/run.done exists without $ROOT/logs/run.exit"
      ssh "$HOST" "cat '$ROOT/logs/run.done' 2>/dev/null; \
        tail -n 20 '$ROOT/logs/run.out' 2>/dev/null" || true
      return 1
      ;;
    0)
      echo "done: exit 0"
      ssh "$HOST" "echo -n 'worker lines: '; wc -l < '$ROOT/logs/events'; \
        grep -v ' exit 0\$' '$ROOT/logs/events' || true"
      ;;
    *)
      echo "failed: exit $status"
      ssh "$HOST" "cat '$ROOT/logs/run.done' 2>/dev/null; \
        grep -v ' exit 0\$' '$ROOT/logs/events' 2>/dev/null || true; \
        tail -n 20 '$ROOT/logs/run.out' 2>/dev/null"
      return 1
      ;;
  esac
}

fetch() {
  local repo_root
  repo_root=$(git rev-parse --show-toplevel)
  ssh "$HOST" "test -f '$ROOT/logs/run.done' && \
    test \"\$(cat '$ROOT/logs/run.exit')\" = 0" \
    || die "the box has no complete run to fetch"
  ssh "$HOST" "set -e
test -f '$ROOT/out/tlog.confirm.mini.jsonl.gz'
test -f '$ROOT/out/meta.confirm.json'
test -f '$ROOT/verify.json'"
  mkdir -p "$repo_root/$DECODES"
  rsync -a "$HOST:$ROOT/out/" "$repo_root/$DECODES/"
  rsync -a "$HOST:$ROOT/verify.json" "$repo_root/$DECODES/"
  echo "mini decodes and its verification record in $repo_root/$DECODES"
}

# --- the box side -------------------------------------------------------------------------------


activate() {
  [ -r "$CONDA" ] || { echo "no conda at $CONDA" >&2; return 1; }
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
}

#: One decode process over its share of a role's gating shards; its exit is an event.
run_worker() {  # <role> <index>
  local role=$1 index=$2 rc=0
  python -m training.mini_flags decode --role "$role" \
    --parts-dir "$ROOT/parts" --cache-dir "$CACHE" \
    --shard-index "$index" --shard-count "$PARALLEL" \
    > "$ROOT/logs/$role.$index.log" 2>&1 || rc=$?
  printf '%s %s worker %s exit %s\n' "$(date -u +%FT%TZ)" "$role" "$index" "$rc" \
    >> "$ROOT/logs/events"
  return "$rc"
}

decode_role() {  # <role>
  local role=$1 index rc status=0 pids=()
  for ((index = 0; index < PARALLEL; index++)); do
    run_worker "$role" "$index" &
    pids+=("$!")
  done
  for index in "${!pids[@]}"; do
    rc=0
    wait "${pids[$index]}" || rc=$?
    [ "$rc" -eq 0 ] || status=$rc
  done
  [ "$status" -eq 0 ] || { echo "$role: a worker failed (status $status)" >&2; return "$status"; }
}

verify_step() {
  # The pinned artifact and the vendored loader, proven before any row is decoded.
  python -m training.mini_flags verify --record "$ROOT/verify.json" >> "$ROOT/logs/verify.log" 2>&1
}

merge_step() {
  # The pooled merge refuses a missing or repeated row: a partial run never looks complete.
  python -m training.mini_flags merge --parts-dir "$ROOT/parts" --out-dir "$ROOT/out" \
    >> "$ROOT/logs/merge.log" 2>&1
}

record_run_result() {  # <status>
  # ``check`` keys off done and then reads exit: write exit first.
  local status=$1 code=$1 done
  mkdir -p "$ROOT/logs"
  if [ "$status" -eq 124 ]; then
    code=2
    done=$(printf '%s budget-exceeded budget=%ss; per-item output kept' \
      "$(date -u +%FT%TZ)" "$BUDGET")
  elif [ "$status" -eq 137 ]; then
    done=$(printf '%s killed (status 137, SIGKILL): timeout escalation or external kill/OOM; \
per-item output kept' "$(date -u +%FT%TZ)")
  else
    done=$(printf '%s exit %s' "$(date -u +%FT%TZ)" "$status")
  fi
  printf '%s\n' "$code" > "$ROOT/logs/run.exit"
  printf '%s\n' "$done" > "$ROOT/logs/run.done"
  return "$code"
}

run() {
  trap 'exit 143' TERM
  trap 'exit 130' INT
  trap 'record_run_result "$?"' EXIT
  mkdir -p "$ROOT"/{parts,out,logs}
  rm -f "$ROOT/logs/run.done" "$ROOT/logs/run.exit"
  : > "$ROOT/logs/events"
  cd "$ROOT/repo/tools"
  activate
  export OMP_NUM_THREADS=1
  verify_step
  local role
  for role in "${ROLES[@]}"; do
    decode_role "$role"
  done
  merge_step
  trap - EXIT TERM INT
  record_run_result 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    plan|launch|check|fetch) command=$1; shift; "$command" ;;
    run) shift; run ;;
    *) usage ;;
  esac
fi

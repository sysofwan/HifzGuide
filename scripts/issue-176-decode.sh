#!/usr/bin/env bash
# issue-176-decode.sh — C95's baseline decodes on the QUD other half and Ship A's held-out
# verified sites (#176). The pass is one flock-held run: ten QUD shards in parallel (each one
# mushaf parquet, downloaded into --cache-dir and deleted once read), then the held-out clips.
#
# Mac side, from the worktree root, once the code is committed:
#   scripts/issue-176-decode.sh launch   # archive HEAD + stage the clips; start detached
#   scripts/issue-176-decode.sh check    # done-file check; 0 = both passes finished cleanly
#   scripts/issue-176-decode.sh fetch    # pull the decodes and C95's caches back
#
# Then, from tools/: merge the QUD shards into the round's committed decode, and read the
# baseline numbers through the harness:
#   python -m training.consonant_baseline merge --decodes-dir "${FETCH_DIR:-/tmp/issue-176-decodes}"
#   python -m training.consonant_baseline read
#
# On the box, launch runs `setup` then `run` under flock /root/scratch/gpu.lock. `run` appends
# one exit line per QUD shard to logs/events and writes, last, logs/baseline.exit and then
# logs/baseline.done; `check` is the readiness signal, never a log tail. A run stopped by the
# budget writes exit 2 and a `budget-exceeded` done-file.
#
# Budget (#176): the QUD half costs ~15.0k decode-seconds for one model at #95's measured
# 1.46 s/item (10,261 complement segments), so P=6 is ~42 wall minutes on the 12-core box.
# `BUDGET` (default two GPU-hours) is that cap, enforced from the lock's acquisition: the
# watchdog stops both passes' process group and keeps their resumable per-item output.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-176}
PARALLEL=${PARALLEL:-6}
#: The issue's cap, in seconds from the GPU lock's acquisition: both passes stop at expiry.
BUDGET=${BUDGET:-7200}
CLIP_DIRS=(/root/scratch/issue-83/stage/clips /root/scratch/issue-126/stage/clips)
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}

setup() {
  mkdir -p "$ROOT"/{decodes,hf_cache,logs,clips}
  # shellcheck source=/dev/null
  ( cd "$ROOT/repo/tools" && source "$CONDA" && conda activate hifzguide \
      && python -m training.consonant_baseline held-out --list-clips ) > "$ROOT/held_clips.txt"
  cd "$ROOT/clips"
  rm -f -- ./*.wav
  local name dir
  while IFS= read -r name; do
    for dir in "${CLIP_DIRS[@]}"; do
      if [ -f "$dir/$name" ]; then ln -sf "$dir/$name" "$name"; break; fi
    done
  done < "$ROOT/held_clips.txt"
  local wanted staged
  wanted=$(wc -l < "$ROOT/held_clips.txt")
  staged=$(find . -maxdepth 1 -name '*.wav' | wc -l)
  if [ "$staged" -ne "$wanted" ]; then
    echo "clips: $staged of $wanted staged from ${CLIP_DIRS[*]}" >&2
    return 1
  fi
  echo "clips: $staged of $wanted staged"
}

# Both decode passes, one job so the budget can stop their whole process group. Each item is
# appended to its shard's jsonl as it decodes, so a stopped pass loses only the item in flight;
# `logs/pass.exit` records each pass's status for the readiness file.
decode_pass() {
  local n qother=0 held=0
  n=$(python -m training.consonant_baseline qother --list-shards | wc -l)
  seq 0 $((n - 1)) | xargs -P "$PARALLEL" -I{} bash -c "
    python -m training.consonant_baseline qother --shard {} \
      --out-dir '$ROOT/decodes' --cache-dir '$ROOT/hf_cache' > '$ROOT/logs/qother.{}.log' 2>&1
    rc=\$?
    printf '%s qother shard %s exit %s\n' \"\$(date -u +%FT%TZ)\" {} \"\$rc\" >> '$ROOT/logs/events'
    exit \"\$rc\"
  " || qother=$?
  python -m training.consonant_baseline held-out --audio-dir "$ROOT/clips" \
    > "$ROOT/logs/held_out.log" 2>&1 || held=$?
  printf '%s qother=%s held_out=%s\n' "$(date -u +%FT%TZ)" "$qother" "$held" \
    > "$ROOT/logs/pass.exit"
  if [ "$qother" -ne 0 ] || [ "$held" -ne 0 ]; then return 1; fi
}

run() {
  mkdir -p "$ROOT/decodes" "$ROOT/logs"
  rm -f "$ROOT/logs/baseline.done" "$ROOT/logs/baseline.exit" "$ROOT/logs/budget.exceeded" \
        "$ROOT/logs/pass.exit"
  : > "$ROOT/logs/events"
  # Globals: the EXIT trap can fire after `run` returns, when its locals are gone.
  SAMPLER=""
  WATCHDOG=""
  # Any exit — failed conda, shard listing, xargs, held-out pass — leaves a readiness marker,
  # so `check` never hangs in `pending`; the sampler and watchdog die with the run.
  finish() {
    local code=$?
    if [ -n "$SAMPLER" ]; then
      kill -- "-$SAMPLER" 2>/dev/null || kill "$SAMPLER" 2>/dev/null || true
    fi
    if [ -n "$WATCHDOG" ]; then
      kill -- "-$WATCHDOG" 2>/dev/null || kill "$WATCHDOG" 2>/dev/null || true
    fi
    if [ ! -e "$ROOT/logs/baseline.done" ]; then
      printf '%s\n' "$code" > "$ROOT/logs/baseline.exit"
      printf '%s early-fail code=%s\n' "$(date -u +%FT%TZ)" "$code" \
        > "$ROOT/logs/baseline.done"
    fi
  }
  trap finish EXIT
  cd "$ROOT/repo/tools"
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
  export OMP_NUM_THREADS=1
  # Job control gives every background job its own process group, so the trap and the deadline
  # stop the sampler and the pass with every decode beneath them, never just a shell.
  set -m
  ( while true; do
      printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$(du -sb "$ROOT" | cut -f1)" \
        "$(df -B1 --output=avail /root/scratch | tail -1)" >> "$ROOT/logs/df.log"
      sleep 60
    done ) >/dev/null 2>&1 &
  SAMPLER=$!
  # The deadline starts here, under the launcher's flock: expiry stops every decode beneath the
  # pass and leaves this shell to write the readiness files; the output already on disk stays.
  decode_pass &
  local pass=$!
  ( sleep "$BUDGET"
    : > "$ROOT/logs/budget.exceeded"
    kill -- "-$pass" 2>/dev/null || true
  ) &
  WATCHDOG=$!
  set +m
  local status=0
  wait "$pass" || status=$?
  if [ -n "$WATCHDOG" ]; then
    kill -- "-$WATCHDOG" 2>/dev/null || kill "$WATCHDOG" 2>/dev/null || true
  fi
  WATCHDOG=""
  local code=0
  if [ -e "$ROOT/logs/budget.exceeded" ]; then
    code=2
    printf '%s budget-exceeded budget=%ss; decode group stopped, per-item output kept\n' \
      "$(date -u +%FT%TZ)" "$BUDGET" > "$ROOT/logs/baseline.done"
  else
    [ "$status" -eq 0 ] || code=1
    if [ ! -s "$ROOT/logs/pass.exit" ]; then
      printf '%s early-fail code=%s\n' "$(date -u +%FT%TZ)" "$status" \
        > "$ROOT/logs/pass.exit"
    fi
    cat "$ROOT/logs/pass.exit" > "$ROOT/logs/baseline.done"
  fi
  printf '%s\n' "$code" > "$ROOT/logs/baseline.exit"
  return "$code"
}

launch() {
  local repo_root
  repo_root=$(git rev-parse --show-toplevel)
  ssh "$HOST" "mkdir -p '$ROOT'"
  git -C "$repo_root" archive HEAD \
    | ssh "$HOST" "rm -rf '$ROOT/repo' && mkdir -p '$ROOT/repo' && tar -x -C '$ROOT/repo'"
  ssh "$HOST" "$ROOT/repo/scripts/issue-176-decode.sh setup"
  ssh "$HOST" "BUDGET=$BUDGET nohup flock /root/scratch/gpu.lock \
    $ROOT/repo/scripts/issue-176-decode.sh run \
    >> '$ROOT/logs/run.out' 2>&1 & echo 'started: '\$!"
}

check() {
  local status
  status=$(ssh "$HOST" "if [ -f '$ROOT/logs/baseline.done' ]; then \
    cat '$ROOT/logs/baseline.exit'; else echo pending; fi")
  case "$status" in
    pending)
      echo "pending: $ROOT/logs/baseline.done is absent"
      ssh "$HOST" "tail -n 3 '$ROOT/logs/events' 2>/dev/null; \
        tail -n 2 '$ROOT/logs/run.out' 2>/dev/null" || true
      return 3
      ;;
    0)
      echo "done: exit 0"
      ssh "$HOST" "echo -n 'shard lines: '; wc -l < '$ROOT/logs/events'; \
        grep -v ' exit 0\$' '$ROOT/logs/events' || true; \
        tail -n 1 '$ROOT/logs/held_out.log' 2>/dev/null || true"
      ;;
    2)
      echo "budget exceeded: the decode group was stopped"
      ssh "$HOST" "cat '$ROOT/logs/baseline.done' 2>/dev/null; \
        grep -v ' exit 0\$' '$ROOT/logs/events' || true"
      return 2
      ;;
    *)
      echo "failed: exit $status"
      ssh "$HOST" "cat '$ROOT/logs/baseline.done' 2>/dev/null; \
        grep -v ' exit 0\$' '$ROOT/logs/events' || true; \
        tail -n 20 '$ROOT/logs/run.out'"
      return 1
      ;;
  esac
}

fetch() {
  local repo_root dest
  repo_root=$(git rev-parse --show-toplevel)
  dest=${FETCH_DIR:-/tmp/issue-176-decodes}
  mkdir -p "$dest"
  rsync -a "$HOST:$ROOT/decodes/" "$dest/"
  mkdir -p "$repo_root/tools/tadabur/truth_baseline/decodes" \
           "$repo_root/tools/tadabur/truth_baseline/checks"
  rsync -a "$HOST:$ROOT/repo/tools/tadabur/truth_baseline/decodes/c95.json" \
    "$repo_root/tools/tadabur/truth_baseline/decodes/"
  rsync -a "$HOST:$ROOT/repo/tools/tadabur/truth_baseline/checks/c95.json" \
    "$repo_root/tools/tadabur/truth_baseline/checks/"
  echo "QUD shards in $dest; C95's caches in $repo_root/tools/tadabur/truth_baseline"
  echo "next: cd tools && python -m training.consonant_baseline merge --decodes-dir $dest"
}

case "${1:-}" in
  setup|run|launch|check|fetch) "$1" ;;
  *)
    echo "usage: $0 {launch|check|fetch}   (setup and run execute on the box)" >&2
    exit 2
    ;;
esac

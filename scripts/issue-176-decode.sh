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
# logs/baseline.done; `check` is the readiness signal, never a log tail.
#
# Budget (#176): the QUD half costs ~15.0k decode-seconds for one model at #95's measured
# 1.46 s/item (10,261 complement segments), so P=6 is ~42 wall minutes on the 12-core box.
# Stop and report above 2 GPU-h.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-176}
PARALLEL=${PARALLEL:-6}
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

run() {
  mkdir -p "$ROOT/decodes" "$ROOT/logs"
  rm -f "$ROOT/logs/baseline.done" "$ROOT/logs/baseline.exit"
  : > "$ROOT/logs/events"
  # Global: the trap can fire after `run` returns, when its locals are gone.
  SAMPLER=""
  # Any exit — failed conda, shard listing, xargs, held-out pass — leaves a readiness marker,
  # so `check` never hangs in `pending`; the sampler dies with the run.
  finish() {
    local code=$?
    [ -z "$SAMPLER" ] || kill "$SAMPLER" 2>/dev/null || true
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
  ( while true; do
      printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$(du -sb "$ROOT" | cut -f1)" \
        "$(df -B1 --output=avail /root/scratch | tail -1)" >> "$ROOT/logs/df.log"
      sleep 60
    done ) &
  SAMPLER=$!
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
  local code=0
  if [ "$qother" -ne 0 ] || [ "$held" -ne 0 ]; then code=1; fi
  printf '%s\n' "$code" > "$ROOT/logs/baseline.exit"
  printf '%s qother=%s held_out=%s\n' "$(date -u +%FT%TZ)" "$qother" "$held" \
    > "$ROOT/logs/baseline.done"
  return "$code"
}

launch() {
  local repo_root
  repo_root=$(git rev-parse --show-toplevel)
  ssh "$HOST" "mkdir -p '$ROOT'"
  git -C "$repo_root" archive HEAD \
    | ssh "$HOST" "rm -rf '$ROOT/repo' && mkdir -p '$ROOT/repo' && tar -x -C '$ROOT/repo'"
  ssh "$HOST" "$ROOT/repo/scripts/issue-176-decode.sh setup"
  ssh "$HOST" "nohup flock /root/scratch/gpu.lock $ROOT/repo/scripts/issue-176-decode.sh run \
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
    *)
      echo "failed: exit $status"
      ssh "$HOST" "grep -v ' exit 0\$' '$ROOT/logs/events' || true; \
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

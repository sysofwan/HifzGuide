#!/usr/bin/env bash
# issue-179-decode.sh — the consonant round's two GPU passes (#179): the look half for the
# #178 candidate's live and EMA weights, then the confirm sets at the weights the look chose.
#
# Mac side, from the worktree root, in this order (the choice commit must exist first):
#   scripts/issue-179-decode.sh launch-look      # archive HEAD; start the look pass detached
#   scripts/issue-179-decode.sh check-look       # done-file check; 0 = look finished cleanly
#   scripts/issue-179-decode.sh fetch-look       # pull the look decodes
#   # merge, look, then commit B (look decodes, models.json, look.json, look.claim.json,
#   # choice.json); only then:
#   scripts/issue-179-decode.sh launch-confirm   # archive HEAD again (commit B); start confirm
#   scripts/issue-179-decode.sh check-confirm
#   scripts/issue-179-decode.sh fetch-confirm    # pull the confirm decodes and caches
#
# `launch-confirm` refuses unless `choice.json`, `look.json` and `look.claim.json` are
# committed at HEAD and no confirm decode is: the decode gate reads them from the archive's
# git history, so the box gets a bundle clone, never a bare tar. `launch-look` refuses once a
# choice is committed: the look half is read once.
#
# On the box, `launch-*` stages the clips and starts `run look|confirm` under
# flock /root/scratch/gpu.lock. `run` appends one exit line per shard to logs/events and
# writes, last, logs/<phase>.exit and then logs/<phase>.done; `check` reads that, never a log
# tail. A run stopped by the budget writes exit 2 and a `budget-exceeded` done-file.
#
# Budget (#179): #95 measured the look half at ~1.6 h for one model at P=3 and the confirm
# sets at ~2.2 h at P=6, so BUDGET defaults to five hours per phase from the lock's
# acquisition. The confirm launch reads the chosen weights from the committed choice and
# decodes the candidate only; C95's, h448's and the teacher's decodes are #95's or #176's.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-179}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
LOOK_PARALLEL=${LOOK_PARALLEL:-3}
CONFIRM_PARALLEL=${CONFIRM_PARALLEL:-6}
BUDGET=${BUDGET:-18000}
CLIP_DIRS=(/root/scratch/issue-94/dev_clips /root/scratch/issue-83/stage/clips \
           /root/scratch/issue-126/stage/clips)
LOOK_MODELS=candidate_live,candidate_ema
#: The confirm sets, in the order the readout reads them; each is one shard range.
CONFIRM_SETS=(tarteel_v1.confirm qdat qud_other qud_half tadabur_dev_unseen)
CHOICE=tools/tadabur/consonant_readout/choice.json
LOOK=tools/tadabur/consonant_readout/look.json
CLAIM=tools/tadabur/consonant_readout/look.claim.json
DECODES=tools/tadabur/consonant_readout/decodes

activate() {
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
}

# --- staging ------------------------------------------------------------------------------------


stage_repo() {
  # A bundle clone, not a tar: the confirm decode's committed_choice reads choice.json,
  # look.json and look.claim.json through git at HEAD, so the box needs the history, and the
  # report records the real commit, not a synthetic one.
  local repo_root=$1 branch sha rev
  repo_root=$(git -C "$repo_root" rev-parse --show-toplevel)
  branch=$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$repo_root" rev-parse HEAD)
  ssh "$HOST" "mkdir -p '$ROOT' && rm -rf '$ROOT/repo' '$ROOT/repo.bundle'"
  if [ "$branch" = HEAD ]; then
    rev=$sha
  else
    rev=$branch
  fi
  git -C "$repo_root" bundle create - "$rev" | ssh "$HOST" "cat > '$ROOT/repo.bundle'"
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

setup_clips() {
  # The Ship A held-out clips (the G6 caches) and the Tadabur dev clips (tadabur_dev_unseen)
  # are staged by symlink from the box's existing clip directories; a missing clip fails here.
  mkdir -p "$ROOT"/{decodes,hf_cache,logs,clips}
  cd "$ROOT/repo/tools"
  activate
  python -m training.consonant_baseline held-out --list-clips > "$ROOT/held_clips.txt"
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

# --- the two decode passes (on the box) ---------------------------------------------------------


run_shard() {  # <set> <shard> <models>
  local set_name=$1 shard=$2 models=$3 rc=0
  python -m training.aug_readout_decode --spec consonant1 --set "$set_name" --shard "$shard" \
    --models "$models" --out-dir "$ROOT/decodes" --cache-dir "$ROOT/hf_cache" \
    --clips-dir "$ROOT/clips" > "$ROOT/logs/${set_name//./_}.$shard.log" 2>&1 || rc=$?
  printf '%s %s shard %s exit %s\n' "$(date -u +%FT%TZ)" "$set_name" "$shard" "$rc" \
    >> "$ROOT/logs/events"
  return "$rc"
}

shard_count() {  # <set>
  python -m training.aug_readout_decode --spec consonant1 --set "$1" --list-shards | wc -l
}

decode_look() {
  local i n
  n=$(shard_count tarteel_v1.look)
  export ROOT
  export -f run_shard
  seq 0 $((n - 1)) | xargs -P "$LOOK_PARALLEL" -I{} bash -c 'run_shard tarteel_v1.look {} "$0"' \
    "$LOOK_MODELS"
}

decode_confirm() {
  local weights i n set_name
  weights=$(cd "$ROOT/repo/tools" && python -c \
    "import json; print(json.load(open('tadabur/consonant_readout/choice.json'))['weights'])")
  export ROOT CONFIRM_MODEL="candidate_$weights"
  export -f run_shard
  : > "$ROOT/logs/jobs.txt"
  for set_name in "${CONFIRM_SETS[@]}"; do
    n=$(shard_count "$set_name")
    for ((i = 0; i < n; i++)); do printf '%s %s\n' "$set_name" "$i"; done \
      >> "$ROOT/logs/jobs.txt"
  done
  # G6's candidate caches at the chosen weights; C95's are the committed baseline.
  xargs -P "$CONFIRM_PARALLEL" -n 2 bash -c 'run_shard "$1" "$2" "$CONFIRM_MODEL"' _ \
    < "$ROOT/logs/jobs.txt"
  python -m training.consonant_baseline held-out --audio-dir "$ROOT/clips" --model c95 \
    > "$ROOT/logs/held_out.c95.log" 2>&1
  python -m training.consonant_baseline held-out --audio-dir "$ROOT/clips" \
    --model "$CONFIRM_MODEL" > "$ROOT/logs/held_out.$CONFIRM_MODEL.log" 2>&1
}

# --- the guarded run ----------------------------------------------------------------------------


run() {
  # Global, not local: the EXIT trap fires after `run` returns, when its locals are gone.
  PHASE=$1
  case "$PHASE" in
    look) ;;
    confirm) ;;
    *)
      echo "usage: $0 run {look|confirm}" >&2
      return 2
      ;;
  esac
  mkdir -p "$ROOT/decodes" "$ROOT/logs"
  rm -f "$ROOT/logs/$PHASE.done" "$ROOT/logs/$PHASE.exit" "$ROOT/logs/$PHASE.pass.exit" \
        "$ROOT/logs/budget.exceeded"
  : > "$ROOT/logs/events"
  # Globals: the EXIT trap can fire after `run` returns, when its locals are gone.
  SAMPLER=""
  WATCHDOG=""
  finish() {
    local code=$?
    if [ -n "$SAMPLER" ]; then
      kill -- "-$SAMPLER" 2>/dev/null || kill "$SAMPLER" 2>/dev/null || true
    fi
    if [ -n "$WATCHDOG" ]; then
      kill -- "-$WATCHDOG" 2>/dev/null || kill "$WATCHDOG" 2>/dev/null || true
    fi
    if [ ! -e "$ROOT/logs/$PHASE.done" ]; then
      printf '%s\n' "$code" > "$ROOT/logs/$PHASE.exit"
      printf '%s early-fail code=%s\n' "$(date -u +%FT%TZ)" "$code" \
        > "$ROOT/logs/$PHASE.done"
    fi
  }
  trap finish EXIT
  cd "$ROOT/repo/tools"
  activate
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
  "decode_$PHASE" &
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
      "$(date -u +%FT%TZ)" "$BUDGET" > "$ROOT/logs/$PHASE.done"
  else
    [ "$status" -eq 0 ] || code=1
    printf '%s %s exit %s\n' "$(date -u +%FT%TZ)" "$PHASE" "$status" \
      > "$ROOT/logs/$PHASE.pass.exit"
    cat "$ROOT/logs/$PHASE.pass.exit" > "$ROOT/logs/$PHASE.done"
  fi
  printf '%s\n' "$code" > "$ROOT/logs/$PHASE.exit"
  return "$code"
}

# --- the Mac side -------------------------------------------------------------------------------


require_no_choice() {  # the look half is read once
  local repo_root=$1
  if git -C "$repo_root" cat-file -e "HEAD:$CHOICE" 2>/dev/null; then
    echo "look refuses: $CHOICE is already committed; the look half is read once" >&2
    return 1
  fi
}

require_committed_choice() {  # the two-phase order: choice before any confirm decode
  local repo_root=$1 path
  for path in "$CHOICE" "$LOOK" "$CLAIM"; do
    if ! git -C "$repo_root" cat-file -e "HEAD:$path" 2>/dev/null; then
      echo "confirm refuses: $path is not committed at HEAD" >&2
      return 1
    fi
  done
  if git -C "$repo_root" ls-tree -r --name-only HEAD -- "$DECODES" \
      | grep -Eq '\.(tarteel_v1\.confirm|qdat|qud_other|qud_half|tadabur_dev_unseen)\.'; then
    echo "confirm refuses: a confirm decode is already committed" >&2
    return 1
  fi
}

launch() {
  local phase=$1 repo_root
  repo_root=$(git rev-parse --show-toplevel)
  if [ "$phase" = look ]; then
    require_no_choice "$repo_root"
  else
    require_committed_choice "$repo_root"
  fi
  stage_repo "$repo_root"
  ssh "$HOST" "$ROOT/repo/scripts/issue-179-decode.sh setup-clips"
  ssh "$HOST" "BUDGET=$BUDGET nohup flock /root/scratch/gpu.lock \
    '$ROOT/repo/scripts/issue-179-decode.sh' run '$phase' \
    >> '$ROOT/logs/run.out' 2>&1 & echo 'started: '\$!"
}

check() {
  local phase=$1 status
  status=$(ssh "$HOST" "if [ -f '$ROOT/logs/$phase.done' ]; then \
    cat '$ROOT/logs/$phase.exit'; else echo pending; fi")
  case "$status" in
    pending)
      echo "pending: $ROOT/logs/$phase.done is absent"
      ssh "$HOST" "tail -n 3 '$ROOT/logs/events' 2>/dev/null; \
        tail -n 2 '$ROOT/logs/run.out' 2>/dev/null" || true
      return 3
      ;;
    0)
      echo "done: exit 0"
      ssh "$HOST" "echo -n 'shard lines: '; wc -l < '$ROOT/logs/events'; \
        grep -v ' exit 0\$' '$ROOT/logs/events' || true"
      ;;
    2)
      echo "budget exceeded: the decode group was stopped"
      ssh "$HOST" "cat '$ROOT/logs/$phase.done' 2>/dev/null; \
        grep -v ' exit 0\$' '$ROOT/logs/events' || true"
      return 2
      ;;
    *)
      echo "failed: exit $status"
      ssh "$HOST" "cat '$ROOT/logs/$phase.done' 2>/dev/null; \
        grep -v ' exit 0\$' '$ROOT/logs/events' || true; \
        tail -n 20 '$ROOT/logs/run.out'"
      return 1
      ;;
  esac
}

fetch() {
  local phase=$1 repo_root dest
  repo_root=$(git rev-parse --show-toplevel)
  dest=${FETCH_DIR:-/tmp/issue-179-$phase}
  mkdir -p "$dest"
  rsync -a "$HOST:$ROOT/decodes/" "$dest/"
  if [ "$phase" = confirm ]; then
    # G6's candidate caches are commit-required data; C95's are already in the tree.
    mkdir -p "$repo_root/tools/tadabur/truth_baseline/decodes" \
             "$repo_root/tools/tadabur/truth_baseline/checks"
    rsync -a "$HOST:$ROOT/repo/tools/tadabur/truth_baseline/decodes/candidate_*.json" \
      "$repo_root/tools/tadabur/truth_baseline/decodes/"
    rsync -a "$HOST:$ROOT/repo/tools/tadabur/truth_baseline/checks/candidate_*.json" \
      "$repo_root/tools/tadabur/truth_baseline/checks/"
  fi
  echo "$phase decodes in $dest"
}

case "${1:-}" in
  setup-clips) setup_clips ;;
  run) run "${2:-}" ;;
  launch-look) launch look ;;
  launch-confirm) launch confirm ;;
  check-look) check look ;;
  check-confirm) check confirm ;;
  fetch-look) fetch look ;;
  fetch-confirm) fetch confirm ;;
  *)
    echo "usage: $0 {launch-look|check-look|fetch-look|launch-confirm|check-confirm|" \
         "fetch-confirm}   (setup-clips and run execute on the box)" >&2
    exit 2
    ;;
esac

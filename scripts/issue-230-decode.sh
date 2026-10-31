#!/usr/bin/env bash
# issue-230-decode.sh — round 2's non-tlog GPU decodes (#230): the chosen candidate and control
# on the guards (qdat, the QUD other half, Tadabur dev) and the reported Tarteel v1 confirm
# half, then G6's held-out caches for both arms.
#
# `plan` reads the **committed** choice and prints one line per decode job and per G6 cache; it
# writes nothing. `launch` runs the same checks, stages the repository as a bundle clone on the
# box (the decoder's `committed_choice` reads git at HEAD, so the box needs the history) and
# starts `run` detached under flock /root/scratch/gpu.lock. A decode of a chosen model that is
# already committed in the round's decodes/ blocks both: the sitting reads once.
#
# On the box, `run` stages the G2 and G6 clips under the lock, decodes every shard at 6
# processes x 1 thread with shared features, then produces both arms' held-out caches under the
# round's own pins. The models are the ones the committed choice names, never a fixed selector.
#
# Mac side, from the worktree root (the choice commit must exist first):
#   scripts/issue-230-decode.sh plan
#   scripts/issue-230-decode.sh launch
#   scripts/issue-230-decode.sh check
#   scripts/issue-230-decode.sh fetch      # pull the decodes and the truth-baseline caches
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-230}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ROUND_DIR=${ROUND_DIR:-tools/tadabur/consonant2_readout}
PARALLEL=${PARALLEL:-6}
BUDGET=${BUDGET:-18000}
#: The sets decoded for the round's two arms, in the order the readout reads them.
SETS=(tarteel_v1.confirm qdat qud_other tadabur_dev_unseen)
CHOICE=$ROUND_DIR/choice.json
LOOK=$ROUND_DIR/look.json
CLAIM=$ROUND_DIR/look.claim.json
DECODES=$ROUND_DIR/decodes
#: The two clip populations: Ship A's held-out clips (G6's caches) and Tadabur's dev clips h448
#: holds no copy of (G2).
HELD_CLIPS=357
DEV_CLIPS=671
CLIP_DIRS=(/root/scratch/issue-94/dev_clips /root/scratch/issue-83/stage/clips \
           /root/scratch/issue-126/stage/clips)


die() {
  echo "$*" >&2
  exit 1
}

# --- the committed inputs -----------------------------------------------------------------------


require_committed() {  # <path>
  git cat-file -e "HEAD:$1" 2>/dev/null || die "$1 is not committed at HEAD"
  git diff --quiet HEAD -- "$1" 2>/dev/null \
    || die "$1 is not committed at HEAD: it changed since its commit"
}

chosen_models() {
  git show "HEAD:$CHOICE" | python -c '
import json, sys
models = json.load(sys.stdin)["models"]
print(models["candidate"] + "," + models["control"])
'
}

require_not_decoded() {  # <candidate,control>
  local model
  for model in ${1//,/ }; do
    if git ls-tree -r --name-only HEAD -- "$DECODES" | grep -q "\.$model\.jsonl\.gz$"; then
      die "a decode of $model is already committed in $DECODES: the sitting reads once"
    fi
  done
}

require_ready() {
  require_committed "$CHOICE"
  require_committed "$LOOK"
  require_committed "$CLAIM"
  MODELS=$(chosen_models)
  [ -n "$MODELS" ] || die "$CHOICE names no candidate and control"
  require_not_decoded "$MODELS"
}

plan() {
  require_ready
  local set_name model
  for set_name in "${SETS[@]}"; do
    printf 'decode %s %s\n' "$set_name" "$MODELS"
  done
  for model in ${MODELS//,/ }; do
    printf 'held-out %s\n' "$model"
  done
}

# --- staging ------------------------------------------------------------------------------------


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
  ssh "$HOST" "mkdir -p '$ROOT/logs' && rm -rf '$ROOT/repo' '$ROOT/repo.bundle'"
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

# --- the Mac side -------------------------------------------------------------------------------


launch() {
  require_ready
  stage_repo
  ssh "$HOST" "ROUND_DIR='$ROUND_DIR' BUDGET=$BUDGET nohup flock /root/scratch/gpu.lock \
    '$ROOT/repo/scripts/issue-230-decode.sh' run \
    >> '$ROOT/logs/run.out' 2>&1 & echo 'started: '\$!"
}

check() {
  local status
  status=$(ssh "$HOST" "if [ -f '$ROOT/logs/run.done' ]; then cat '$ROOT/logs/run.exit'; \
    else echo pending; fi")
  case "$status" in
    pending)
      echo "pending: $ROOT/logs/run.done is absent"
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
      ssh "$HOST" "cat '$ROOT/logs/run.done' 2>/dev/null; \
        grep -v ' exit 0\$' '$ROOT/logs/events' || true"
      return 2
      ;;
    *)
      echo "failed: exit $status"
      ssh "$HOST" "cat '$ROOT/logs/run.done' 2>/dev/null; \
        grep -v ' exit 0\$' '$ROOT/logs/events' || true; \
        tail -n 20 '$ROOT/logs/run.out'"
      return 1
      ;;
  esac
}

fetch() {
  local repo_root dest model
  repo_root=$(git rev-parse --show-toplevel)
  dest=${FETCH_DIR:-/tmp/issue-230}
  mkdir -p "$dest"
  rsync -a "$HOST:$ROOT/decodes/" "$dest/"
  # G6's arm caches are commit-required data.
  mkdir -p "$repo_root/tools/tadabur/truth_baseline/decodes" \
           "$repo_root/tools/tadabur/truth_baseline/checks"
  for model in ${MODELS//,/ }; do
    rsync -a "$HOST:$ROOT/repo/tools/tadabur/truth_baseline/decodes/$model.json" \
      "$repo_root/tools/tadabur/truth_baseline/decodes/"
    rsync -a "$HOST:$ROOT/repo/tools/tadabur/truth_baseline/checks/$model.json" \
      "$repo_root/tools/tadabur/truth_baseline/checks/"
  done
  echo "decodes in $dest; G6 caches for $MODELS committed beside the truth baseline's"
}

# --- the box side -------------------------------------------------------------------------------


activate() {
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
}

stage_names() {  # <list> <expected> <label>
  local list=$1 expected=$2 label=$3 name dir staged=0 missing=0
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    for dir in "${CLIP_DIRS[@]}"; do
      if [ -f "$dir/$name" ]; then ln -sf "$dir/$name" "$name"; break; fi
    done
    if [ -e "$name" ]; then
      staged=$((staged + 1))
    else
      echo "$label: no $name in ${CLIP_DIRS[*]}" >&2
      missing=$((missing + 1))
    fi
  done < "$list"
  if [ "$staged" -ne "$expected" ] || [ "$missing" -ne 0 ]; then
    echo "$label: $staged of $expected staged, $missing missing" >&2
    return 1
  fi
  echo "$label: $staged of $expected staged"
}

setup_clips() {
  mkdir -p "$ROOT"/{decodes,hf_cache,logs,clips}
  cd "$ROOT/repo/tools"
  activate
  python -m training.consonant_baseline held-out --list-clips > "$ROOT/held_clips.txt"
  # The filter ``tadabur_recordings`` applies: every dev clip h448 did not see.
  python - <<'PY' > "$ROOT/dev_clips.txt"
from training.nonpro_gap import committed_grid, dev_clips

seen = set(committed_grid()["sample"]["h448_seen"])
print("\n".join(sorted(c.filename for c in dev_clips() if c.filename not in seen)))
PY
  cd "$ROOT/clips"
  rm -f -- ./*.wav
  stage_names "$ROOT/held_clips.txt" "$HELD_CLIPS" "Ship A held-out"
  stage_names "$ROOT/dev_clips.txt" "$DEV_CLIPS" "Tadabur dev-unseen"
}

run_shard() {  # <set> <shard>
  local set_name=$1 shard=$2 rc=0
  python -m training.aug_readout_decode --spec consonant2 --set "$set_name" --shard "$shard" \
    --models "$MODELS" --out-dir "$ROOT/decodes" --cache-dir "$ROOT/hf_cache" \
    --clips-dir "$ROOT/clips" --share-features \
    > "$ROOT/logs/${set_name//./_}.$shard.log" 2>&1 || rc=$?
  printf '%s %s shard %s exit %s\n' "$(date -u +%FT%TZ)" "$set_name" "$shard" "$rc" \
    >> "$ROOT/logs/events"
  return "$rc"
}

decode() {
  local set_name i n model
  : > "$ROOT/logs/jobs.txt"
  for set_name in "${SETS[@]}"; do
    n=$(python -m training.aug_readout_decode --spec consonant2 --set "$set_name" \
      --list-shards | wc -l)
    for ((i = 0; i < n; i++)); do printf '%s %s\n' "$set_name" "$i"; done \
      >> "$ROOT/logs/jobs.txt"
  done
  export ROOT MODELS
  export -f run_shard
  xargs -P "$PARALLEL" -n 2 bash -c 'run_shard "$1" "$2"' _ < "$ROOT/logs/jobs.txt"
  # G6's caches, under the round's own pins: the baseline's already exist from round 1.
  for model in ${MODELS//,/ }; do
    python -m training.consonant_baseline held-out --spec consonant2 --audio-dir "$ROOT/clips" \
      --model "$model" > "$ROOT/logs/held_out.$model.log" 2>&1
  done
}

run() {
  mkdir -p "$ROOT/decodes" "$ROOT/logs"
  rm -f "$ROOT/logs/run.done" "$ROOT/logs/run.exit"
  : > "$ROOT/logs/events"
  MODELS=$(cd "$ROOT/repo/$ROUND_DIR" && python -c \
    'import json; m = json.load(open("choice.json"))["models"]; print(m["candidate"] + "," + m["control"])')
  export MODELS
  local status=0
  timeout --signal=TERM --kill-after=60 "$BUDGET" bash -c 'set -e; setup_clips; decode' \
    || status=$?
  local code=$status
  if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    code=2
    printf '%s budget-exceeded budget=%ss; per-item output kept\n' \
      "$(date -u +%FT%TZ)" "$BUDGET" > "$ROOT/logs/run.done"
  else
    printf '%s exit %s\n' "$(date -u +%FT%TZ)" "$status" > "$ROOT/logs/run.done"
  fi
  printf '%s\n' "$code" > "$ROOT/logs/run.exit"
  return "$code"
}

# Sourceable for the local helpers; only a direct run dispatches.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    plan) plan ;;
    launch) launch ;;
    check) check ;;
    fetch) fetch ;;
    run) run ;;
    *)
      echo "usage: $0 {plan|launch|check|fetch}   (run executes on the box)" >&2
      exit 2
      ;;
  esac
fi

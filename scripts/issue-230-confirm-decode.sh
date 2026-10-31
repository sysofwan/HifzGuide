#!/usr/bin/env bash
# issue-230-confirm-decode.sh — round 2's P3 confirm decode (#230): the teacher, C95, the chosen
# candidate and the chosen control on confirm.1 + confirm.2 gating.
#
# The pre-registration's step 6 decodes four models on the two claimed tranches once. This is the
# tlog decoder's job (`tadabur.tlog_eval_decode`), not the non-tlog launcher's: confirm rows are
# sealed until their claims are in HEAD, and the decoder itself refuses an unclaimed tranche or a
# model outside the frozen set. `scripts/issue-230-decode.sh` stays the four non-tlog sets.
#
# The two tranches are decoded one after the other, each at PARALLEL processes x 1 thread with
# shared features. The default is 4 rather than the 6 the look run used: the box has 24 GB of
# host RAM, and a 6-process non-tlog run OOM-killed a shard (#230). GPU peak at 6 processes was
# 12,419 of 16,311 MiB (#217), so 4 fits.
#
# `plan` reads the **committed** choice and the two committed claims and prints the jobs; it
# writes nothing. `launch` stages the repository as a bundle clone on the box (the decoder's
# `committed_choice` and claim checks read git at HEAD) and starts `run` detached under
# flock /root/scratch/gpu.lock. `resume` restages and retries at RESUME_PARALLEL, keeping the
# parts already decoded (the decoder resumes by row key). A confirm decode already committed in
# the round's decodes/ blocks every command: the sitting reads once.
#
# On the box, `run` decodes both tranches, then pools them into `out/` with `merge-confirm`:
# one sorted `tlog.confirm.<model>.jsonl.gz` per model plus `meta.confirm.json`. The merge
# refuses a missing or repeated row, so a partial run never looks complete.
#
# Mac side, from the worktree root (the claims must exist first):
#   scripts/issue-230-confirm-decode.sh plan
#   scripts/issue-230-confirm-decode.sh launch
#   scripts/issue-230-confirm-decode.sh resume     # after a failed or interrupted run
#   scripts/issue-230-confirm-decode.sh check      # run.done / run.exit
#   scripts/issue-230-confirm-decode.sh fetch      # pull the pooled decodes into the round
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-230-confirm}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ROUND_DIR=${ROUND_DIR:-tools/tadabur/consonant2_readout}
TRANCHES_DIR=${TRANCHES_DIR:-tools/tadabur/tlog/tranches}
PARALLEL=${PARALLEL:-4}
#: A resume retries after an OOM kill: fewer processes, whatever the first launch used.
RESUME_PARALLEL=${RESUME_PARALLEL:-2}
BUDGET=${BUDGET:-28800}
#: The two claimable tranches, in claim order. The decoder refuses any other role.
ROLES=(confirm.1 confirm.2)
#: The frozen producer set: the teacher, the baseline and the committed choice's two arms.
FIXED_MODELS=(base c95)
CHOICE=$ROUND_DIR/choice.json
DECODES=$ROUND_DIR/decodes
PARTS=$ROOT/parts
OUT=$ROOT/out

die() { echo "$*" >&2; exit 1; }

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
print(",".join(["base", "c95", models["candidate"], models["control"]]))
'
}

require_not_decoded() {  # <models>
  local model
  for model in ${1//,/ }; do
    if git cat-file -e "HEAD:$DECODES/tlog.confirm.$model.jsonl.gz" 2>/dev/null; then
      die "a confirm decode of $model is already committed in $DECODES: the sitting reads once"
    fi
  done
}

require_ready() {
  require_committed "$CHOICE"
  local role
  for role in "${ROLES[@]}"; do
    require_committed "$TRANCHES_DIR/$role.claim.json"
  done
  MODELS=$(chosen_models)
  [ -n "$MODELS" ] || die "$CHOICE names no candidate and control"
  require_not_decoded "$MODELS"
}

# --- the Mac side -------------------------------------------------------------------------------


plan() {
  require_ready
  local role
  for role in "${ROLES[@]}"; do
    printf 'decode %s %s at %s processes\n' "$role" "$MODELS" "$PARALLEL"
  done
  printf 'merge-confirm %s\n' "$MODELS"
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

start_run() {  # <parallel> <run args>
  ssh "$HOST" "ISSUE_ROOT='$ROOT' ROUND_DIR='$ROUND_DIR' BUDGET=$BUDGET PARALLEL=$1 \
    nohup flock /root/scratch/gpu.lock '$ROOT/repo/scripts/issue-230-confirm-decode.sh' run $2 \
    >> '$ROOT/logs/run.out' 2>&1 & echo 'started: '\$!"
}

launch() {
  require_ready
  stage_repo
  start_run "$PARALLEL" ""
}

resume() {
  require_ready
  stage_repo
  start_run "$RESUME_PARALLEL" --resume
}

check() {
  local status
  status=$(ssh "$HOST" "if [ ! -f '$ROOT/logs/run.done' ]; then echo pending; \
    elif [ ! -f '$ROOT/logs/run.exit' ]; then echo no-exit; \
    else cat '$ROOT/logs/run.exit'; fi")
  case "$status" in
    pending)
      echo "pending: $ROOT/logs/run.done is absent"
      ssh "$HOST" "tail -n 3 '$ROOT/logs/events' 2>/dev/null; \
        tail -n 2 '$ROOT/logs/run.out' 2>/dev/null" || true
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

require_decoded() {
  # ``fetch`` copies the run's output as final, so a partial run must refuse: the run must have
  # exited 0 and the pooled merge must hold one file per model and its meta.
  ssh "$HOST" bash -s <<EOF || die "the box has no complete run to fetch"
set -e
test -f '$ROOT/logs/run.done'
test "\$(cat '$ROOT/logs/run.exit')" = 0
test -f '$OUT/meta.confirm.json'
for model in ${MODELS//,/ }; do
  test -f '$OUT/tlog.confirm.'\$model'.jsonl.gz'
done
for role in ${ROLES[*]}; do
  grep -q " \$role worker " '$ROOT/logs/events'
done
EOF
}

fetch() {
  local repo_root dest
  require_ready
  require_decoded
  repo_root=$(git rev-parse --show-toplevel)
  dest=${FETCH_DIR:-$repo_root/$DECODES}
  mkdir -p "$dest"
  rsync -a "$HOST:$OUT/" "$dest/"
  echo "confirm decodes in $dest: commit them with their provenance"
}

# --- the box side -------------------------------------------------------------------------------


activate() {
  [ -r "$CONDA" ] || { echo "no conda at $CONDA" >&2; return 1; }
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
}

run_worker() {  # <role> <index>
  local role=$1 index=$2 rc=0
  python -m tadabur.tlog_eval_decode decode --parts-dir "$PARTS" --cache-dir "$ROOT/cache" \
    --models ${MODELS//,/ } --role "$role" --shard-index "$index" --shard-count "$PARALLEL" \
    > "$ROOT/logs/$role.$index.log" 2>&1 || rc=$?
  printf '%s %s worker %s exit %s\n' "$(date -u +%FT%TZ)" "$role" "$index" "$rc" \
    >> "$ROOT/logs/events"
  return "$rc"
}

decode() {
  local role index status=0 rc pids
  for role in "${ROLES[@]}"; do
    status=0
    pids=()
    for ((index = 0; index < PARALLEL; index++)); do
      run_worker "$role" "$index" &
      pids+=("$!")
    done
    for index in "${!pids[@]}"; do
      rc=0
      wait "${pids[$index]}" || rc=$?
      [ "$rc" -eq 0 ] || status=$rc
    done
    [ "$status" -eq 0 ] || {
      echo "$role: a worker failed (status $status)" >&2
      return "$status"
    }
  done
  # The pooled merge refuses a missing or repeated row: a partial run never looks complete.
  python -m tadabur.tlog_eval_decode merge-confirm --parts-dir "$PARTS" --out-dir "$OUT" \
    --models ${MODELS//,/ }
}

decode_phase() {
  # The timed body: the parts and output live outside the staged clone, so a restaged resume
  # keeps what the failed run decoded. Clips are streamed from the source shards (which the
  # decoder deletes once read); no confirm clip cache is kept.
  set -e
  mkdir -p "$PARTS" "$OUT" "$ROOT/cache" "$ROOT/logs"
  cd "$ROOT/repo/tools"
  activate
  export OMP_NUM_THREADS=1
  decode
}

record_run_result() {  # <status>
  # ``check`` keys off done and then reads exit: write exit first.
  local status=$1 code=$1 done
  if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    code=2
    done=$(printf '%s budget-exceeded budget=%ss; per-item output kept' \
      "$(date -u +%FT%TZ)" "$BUDGET")
  else
    done=$(printf '%s exit %s' "$(date -u +%FT%TZ)" "$status")
  fi
  printf '%s\n' "$code" > "$ROOT/logs/run.exit"
  printf '%s\n' "$done" > "$ROOT/logs/run.done"
  return "$code"
}

run() {  # [--resume]
  # A signal's EXIT trap reads the last successful command's status, so translate TERM and INT
  # into their 128+signal codes here; SIGKILL cannot be recorded, so a hard-killed run stays
  # pending. The traps are armed before anything can fail.
  local resume=""
  case "${1:-}" in
    --resume) resume=1 ;;
    "") ;;
    *) die "unknown run argument: $1" ;;
  esac
  trap 'exit 143' TERM
  trap 'exit 130' INT
  trap 'record_run_result "$?"' EXIT
  mkdir -p "$PARTS" "$OUT" "$ROOT/logs"
  rm -f "$ROOT/logs/run.done" "$ROOT/logs/run.exit"
  # A resume keeps the failed run's worker lines; a fresh run starts them clean.
  if [ -n "$resume" ]; then touch "$ROOT/logs/events"; else : > "$ROOT/logs/events"; fi
  activate
  MODELS=$(cd "$ROOT/repo/$ROUND_DIR" && python -c \
    'import json; m = json.load(open("choice.json"))["models"]; print(",".join(["base", "c95", m["candidate"], m["control"]]))')
  export MODELS
  local status=0
  timeout --signal=TERM --kill-after=60 "$BUDGET" bash "$0" decode-phase || status=$?
  trap - EXIT TERM INT
  record_run_result "$status"
}

# Sourceable for the local helpers; only a direct run dispatches.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    plan) plan ;;
    launch) launch ;;
    resume) resume ;;
    check) check ;;
    fetch) fetch ;;
    run)
      shift
      run "$@"
      ;;
    decode-phase) decode_phase ;;
    *)
      echo "usage: $0 {plan|launch|resume|check|fetch}
       (run [--resume] and decode-phase execute on the box)" >&2
      exit 2
      ;;
  esac
fi

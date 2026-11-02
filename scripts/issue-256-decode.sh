#!/usr/bin/env bash
# issue-256-decode.sh — round 3's GPU decodes (#256): the look sitting, the non-tlog guard
# sets and the two confirm tranches.
#
# Three phases, each with its own box directory so their done/exit files never collide:
#
#   look     `r3_candidate_live,r3_candidate_ema` on the tlog look gating rows, then one merge.
#            The teacher's, C95's and round 2's look decodes are committed already, so they are
#            reused, never decoded again. The look reads no choice.
#   gates    the committed choice's candidate (`models.candidate`; round 3 has no control) on
#            tarteel_v1.confirm, qdat, qud_other and tadabur_dev_unseen, then Ship A's held-out
#            cache. The choice's control, if it still names one, is never decoded.
#   confirm  the teacher, C95 and the chosen round-3 weights on confirm.3 then confirm.4, then
#            `merge-confirm`. Both tranches' claims must be committed first: the decoder refuses
#            an unclaimed tranche itself, and this script refuses before it stages anything.
#
# `plan` reads the **committed** inputs and prints the jobs; it writes nothing. `launch` stages
# the repository as a bundle clone on the box (the decoder's `committed_choice` and claim checks
# read git at HEAD) and starts `run` detached under flock /root/scratch/gpu.lock. `resume`
# restages and retries at RESUME_PARALLEL, keeping what the failed run decoded. A decode of one
# of the phase's own models already committed in the round's decodes/ blocks every command: the
# sitting reads once. The tlog decoder writes parts under `<phase>/parts` and merges into
# `<phase>/out`; `fetch` copies them into `tools/tadabur/consonant3_readout/decodes/`.
#
# Mac side, from the worktree root (each phase's inputs must exist first):
#   scripts/issue-256-decode.sh look plan
#   scripts/issue-256-decode.sh look launch
#   scripts/issue-256-decode.sh confirm resume     # after a failed or interrupted run
#   scripts/issue-256-decode.sh gates check        # run.done / run.exit
#   scripts/issue-256-decode.sh gates fetch        # pull the decodes and the G6 caches
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-256}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ROUND_DIR=${ROUND_DIR:-tools/tadabur/consonant3_readout}
TRANCHES_DIR=${TRANCHES_DIR:-tools/tadabur/tlog/tranches}
PARALLEL=${PARALLEL:-4}
#: A resume retries after an OOM kill: fewer processes, whatever the first launch used (#230).
RESUME_PARALLEL=${RESUME_PARALLEL:-2}
BUDGET=${BUDGET:-28800}
#: Round 3's two look arms, at the one checkpoint's live and averaged weights.
LOOK_MODELS=r3_candidate_live,r3_candidate_ema
#: The confirm tranches' fixed producer prefix, in the order `merge-confirm` pools them.
PREFIX_MODELS=base,c95
#: The two claimable tranches, in claim order. The decoder refuses any other role.
ROLES=(confirm.3 confirm.4)
#: The non-tlog sets decoded for the chosen candidate. qdat_bench is qdat's stream_b0 store and
#: ship_a_verified is the G6 held-out cache, so neither is a decode set of its own.
SETS=(tarteel_v1.confirm qdat qud_other tadabur_dev_unseen)
CHOICE=$ROUND_DIR/choice.json
LOOK=$ROUND_DIR/look.json
LOOK_CLAIM=$ROUND_DIR/look.claim.json
DECODES=$ROUND_DIR/decodes
#: The two clip populations: Ship A's held-out clips (G6's caches) and Tadabur's dev clips h448
#: holds no copy of (G2).
HELD_CLIPS=357
DEV_CLIPS=671
CLIP_DIRS=(/root/scratch/issue-94/dev_clips /root/scratch/issue-83/stage/clips \
           /root/scratch/issue-126/stage/clips)


die() { echo "$*" >&2; exit 1; }

valid_phase() {  # <phase>
  case "$1" in
    look|gates|confirm) return 0 ;;
    *) return 1 ;;
  esac
}

usage() {
  cat >&2 <<'EOF'
usage: issue-256-decode.sh <phase> <command>
       phase: look | gates | confirm
       command: plan | launch | resume | check | fetch
       (on the box: run <phase> [--resume], decode-phase <phase>)
EOF
  exit 2
}

# --- the committed inputs -----------------------------------------------------------------------


require_committed() {  # <path>
  git cat-file -e "HEAD:$1" 2>/dev/null || die "$1 is not committed at HEAD"
  git diff --quiet HEAD -- "$1" 2>/dev/null \
    || die "$1 is not committed at HEAD: it changed since its commit"
}

choice_candidate() {  # <round dir>
  python -c '
import json, sys
print(json.load(open(sys.argv[1]))["models"]["candidate"])
' "$1/choice.json"
}

#: Each phase's models, in the order its decodes and merge list them. The look's are fixed; the
#: other phases follow the committed choice, and round 3 never decodes a control.
phase_models() {  # <round dir>
  local candidate
  case "$PHASE" in
    look) echo "$LOOK_MODELS" ;;
    gates) choice_candidate "$1" ;;
    confirm)
      candidate=$(choice_candidate "$1") || return $?
      printf '%s,%s\n' "$PREFIX_MODELS" "$candidate"
      ;;
    *) die "unknown phase: $PHASE" ;;
  esac
}

require_not_decoded() {  # <models>
  # Only the decodes this phase produces count: another phase's files, and the look's reused
  # teacher/C95/round-2 decodes, never block it.
  local model name
  for model in ${1//,/ }; do
    case "$PHASE" in
      look) name=tlog.look.$model ;;
      gates)
        for set_name in "${SETS[@]}"; do
          if git cat-file -e "HEAD:$DECODES/$set_name.$model.jsonl.gz" 2>/dev/null; then
            die "a decode of $model on $set_name is already committed in $DECODES: the sitting reads once"
          fi
        done
        continue
        ;;
      confirm) name=tlog.confirm.$model ;;
      *) die "unknown phase: $PHASE" ;;
    esac
    if git cat-file -e "HEAD:$DECODES/$name.jsonl.gz" 2>/dev/null; then
      die "a $PHASE decode of $model is already committed in $DECODES: the sitting reads once"
    fi
  done
}

require_ready() {
  local repo_root
  case "$PHASE" in
    look) ;;
    gates)
      require_committed "$CHOICE"
      require_committed "$LOOK"
      require_committed "$LOOK_CLAIM"
      ;;
    confirm)
      require_committed "$CHOICE"
      local role
      for role in "${ROLES[@]}"; do
        require_committed "$TRANCHES_DIR/$role.claim.json"
      done
      ;;
    *) die "unknown phase: $PHASE" ;;
  esac
  repo_root=$(git rev-parse --show-toplevel)
  MODELS=$(phase_models "$repo_root/$ROUND_DIR")
  [ -n "$MODELS" ] || die "$CHOICE names no candidate"
  require_not_decoded "$MODELS"
}

# --- the Mac side -------------------------------------------------------------------------------


plan() {
  require_ready
  local role set_name
  case "$PHASE" in
    look)
      printf 'decode look %s at %s processes\n' "$MODELS" "$PARALLEL"
      printf 'merge %s\n' "$MODELS"
      ;;
    gates)
      for set_name in "${SETS[@]}"; do
        printf 'decode %s %s\n' "$set_name" "$MODELS"
      done
      printf 'held-out %s\n' "$MODELS"
      ;;
    confirm)
      for role in "${ROLES[@]}"; do
        printf 'decode %s %s at %s processes\n' "$role" "$MODELS" "$PARALLEL"
      done
      printf 'merge-confirm %s\n' "$MODELS"
      ;;
  esac
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
  ssh "$HOST" "mkdir -p '$ROOT' && rm -rf '$ROOT/repo' '$ROOT/repo.bundle'"
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
  ssh "$HOST" "ISSUE_ROOT='$ROOT' PHASE='$PHASE' ROUND_DIR='$ROUND_DIR' BUDGET=$BUDGET \
    PARALLEL=$1 nohup flock /root/scratch/gpu.lock '$ROOT/repo/scripts/issue-256-decode.sh' \
    run '$PHASE' $2 >> '$ROOT/$PHASE/logs/run.out' 2>&1 & echo 'started: '\$!"
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
  status=$(ssh "$HOST" "if [ ! -f '$ROOT/$PHASE/logs/run.done' ]; then echo pending; \
    elif [ ! -f '$ROOT/$PHASE/logs/run.exit' ]; then echo no-exit; \
    else cat '$ROOT/$PHASE/logs/run.exit' | tr -d '[:space:]'; fi")
  case "$status" in
    pending)
      echo "pending: $ROOT/$PHASE/logs/run.done is absent"
      ssh "$HOST" "tail -n 3 '$ROOT/$PHASE/logs/events' 2>/dev/null; \
        tail -n 2 '$ROOT/$PHASE/logs/run.out' 2>/dev/null" || true
      return 3
      ;;
    no-exit)
      echo "failed: $ROOT/$PHASE/logs/run.done exists without $ROOT/$PHASE/logs/run.exit"
      ssh "$HOST" "cat '$ROOT/$PHASE/logs/run.done' 2>/dev/null; \
        tail -n 20 '$ROOT/$PHASE/logs/run.out' 2>/dev/null" || true
      return 1
      ;;
    0)
      echo "done: exit 0"
      ssh "$HOST" "echo -n 'worker lines: '; wc -l < '$ROOT/$PHASE/logs/events'; \
        grep -v ' exit 0\$' '$ROOT/$PHASE/logs/events' || true"
      ;;
    2)
      echo "budget exceeded: the decode group was stopped"
      ssh "$HOST" "cat '$ROOT/$PHASE/logs/run.done' 2>/dev/null; \
        grep -v ' exit 0\$' '$ROOT/$PHASE/logs/events' || true"
      return 2
      ;;
    *)
      echo "failed: exit $status"
      ssh "$HOST" "cat '$ROOT/$PHASE/logs/run.done' 2>/dev/null; \
        grep -v ' exit 0\$' '$ROOT/$PHASE/logs/events' || true; \
        tail -n 20 '$ROOT/$PHASE/logs/run.out'"
      return 1
      ;;
  esac
}

require_decoded() {
  # ``fetch`` copies the run's output as final, so a partial run must refuse: the run must have
  # exited 0, and the phase's own outputs must be complete.
  ssh "$HOST" "test -f '$ROOT/$PHASE/logs/run.done' && \
    test \"\$(cat '$ROOT/$PHASE/logs/run.exit')\" = 0" \
    || die "the box has no complete run to fetch"
  case "$PHASE" in
    gates)
      ssh "$HOST" bash -s <<EOF || die "the box's $PHASE run has no complete decodes to fetch"
set -e
test -s '$ROOT/$PHASE/logs/jobs.txt'
while read -r set_name shard; do
  grep -Eq " \$set_name shard \$shard exit 0\$" '$ROOT/$PHASE/logs/events'
  for model in ${MODELS//,/ }; do
    test -f '$ROOT/$PHASE/decodes/'\$set_name'.'\$shard'.'\$model'.jsonl'
  done
done < '$ROOT/$PHASE/logs/jobs.txt'
for model in ${MODELS//,/ }; do
  test -f '$ROOT/repo/tools/tadabur/truth_baseline/decodes/'\$model'.json'
  test -f '$ROOT/repo/tools/tadabur/truth_baseline/checks/'\$model'.json'
done
EOF
      ;;
    look) require_tlog_decoded meta.json look ;;
    confirm) require_tlog_decoded meta.confirm.json "${ROLES[@]}" ;;
  esac
}

require_tlog_decoded() {  # <meta> <role>...
  local meta=$1 role
  shift
  ssh "$HOST" bash -s <<EOF || die "the box's $PHASE run has no complete decodes to fetch"
set -e
test -f '$ROOT/$PHASE/out/$meta'
for model in ${MODELS//,/ }; do
  test -f '$ROOT/$PHASE/out/tlog.$PHASE.'\$model'.jsonl.gz'
done
for role in $*; do
  grep -q " \$role worker " '$ROOT/$PHASE/logs/events'
done
EOF
}

fetch() {
  local repo_root dest model
  require_ready
  require_decoded
  repo_root=$(git rev-parse --show-toplevel)
  dest=${FETCH_DIR:-$repo_root/$DECODES}
  mkdir -p "$dest"
  if [ "$PHASE" = gates ]; then
    rsync -a "$HOST:$ROOT/$PHASE/decodes/" "$dest/"
    mkdir -p "$repo_root/tools/tadabur/truth_baseline/decodes" \
             "$repo_root/tools/tadabur/truth_baseline/checks"
    for model in ${MODELS//,/ }; do
      rsync -a "$HOST:$ROOT/repo/tools/tadabur/truth_baseline/decodes/$model.json" \
        "$repo_root/tools/tadabur/truth_baseline/decodes/"
      rsync -a "$HOST:$ROOT/repo/tools/tadabur/truth_baseline/checks/$model.json" \
        "$repo_root/tools/tadabur/truth_baseline/checks/"
    done
    echo "gate decodes in $dest; G6 caches for $MODELS committed beside the truth baseline's"
  else
    rsync -a "$HOST:$ROOT/$PHASE/out/" "$dest/"
    echo "$PHASE decodes in $dest: commit them with their provenance"
  fi
}

# --- the box side -------------------------------------------------------------------------------


activate() {
  [ -r "$CONDA" ] || { echo "no conda at $CONDA" >&2; return 1; }
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
}

#: One tlog decoder process over its share of a role's gating shards; its exit is an event so
#: a resumed run can tell which tranches already ran to the end.
run_worker() {  # <role> <index>
  local role=$1 index=$2 rc=0
  python -m tadabur.tlog_eval_decode decode --spec consonant3 \
    --parts-dir "$ROOT/$PHASE/parts" --cache-dir "$ROOT/$PHASE/cache" \
    --models ${MODELS//,/ } --role "$role" --shard-index "$index" --shard-count "$PARALLEL" \
    > "$ROOT/$PHASE/logs/$role.$index.log" 2>&1 || rc=$?
  printf '%s %s worker %s exit %s\n' "$(date -u +%FT%TZ)" "$role" "$index" "$rc" \
    >> "$ROOT/$PHASE/logs/events"
  return "$rc"
}

decode_tlog_role() {  # <role>
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
  [ "$status" -eq 0 ] || {
    echo "$role: a worker failed (status $status)" >&2
    return "$status"
  }
}

decode_look() {
  decode_tlog_role look
  python -m tadabur.tlog_eval_decode merge --spec consonant3 \
    --parts-dir "$ROOT/$PHASE/parts" --out-dir "$ROOT/$PHASE/out" --models ${MODELS//,/ }
}

decode_confirm() {
  local role
  for role in "${ROLES[@]}"; do
    decode_tlog_role "$role" || return $?
  done
  # The pooled merge refuses a missing or repeated row: a partial run never looks complete.
  python -m tadabur.tlog_eval_decode merge-confirm --spec consonant3 \
    --parts-dir "$ROOT/$PHASE/parts" --out-dir "$ROOT/$PHASE/out" --models ${MODELS//,/ }
}

# --- the guard sets ------------------------------------------------------------------------------


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
  mkdir -p "$ROOT/$PHASE"/{decodes,hf_cache,logs,clips}
  cd "$ROOT/repo/tools"
  activate
  python -m training.consonant_baseline held-out --list-clips > "$ROOT/$PHASE/held_clips.txt"
  # The filter ``tadabur_recordings`` applies: every dev clip h448 did not see.
  python - <<'PY' > "$ROOT/$PHASE/dev_clips.txt"
from training.nonpro_gap import committed_grid, dev_clips

seen = set(committed_grid()["sample"]["h448_seen"])
print("\n".join(sorted(c.filename for c in dev_clips() if c.filename not in seen)))
PY
  cd "$ROOT/$PHASE/clips"
  rm -f -- ./*.wav
  stage_names "$ROOT/$PHASE/held_clips.txt" "$HELD_CLIPS" "Ship A held-out"
  stage_names "$ROOT/$PHASE/dev_clips.txt" "$DEV_CLIPS" "Tadabur dev-unseen"
}

run_shard() {  # <set> <shard>
  local set_name=$1 shard=$2 rc=0
  python -m training.aug_readout_decode --spec consonant3 --set "$set_name" --shard "$shard" \
    --models "$MODELS" --out-dir "$ROOT/$PHASE/decodes" --cache-dir "$ROOT/hf_cache" \
    --clips-dir "$ROOT/$PHASE/clips" --share-features \
    > "$ROOT/$PHASE/logs/${set_name//./_}.$shard.log" 2>&1 || rc=$?
  printf '%s %s shard %s exit %s\n' "$(date -u +%FT%TZ)" "$set_name" "$shard" "$rc" \
    >> "$ROOT/$PHASE/logs/events"
  return "$rc"
}

queue_jobs() {
  local set_name i n
  : > "$ROOT/$PHASE/logs/jobs.txt"
  for set_name in "${SETS[@]}"; do
    n=$(python -m training.aug_readout_decode --spec consonant3 --set "$set_name" \
      --list-shards | wc -l)
    for ((i = 0; i < n; i++)); do printf '%s %s\n' "$set_name" "$i"; done \
      >> "$ROOT/$PHASE/logs/jobs.txt"
  done
}

pending_jobs() {
  # A resumed run retries only the shards with no exit-0 event: a shard that ran is decoded
  # once, and its output stays as the sitting wrote it.
  local set_name shard
  while read -r set_name shard; do
    grep -Eq " $set_name shard $shard exit 0\$" "$ROOT/$PHASE/logs/events" 2>/dev/null \
      || printf '%s %s\n' "$set_name" "$shard"
  done < "$ROOT/$PHASE/logs/jobs.txt"
}

decode_gates() {
  local model status=0 cache_status
  queue_jobs
  pending_jobs > "$ROOT/$PHASE/logs/pending.txt"
  export ROOT PHASE MODELS
  export -f run_shard
  # GNU xargs runs its command once on empty input (no --no-run-if-empty), handing the decoder
  # empty --set/--shard and failing a resume whose only work is G6. G6 stays outside this guard.
  if [ -s "$ROOT/$PHASE/logs/pending.txt" ]; then
    xargs -P "$PARALLEL" -n 2 bash -c 'run_shard "$1" "$2"' _ \
      < "$ROOT/$PHASE/logs/pending.txt" || status=$?
  fi
  # G6's caches, under the round's own pins. They are independent of the shard decodes, so a
  # still-failing shard must not leave them unwritten.
  for model in ${MODELS//,/ }; do
    cache_status=0
    python -m training.consonant_baseline held-out --spec consonant3 \
      --audio-dir "$ROOT/$PHASE/clips" --model "$model" \
      > "$ROOT/$PHASE/logs/held_out.$model.log" 2>&1 || cache_status=$?
    [ "$cache_status" -eq 0 ] || [ "$status" -ne 0 ] || status=$cache_status
  done
  [ "$status" -eq 0 ] || return "$status"
}

decode() {
  case "$PHASE" in
    look) decode_look ;;
    gates) decode_gates ;;
    confirm) decode_confirm ;;
    *) die "unknown phase: $PHASE" ;;
  esac
}

decode_phase() {
  # The timed body: the parts and output live outside the staged clone, so a restaged resume
  # keeps what the failed run decoded.
  set -e
  local dir=$ROOT/$PHASE
  mkdir -p "$dir/parts" "$dir/out" "$dir/decodes" "$dir/logs"
  if [ "$PHASE" = gates ]; then setup_clips; fi
  cd "$ROOT/repo/tools"
  activate
  export OMP_NUM_THREADS=1
  decode
}

record_run_result() {  # <status>
  # ``check`` keys off done and then reads exit: write exit first.
  local status=$1 code=$1 done
  mkdir -p "$ROOT/$PHASE/logs"
  if [ "$status" -eq 124 ]; then
    code=2
    done=$(printf '%s budget-exceeded budget=%ss; per-item output kept' \
      "$(date -u +%FT%TZ)" "$BUDGET")
  elif [ "$status" -eq 137 ]; then
    # 137 is ambiguous: ``timeout --kill-after`` escalates to SIGKILL (GNU timeout's
    # documented escalation status), and an external kill or the OOM killer looks the same
    # here. Record both possibilities rather than guessing one.
    done=$(printf '%s killed (status 137, SIGKILL): timeout escalation or external kill/OOM; \
per-item output kept' "$(date -u +%FT%TZ)")
  else
    done=$(printf '%s exit %s' "$(date -u +%FT%TZ)" "$status")
  fi
  printf '%s\n' "$code" > "$ROOT/$PHASE/logs/run.exit"
  printf '%s\n' "$done" > "$ROOT/$PHASE/logs/run.done"
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
  local dir=$ROOT/$PHASE
  trap 'exit 143' TERM
  trap 'exit 130' INT
  trap 'record_run_result "$?"' EXIT
  mkdir -p "$dir/parts" "$dir/out" "$dir/decodes" "$dir/logs"
  rm -f "$dir/logs/run.done" "$dir/logs/run.exit"
  # A resume keeps the failed run's worker lines; a fresh run starts them clean.
  if [ -n "$resume" ]; then touch "$dir/logs/events"; else : > "$dir/logs/events"; fi
  activate
  MODELS=$(phase_models "$ROOT/repo/$ROUND_DIR")
  export MODELS PHASE
  local status=0
  timeout --signal=TERM --kill-after=60 "$BUDGET" bash "$0" decode-phase "$PHASE" || status=$?
  trap - EXIT TERM INT
  record_run_result "$status"
}

# Sourceable for the local helpers; only a direct run dispatches.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    look|gates|confirm)
      PHASE=$1
      shift
      case "${1:-}" in
        plan|launch|resume|check|fetch) command=$1; shift ;;
        *) usage ;;
      esac
      "$command"
      ;;
    run)
      shift
      PHASE=${1:-}
      valid_phase "$PHASE" || usage
      shift
      run "$@"
      ;;
    decode-phase)
      shift
      PHASE=${1:-}
      valid_phase "$PHASE" || usage
      decode_phase
      ;;
    *) usage ;;
  esac
fi

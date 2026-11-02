#!/usr/bin/env bash
# issue-263-mining-decode.sh — phase A of #263: the base teacher's word times for every look and
# confirm.1/2 gating clip.
#
# Each clip is run whole through the base teacher, its filename ayah's phoneme reference is
# CTC-force-aligned to the posteriors, and the alignment is mapped to Uthmani word boundaries.
# The rows are `tadabur.tlog_mining.selection.phase_a_rows`: the gating stratum of the look and
# the two claimed confirm tranches, cross-checked against the committed split tree. Phases B and
# C are refused — B waits for #262, C for round 3 — and every command that takes a phase refuses
# them before anything is staged.
#
# Round 3 has the box. This launcher never holds the GPU lock across a run: the Python loop
# (`tadabur.tlog_mining.give_way`) takes /root/scratch/gpu.lock non-blocking for one batch, at
# most BATCH_SECONDS, releases it, sleeps at least YIELD_SECONDS, and tries again. A held lock is
# waited out by sleeping, never by blocking. The loop stops at a batch boundary on the PAUSE file
# or when free disk or available RAM cross their floors. The run's line in logs/run.out reports
# the GPU seconds held and the seconds yielded, and the counts the run note cites.
#
# `plan` reads the **committed** choice and the two committed confirm claims and prints the jobs;
# it writes nothing. `launch` stages the repository as a bundle clone on the box and starts `run`
# detached with `setsid nohup`. `pause` touches PAUSE; `resume` removes it, restages and starts
# `run --resume` (the batch resumes by row key from the parts). `status` reads the box's
# run.done/run.exit and the run's line. `fetch` copies a complete merge into
# tools/tadabur/tlog_mining/word_times/ for the commit.
#
# Mac side, from the worktree root:
#   scripts/issue-263-mining-decode.sh plan
#   scripts/issue-263-mining-decode.sh launch
#   scripts/issue-263-mining-decode.sh status
#   scripts/issue-263-mining-decode.sh pause       # give the box to round 3 now
#   scripts/issue-263-mining-decode.sh resume
#   scripts/issue-263-mining-decode.sh fetch
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog-mining}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ROUND_DIR=${ROUND_DIR:-tools/tadabur/consonant2_readout}
TRANCHES_DIR=${TRANCHES_DIR:-tools/tadabur/tlog/tranches}
PARALLEL=${PARALLEL:-2}
BUDGET=${BUDGET:-28800}
BATCH_SECONDS=${BATCH_SECONDS:-300}
YIELD_SECONDS=${YIELD_SECONDS:-30}
DISK_RESERVE_GB=${DISK_RESERVE_GB:-10}
RAM_FLOOR_GB=${RAM_FLOOR_GB:-16}
#: The look gating clips round 2 already retained (#212): hardlinked, never re-downloaded.
LOOK_CLIP_CACHE=${LOOK_CLIP_CACHE:-/root/scratch/tlog/r2/clips}
PR=${PR:-263}
CHOICE=$ROUND_DIR/choice.json
#: The three phase A tranches, in the order the launcher aligns them.
ROLES=(look confirm.1 confirm.2)
PARTS=$ROOT/parts
OUT=$ROOT/out
CLIPS=$ROOT/clips
SHARDS=$ROOT/shards
PAUSE_FILE=$ROOT/PAUSE
#: Where the fetched merge is committed.
WORD_TIMES_DIR=tools/tadabur/tlog_mining/word_times

die() { echo "$*" >&2; exit 1; }

# --- the phase and the committed inputs ---------------------------------------------------------


require_phase() {  # <phase>
  case "$1" in
    A) ;;
    B) die "phase B is not runnable here: it waits for #262's amendment (phase A only)" ;;
    C) die "phase C is not runnable here: it waits for round 3's verdict (phase A only)" ;;
    *) die "unknown phase $1: this launcher runs phase A only" ;;
  esac
}

require_committed() {  # <path>
  git cat-file -e "HEAD:$1" 2>/dev/null || die "$1 is not committed at HEAD"
  git diff --quiet HEAD -- "$1" 2>/dev/null \
    || die "$1 is not committed at HEAD: it changed since its commit"
}

# --- the Mac side -------------------------------------------------------------------------------


plan() {  # [phase]
  require_phase "${1:-A}"
  require_committed "$CHOICE"
  local role
  for role in confirm.1 confirm.2; do
    require_committed "$TRANCHES_DIR/$role.claim.json"
  done
  for role in "${ROLES[@]}"; do
    printf 'align %s base at %s processes\n' "$role" "$PARALLEL"
  done
  printf 'merge word_times\n'
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

start_run() {  # <run args>
  # No flock wrapper: the Python loop takes the lock non-blocking for one batch at a time.
  ssh "$HOST" "ISSUE_ROOT='$ROOT' PARALLEL=$PARALLEL BUDGET=$BUDGET PR=$PR \
    BATCH_SECONDS=$BATCH_SECONDS YIELD_SECONDS=$YIELD_SECONDS \
    DISK_RESERVE_GB=$DISK_RESERVE_GB RAM_FLOOR_GB=$RAM_FLOOR_GB \
    LOOK_CLIP_CACHE='$LOOK_CLIP_CACHE' \
    setsid nohup '$ROOT/repo/scripts/issue-263-mining-decode.sh' run $1 \
    >> '$ROOT/logs/run.out' 2>&1 < /dev/null & echo 'started: '\$!"
}

launch() {  # [phase]
  require_phase "${1:-A}"
  plan "${1:-A}" > /dev/null
  stage_repo
  start_run ""
}

resume() {
  plan A > /dev/null
  ssh "$HOST" "rm -f '$PAUSE_FILE'"
  stage_repo
  start_run --resume
}

pause() {
  ssh "$HOST" "mkdir -p '$ROOT' && touch '$PAUSE_FILE'"
  echo "paused: the loop stops at its next batch boundary ($PAUSE_FILE)"
}

status() {
  local state
  state=$(ssh "$HOST" "if [ ! -f '$ROOT/logs/run.done' ]; then echo pending; \
    elif [ ! -f '$ROOT/logs/run.exit' ]; then echo no-exit; \
    else cat '$ROOT/logs/run.exit'; fi")
  case "$state" in
    pending)
      echo "pending: $ROOT/logs/run.done is absent"
      ssh "$HOST" "tail -n 2 '$ROOT/logs/run.out' 2>/dev/null" || true
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
      ssh "$HOST" "grep -E '^(stopped=|look:|confirm)' '$ROOT/logs/run.out' || true"
      ;;
    3)
      echo "yielded: a gate stopped the loop (pause, disk or RAM); resume restarts it"
      ssh "$HOST" "grep -E '^stopped=' '$ROOT/logs/run.out' || true"
      return 3
      ;;
    *)
      echo "failed: exit $state"
      ssh "$HOST" "cat '$ROOT/logs/run.done' 2>/dev/null; \
        tail -n 20 '$ROOT/logs/run.out'" || true
      return 1
      ;;
  esac
}

require_merged() {
  # ``fetch`` copies the merge as final: a partial run must refuse.
  ssh "$HOST" bash -s <<EOF || die "the box has no complete merge to fetch"
set -e
test -f '$ROOT/logs/run.exit' && test "\$(cat '$ROOT/logs/run.exit')" = 0
test -f '$OUT/meta.json'
for role in ${ROLES[*]}; do
  test -f '$OUT/'\$role'.jsonl.gz'
done
EOF
}

fetch() {
  local repo_root dest
  require_merged
  repo_root=$(git rev-parse --show-toplevel)
  dest=${FETCH_DIR:-$repo_root/$WORD_TIMES_DIR}
  mkdir -p "$dest"
  rsync -a "$HOST:$OUT/" "$dest/"
  echo "word times in $dest: commit them with their provenance"
}

# --- the box side -------------------------------------------------------------------------------


activate() {
  [ -r "$CONDA" ] || { echo "no conda at $CONDA" >&2; return 1; }
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
}

record_run_result() {  # <status>
  # ``status`` keys off done and then reads exit: write exit first. 3 is a gate's clean stop.
  local status=$1 done
  case "$status" in
    0) done=$(printf '%s exit 0' "$(date -u +%FT%TZ)") ;;
    3) done=$(printf '%s stopped at a gate (pause/disk/ram); parts kept, resume restarts' \
         "$(date -u +%FT%TZ)") ;;
    *) done=$(printf '%s exit %s' "$(date -u +%FT%TZ)" "$status") ;;
  esac
  printf '%s\n' "$status" > "$ROOT/logs/run.exit"
  printf '%s\n' "$done" > "$ROOT/logs/run.done"
  return "$status"
}

run() {  # [--resume]
  local resume=0
  case "${1:-}" in
    --resume) resume=1 ;;
    "") ;;
    *) die "unknown run argument: $1" ;;
  esac
  # A signal's EXIT trap reads the last command's status, so translate TERM and INT here.
  trap 'exit 143' TERM
  trap 'exit 130' INT
  trap 'record_run_result "$?"' EXIT
  mkdir -p "$PARTS" "$OUT" "$CLIPS" "$SHARDS" "$ROOT/logs"
  rm -f "$ROOT/logs/run.done" "$ROOT/logs/run.exit"
  cd "$ROOT/repo/tools"
  activate
  local status=0 flags=""
  if [ "$resume" -eq 1 ]; then flags="--resume"; fi
  python -m tadabur.tlog_mining run --parts-dir "$PARTS" --clip-dir "$CLIPS" \
    --shard-cache "$SHARDS" --pause-file "$PAUSE_FILE" --lock /root/scratch/gpu.lock \
    --parallel "$PARALLEL" --batch-seconds "$BATCH_SECONDS" --yield-seconds "$YIELD_SECONDS" \
    --disk-reserve-gb "$DISK_RESERVE_GB" --ram-floor-gb "$RAM_FLOOR_GB" \
    --look-clip-cache "$LOOK_CLIP_CACHE" $flags || status=$?
  if [ "$status" -eq 0 ]; then
    python -m tadabur.tlog_mining merge --parts-dir "$PARTS" --out-dir "$OUT" \
      --pull-request "$PR" || status=$?
  fi
  trap - EXIT TERM INT
  record_run_result "$status"
}

# Sourceable for the local helpers; only a direct run dispatches.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    plan) shift; plan "${1:-A}" ;;
    launch) shift; launch "${1:-A}" ;;
    resume) resume ;;
    pause) pause ;;
    status) status ;;
    fetch) fetch ;;
    run) shift; run "$@" ;;
    *)
      echo "usage: $0 {plan [A]|launch [A]|resume|pause|status|fetch}
       (run [--resume] executes on the box)" >&2
      exit 2
      ;;
  esac
fi

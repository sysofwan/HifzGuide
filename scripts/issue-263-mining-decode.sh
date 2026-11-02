#!/usr/bin/env bash
# issue-263-mining-decode.sh — #263's idle-time mining, phases A and B, in one preemptible
# launcher.
#
# Phase A (the default): each look and confirm.1/2 gating clip is run whole through the base
# teacher, its filename ayah's phoneme reference is CTC-force-aligned to the posteriors, and the
# alignment is mapped to Uthmani word boundaries. The rows are
# `tadabur.tlog_mining.selection.phase_a_rows`: the gating stratum of the look and the two
# claimed confirm tranches, cross-checked against the committed split tree.
#
# Phase B: the labelled `reported:*` rows of those two confirm tranches (straddles, identified,
# below-bar, unclean-identified — never an unlabelled unclean row) are decoded whole by the
# teacher, C95 and round 2's EMA candidate into `checks` and `spans`. The rows whose
# `assigned_label` the phonetizer cannot read (#248) are listed in the meta, not decoded. B runs
# only under slice 1's amendment at its pinned SHA-256, and only when the committed choice still
# names `r2_candidate_ema` as the candidate (the decoder refuses any other producer set).
#
# Phase C waits for round 3's verdict and its committed live/EMA choice, and is refused by every
# command that takes a phase, as is any other tranche: `plan`/`launch`/`resume` check the phase
# before they stage anything.
#
# Round 3 has the box. This launcher never holds the GPU lock across a run: the Python loop
# (`tadabur.tlog_mining.give_way`) takes /root/scratch/gpu.lock non-blocking for one batch, at
# most BATCH_SECONDS, releases it, sleeps at least YIELD_SECONDS, and tries again. A held lock is
# waited out by sleeping, never by blocking. The loop stops at a batch boundary on the PAUSE file
# or when free disk or available RAM cross their floors. The run's line in the phase's
# logs/run.out reports the GPU seconds held and the seconds yielded, and the counts the run note
# cites.
#
# `plan [A|B]` reads the **committed** choice, the two committed confirm claims and (for B) the
# committed amendment, and prints the jobs; it writes nothing. `launch [A|B]` stages the
# repository as a bundle clone on the box and starts `run` detached with `setsid nohup`.
# `pause` touches PAUSE; `resume [A|B]` removes it, restages and starts `run --resume` (the batch
# resumes by row key from the parts). `status [A|B]` reads the box's run.done/run.exit and the
# run's line. `fetch [A|B]` copies a complete merge into the phase's committed directory.
#
# Each phase has its own parts, clips, logs and output directory under $ROOT (A) and $ROOT/B
# (phase B), so no part, done or exit file of one phase is ever read as the other's. The staged
# clone at $ROOT/repo **is** shared, so `launch`/`resume` refuse while either phase's loop is
# still live.
#
# Two interpreters, two requirements. Mac side: `plan`/`launch`/`resume` read the committed
# choice and the pinned amendment through `python`, which must be a `python` from the checkout's
# requirements (it imports `tadabur.tlog_mining.selection`, i.e. numpy) — the same interpreter
# `scripts/test.sh` resolves. Box side: `run` is a non-interactive ssh shell, so it activates the
# conda env (`$CONDA`) before it reads anything through `git` or `python`; a refusal there writes
# the phase's `run.exit`/`run.done`, so `status` reports it as a failed run, not as pending.
#
# Mac side, from the worktree root:
#   scripts/issue-263-mining-decode.sh plan B
#   scripts/issue-263-mining-decode.sh launch B
#   scripts/issue-263-mining-decode.sh status B
#   scripts/issue-263-mining-decode.sh pause       # give the box to round 3 now
#   scripts/issue-263-mining-decode.sh resume B
#   scripts/issue-263-mining-decode.sh fetch B
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog-mining}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ROUND_DIR=${ROUND_DIR:-tools/tadabur/consonant2_readout}
TRANCHES_DIR=${TRANCHES_DIR:-tools/tadabur/tlog/tranches}
PARALLEL=${PARALLEL:-2}
BATCH_SECONDS=${BATCH_SECONDS:-300}
YIELD_SECONDS=${YIELD_SECONDS:-30}
DISK_RESERVE_GB=${DISK_RESERVE_GB:-10}
RAM_FLOOR_GB=${RAM_FLOOR_GB:-16}
#: The look gating clips round 2 already retained (#212): hardlinked, never re-downloaded. The
#: same flag reaches either phase's worker; a clip the cache does not hold is streamed once.
LOOK_CLIP_CACHE=${LOOK_CLIP_CACHE:-/root/scratch/tlog/r2/clips}
#: The pull request that will carry the merge, when one exists; empty records no ref. The issue
#: number is not a PR: `refs/pull/263/head` would 404, and the branch SHA is the ref until then.
PR=${PR:-}
CHOICE=$ROUND_DIR/choice.json
#: Slice 1's amendment: phase B refuses anything else, at the bytes the code pins.
AMENDMENT=docs/tlog-mining-amendment.md
#: The frozen producer prefix phase B's choice extends: the teacher and the baseline.
PREFIX_MODELS=base,c95
#: The candidate phase B decodes at — `MINED_MODELS` in `tadabur.tlog_mining.mined_decode`.
#: Another choice's candidate is a different run: the merged files are named for these weights.
MINED_CANDIDATE=r2_candidate_ema
#: This script's own checkout, for the pinned amendment. The committed inputs are read from the
#: current repository (the caller's worktree); the pin is the one this code carries.
CODE_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools" && pwd)
#: One pause file for the whole launcher: the orchestrator touches it before any round-3 launch.
PAUSE_FILE=$ROOT/PAUSE
#: The phase-scoped paths, set by `phase_dirs`: PARTS, OUT, CLIPS, SHARDS, LOGS, ROLES and the
#: merged output names MODELS (phase B) — never shared between phases.
PARTS=
OUT=
CLIPS=
SHARDS=
LOGS=
MODELS=

die() { echo "$*" >&2; exit 1; }

# --- the phase and the committed inputs ---------------------------------------------------------


require_phase() {  # <phase>
  case "$1" in
    A) ;;
    B) require_amendment ;;
    C) die "phase C is not runnable here: it waits for round 3's verdict and its committed \
live/EMA choice (phases A and B only)" ;;
    *) die "unknown phase $1: this launcher runs phase A and phase B only" ;;
  esac
}

require_amendment() {
  local found
  git cat-file -e "HEAD:$AMENDMENT" 2>/dev/null \
    || die "$AMENDMENT is not committed at HEAD: phase B needs the mining amendment at its \
pinned SHA-256"
  found=$(git show "HEAD:$AMENDMENT" | PYTHONPATH="$CODE_TOOLS" python -c '
import hashlib, sys
from tadabur.tlog_mining.selection import MINING_AMENDMENT_SHA256
found = hashlib.sha256(sys.stdin.buffer.read()).hexdigest()
print("" if found == MINING_AMENDMENT_SHA256 else found)
') || die "the pinned amendment SHA-256 in $CODE_TOOLS cannot be read: phase B refuses"
  [ -z "$found" ] \
    || die "$AMENDMENT at HEAD is $found, not the pinned SHA-256: phase B needs slice 1's \
amendment pinned"
}

require_committed() {  # <path>
  git cat-file -e "HEAD:$1" 2>/dev/null || die "$1 is not committed at HEAD"
  git diff --quiet HEAD -- "$1" 2>/dev/null \
    || die "$1 is not committed at HEAD: it changed since its commit"
}

phase_dirs() {  # <phase>: set the phase's paths and roles
  case "$1" in
    A)
      PARTS=$ROOT/parts
      OUT=$ROOT/out
      CLIPS=$ROOT/clips
      SHARDS=$ROOT/shards
      LOGS=$ROOT/logs
      ROLES=(look confirm.1 confirm.2)
      ;;
    B)
      PARTS=$ROOT/B/parts
      OUT=$ROOT/B/out
      CLIPS=$ROOT/B/clips
      SHARDS=$ROOT/B/shards
      LOGS=$ROOT/B/logs
      ROLES=(confirm.1 confirm.2)
      ;;
  esac
}

# --- the Mac side -------------------------------------------------------------------------------


plan() {  # [phase]
  local phase=${1:-A}
  require_phase "$phase"
  require_committed "$CHOICE"
  local role
  for role in confirm.1 confirm.2; do
    require_committed "$TRANCHES_DIR/$role.claim.json"
  done
  phase_dirs "$phase"
  case "$phase" in
    A)
      for role in "${ROLES[@]}"; do
        printf 'align %s base at %s processes\n' "$role" "$PARALLEL"
      done
      printf 'merge word_times\n'
      ;;
    B)
      require_candidate
      MODELS=$(chosen_models)
      for role in "${ROLES[@]}"; do
        printf 'decode %s %s at %s processes\n' "$role" "$MODELS" "$PARALLEL"
      done
      printf 'merge mined\n'
      ;;
  esac
}

chosen_models() {
  # The teacher, the baseline and the candidate the committed round chose, by name.
  git show "HEAD:$CHOICE" | python -c '
import json, sys
candidate = json.load(sys.stdin)["models"]["candidate"]
print(",".join([*sys.argv[1].split(","), candidate]))
' "$PREFIX_MODELS"
}

require_candidate() {
  # Phase B's decode set is frozen at round 2's EMA candidate. A choice that names another one
  # is a different run (other weights, other file names), and the decoder refuses it inside the
  # first worker — after the clone, the model loads and a lock-held batch. Refuse it here.
  local candidate
  candidate=$(git show "HEAD:$CHOICE" | python -c \
    'import json, sys; print(json.load(sys.stdin)["models"]["candidate"])')
  [ "$candidate" = "$MINED_CANDIDATE" ] \
    || die "the committed choice's candidate is $candidate, not phase B's $MINED_CANDIDATE: \
phase B decodes round 2's EMA weights only"
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
  ssh "$HOST" "mkdir -p '$LOGS' && rm -rf '$ROOT/repo' '$ROOT/repo.bundle'"
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

start_run() {  # <phase> <run flags>
  # No flock wrapper: the Python loop takes the lock non-blocking for one batch at a time. The
  # phase's log directory is made before the redirect opens: `status` reads that directory, and
  # a phase B run's first start is its first use.
  ssh "$HOST" "mkdir -p '$LOGS' && ISSUE_ROOT='$ROOT' PARALLEL=$PARALLEL PR=$PR \
    BATCH_SECONDS=$BATCH_SECONDS YIELD_SECONDS=$YIELD_SECONDS \
    DISK_RESERVE_GB=$DISK_RESERVE_GB RAM_FLOOR_GB=$RAM_FLOOR_GB \
    LOOK_CLIP_CACHE='$LOOK_CLIP_CACHE' \
    setsid nohup '$ROOT/repo/scripts/issue-263-mining-decode.sh' run $2 --phase $1 \
    >> '$LOGS/run.out' 2>&1 < /dev/null & echo 'started: '\$!"
}

live_pid() {  # <logs dir>
  # The box's loop pid for that phase's logs, when the process is still alive (a pause lands at
  # its next boundary).
  ssh "$HOST" "if [ -f '$1/run.pid' ]; then pid=\$(cat '$1/run.pid' 2>/dev/null); \
    if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then echo \"\$pid\"; fi; fi"
}

require_stopped() {
  # The clone is shared by both phases (`stage_repo` replaces `$ROOT/repo`), so a live run of
  # *either* phase must block a restage: deleting the tree under a running loop breaks its next
  # batch.
  local dir pid
  for dir in "$ROOT/logs" "$ROOT/B/logs"; do
    pid=$(live_pid "$dir")
    [ -z "$pid" ] || die "a mining run is still live (pid $pid, logs in $dir): pause it and \
      wait for run.exit, then resume"
  done
}

launch() {  # [phase]
  local phase=${1:-A}
  plan "$phase" > /dev/null
  phase_dirs "$phase"
  require_stopped
  stage_repo
  start_run "$phase" ""
}

resume() {  # [phase]
  local phase=${1:-A}
  plan "$phase" > /dev/null
  phase_dirs "$phase"
  require_stopped
  ssh "$HOST" "rm -f '$PAUSE_FILE'"
  stage_repo
  start_run "$phase" --resume
}

pause() {
  ssh "$HOST" "mkdir -p '$ROOT' && touch '$PAUSE_FILE'"
  echo "paused: the loop stops at its next batch boundary ($PAUSE_FILE)"
}

status() {  # [phase]
  local phase=${1:-A} state
  require_phase "$phase"
  phase_dirs "$phase"
  state=$(ssh "$HOST" "if [ ! -f '$LOGS/run.done' ]; then echo pending; \
    elif [ ! -f '$LOGS/run.exit' ]; then echo no-exit; \
    else cat '$LOGS/run.exit'; fi")
  case "$state" in
    pending)
      echo "pending: $LOGS/run.done is absent"
      ssh "$HOST" "tail -n 2 '$LOGS/run.out' 2>/dev/null" || true
      return 3
      ;;
    no-exit)
      echo "failed: $LOGS/run.done exists without $LOGS/run.exit"
      ssh "$HOST" "cat '$LOGS/run.done' 2>/dev/null; \
        tail -n 20 '$LOGS/run.out' 2>/dev/null" || true
      return 1
      ;;
    0)
      echo "done: exit 0"
      ssh "$HOST" "grep -E \
        '^(stopped=|look:|confirm\\.|base:|c95:|r2_|label sources:|unphonetizable:)' \
        '$LOGS/run.out' || true"
      ;;
    3)
      echo "yielded: a gate stopped the loop (pause, disk or RAM); resume restarts it"
      ssh "$HOST" "grep -E '^stopped=' '$LOGS/run.out' || true"
      return 3
      ;;
    *)
      echo "failed: exit $state"
      ssh "$HOST" "cat '$LOGS/run.done' 2>/dev/null; \
        tail -n 20 '$LOGS/run.out'" || true
      return 1
      ;;
  esac
}

output_names() {  # <phase>: the merged files a complete run leaves in its out dir
  local model
  case "$1" in
    A) printf '%s.jsonl.gz\n' "${ROLES[@]}" ;;
    B) for model in ${MODELS//,/ }; do printf 'tlog.mined.%s.jsonl.gz\n' "$model"; done ;;
  esac
}

require_merged() {
  # ``fetch`` copies the merge as final: a partial run must refuse.
  ssh "$HOST" bash -s <<EOF || die "the box has no complete merge to fetch"
set -e
test -f '$LOGS/run.exit' && test "\$(cat '$LOGS/run.exit')" = 0
test -f '$OUT/meta.json'
for name in $(output_names "$1"); do test -f '$OUT/'\$name; done
EOF
}

fetch() {  # [phase]
  local phase=${1:-A} repo_root dest default_dir
  require_phase "$phase"
  phase_dirs "$phase"
  if [ "$phase" = B ]; then
    MODELS=$(chosen_models)
    default_dir=tools/tadabur/tlog_mining/decodes
  else
    default_dir=tools/tadabur/tlog_mining/word_times
  fi
  require_merged "$phase"
  repo_root=$(git rev-parse --show-toplevel)
  dest=${FETCH_DIR:-$repo_root/$default_dir}
  mkdir -p "$dest"
  rsync -a "$HOST:$OUT/" "$dest/"
  echo "phase $phase outputs in $dest: commit them with their provenance"
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
  printf '%s\n' "$status" > "$LOGS/run.exit"
  printf '%s\n' "$done" > "$LOGS/run.done"
  return "$status"
}

run() {  # [--resume] [--phase A|B]
  local resume=0 phase=A status=0 flags="" merge_pr=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --resume) resume=1 ;;
      --phase) shift; phase=$1 ;;
      *) die "unknown run argument: $1" ;;
    esac
    shift
  done
  phase_dirs "$phase"
  # A signal's EXIT trap reads the last command's status, so translate TERM and INT here. The
  # trap, the pid and the log directory come before *any* check: a refusal (the conda env, the
  # clone, the phase, the choice) is a failed run, and `status` must see its exit file instead of
  # "pending" forever.
  trap 'exit 143' TERM
  trap 'exit 130' INT
  trap 'record_run_result "$?"' EXIT
  mkdir -p "$PARTS" "$OUT" "$CLIPS" "$SHARDS" "$LOGS"
  printf '%s\n' "$$" > "$LOGS/run.pid"
  rm -f "$LOGS/run.done" "$LOGS/run.exit"
  # The box's interpreter comes before anything read through `python`: this shell is
  # non-interactive (setsid from ssh), so a bare `python` is whatever sshd's PATH holds. The
  # phase check reads the pinned amendment through the tools (numpy included) and must not
  # decide on that PATH.
  activate
  # Phase B's amendment check reads HEAD of the *staged* clone on the box (and of the caller's
  # worktree on the Mac side), so the phase is read from inside the repository, not the ssh
  # session's home directory.
  cd "$ROOT/repo" || die "the box's clone $ROOT/repo is missing: stage the repository first"
  require_phase "$phase"
  if [ "$phase" = B ]; then require_candidate; fi
  cd "$ROOT/repo/tools"
  if [ "$resume" -eq 1 ]; then flags="--resume"; fi
  python -m tadabur.tlog_mining run --phase "$phase" --parts-dir "$PARTS" --clip-dir "$CLIPS" \
    --shard-cache "$SHARDS" --pause-file "$PAUSE_FILE" --lock /root/scratch/gpu.lock \
    --parallel "$PARALLEL" --batch-seconds "$BATCH_SECONDS" --yield-seconds "$YIELD_SECONDS" \
    --disk-reserve-gb "$DISK_RESERVE_GB" --ram-floor-gb "$RAM_FLOOR_GB" \
    --look-clip-cache "$LOOK_CLIP_CACHE" $flags || status=$?
  if [ "$status" -eq 0 ]; then
    # Only a real PR gets a ref; an issue-only run records the branch SHA and no pull request.
    if [ -n "$PR" ]; then merge_pr="--pull-request $PR"; fi
    python -m tadabur.tlog_mining merge --phase "$phase" --parts-dir "$PARTS" --out-dir "$OUT" \
      $merge_pr || status=$?
  fi
  trap - EXIT TERM INT
  record_run_result "$status"
}

# Sourceable for the local helpers; only a direct run dispatches.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    plan) shift; plan "${1:-A}" ;;
    launch) shift; launch "${1:-A}" ;;
    resume) shift; resume "${1:-A}" ;;
    pause) pause ;;
    status) shift; status "${1:-A}" ;;
    fetch) shift; fetch "${1:-A}" ;;
    run) shift; run "$@" ;;
    *)
      echo "usage: $0 {plan [A|B]|launch [A|B]|resume [A|B]|pause|status [A|B]|fetch [A|B]}
       (run [--phase A|B] [--resume] executes on the box)" >&2
      exit 2
      ;;
  esac
fi

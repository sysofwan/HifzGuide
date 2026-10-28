#!/usr/bin/env bash
# issue-212-decode.sh — the tlog look decode and its box clip cache on cuda-dev (#212).
#
# Mac side, from the worktree root:
#   scripts/issue-212-decode.sh decode launch    # gating rows, 3 models, checks, 6 x OMP=1
#   scripts/issue-212-decode.sh decode check     # 0 = done, 3 = the gate stopped it
#   scripts/issue-212-decode.sh decode fetch DIR # pull the merged decodes and meta.json
#   scripts/issue-212-decode.sh decode merge     # re-run the merge over the parts already there
#   scripts/issue-212-decode.sh cache launch     # retain the gating clips in $CLIPS
#   scripts/issue-212-decode.sh cache check
#
# The run is the look tranche's 2,853 `gating` rows (the frozen manifest's stratum) decoded by
# the teacher, C95 and the #178 candidate's EMA, `checks` only, as PROCESSES processes with
# OMP_NUM_THREADS=1 each (round 1 measured that configuration at ~3x this box's serial
# per-model rate). After GATE_WAIT seconds it projects the selection from the wall the run has
# taken, or — when fewer than GATE_ROWS timings arrived and workers are still live — it reaps
# them and exits 3 rather than run ungated; over LIMIT_HOURS it does the same. Either way it
# writes `logs/decode.needs-post`, so the orchestrator decides — the architect's fallback is
# the teacher and C95 alone. Nothing is graded or read.
#
# The cache job streams the same shards once, keeps every selected clip under its recorded
# checksum in $CLIPS, and deletes the shard; a later decode pass reads the clips instead of the
# source. $CLIPS is outside the repo and outside what `fetch` copies. It decodes and grades
# nothing, so it does not hold gpu.lock.
#
# Either job stages the current branch as a bundle on the box, writes one log, and writes its
# exit file last, so `check` never reads a log tail. It runs under setsid with stdin closed, so
# it survives the ssh session; the decode runner holds /root/scratch/gpu.lock. Shards are
# downloaded into $ROOT/cache and deleted once read; no audio leaves the box. `launch` (and a
# merge) refuses unless PR #211 is merged (origin/main holds the look manifest), the worktree is
# clean, and neither job's runner is live — both run from the one staged checkout, so staging
# under either would pull it out from under that runner. The fresh parts directory is the
# default: thread settings are never mixed in one output.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog/r2}
LOGS=$ROOT/logs
REPO=$ROOT/repo
CACHE=$ROOT/cache
# The retained look clips: box scratch, never the repo or the fetched output directory.
CLIPS=${CLIP_CACHE_DIR:-$ROOT/clips}
PARTS=${PARTS_DIR:-$ROOT/parts-gating-o1x6}
OUT=tools/tadabur/consonant2_readout/decodes
MANIFEST=tools/tadabur/tlog/tranches/look.jsonl
MODELS=${MODELS:-base,c95,candidate_ema}
PROCESSES=${PROCESSES:-6}
THREADS=${THREADS:-1}
GATE_ROWS=${GATE_ROWS:-12}
LIMIT_HOURS=${LIMIT_HOURS:-11}
# Seconds the gate waits for GATE_ROWS timing lines before projecting what it has.
GATE_WAIT=${GATE_WAIT:-1200}
# The PR that keeps the cited code SHA resolvable after the squash merge (refs/pull/<N>/head).
PR=${PR:-}

job=${1:-}
action=${2:-}
case "$job" in
  decode|cache) NAME=$job ;;
  *) echo "usage: $0 {decode|cache} [launch|check]" >&2
     exit 2 ;;
esac

remote() { ssh -n "$HOST" "$@"; }

pr_args() {
  if [ -n "$PR" ]; then printf -- '--pr %s' "$PR"; fi
}

stage_repo() {
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT' '$LOGS' '$CACHE' '$PARTS' && rm -rf '$REPO' '$ROOT/repo.bundle'"
  git -C "$REPO_DIR" bundle create - "$branch" | ssh "$HOST" "cat > '$ROOT/repo.bundle'"
  if [ "$branch" = HEAD ]; then
    remote "git clone -q '$ROOT/repo.bundle' '$REPO'"
  else
    remote "git clone -q -b '$branch' '$ROOT/repo.bundle' '$REPO'"
  fi
  remote "rm -f '$ROOT/repo.bundle' && test \"\$(git -C '$REPO' rev-parse HEAD)\" = '$sha'" \
    || { echo "the box's clone is not $sha" >&2; return 1; }
}

write_runner() {
  local manifest_commit=$1
  remote "cat > '$ROOT/run-$NAME.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
run() {
  local start=\$SECONDS
  source '$CONDA' || return 1
  conda activate $ENV_NAME || return 1
  cd '$REPO/tools' || return 1
  export OMP_NUM_THREADS=$THREADS
  local pids=() index lines gate_code failed pid
  for index in \$(seq 0 $((PROCESSES - 1))); do
    python -u -m tadabur.tlog_eval_decode decode --parts-dir '$PARTS' --cache-dir '$CACHE' \\
        --clip-cache '$CLIPS' --models ${MODELS//,/ } \\
        --shard-index \"\$index\" --shard-count $PROCESSES &
    pids+=(\"\$!\")
  done
  lines=0
  for _ in \$(seq 1 $((GATE_WAIT / 15))); do
    lines=\$(wc -l < '$PARTS/timing.jsonl' 2>/dev/null || echo 0)
    [ \"\$lines\" -ge $GATE_ROWS ] && break
    [ \"\$(jobs -pr | wc -l)\" -eq 0 ] && break
    sleep 15
  done
  lines=\$(wc -l < '$PARTS/timing.jsonl' 2>/dev/null || echo 0)
  lines=\${lines// /}
  if [ \"\$lines\" -ge $GATE_ROWS ]; then
    python -u -m tadabur.tlog_eval_decode gate --parts-dir '$PARTS' \\
        --elapsed-seconds \$((SECONDS - start)) --limit-hours $LIMIT_HOURS \\
        --processes $PROCESSES --threads $THREADS
    gate_code=\$?
    if [ \"\$gate_code\" -eq 3 ]; then
      kill \"\${pids[@]}\" 2>/dev/null
      wait
      return 3
    elif [ \"\$gate_code\" -ne 0 ]; then
      kill \"\${pids[@]}\" 2>/dev/null
      wait
      return 1
    fi
  elif [ \"\$(jobs -pr | wc -l)\" -gt 0 ]; then
    kill \"\${pids[@]}\" 2>/dev/null
    wait
    printf 'insufficient sample: %s timing lines in %s s, need %s\\n' \\
        \"\$lines\" \$((SECONDS - start)) $GATE_ROWS > '$LOGS/$NAME.needs-post'
    return 3
  fi
  failed=0
  for pid in \"\${pids[@]}\"; do wait \"\$pid\" || failed=1; done
  [ \"\$failed\" -eq 0 ] || return 1
  python -u -m tadabur.tlog_eval_decode merge --parts-dir '$PARTS' --out-dir '$REPO/$OUT' \\
      --models ${MODELS//,/ } --processes $PROCESSES --threads $THREADS $(pr_args) \\
      --elapsed-seconds \$((SECONDS - start)) --manifest-commit $manifest_commit
}
run
code=\$?
if [ \$code -eq 3 ]; then touch '$LOGS/$NAME.needs-post'; fi
echo \"\$code\" > '$LOGS/$NAME.exit'
exit \$code
SH
chmod +x '$ROOT/run-$NAME.sh'"
}

write_cache_runner() {
  remote "cat > '$ROOT/run-$NAME.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
run() {
  source '$CONDA' || return 1
  conda activate $ENV_NAME || return 1
  cd '$REPO/tools' || return 1
  export OMP_NUM_THREADS=$THREADS
  local pids=() index failed pid
  for index in \$(seq 0 $((PROCESSES - 1))); do
    python -u -m tadabur.tlog_eval_decode cache --clip-cache '$CLIPS' --shard-cache '$CACHE' \\
        --shard-index \"\$index\" --shard-count $PROCESSES &
    pids+=(\"\$!\")
  done
  failed=0
  for pid in \"\${pids[@]}\"; do wait \"\$pid\" || failed=1; done
  python -u -m tadabur.tlog_eval_decode coverage --clip-cache '$CLIPS' || failed=1
  [ \"\$failed\" -eq 0 ]
}
run
code=\$?
echo \"\$code\" > '$LOGS/$NAME.exit'
exit \$code
SH
chmod +x '$ROOT/run-$NAME.sh'"
}

require_manifest_on_main() {
  git -C "$REPO_DIR" fetch -q origin main
  git -C "$REPO_DIR" cat-file -e "origin/main:$MANIFEST" 2>/dev/null \
    || { echo "PR #211 is not merged: origin/main does not hold $MANIFEST" >&2; return 1; }
  git -C "$REPO_DIR" log -1 --format=%H origin/main -- "$MANIFEST"
}

# Both jobs run from the one checked-out $REPO, and staging replaces it. Every stage_repo is
# guarded by this check, for both runner names, so a live job is never pulled out from under
# its own imports; RESUME deliberately does not bypass it.
refuse_if_live() {
  local other
  for other in decode cache; do
    if remote "test -f '$LOGS/$other.pid' && test ! -f '$LOGS/$other.exit' \\
               && kill -0 \"\$(cat '$LOGS/$other.pid')\" 2>/dev/null"; then
      echo "the $other runner is live on $HOST; not replacing $REPO under it" >&2
      return 1
    fi
  done
}

# The checks every launch shares: a clean worktree, no live runner of either job, no second
# run in a parts dir ("parts" only for the decode job), and PR #211 on main. Echoes the
# manifest commit.
preflight() {
  [ -z "$(git -C "$REPO_DIR" status --porcelain)" ] \
    || { echo "the worktree is dirty; commit before launching" >&2; return 1; }
  refuse_if_live || return 1
  if [ "${RESUME:-0}" != 1 ]; then
    if [ "${1:-}" = parts ] && remote "test -f '$PARTS/timing.jsonl'"; then
      echo "$HOST:$PARTS already holds timings; pass PARTS_DIR=<fresh dir> " \
           "(RESUME=1 to continue it)" >&2
      return 1
    fi
  fi
  require_manifest_on_main
}

# Detach the runner that write_*_runner just staged; $1 is a lock file to hold, if any.
start_job() {
  local launch_command
  if [ -n "${1:-}" ]; then
    launch_command="flock '$1' bash '$ROOT/run-$NAME.sh'"
  else
    launch_command="bash '$ROOT/run-$NAME.sh'"
  fi
  remote "rm -f '$LOGS/$NAME.exit' '$LOGS/$NAME.needs-post' '$LOGS/$NAME.pid'
          setsid $launch_command </dev/null >'$LOGS/$NAME.log' 2>&1 &
          echo \$! > '$LOGS/$NAME.pid'"
  echo "job started on $HOST"
  echo "  pid:  $(remote "cat '$LOGS/$NAME.pid'")"
  echo "  log:  $HOST:$LOGS/$NAME.log"
  echo "  exit: $HOST:$LOGS/$NAME.exit (0 = clean; 3 = the gate stopped it; absent while running)"
  echo "  check with: scripts/issue-212-decode.sh $job check"
}

launch_decode() {
  local manifest_commit
  manifest_commit=$(preflight parts) || return 1
  remote "mkdir -p '$ROOT' '$LOGS' '$CACHE' '$PARTS'"
  stage_repo
  write_runner "$manifest_commit"
  start_job /root/scratch/gpu.lock
}

launch_cache() {
  preflight >/dev/null || return 1
  remote "mkdir -p '$ROOT' '$LOGS' '$CACHE' '$CLIPS'"
  stage_repo
  write_cache_runner
  start_job
}

check() {
  if remote "test -f '$LOGS/$NAME.needs-post'"; then
    echo "needs-post: $(remote "cat '$LOGS/$NAME.needs-post'")"
    remote "cat '$PARTS/gate_timing.json' 2>/dev/null || true"
  fi
  if remote "test -f '$LOGS/$NAME.exit'"; then
    echo "exit $(remote "cat '$LOGS/$NAME.exit'")"
  else
    echo "running (pid $(remote "cat '$LOGS/$NAME.pid' 2>/dev/null || echo unknown"))"
    remote "ps -p \"\$(cat '$LOGS/$NAME.pid')\" -o pid=,etime=,stat= 2>/dev/null || true"
  fi
}

merge_job() {
  local manifest_commit elapsed
  [ -z "$(git -C "$REPO_DIR" status --porcelain)" ] \
    || { echo "the worktree is dirty; commit before merging" >&2; return 1; }
  refuse_if_live || return 1
  manifest_commit=$(require_manifest_on_main) \
    || { echo "PR #211 is not merged: origin/main does not hold $MANIFEST" >&2; return 1; }
  elapsed=''
  [ -n "${ELAPSED_SECONDS:-}" ] && elapsed="--elapsed-seconds $ELAPSED_SECONDS"
  stage_repo
  remote "source '$CONDA' && conda activate $ENV_NAME && cd '$REPO/tools' && \\
          OMP_NUM_THREADS=$THREADS python -u -m tadabur.tlog_eval_decode merge \\
          --parts-dir '$PARTS' --out-dir '$REPO/$OUT' --models ${MODELS//,/ } \\
          --processes $PROCESSES --threads $THREADS $(pr_args) \\
          --manifest-commit $manifest_commit $elapsed" \
    || return 1
  remote "echo 0 > '$LOGS/$NAME.exit'"
  echo "merged into $HOST:$REPO/$OUT; $LOGS/$NAME.exit is 0"
}

fetch() {
  local destination=${1:?fetch needs a destination directory}
  mkdir -p "$destination"
  rsync -a "$HOST:$REPO/$OUT/" "$destination/"
  echo "fetched $job into $destination"
}

case "$action" in
  launch) "launch_$NAME" ;;
  check) check ;;
  fetch) [ "$NAME" = decode ] || { echo "fetch is decode-only" >&2; exit 2; }
         fetch "${3:-}" ;;
  merge) [ "$NAME" = decode ] || { echo "merge is decode-only" >&2; exit 2; }
         merge_job ;;
  "") "launch_$NAME" ;;
  *) echo "usage: $0 {decode|cache} [launch|check] (decode also: merge|fetch DIR)" >&2
     exit 2 ;;
esac

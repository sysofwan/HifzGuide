#!/usr/bin/env bash
# issue-212-decode.sh — the tlog look decode of the four round-2 models on cuda-dev (#212).
#
# Mac side, from the worktree root:
#   scripts/issue-212-decode.sh slice launch    # the timed 500-row slice; never the full pass
#   scripts/issue-212-decode.sh slice check     # 0 = clean, 3 = over the 8 h budget (needs-post)
#   scripts/issue-212-decode.sh slice fetch DIR # pull slice_timing.json back
#   scripts/issue-212-decode.sh decode launch   # after PR #211 merges and the slice passes
#   scripts/issue-212-decode.sh decode check
#   scripts/issue-212-decode.sh decode fetch DIR # pull the merged decodes and meta.json
#
# The job stages the current branch as a bundle on the box, writes one log, and writes its exit
# file last, so `check` never reads a log tail. It runs under setsid with stdin closed, so it
# survives the ssh session, and the whole runner holds /root/scratch/gpu.lock: the slice and
# the full pass both decode on the GPU. Logs carry aggregate counts only.
#
# The look shards are cached in $ROOT/cache and deleted once read; no audio leaves the box.
# `decode launch` refuses unless PR #211 is merged (origin/main holds the look manifest), the
# worktree is clean, and the slice's projection fits the 8 h budget. A slice over budget
# writes `logs/slice.needs-post` and exits 3: the orchestrator decides, nothing auto-continues.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog/r2}
LOGS=$ROOT/logs
REPO=$ROOT/repo
CACHE=$ROOT/cache
PARTS=$ROOT/parts
OUT=tools/tadabur/consonant2_readout/decodes
MANIFEST=tools/tadabur/tlog/tranches/look.jsonl
PARALLEL=${PARALLEL:-2}
SLICE_ROWS=${SLICE_ROWS:-500}
# The PR that keeps the cited code SHA resolvable after the squash merge (refs/pull/<N>/head).
PR=${PR:-}

job=${1:-}
action=${2:-}
case "$job" in
  slice)  NAME=slice ;;
  decode) NAME=decode ;;
  *) echo "usage: $0 slice|decode [launch|check|fetch DIR]" >&2
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

write_slice_runner() {
  remote "cat > '$ROOT/run-$NAME.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
run() {
  source '$CONDA' || return 1
  conda activate $ENV_NAME || return 1
  cd '$REPO/tools' || return 1
  python -u -m tadabur.tlog_eval_decode slice --parts-dir '$PARTS' --cache-dir '$CACHE' \\
      --rows $SLICE_ROWS
}
run
code=\$?
if [ \$code -eq 3 ]; then touch '$LOGS/$NAME.needs-post'; fi
echo \"\$code\" > '$LOGS/$NAME.exit'
exit \$code
SH
chmod +x '$ROOT/run-$NAME.sh'"
}

write_decode_runner() {
  remote "cat > '$ROOT/run-$NAME.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
run() {
  source '$CONDA' || return 1
  conda activate $ENV_NAME || return 1
  cd '$REPO/tools' || return 1
  pids=()
  for index in \$(seq 0 $((PARALLEL - 1))); do
    python -u -m tadabur.tlog_eval_decode decode --parts-dir '$PARTS' --cache-dir '$CACHE' \\
        --shard-index \"\$index\" --shard-count $PARALLEL &
    pids+=(\"\$!\")
  done
  failed=0
  for pid in \"\${pids[@]}\"; do wait \"\$pid\" || failed=1; done
  [ \"\$failed\" -eq 0 ] || return 1
  python -u -m tadabur.tlog_eval_decode merge --parts-dir '$PARTS' --out-dir '$REPO/$OUT' $(pr_args)
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
}

require_within_budget() {
  local verdict
  verdict=$(remote "source '$CONDA' && conda activate $ENV_NAME && python -c \"import json; print(json.load(open('$PARTS/slice_timing.json'))['within_budget'])\"" 2>/dev/null) \
    || { echo "no slice record on $HOST; run '$0 slice launch' first" >&2; return 1; }
  [ "$verdict" = "True" ] \
    || { echo "the slice projected the full pass over the 8 h budget; post it and wait" >&2;
         return 1; }
}

launch() {
  remote "mkdir -p '$ROOT' '$LOGS' '$CACHE' '$PARTS'"
  case "$job" in
    slice)
      stage_repo
      write_slice_runner ;;
    decode)
      [ -z "$(git -C "$REPO_DIR" status --porcelain)" ] \
        || { echo "the worktree is dirty; commit before launching" >&2; return 1; }
      require_manifest_on_main
      require_within_budget
      stage_repo
      write_decode_runner ;;
  esac
  remote "rm -f '$LOGS/$NAME.exit' '$LOGS/$NAME.needs-post' '$LOGS/$NAME.pid'
          setsid flock /root/scratch/gpu.lock bash '$ROOT/run-$NAME.sh' </dev/null \\
            >'$LOGS/$NAME.log' 2>&1 &
          echo \$! > '$LOGS/$NAME.pid'"
  echo "job started on $HOST"
  echo "  pid:  $(remote "cat '$LOGS/$NAME.pid'")"
  echo "  log:  $HOST:$LOGS/$NAME.log"
  echo "  exit: $HOST:$LOGS/$NAME.exit (0 = clean; absent while running)"
  echo "  check with: scripts/issue-212-decode.sh $job check"
}

check() {
  if remote "test -f '$LOGS/$NAME.needs-post'"; then
    echo "needs-post: $HOST:$PARTS/slice_timing.json"
    remote "cat '$PARTS/slice_timing.json'"
  fi
  if remote "test -f '$LOGS/$NAME.exit'"; then
    echo "exit $(remote "cat '$LOGS/$NAME.exit'")"
  else
    echo "running (pid $(remote "cat '$LOGS/$NAME.pid' 2>/dev/null || echo unknown"))"
    remote "ps -p \"\$(cat '$LOGS/$NAME.pid')\" -o pid=,etime=,stat= 2>/dev/null || true"
  fi
}

fetch() {
  local destination=${1:?fetch needs a destination directory}
  mkdir -p "$destination"
  case "$job" in
    slice) rsync -a "$HOST:$PARTS/slice_timing.json" "$destination/" ;;
    decode) rsync -a "$HOST:$REPO/$OUT/" "$destination/" ;;
  esac
  echo "fetched $job into $destination"
}

case "$action" in
  launch) launch ;;
  check) check ;;
  fetch) fetch "${3:-}" ;;
  "") launch ;;
  *) echo "usage: $0 $job [launch|check|fetch DIR]" >&2; exit 2 ;;
esac

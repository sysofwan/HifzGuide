#!/usr/bin/env bash
# issue-212-decode.sh — the narrowed tlog look decode for round 2 on cuda-dev (#212, r26).
#
# Mac side, from the worktree root:
#   scripts/issue-212-decode.sh decode launch   # the gating rows, serial, under gpu.lock
#   scripts/issue-212-decode.sh decode check    # 0 = done, 3 = the 11 h gate stopped it
#   scripts/issue-212-decode.sh decode fetch DIR # pull the merged decodes and meta.json
#
# The run is the look tranche's 2,853 `gating` rows (the frozen manifest's stratum) decoded by
# the teacher, C95 and the #178 candidate's EMA, `checks` only, one process (the work is
# CPU-bound; parallel processes are unmeasured). After 20 shards it projects the rest from the
# audio covered; over 11 h it stops, writes `logs/decode.needs-post` and exits 3, so the
# orchestrator decides — the architect's fallback is the teacher and C95 alone. Nothing is
# graded or read.
#
# The job stages the current branch as a bundle on the box, writes one log, and writes its
# exit file last, so `check` never reads a log tail. It runs under setsid with stdin closed, so
# it survives the ssh session, and the whole runner holds /root/scratch/gpu.lock. The look
# shards are cached in $ROOT/cache and deleted once read; no audio leaves the box.
# `launch` refuses unless PR #211 is merged (origin/main holds the look manifest) and the
# worktree is clean. A fresh parts directory is the default, so a stopped or earlier run's
# parts can never be resumed into this one by accident.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog/r2}
LOGS=$ROOT/logs
REPO=$ROOT/repo
CACHE=$ROOT/cache
PARTS=${PARTS_DIR:-$ROOT/parts-gating}
OUT=tools/tadabur/consonant2_readout/decodes
MANIFEST=tools/tadabur/tlog/tranches/look.jsonl
MODELS=${MODELS:-base,c95,candidate_ema}
GATE_SHARDS=${GATE_SHARDS:-20}
LIMIT_HOURS=${LIMIT_HOURS:-11}
# The PR that keeps the cited code SHA resolvable after the squash merge (refs/pull/<N>/head).
PR=${PR:-}

job=${1:-}
action=${2:-}
case "$job" in
  decode) NAME=decode ;;
  *) echo "usage: $0 decode [launch|check|fetch DIR]" >&2
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
  remote "cat > '$ROOT/run-$NAME.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
run() {
  source '$CONDA' || return 1
  conda activate $ENV_NAME || return 1
  cd '$REPO/tools' || return 1
  python -u -m tadabur.tlog_eval_decode decode --parts-dir '$PARTS' --cache-dir '$CACHE' \\
      --models ${MODELS//,/ } --gate-shards $GATE_SHARDS --limit-hours $LIMIT_HOURS || return \$?
  python -u -m tadabur.tlog_eval_decode merge --parts-dir '$PARTS' --out-dir '$REPO/$OUT' \\
      --models ${MODELS//,/ } $(pr_args)
}
run
code=\$?
if [ \$code -eq 3 ]; then touch '$LOGS/$NAME.needs-post'; fi
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

launch() {
  [ -z "$(git -C "$REPO_DIR" status --porcelain)" ] \
    || { echo "the worktree is dirty; commit before launching" >&2; return 1; }
  require_manifest_on_main
  remote "mkdir -p '$ROOT' '$LOGS' '$CACHE' '$PARTS'"
  stage_repo
  write_runner
  remote "rm -f '$LOGS/$NAME.exit' '$LOGS/$NAME.needs-post' '$LOGS/$NAME.pid'
          setsid flock /root/scratch/gpu.lock bash '$ROOT/run-$NAME.sh' </dev/null \\
            >'$LOGS/$NAME.log' 2>&1 &
          echo \$! > '$LOGS/$NAME.pid'"
  echo "job started on $HOST"
  echo "  pid:  $(remote "cat '$LOGS/$NAME.pid'")"
  echo "  log:  $HOST:$LOGS/$NAME.log"
  echo "  exit: $HOST:$LOGS/$NAME.exit (0 = clean; 3 = the gate stopped it; absent while running)"
  echo "  check with: scripts/issue-212-decode.sh $job check"
}

check() {
  if remote "test -f '$LOGS/$NAME.needs-post'"; then
    echo "needs-post: the gate stopped the run"
    remote "cat '$PARTS/gate_timing.json' 2>/dev/null || true"
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
  rsync -a "$HOST:$REPO/$OUT/" "$destination/"
  echo "fetched $job into $destination"
}

case "$action" in
  launch) launch ;;
  check) check ;;
  fetch) fetch "${3:-}" ;;
  "") launch ;;
  *) echo "usage: $0 $job [launch|check|fetch DIR]" >&2; exit 2 ;;
esac

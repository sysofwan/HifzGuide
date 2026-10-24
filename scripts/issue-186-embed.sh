#!/usr/bin/env bash
# issue-186-embed.sh — the tlog S2 embedding pass on cuda-dev (#186).
#
# Mac side, from the worktree root, in this order:
#   scripts/issue-186-embed.sh launch   # stage HEAD, install speechbrain, smoke one clip
#   scripts/issue-186-embed.sh sweep    # start the full pass detached; prints its PID
#   scripts/issue-186-embed.sh check    # 0 = the pass finished cleanly; prints the done-file
#   scripts/issue-186-embed.sh fetch DIR  # pull pins.json + the npz files (scratch, never committed)
#
# The pass is `python -m tadabur.tlog_s2 embed`: it pins both model revisions to
# out-dir/pins.json (and the run log) before the first embedding, then streams the 20 committed
# sample shards one at a time. On the box this same file's `run` mode does the work under
# /root/scratch/gpu.lock: it writes logs/embed.exit, then logs/embed.done, so `check` reads a
# file rather than a log tail. BUDGET (default 45 min) stops the pass and records exit 2.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-186}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
SPEECHBRAIN=${SPEECHBRAIN:-speechbrain==1.1.2}
BUDGET=${BUDGET:-2700}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

remote() { ssh "$HOST" "$@"; }

bundle_ref() {  # <repo_root>
  local branch
  branch=$(git -C "$1" rev-parse --abbrev-ref HEAD)
  [ "$branch" = HEAD ] && echo HEAD || echo "$branch"
}

stage_repo() {
  # A bundle clone: the clone's HEAD must be the commit the pins record.
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  remote "mkdir -p '$ROOT/logs' '$ROOT/embeddings' '$ROOT/smoke' && \
          rm -rf '$ROOT/repo' '$ROOT/repo.bundle'"
  git -C "$REPO_DIR" bundle create - "$(bundle_ref "$REPO_DIR")" \
    | remote "cat > '$ROOT/repo.bundle'"
  if [ "$branch" = HEAD ]; then
    remote "git clone -q '$ROOT/repo.bundle' '$ROOT/repo'"
  else
    remote "git clone -q -b '$branch' '$ROOT/repo.bundle' '$ROOT/repo'"
  fi
  remote "rm -f '$ROOT/repo.bundle' && \
          test \"\$(git -C '$ROOT/repo' rev-parse HEAD)\" = '$sha'" \
    || { echo "the box's clone is not $sha" >&2; return 1; }
  echo "staged $sha"
}

launch() {
  stage_repo
  remote "source '$CONDA' && conda activate '$ENV_NAME' && pip install -q '$SPEECHBRAIN'" \
    || { echo "speechbrain install failed" >&2; return 1; }
  # One clip, both models: proves the snapshot loads and the speechbrain interface works
  # before a 20-shard sweep; it writes into $ROOT/smoke, never the real out-dir.
  remote "cd '$ROOT/repo/tools' && source '$CONDA' && conda activate '$ENV_NAME' && \
          python -m tadabur.tlog_s2 embed --out-dir '$ROOT/smoke' --limit 1"
  echo "smoke ok"
}

sweep() {
  remote "cd '$ROOT/repo' && source '$CONDA' && conda activate '$ENV_NAME' && \
          rm -f '$ROOT/logs/embed.done' '$ROOT/logs/embed.exit' && \
          nohup flock /root/scratch/gpu.lock bash scripts/issue-186-embed.sh run \
            > '$ROOT/logs/embed.log' 2>&1 & echo \$!"
}

check() {
  remote "cat '$ROOT/logs/embed.done' 2>/dev/null || { echo 'still running (or not started)' >&2; exit 3; }; \
          test \"\$(cat '$ROOT/logs/embed.exit')\" = 0"
}

fetch() {  # <local dir>
  local target=$1
  mkdir -p "$target"
  remote "tar -C '$ROOT/embeddings' -czf - ." | tar -C "$target" -xzf -
  echo "fetched into $target"
}

run() {
  # Executed on the box, under flock. Never `set -e`: the exit code is the run's result.
  set +e
  cd "$REPO_DIR/tools" || exit 1
  source "$CONDA" && conda activate "$ENV_NAME" || exit 1
  mkdir -p "$ROOT/logs" "$ROOT/embeddings"
  python -m tadabur.tlog_s2 embed --out-dir "$ROOT/embeddings" &
  local pass=$!
  ( sleep "$BUDGET"
    echo "$(date -u +%FT%TZ) budget exceeded; stopping $pass" >> "$ROOT/logs/embed.log"
    : > "$ROOT/logs/budget.exceeded"
    kill -TERM "$pass" 2>/dev/null ) &
  local watchdog=$!
  local status=0
  wait "$pass" || status=$?
  kill "$watchdog" 2>/dev/null
  if [ -e "$ROOT/logs/budget.exceeded" ]; then
    status=2
  fi
  echo "$status" > "$ROOT/logs/embed.exit"
  echo "$(date -u +%FT%TZ) exit $status" > "$ROOT/logs/embed.done"
}

case "${1:-}" in
  launch) launch ;;
  sweep) sweep ;;
  check) check ;;
  fetch) fetch "${2:?fetch needs a local directory}" ;;
  run) run ;;
  *)
    echo "usage: $0 {launch|sweep|check|fetch DIR}" >&2
    exit 2
    ;;
esac

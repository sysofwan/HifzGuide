#!/usr/bin/env bash
# issue-188-pass1.sh — the tlog S3b gate and full pass 1 on cuda-dev (#188).
#
# Mac side, from the worktree root, in this order:
#   PR=198 scripts/issue-188-pass1.sh launch        # stage HEAD as a clean clone on the box
#   PR=198 scripts/issue-188-pass1.sh gate          # gate tranche, detached, 3 workers
#   scripts/issue-188-pass1.sh check gate           # done file, projection, disk report
#   PR=198 scripts/issue-188-pass1.sh full          # all 743 shards, detached, 3 workers
#   scripts/issue-188-pass1.sh check full
#   scripts/issue-188-pass1.sh fetch DIR            # pull the pass-1 rows + meta (never audio)
#
# The gate names clean 0-5 and unclean 0-5; clean 0 and unclean 0 are the S3a tracer's
# committed outputs, so the resume guard verifies them and skips them, and they do not enter
# the measured rate. Each split is timed as its own phase at three workers. `check gate` prints
# `throughput.gate_projection`; run `full` only when its `within_owner_envelope` is true
# (<= 8 wall hours, 0 GPU hours) and the peak disk is <= 5 GB.
#
# Outputs go to /root/scratch/tlog/s3b/pass1, never into the staged checkout: the tree stays
# clean for the code SHA the meta cites, and a resumed run re-verifies what is already there
# instead of downloading it again. Pass 1 is CPU-only (decode, VAD, fingerprint): it takes no
# GPU lock. tlog audio is gated and never leaves cuda-dev; only rows and metadata are fetched.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog/s3b}
OUT=${PASS1_OUT:-/root/scratch/tlog/s3b/pass1}
CACHE=${PASS1_CACHE:-/root/scratch/tlog/cache}
FP=${FP_DIR:-/root/scratch/tlog/fp}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
PR=${PR:-}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

remote() { ssh "$HOST" "$@"; }

stage_repo() {
  # A bundle clone: require_clean_tree sees a clean checkout, and HEAD is the cited SHA.
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT/logs' '$OUT' && rm -rf '$ROOT/repo' '$ROOT/repo.bundle'"
  git -C "$REPO_DIR" bundle create - "$branch" | remote "cat > '$ROOT/repo.bundle'"
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
  remote "df -h '$OUT'; \
          cd '$ROOT/repo' && source '$CONDA' && conda activate '$ENV_NAME' && \
          python -c 'import av, silero_vad; print(\"av\", av.__version__)'"
  echo "scratch: $OUT (audio cache: $CACHE, fingerprints: $FP)"
}

sweep() {  # gate|full
  local name=$1
  [ -n "$PR" ] || { echo "PR=<N> is required" >&2; return 1; }
  remote "cd '$ROOT/repo' && rm -f '$ROOT/logs/$name.done' '$ROOT/logs/$name.exit' \
            '$ROOT/logs/$name.pid' && \
          nohup env PR='$PR' bash scripts/issue-188-pass1.sh run '$name' \
            > '$ROOT/logs/$name.log' 2>&1 & echo \$!"
}

check() {  # gate|full
  local name=$1
  remote "if [ ! -f '$ROOT/logs/$name.done' ]; then \
            echo 'still running (or not started); log: $ROOT/logs/$name.log' >&2; \
            test -f '$ROOT/logs/$name.pid' && echo \"pid \$(cat '$ROOT/logs/$name.pid')\"; \
            exit 3; fi; \
          cat '$ROOT/logs/$name.done'; \
          test \"\$(cat '$ROOT/logs/$name.exit')\" = 0 || exit 4; \
          echo '--- disk before/after (bytes)'; cat '$ROOT/logs/$name.df.before' '$ROOT/logs/$name.df.after'; \
          python -c \"import json; m=json.load(open('$OUT/meta.json')); \
t=m['throughput']; print(json.dumps({'counts': m['counts']['rows'], 'workers': t['workers'], \
'shards_measured': t['shards_measured'], 'shards_skipped': t['shards_skipped'], \
'wall_seconds': t['wall_seconds'], 'peak_disk_bytes': t['peak_disk_bytes'], \
'gate_projection': t['gate_projection']}, indent=1, sort_keys=True))\""
}

fetch() {  # <local dir>
  local target=$1
  mkdir -p "$target"
  remote "tar -C '$OUT' -czf - ." | tar -C "$target" -xzf -
  echo "fetched pass-1 rows and meta into $target (no audio)"
}

run() {  # executed on the box; never `set -e`: the exit code is the run's result
  local name=$1 selection
  set +e
  cd "$REPO_DIR/tools" || exit 1
  source "$CONDA" && conda activate "$ENV_NAME" || exit 1
  mkdir -p "$ROOT/logs" "$OUT"
  if [ "$name" = gate ]; then selection=--gate; else selection=--all; fi
  df -B1 "$OUT" > "$ROOT/logs/$name.df.before"
  python -m tadabur.tlog_pass1 run "$selection" --workers 3 --pr "$PR" \
    --cache-dir "$CACHE" --out-dir "$OUT" --fp-dir "$FP" &
  local pass=$!
  echo "$pass" > "$ROOT/logs/$name.pid"
  wait "$pass"
  local status=$?
  df -B1 "$OUT" > "$ROOT/logs/$name.df.after"
  echo "$status" > "$ROOT/logs/$name.exit"
  echo "$(date -u +%FT%TZ) $name exit $status (pid $pass)" > "$ROOT/logs/$name.done"
}

case "${1:-}" in
  launch) launch ;;
  gate|full) sweep "$1" ;;
  check) check "${2:?check needs gate|full}" ;;
  fetch) fetch "${2:?fetch needs a local directory}" ;;
  run) run "${2:?run needs gate|full}" ;;
  *)
    echo "usage: $0 {launch|gate|full|check gate|full|fetch DIR}" >&2
    exit 2
    ;;
esac

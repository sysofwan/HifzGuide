#!/usr/bin/env bash
# issue-189-overlap.sh — the tlog S4 audio-overlap build on cuda-dev (#189).
#
# Mac side, from the worktree root, in this order:
#   PR=NNN scripts/issue-189-overlap.sh launch          # stage HEAD as a clean clone on the box
#   scripts/issue-189-overlap.sh external               # fingerprint every frozen eval set
#   scripts/issue-189-overlap.sh check external
#   PR=NNN scripts/issue-189-overlap.sh build           # index, controls, records, meta
#   scripts/issue-189-overlap.sh check build
#   scripts/issue-189-overlap.sh fetch DIR              # pull the committed overlap tree back
#
# Both stages are CPU-only (decode, fingerprint, index) and take no GPU lock. Each is
# resumable and detached: `setsid` with stdin closed, its pid/exit/done files under
# $LOGS, and `python -u` so every per-unit and per-shard line lands in the log as it
# happens. The external pass writes only fingerprints and control candidates to scratch
# ($FP); the build reads S3b's tlog fingerprint store ($STORE) and writes the committed
# tree to $OUT — never into the staged checkout, so the cited code SHA stays clean.
# Audio is streamed and deleted; nothing but fingerprints and the overlap tree leaves.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/tlog/s4}
LOGS=${ISSUE_LOGS:-$ROOT/logs}
OUT=${OVERLAP_OUT:-$ROOT/overlap}
FP=${FP_DIR:-/root/scratch/tlog/overlap/fp}
STORE=${STORE_DIR:-/root/scratch/tlog/fp}
CACHE=${AUDIO_CACHE:-/root/scratch/tlog/cache}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
PR=${PR:-}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

remote() { ssh "$HOST" "$@"; }

stage_repo() {
  # A bundle clone: the checkout is its own repository, and HEAD is the cited SHA.
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT' '$LOGS' '$OUT' && rm -rf '$ROOT/repo' '$ROOT/repo.bundle'"
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
  remote "df -h '$OUT' '$CACHE'; \
          cd '$ROOT/repo' && source '$CONDA' && conda activate '$ENV_NAME' && \
          python -c 'import numpy, soundfile, librosa, pyarrow, huggingface_hub; \
print(\"numpy\", numpy.__version__, \"soundfile\", soundfile.__version__)'; \
          test -e '$STORE' && ls '$STORE' | wc -l && du -sh '$STORE'"
  echo "external store: $FP; committed tree: $OUT; tlog fingerprints (read only): $STORE"
}

fail() {  # <name> <reason>: a launch that died before the run, visible to `check`
  local name=$1 reason=$2
  mkdir -p "$LOGS"
  echo 1 > "$LOGS/$name.exit"
  echo "$(date -u +%FT%TZ) $name did not run: $reason" > "$LOGS/$name.done"
  echo "$name did not run: $reason" >&2
}

sweep() {  # external|build
  local name=$1
  if [ "$name" = build ]; then
    [ -n "$PR" ] || { echo "PR=<N> is required for the build" >&2; return 1; }
  fi
  # The checkout is the box's, not this Mac's: the test belongs in the remote command.
  remote "test -d '$ROOT/repo' || { echo \"run 'launch' first\" >&2; exit 1; }; \
          mkdir -p '$LOGS' '$OUT'; cd '$ROOT/repo' || exit 1; \
          rm -f '$LOGS/$name.done' '$LOGS/$name.exit' '$LOGS/$name.pid'; \
          setsid env PR='$PR' ISSUE_ROOT='$ROOT' OVERLAP_OUT='$OUT' FP_DIR='$FP' \
            STORE_DIR='$STORE' AUDIO_CACHE='$CACHE' \
            bash scripts/issue-189-overlap.sh run '$name' \
            < /dev/null > '$LOGS/$name.log' 2>&1 & \
          pid=\$!; echo \$pid > '$LOGS/$name.pid'; echo \"started $name: pid \$pid\""
}

check() {  # external|build
  local name=$1
  remote "if [ ! -f '$LOGS/$name.done' ]; then \
            echo 'still running (or not started); log: $LOGS/$name.log' >&2; \
            test -f '$LOGS/$name.pid' && echo \"pid \$(cat '$LOGS/$name.pid')\"; \
            tail -n 3 '$LOGS/$name.log' 2>/dev/null || true; exit 3; fi; \
          cat '$LOGS/$name.done'; \
          test \"\$(cat '$LOGS/$name.exit')\" = 0 || exit 4; \
          tail -n 20 '$LOGS/$name.log'"
}

fetch() {  # <local dir>
  local target=$1
  mkdir -p "$target"
  remote "tar -C '$OUT' -czf - ." | tar -C "$target" -xzf -
  echo "fetched the overlap tree into $target (no audio)"
}

run() {  # executed on the box; never `set -e`: the exit code is the run's result
  local name=$1 code
  set +e
  mkdir -p "$LOGS" "$OUT"
  cd "$REPO_DIR/tools" || { fail "$name" "no $REPO_DIR/tools on the box"; return 1; }
  # shellcheck source=/dev/null
  source "$CONDA" || { fail "$name" "no $CONDA on the box"; return 1; }
  conda activate "$ENV_NAME" || { fail "$name" "conda activate $ENV_NAME"; return 1; }
  if [ "$name" = external ]; then
    python -u -m tadabur.tlog_overlap external --fp-dir "$FP" --cache-dir "$CACHE"
    code=$?
  elif [ "$name" = build ]; then
    [ -n "$PR" ] || { fail "$name" "PR=<N> is required"; return 1; }
    python -u -m tadabur.tlog_overlap build --pr "$PR" --out "$OUT" \
      --fp-dir "$FP" --store-dir "$STORE" \
      --code-sha "$(git -C "$REPO_DIR" rev-parse HEAD)"
    code=$?
  else
    fail "$name" "unknown stage $name"
    return 1
  fi
  echo "$code" > "$LOGS/$name.exit"
  echo "$(date -u +%FT%TZ) $name exit=$code" > "$LOGS/$name.done"
  exit "$code"
}

case "${1:-}" in
  launch) launch ;;
  external|build) sweep "$1" ;;
  check) check "${2:?stage}" ;;
  fetch) fetch "${2:?local dir}" ;;
  run) run "${2:?stage}" ;;
  *) echo "usage: $0 {launch|external|build|check <stage>|fetch <dir>}" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# zipformer-screen.sh — the #245 zipformer screen's steps on cuda-dev.
#
# On the Mac, from the worktree root, one subcommand per step (`fetch` must run first; each
# later step reads the earlier one's committed output and the fetched files in the work
# directory):
#
#   scripts/zipformer-screen.sh fetch          # Hugging Face fetch, SHA-256 pinned
#   scripts/zipformer-screen.sh disjointness   # 600 benchmark clips vs every tlog row
#   scripts/zipformer-screen.sh gate           # card reproduction, scale pin, identity
#   scripts/zipformer-screen.sh check <step>   # pid, status and log tail of a running/ended step
#   scripts/zipformer-screen.sh fetchout <step> DIR   # pull the committed JSON outputs back
#
# Each step stages the current branch as a clean bundle clone on the box (so the cited code
# SHA is the one that ran), then runs detached under setsid with stdin closed: one log, one pid
# and one exit-code file under $ROOT/logs. The three steps are CPU-only — zipformer runs on
# ONNX Runtime's CPU provider — so none takes the GPU lock. The gated Hub token lives only on
# the box; the benchmark audio and the ONNX files stay in $ROOT/data and are never committed.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/zipformer}
REPO=$ROOT/repo
LOGS=$ROOT/logs
WORK=$ROOT/data
STORE=${STORE_DIR:-/root/scratch/tlog/fp}
WORKERS=${WORKERS:-12}
OUT=tools/tadabur/zipformer_screen
#: The committed files a step's `fetchout` pulls back.
FETCH_OUT_FILES=(disjointness.json card_gate.json identity.json)

step=${1:-}
action=${2:-}
case "$step" in
  fetch|disjointness|gate) ;;
  *)
    echo "usage: $0 {fetch|disjointness|gate} [launch|check|fetchout DIR]" >&2
    exit 2
    ;;
esac

remote() { ssh -n "$HOST" "$@"; }

stage_repo() {
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT' '$LOGS' '$WORK' && rm -rf '$REPO' '$ROOT/repo.bundle'"
  git -C "$REPO_DIR" bundle create - "$branch" | ssh "$HOST" "cat > '$ROOT/repo.bundle'"
  if [ "$branch" = HEAD ]; then
    remote "git clone -q '$ROOT/repo.bundle' '$REPO'"
  else
    remote "git clone -q -b '$branch' '$ROOT/repo.bundle' '$REPO'"
  fi
  remote "rm -f '$ROOT/repo.bundle' && test \"\$(git -C '$REPO' rev-parse HEAD)\" = '$sha'" \
    || { echo "the box's clone is not $sha" >&2; return 1; }
}

command_for() {
  case "$step" in
    fetch) echo "python -u -m training.zipformer_run fetch --work-dir '$WORK'" ;;
    disjointness)
      echo "python -u -m training.zipformer_run disjointness --work-dir '$WORK' --store-dir '$STORE'" ;;
    gate) echo "python -u -m training.zipformer_run gate --work-dir '$WORK' --workers '$WORKERS'" ;;
  esac
}

write_runner() {
  remote "cat > '$ROOT/run-$step.sh' <<'SH'
#!/usr/bin/env bash
set -uo pipefail
source '$CONDA'
conda activate '$ENV_NAME'
cd '$REPO/tools' || exit 1
$(command_for)
code=\$?
echo \"\$code\" > '$LOGS/$step.exit'
exit \$code
SH
chmod +x '$ROOT/run-$step.sh'"
}

launch() {
  stage_repo
  write_runner
  remote "rm -f '$LOGS/$step.exit' '$LOGS/$step.pid'
          setsid bash '$ROOT/run-$step.sh' </dev/null >'$LOGS/$step.log' 2>&1 &
          echo \$! > '$LOGS/$step.pid'"
  echo "step $step started on $HOST"
  echo "  pid:  $(remote "cat '$LOGS/$step.pid'")"
  echo "  log:  $HOST:$LOGS/$step.log"
  echo "  exit: $HOST:$LOGS/$step.exit (0 = clean; absent while running)"
}

check() {
  if remote "test -f '$LOGS/$step.exit'"; then
    echo "step $step exited $(remote "cat '$LOGS/$step.exit'")"
  else
    echo "step $step is running (pid $(remote "cat '$LOGS/$step.pid' 2>/dev/null || echo unknown"))"
    remote "ps -p \"\$(cat '$LOGS/$step.pid')\" -o pid=,etime=,stat= 2>/dev/null || true"
  fi
  remote "tail -n 12 '$LOGS/$step.log' 2>/dev/null || true"
}

fetchout() {
  local destination=${1:?fetchout needs a destination directory}
  mkdir -p "$destination"
  for name in "${FETCH_OUT_FILES[@]}"; do
    if remote "test -f '$REPO/$OUT/$name'"; then
      rsync -a "$HOST:$REPO/$OUT/$name" "$destination/"
    fi
  done
  echo "fetched the $step step's committed outputs into $destination"
}

case "$action" in
  launch|"") launch ;;
  check) check ;;
  fetchout) fetchout "${3:-}" ;;
  *) echo "usage: $0 {fetch|disjointness|gate} [launch|check|fetchout DIR]" >&2; exit 2 ;;
esac

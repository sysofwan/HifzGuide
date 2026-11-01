#!/usr/bin/env bash
# zip-screen-batch.sh — #245's gate-independent steps on cuda-dev, one detached batch.
#
# On the Mac, from the worktree root:
#
#   scripts/zip-screen-batch.sh launch           # stage, then run every step below detached
#   scripts/zip-screen-batch.sh check            # the batch's exit files, and the newest log lines
#   scripts/zip-screen-batch.sh fetchout DIR     # pull the decodes, the renders and the record back
#
# The steps, in order (each writes $LOGS/<name>.exit; $LOGS/batch.exit is 0 only if all did):
#
#   1. edits-gen     render the 300 synthetic-edit audios in a *scratch copy* of tools/
#   2. edits-verify  every item and its audio's SHA-256 against the committed manifest
#   3. sherpa        a throwaway sherpa-onnx venv under /root/scratch (never a repo dependency),
#                    pinned, recording its version and whether the zipformer2-CTC entry point exists
#   4. gpu           our three models on the benchmark (the plan's 577 clips), the 300 edits and
#                    qdat's spans arm, three model processes per set, one gpu.lock for the group
#
# Nothing here decodes zipformer: the parity gate decides that, and this batch is what runs
# while it is cross-checked. Each step resumes: a relaunch decodes only the items still missing.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-hifzguide}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=${ISSUE_ROOT:-/root/scratch/zip-screen/batch}
REPO=$ROOT/repo
EDITS_REPO=$ROOT/edits-repo
LOGS=$ROOT/logs
DECODES=$ROOT/decodes
REGEN=$ROOT/edits-regen
WORK=${WORK_DIR:-/root/scratch/zipformer/data}
CLIPS=${CLIPS_DIR:-/root/scratch/issue-88/stage/clips}
SHERPA=${SHERPA_DIR:-/root/scratch/sherpa-venv}
SHERPA_PIN=${SHERPA_PIN:-2.0.1}
GPU_LOCK=${GPU_LOCK:-/root/scratch/gpu.lock}

action=${1:-}
case "$action" in
  launch|check|fetchout) ;;
  *) echo "usage: $0 {launch|check|fetchout DIR}" >&2; exit 2 ;;
esac

remote() { ssh -n "$HOST" "$@"; }

stage_repo() {
  local branch sha
  branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
  sha=$(git -C "$REPO_DIR" rev-parse HEAD)
  [ "$branch" = HEAD ] && branch=HEAD
  remote "mkdir -p '$ROOT' '$LOGS' '$DECODES' && rm -rf '$REPO' '$EDITS_REPO' '$ROOT/repo.bundle'"
  git -C "$REPO_DIR" bundle create - "$branch" | ssh "$HOST" "cat > '$ROOT/repo.bundle'"
  if [ "$branch" = HEAD ]; then
    remote "git clone -q '$ROOT/repo.bundle' '$REPO' && git clone -q '$ROOT/repo.bundle' '$EDITS_REPO'"
  else
    remote "git clone -q -b '$branch' '$ROOT/repo.bundle' '$REPO' &&
            git clone -q -b '$branch' '$ROOT/repo.bundle' '$EDITS_REPO'"
  fi
  remote "rm -f '$ROOT/repo.bundle' &&
          test \"\$(git -C '$REPO' rev-parse HEAD)\" = '$sha' &&
          test \"\$(git -C '$EDITS_REPO' rev-parse HEAD)\" = '$sha'" \
    || { echo "the box's clones are not $sha" >&2; return 1; }
}

write_runner() {
  remote "cat > '$ROOT/run-batch.sh' <<'RUN'
#!/usr/bin/env bash
set -uo pipefail
source '$CONDA'
conda activate '$ENV_NAME'
ROOT='$ROOT'; REPO='$REPO'; EDITS_REPO='$EDITS_REPO'; LOGS='$LOGS'
DECODES='$DECODES'; REGEN='$REGEN'; WORK='$WORK'; CLIPS='$CLIPS'
SHERPA='$SHERPA'; SHERPA_PIN='$SHERPA_PIN'; GPU_LOCK='$GPU_LOCK'
MODELS='base c95 r2_candidate_ema'
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

step() {  # name dir command...
  local name=\$1 dir=\$2; shift 2
  { echo \"=== \$name on \$(date -Is)\"; ( cd \"\$dir\" && \"\$@\" ); } >> \"\$LOGS/\$name.log\" 2>&1
  local code=\$?
  echo \"\$code\" > \"\$LOGS/\$name.exit\"
  echo \"\$name exited \$code\"
  return \$code
}

sherpa() {  # a throwaway venv: pinned, never a dependency of tools/
  [ -x \"\$SHERPA/bin/python\" ] || python -m venv \"\$SHERPA\" || return 1
  \"\$SHERPA/bin/python\" -m pip install --quiet --upgrade pip || return 1
  \"\$SHERPA/bin/python\" -m pip install --quiet \"sherpa-onnx==\$SHERPA_PIN\" || return 1
  \"\$SHERPA/bin/python\" -c \"import json,sys,sherpa_onnx as so; from sherpa_onnx import OnlineRecognizer as R; json.dump({'version': so.__version__, 'pin': sys.argv[2], 'from_zipformer2_ctc': hasattr(R, 'from_zipformer2_ctc')}, open(sys.argv[1], 'w'), indent=1)\" \"\$ROOT/sherpa.json\" \"\$SHERPA_PIN\" || return 1
  cat \"\$ROOT/sherpa.json\"
}

decode_group() {  # set [args...]: one model at a time (three teacher processes OOM the card)
  local set=\$1; shift
  local m rc code=0
  for m in \$MODELS; do
    ( cd \"\$REPO/tools\" && python -u -m training.zip_screen_decode --set \"\$set\" --models \"\$m\" \
        --out-dir \"\$DECODES\" \"\$@\" ) >> \"\$LOGS/decode-\$set-\$m.log\" 2>&1
    rc=\$?
    echo \"\$rc\" > \"\$LOGS/decode-\$set-\$m.exit\"
    echo \"decode \$set \$m exited \$rc\"
    [ \"\$rc\" -eq 0 ] || code=1
  done
  return \$code
}

edits_verify() {
  local stale
  stale=\$(git -C \"\$EDITS_REPO\" status --porcelain tools/tadabur/synthetic_edits/)
  if [ -n \"\$stale\" ]; then
    echo \"the render is not byte-identical to the commit:\"; echo \"\$stale\"; return 1
  fi
  echo \"manifest, summary and worklist are byte-identical to the commit\"
  python -u -m tadabur.synthetic_edits verify --manifest \"\$EDITS_REPO/tools/tadabur/synthetic_edits/edits.jsonl\" --audio-dir \"\$REGEN/audio\"
}

gpu_phase() {
  local code=0
  decode_group benchmark --work-dir \"\$WORK\" || code=1
  decode_group edits --audio-dir \"\$REGEN/audio\" || code=1
  decode_group qdat-spans --cache-dir \"\$ROOT/qdat_cache\" || code=1
  return \$code
}

mode=\${1:-all}
if [ \"\$mode\" = gpu ]; then
  step gpu-decodes \"\$ROOT\" gpu_phase
  exit \$?
fi

failed=0
step edits-gen \"\$EDITS_REPO/tools\" python -u -m tadabur.synthetic_edits generate \
  --audio-dir \"\$CLIPS\" --out-dir \"\$REGEN\" || failed=1
step edits-verify \"\$REPO/tools\" edits_verify || failed=1
step sherpa \"\$ROOT\" sherpa || failed=1
step gpu \"\$ROOT\" flock \"\$GPU_LOCK\" bash \"\$ROOT/run-batch.sh\" gpu || failed=1
echo \"\$failed\" > \"\$LOGS/batch.exit\"
echo \"batch finished with \$failed (0 = every step clean)\"
exit \"\$failed\"
RUN
chmod +x '$ROOT/run-batch.sh' && bash -n '$ROOT/run-batch.sh'"
}

launch() {
  stage_repo
  write_runner
  remote "rm -f '$LOGS'/batch.exit '$LOGS'/batch.pid
          setsid bash '$ROOT/run-batch.sh' </dev/null >'$LOGS/batch.log' 2>&1 &
          echo \$! > '$LOGS/batch.pid'"
  echo "batch started on $HOST"
  echo "  pid:  $(remote "cat '$LOGS/batch.pid'")"
  echo "  log:  $HOST:$LOGS/batch.log"
  echo "  done: $HOST:$LOGS/batch.exit (0 = every step clean; absent while running)"
  echo "  steps: edits-gen edits-verify sherpa gpu -> $LOGS/<name>.exit"
}

check() {
  if remote "test -f '$LOGS/batch.exit'"; then
    echo "batch exited $(remote "cat '$LOGS/batch.exit'")"
  else
    echo "batch is running (pid $(remote "cat '$LOGS/batch.pid' 2>/dev/null || echo unknown"))"
  fi
  remote "for f in '$LOGS'/*.exit; do [ -e \"\$f\" ] && echo \"\$(basename \"\$f\" .exit) \$(cat \"\$f\")\"; done"
  remote "tail -n 20 '$LOGS/batch.log' 2>/dev/null || true"
}

fetchout() {
  local destination=${1:?fetchout needs a destination directory}
  mkdir -p "$destination"
  rsync -a "$HOST:$LOGS/" "$destination/logs/" || true
  rsync -a "$HOST:$DECODES/" "$destination/decodes/"
  rsync -a "$HOST:$REGEN/edits.jsonl" "$HOST:$REGEN/summary.json" "$destination/" || true
  rsync -a "$HOST:$ROOT/sherpa.json" "$destination/" || true
  echo "fetched the batch's decodes and records into $destination"
}

case "$action" in
  launch) launch ;;
  check) check ;;
  fetchout) fetchout "${2:-}" ;;
esac

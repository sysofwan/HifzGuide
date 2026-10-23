#!/usr/bin/env bash
# issue-178-train.sh — #178's consonant round: continue C95 toward the teacher, candidate arm.
#
# The candidate is C95's averaged weights continued for 10,000 steps under #94's recipe
# (augmented student, KL to the clean base teacher, no supervised term), on the 278 shards
# #94's run did not consume. Data derivation, recipe and blockers: docs/consonant-round-run.md.
#
# Mac side, from the worktree root, once the code and the run note are committed:
#   scripts/issue-178-train.sh launch   # archive HEAD, start detached under gpu.lock
#   scripts/issue-178-train.sh check    # done-file check; 0 = smoke and the full run finished
#   scripts/issue-178-train.sh report   # final step, checkpoint SHA-256, loss and throughput
#
# `launch` refuses while the GPU lock is held (another issue's run owns the box) and starts
# `run` under `flock /root/scratch/gpu.lock` when it is free. `run` asserts the data first
# (the derivation, its committed registration, the warm start's checksum), smoke-tests 300
# steps, gates the smoke on throughput and health, then runs 10,000 steps. It appends one
# line per stage to logs/events and writes, last, logs/train.exit and then logs/train.done;
# `check` reads that, never a log tail. A run stopped by the budget writes exit 2 and a
# `budget-exceeded` done-file.
#
# Budget (#178): the smoke is ~0.1 GPU-h and 10,000 steps ~3.5 GPU-h at #94's 0.8-0.9
# steps/s, so BUDGET defaults to six hours from the lock's acquisition. There is no control
# arm: C95 itself is the run without the round's only ingredient.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-178}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
BUDGET=${BUDGET:-21600}

# The warm start: C95's checkpoint, whose averaged weights #174 names.
C95=${C95:-/root/scratch/issue-94/repo-fix/tools/runs/aug94_candidate_v2/checkpoint.pt}
C95_SHA=8490b63d0664827d5a2e2e3121eb4af7b22069273db39268c7638cc7d8839cd6
POOL=${POOL:-/root/scratch/issue-94/noise_pool.npz}
POOL_SHA=d2978d0ff3fcc14b08b21b9360d0933202e1a9f7b92fae5de9376d6ba2b3fd43
# In-training validation is monitoring only and must touch no evaluation set (#178), so it
# reads clips staged from the run's own shards -- from the last shard of its seed-178
# permutation, which 10,000 steps never reaches -- rather than #94's dev clips.
VAL_CLIPS=${VAL_CLIPS:-$ROOT/val_clips}
VAL_CLIPS_COUNT=${VAL_CLIPS_COUNT:-200}

# The source run whose consumed prefix the data is derived from, and the shard index the
# registration counts its rows from.
SOURCE_LOG=${SOURCE_LOG:-/root/scratch/issue-94/logs/aug94_candidate_v2.log}
INDEX=${INDEX:-/root/scratch/issue-89/stage/full_index.jsonl}

SEED=178
STEPS=10000
SMOKE_STEPS=300
OUT=runs/consonant1_candidate
SMOKE=runs/consonant1_smoke
EXPOSURE_USE=consonant1.training

# --- on the box -------------------------------------------------------------------------


setup() {
  mkdir -p "$ROOT/logs" "$VAL_CLIPS"
  cd "$ROOT/repo/tools"
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
  python - "$VAL_CLIPS" "$VAL_CLIPS_COUNT" <<'PY'
import pathlib
import sys

import soundfile as sf

from tadabur.audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
from tadabur.exposure import CONSONANT_TRAINING, RowExclusion, load_registry
from tadabur.shard_reader import iter_shard_rows
from training.consumed_prefix import shard_permutation

out = pathlib.Path(sys.argv[1])
wanted = int(sys.argv[2])
# The shards the run is registered to stream, and the last of its own seed-178 permutation:
# 10,000 steps at ~150 steps per shard never reaches it, so the validation clips are audio
# the run does not train on while still belonging to no evaluation set.
shards = list(load_registry().shard_uses[CONSONANT_TRAINING].shards)
shard = shard_permutation(shards, 178)[-1]
for stale in out.glob("*.wav"):
    stale.unlink()

exclusion = RowExclusion.for_use(CONSONANT_TRAINING)
written = 0
for row_index, row in enumerate(iter_shard_rows([shard], delete_after=True)):
    if written >= wanted:
        continue  # keep consuming the shard so its cached blob is deleted
    if not exclusion.keeps((shard, row_index), row):
        continue
    try:
        waveform = decode_to_mono_16k(row["audio"]["bytes"])
    except Exception:  # a corrupt row is not a reason to stage nothing
        continue
    sf.write(out / f"{shard}_{row_index:04d}.wav", waveform, TARGET_SAMPLE_RATE)
    written += 1
print(f"validation clips: {written} from shard {shard} of {len(shards)} training shards")
if written < 50:
    raise SystemExit(f"only {written} clips staged: too few for a validation split")
PY
}

assert_ready() {
  echo "[assert] warm start $C95"
  local actual
  actual=$(sha256sum "$C95" | cut -d' ' -f1)
  if [ "$actual" != "$C95_SHA" ]; then
    echo "[assert] C95 checkpoint is $actual, expected $C95_SHA" >&2
    return 1
  fi
  actual=$(sha256sum "$POOL" | cut -d' ' -f1)
  if [ "$actual" != "$POOL_SHA" ]; then
    echo "[assert] noise pool is $actual, expected $POOL_SHA" >&2
    return 1
  fi
  assert_stream
}

assert_stream() {
  # The data: rebuild #94's consumed prefix from its log and refuse if the remainder has
  # drifted from the committed registration. The derivation reads only `h448`'s 345-shard
  # spec, and the trainer refuses a held-out shard before it downloads anything, so the
  # reserve cannot reach training by this path.
  #
  # Every failure is an explicit `|| return 1`: called as `assert_ready || return 1`, bash
  # disables errexit for the whole body, so a bare failing command would be swallowed and
  # a drift would launch anyway.
  mkdir -p "$ROOT/logs"
  local spec
  spec=$(python -m training.consumed_prefix --log "$SOURCE_LOG") || return 1
  echo "[assert] stream shards: $spec"
  python -m tadabur.exposure training-stream --use "$EXPOSURE_USE" --index "$INDEX" \
    --shards "$spec" --membership exact --note "launch assertion" --check || return 1
  printf '%s\n' "$spec" > "$ROOT/logs/spec.txt"
}

gate_smoke() {
  python - "$1" <<'PY'
import json
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        if "split" not in record and "step" in record:
            rows.append(record)
if len(rows) < 2:
    raise SystemExit(f"{sys.argv[1]}: {len(rows)} training records, too few to gate")
first, last = rows[0], rows[-1]
rate = (last["step"] - first["step"]) / (last["elapsed_s"] - first["elapsed_s"])
losses = [row["total"] for row in rows]
if not all(loss == loss and abs(loss) != float("inf") for loss in losses):
    raise SystemExit("KL is not finite")
if rate < 0.6:
    raise SystemExit(f"throughput {rate:.2f} steps/s is below the 0.6 threshold")
print(
    f"[gate] {rate:.2f} steps/s over steps {first['step']}-{last['step']}, "
    f"KL {min(losses):.3f}-{max(losses):.3f}, last {losses[-1]:.3f}"
)
PY
}

smoke_has_ema() {
  python - "$1" <<'PY'
import sys

import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if "ema_state" not in state:
    raise SystemExit(f"{sys.argv[1]} carries no averaged weights: the smoke kept none")
print(f"[gate] averaged weights saved beside the live ones at step {state['step']}")
PY
}

train_arm() {  # <out-dir> <steps> <log-every>
  local out=$1 steps=$2 log_every=$3
  python -m training.distill_train \
    --preset h448 --stream-shards "$(cat "$ROOT/logs/spec.txt")" \
    --audio-root "$VAL_CLIPS" --out-dir "$out" --steps "$steps" --batch-size 32 \
    --learning-rate 5e-5 --warmup-steps 500 --cooldown-steps 3000 \
    --temperature 2.0 --nonblank-weight 3.0 --confirm-weight 2.0 \
    --ema-decay 0.999 --seed "$SEED" --save-every 2000 --log-every "$log_every" \
    --eval-every 2000 --num-workers 1 --exposure-use "$EXPOSURE_USE" \
    --augment --noise-pool "$POOL" --init-from "$C95" --init-weights ema
}

chain() {
  assert_ready || return 1
  printf '%s smoke start\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
  if ! train_arm "$SMOKE" "$SMOKE_STEPS" 25 > "$ROOT/logs/smoke.log" 2>&1; then
    printf '%s smoke failed\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
    return 1
  fi
  gate_smoke "$SMOKE/metrics.jsonl" >> "$ROOT/logs/events" || return 1
  smoke_has_ema "$SMOKE/checkpoint.pt" >> "$ROOT/logs/events" || return 1
  rm -f "$SMOKE/checkpoint.pt"  # the metrics stay; the run's weights supersede them
  printf '%s full start\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
  if ! train_arm "$OUT" "$STEPS" 50 > "$ROOT/logs/train.log" 2>&1; then
    printf '%s full failed\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
    return 1
  fi
  printf '%s full done\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
}

run() {
  mkdir -p "$ROOT/logs"
  rm -f "$ROOT/logs/train.done" "$ROOT/logs/train.exit" "$ROOT/logs/budget.exceeded"
  : > "$ROOT/logs/events"
  # Globals: the EXIT trap can fire after `run` returns, when its locals are gone.
  SAMPLER=""
  WATCHDOG=""
  CHAIN=""
  finish() {
    local code=$?
    local group
    for group in "$CHAIN" "$SAMPLER" "$WATCHDOG"; do
      if [ -n "$group" ]; then
        kill -- "-$group" 2>/dev/null || kill "$group" 2>/dev/null || true
      fi
    done
    if [ ! -e "$ROOT/logs/train.done" ]; then
      printf '%s\n' "$code" > "$ROOT/logs/train.exit"
      printf '%s early-fail code=%s\n' "$(date -u +%FT%TZ)" "$code" \
        > "$ROOT/logs/train.done"
    fi
  }
  trap finish EXIT
  cd "$ROOT/repo/tools"
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
  export OMP_NUM_THREADS=1
  # Job control gives every background job its own process group, so the trap and the
  # deadline stop the chain and the trainer with every worker beneath them.
  set -m
  ( while true; do
      printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$(du -sb "$ROOT" | cut -f1)" \
        "$(df -B1 --output=avail /root/scratch | tail -1)" >> "$ROOT/logs/df.log"
      sleep 60
    done ) >/dev/null 2>&1 &
  SAMPLER=$!
  chain > "$ROOT/logs/chain.out" 2>&1 &
  CHAIN=$!
  # The deadline starts here, under the launcher's flock: expiry stops the chain and leaves
  # this shell to write the readiness files; the checkpoints already on disk stay.
  ( sleep "$BUDGET"
    : > "$ROOT/logs/budget.exceeded"
    kill -- "-$CHAIN" 2>/dev/null || true
  ) &
  WATCHDOG=$!
  set +m
  local status=0
  wait "$CHAIN" || status=$?
  if [ -n "$WATCHDOG" ]; then
    kill -- "-$WATCHDOG" 2>/dev/null || kill "$WATCHDOG" 2>/dev/null || true
  fi
  WATCHDOG=""
  CHAIN=""
  local code=$status
  if [ -e "$ROOT/logs/budget.exceeded" ]; then
    code=2
    printf '%s budget-exceeded budget=%ss; run group stopped, checkpoints kept\n' \
      "$(date -u +%FT%TZ)" "$BUDGET" > "$ROOT/logs/train.done"
  else
    [ "$code" -eq 0 ] || code=1
    printf '%s chain exit %s\n' "$(date -u +%FT%TZ)" "$status" > "$ROOT/logs/train.done"
  fi
  printf '%s\n' "$code" > "$ROOT/logs/train.exit"
  return "$code"
}

report() {
  ssh "$HOST" "python3 - '$ROOT' <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / 'repo' / 'tools' / 'runs' / 'consonant1_candidate'
checkpoint = root / 'checkpoint.pt'
digest = hashlib.sha256()
with checkpoint.open('rb') as handle:
    for block in iter(lambda: handle.read(1 << 20), b''):
        digest.update(block)
rows = []
final = None
with (root / 'metrics.jsonl').open() as handle:
    for line in handle:
        record = json.loads(line)
        if 'split' in record:
            final = record
        else:
            rows.append(record)
step = rows[-1]['step']
rate = (rows[-1]['step'] - rows[0]['step']) / (rows[-1]['elapsed_s'] - rows[0]['elapsed_s'])
losses = [row['total'] for row in rows]
grads = [row['grad_norm_preclip'] for row in rows]
print('step', step)
print('sha256', digest.hexdigest())
print('throughput {:.2f} steps/s over {} logged steps'.format(rate, len(rows)))
print('kl min {:.4f} max {:.4f} last {:.4f}'.format(min(losses), max(losses), losses[-1]))
print('grad_norm_preclip max {:.2f}'.format(max(grads)))
if final is not None:
    print('final val', json.dumps(final, sort_keys=True))
PY"
}

# --- on the Mac -------------------------------------------------------------------------


launch() {
  local repo_root head held
  repo_root=$(git rev-parse --show-toplevel)
  head=$(git -C "$repo_root" rev-parse HEAD)
  ssh "$HOST" "mkdir -p '$ROOT/logs'"
  if ! ssh "$HOST" "flock -n /root/scratch/gpu.lock true" 2>/dev/null; then
    held=$(ssh "$HOST" "pgrep -af 'flock /root/scratch/gpu.lock' || true")
    echo "gpu.lock is held; not launching. Holder:" >&2
    printf '%s\n' "$held" >&2
    return 3
  fi
  git -C "$repo_root" archive "$head" \
    | ssh "$HOST" "rm -rf '$ROOT/repo' && mkdir -p '$ROOT/repo' && tar -x -C '$ROOT/repo'"
  ssh "$HOST" "printf '%s %s\n' '$head' \"\$(date -u +%FT%TZ)\" > '$ROOT/logs/launch.txt'"
  echo "launch commit $head"
  ssh "$HOST" "'$ROOT/repo/scripts/issue-178-train.sh' setup"
  ssh "$HOST" "BUDGET=$BUDGET nohup flock /root/scratch/gpu.lock \
    '$ROOT/repo/scripts/issue-178-train.sh' run \
    >> '$ROOT/logs/run.out' 2>&1 & echo 'started: '\$!"
}

check() {
  local status
  status=$(ssh "$HOST" "if [ -f '$ROOT/logs/train.done' ]; then \
    cat '$ROOT/logs/train.exit'; else echo pending; fi")
  case "$status" in
    pending)
      echo "pending: $ROOT/logs/train.done is absent"
      ssh "$HOST" "tail -n 3 '$ROOT/logs/events' 2>/dev/null; \
        tail -n 2 '$ROOT/logs/run.out' 2>/dev/null" || true
      return 3
      ;;
    0)
      echo "done: exit 0"
      ssh "$HOST" "cat '$ROOT/logs/train.done'; tail -n 2 '$ROOT/logs/events'"
      ;;
    2)
      echo "budget exceeded: the run group was stopped"
      ssh "$HOST" "cat '$ROOT/logs/train.done'"
      return 2
      ;;
    *)
      echo "failed: exit $status"
      ssh "$HOST" "cat '$ROOT/logs/train.done' 2>/dev/null; \
        tail -n 20 '$ROOT/logs/smoke.log' '$ROOT/logs/train.log' 2>/dev/null"
      return 1
      ;;
  esac
}

case ${1:-} in
  launch) launch ;;
  check) check ;;
  report) report ;;
  setup) setup ;;
  run) run ;;
  # The data assertion alone, in the shape `chain` reaches it: through `assert_ready || return
  # 1`, where bash disables errexit inside the function. Run it from `tools/` with the env
  # active; it exists so a test can prove the assertion can fail at all.
  verify)
    status=0
    assert_stream || status=$?
    exit "$status"
    ;;
  *)
    echo "usage: $0 {launch|check|report} (on the box: setup|run|verify)" >&2
    exit 2
    ;;
esac

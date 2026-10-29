#!/usr/bin/env bash
# issue-223-control.sh — round 2's control: #178's recipe one round further, on Tadabur only.
#
# The round-2 candidate adds one ingredient, tlog app-user audio. This arm trains the same warm
# start under the same schedule and seed with Tadabur in its place, so candidate minus control
# shows what the tlog data added rather than what 10,000 more steps did. It never enters the
# round-2 verdict and nothing decodes it before the round's readout, so it cannot bias the bars.
# Recipe, data and readings: docs/consonant-round2-control.md.
#
# Mac side, from the worktree root, once the code is committed:
#   scripts/issue-223-control.sh launch   # archive HEAD, start detached under gpu.lock
#   scripts/issue-223-control.sh check    # done-file check; 0 = smoke and the full run finished
#   scripts/issue-223-control.sh report   # final checkpoint SHA-256, loss and throughput
#
# `launch` refuses while the GPU lock is held (another issue's run owns the box) and while a
# previous run's directory exists; `launch --resume` continues it instead (#221: a retry must
# never destroy the checkpoints, which is why the run directories sit outside the archived
# repo). `run` asserts the data first (both source runs' consumed prefixes, the committed
# registration, the warm start's checksum), writes the frame manifest the round-2 candidate
# must replay (shard list and digest, both RNG seeds, the recipe record), smoke-tests 300
# steps with an early evaluation, gates the smoke on throughput, finite KL, a real validation
# reading and the averaged warm start, then runs 10,000 steps. It appends one line per stage
# to logs/events and writes, last, logs/train.exit and then logs/train.done; `check` reads
# that, never a log tail. A run stopped by the budget writes exit 2 and a `budget-exceeded`
# done-file.
#
# Budget: the smoke is ~0.1 GPU-h and 10,000 steps ~3.5 GPU-h at #178's 0.83 steps/s, so
# BUDGET defaults to six hours from the lock's acquisition.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-223}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
BUDGET=${BUDGET:-21600}

# The warm start: #178's candidate checkpoint, whose averaged weights continue C95. Its
# SHA-256 is the file the #178 run note records -- live and averaged weights in one file.
INIT=${INIT:-/root/scratch/issue-178/repo/tools/runs/consonant1_candidate/checkpoint.pt}
INIT_SHA=057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2
POOL=${POOL:-/root/scratch/issue-94/noise_pool.npz}
POOL_SHA=d2978d0ff3fcc14b08b21b9360d0933202e1a9f7b92fae5de9376d6ba2b3fd43
# In-training validation is monitoring only and must touch no evaluation set, so it reads
# clips staged from the run's own shards -- from the last shard of its seed-2026 permutation,
# which `assert_stream` splits *off* the stream -- rather than #94's dev clips.
VAL_CLIPS=${VAL_CLIPS:-$ROOT/val_clips}
VAL_CLIPS_COUNT=${VAL_CLIPS_COUNT:-200}

# The runs whose consumed prefixes the data excludes, in the order they happened: #94's C95
# run over `h448`'s 345 shards (seed 9494), then #178's continuation of what it left (#178's
# `--seed 178` is its stream's permutation seed). Each log's order is cross-checked against
# its own seed, so a log from another run cannot be chained in.
SOURCE_LOG_94=${SOURCE_LOG_94:-/root/scratch/issue-94/logs/aug94_candidate_v2.log}
SOURCE_LOG_178=${SOURCE_LOG_178:-/root/scratch/issue-178/logs/train.log}
SEED_94=9494
SEED_178=178
# The shard index the registration counts its rows from.
INDEX=${INDEX:-/root/scratch/issue-89/stage/full_index.jsonl}

# The candidate's seed, so candidate minus control is the data ingredient alone.
SEED=2026
STEPS=10000
SMOKE_STEPS=300
# The teacher commit #174 pre-registers (`obadx/muaalem-model-v3_2` @ 01a1ef9). Pinned
# through preflight and training: the hub's default branch can move between the two loads,
# which would silently change the run the smoke gated.
TEACHER_REVISION=01a1ef9fbe40d144ef845101e89ff924aed3fef5
# The smoke evaluates early enough to exercise the staged validation clips before the gate;
# the full run keeps #94's cadence.
SMOKE_EVAL_EVERY=150
EVAL_EVERY=2000
OUT=$ROOT/runs/consonant2_control
SMOKE=$ROOT/runs/consonant2_control_smoke
EXPOSURE_USE=consonant2.control.training

# --- on the box -------------------------------------------------------------------------


activate() {
  # The one place the box's environment is entered. Every subcommand that runs Python on the
  # box goes through it (`report` reaches `monitor` from a fresh SSH shell, where the conda
  # env -- and a bare `python` -- are not on PATH).
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
}


setup() {
  mkdir -p "$ROOT/logs" "$VAL_CLIPS"
  cd "$ROOT/repo/tools"
  activate
  python - "$VAL_CLIPS" "$VAL_CLIPS_COUNT" "$SOURCE_LOG_94" "$SOURCE_LOG_178" "$SEED" <<'PY'
import pathlib
import sys

import soundfile as sf

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.exposure import CONSONANT2_CONTROL, RowExclusion
from tadabur.shard_reader import iter_shard_rows
from training.consumed_prefix import monitoring_shard, remaining_after_logs
from training.distill_data import clip_split
from training.distill_stream import decode_row_waveform

out = pathlib.Path(sys.argv[1])
wanted = int(sys.argv[2])
# The shard the run monitors on: the last of its seed permutation. The run's step budget
# never completes the shards before it; `assert_stream` logs the position and `run` asserts
# after the full run that the log never reached it.
remaining = remaining_after_logs(
    [(pathlib.Path(sys.argv[3]), 9494), (pathlib.Path(sys.argv[4]), 178)]
)
monitor, position = monitoring_shard(remaining, int(sys.argv[5]))
for stale in out.glob("*.wav"):
    stale.unlink()

exclusion = RowExclusion.for_use(CONSONANT2_CONTROL)
written = 0
for row_index, row in enumerate(iter_shard_rows([monitor], delete_after=True)):
    if written >= wanted:
        continue  # keep consuming the shard so its cached blob is deleted
    if not exclusion.keeps((monitor, row_index), row):
        continue
    try:
        # The stream's own decoder, so staging and training decode identically by
        # construction: these clips are fed back to the same trainer as validation.
        waveform = decode_row_waveform(row)
    except Exception:  # a corrupt row is not a reason to stage nothing
        continue
    sf.write(out / f"{monitor}_{row_index:04d}.wav", waveform, TARGET_SAMPLE_RATE)
    written += 1
# The trainer's hash split decides which staged clips validation reads; it must not be empty.
val = [path for path in sorted(out.glob("*.wav")) if clip_split(path.name) == "val"]
print(f"validation clips: {written} from monitor shard {monitor} (position {position} of the "
      f"{len(remaining)} the run streams); {len(val)} in the val split")
if written < 50:
    raise SystemExit(f"only {written} clips staged: too few for a validation split")
if not val:
    raise SystemExit(
        f"none of the {written} staged clips falls in the val split: the run would log no "
        f"validation"
    )
PY
}

assert_ready() {
  echo "[assert] warm start $INIT"
  local actual
  actual=$(sha256sum "$INIT" | cut -d' ' -f1)
  if [ "$actual" != "$INIT_SHA" ]; then
    echo "[assert] init checkpoint is $actual, expected $INIT_SHA" >&2
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
  # The data: rebuild what #94's and #178's runs consumed, name the shard this run monitors
  # on (validation clips are staged from it), and refuse if the remainder has drifted from
  # the committed registration. The derivation reads only `h448`'s 345-shard spec, and the
  # trainer refuses a held-out shard before it downloads anything, so the reserve cannot
  # reach training by this path.
  #
  # Every failure is an explicit `|| return 1`: called as `assert_ready || return 1`, bash
  # disables errexit for the whole body, so a bare failing command would be swallowed and
  # a drift would launch anyway.
  mkdir -p "$ROOT/logs"
  local derivation monitor="" position="" spec=""
  derivation=$(python - "$SOURCE_LOG_94" "$SOURCE_LOG_178" "$SEED" <<'PY'
import pathlib
import sys

from tadabur.shard_reader import shard_spec
from training.consumed_prefix import monitoring_shard, remaining_after_logs

remaining = remaining_after_logs(
    [(pathlib.Path(sys.argv[1]), 9494), (pathlib.Path(sys.argv[2]), 178)]
)
monitor, position = monitoring_shard(remaining, int(sys.argv[3]))
print(f"monitor={monitor}")
print(f"position={position}")
print(f"stream={shard_spec(remaining)}")
PY
) || return 1
  while IFS='=' read -r key value; do
    case $key in
      monitor) monitor=$value ;;
      position) position=$value ;;
      stream) spec=$value ;;
    esac
  done <<<"$derivation"
  if [ -z "$monitor" ] || [ -z "$position" ] || [ -z "$spec" ]; then
    echo "[assert] derivation printed no monitor shard, position and spec:" \
      "(${derivation})" >&2
    return 1
  fi
  echo "[assert] monitor shard $monitor: validation clips are staged from it, and the run" \
    "enters it only after $position completions -- asserted by assert_monitor"
  echo "[assert] stream shards: $spec"
  python -m tadabur.exposure training-stream --use "$EXPOSURE_USE" --index "$INDEX" \
    --shards "$spec" --membership exact --note "launch assertion" --check || return 1
  printf '%s\n' "$spec" > "$ROOT/logs/spec.txt"
  printf '%s\n' "$monitor" "$position" > "$ROOT/logs/val_shard.txt"
}

assert_monitor() {  # <log>
  # The guard S1.3 asked for, kept for this arm: the monitor shard is inside the run's
  # --stream-shards, so the only thing keeping validation out of training is the step budget.
  # This reads the finished log's shard-completion lines and refuses if the run entered the
  # shard the validation clips were decoded from -- the val rows would be in-sample.
  python - "$1" "$SOURCE_LOG_94" "$SOURCE_LOG_178" "$SEED" <<'PY'
import pathlib
import sys

from training.consumed_prefix import (
    completion_order,
    monitor_reached,
    monitoring_shard,
    remaining_after_logs,
)

log = pathlib.Path(sys.argv[1])
remaining = remaining_after_logs(
    [(pathlib.Path(sys.argv[2]), 9494), (pathlib.Path(sys.argv[3]), 178)]
)
completed = completion_order(log.read_text(encoding="utf-8"))
monitor, position = monitoring_shard(remaining, int(sys.argv[4]))
if monitor_reached(remaining, completed, int(sys.argv[4])):
    raise SystemExit(
        f"{log}: {len(completed)} shards completed, so the run entered monitor shard "
        f"{monitor} at position {position}: the validation clips are training audio"
    )
print(
    f"[monitor] {len(completed)} shards completed, monitor shard {monitor} entered at "
    f"{position}: never streamed"
)
PY
}

gate_smoke() {  # <metrics.jsonl> <smoke.log>
  python - "$1" "$2" <<'PY'
import json
import sys

metrics_path, log_path = sys.argv[1], sys.argv[2]
rows, evaluations = [], []
with open(metrics_path, encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        if record.get("split") == "val":
            evaluations.append(record)
        elif "step" in record:
            rows.append(record)
if len(rows) < 2:
    raise SystemExit(f"{metrics_path}: {len(rows)} training records, too few to gate")
first, last = rows[0], rows[-1]
rate = (last["step"] - first["step"]) / (last["elapsed_s"] - first["elapsed_s"])
losses = [row["total"] for row in rows]
if not all(loss == loss and abs(loss) != float("inf") for loss in losses):
    raise SystemExit("KL is not finite")
if rate < 0.6:
    raise SystemExit(f"throughput {rate:.2f} steps/s is below the 0.6 threshold")

# The staged validation clips are the launcher's one new data path. A smoke that never
# evaluated them -- or evaluated an empty val split -- says nothing about the full run, which
# would then find out at its first evaluation.
def metrics(record):
    return {key: value for key, value in record.items() if key not in ("step", "split")}

scored = [record for record in evaluations if metrics(record)]
empty = [record["step"] for record in evaluations if not metrics(record)]
if empty:
    raise SystemExit(f"{metrics_path}: the val split was empty at steps {empty}")
if not scored:
    raise SystemExit(
        f"{metrics_path}: no evaluation of the staged clips in the smoke ({len(evaluations)} "
        f"val records): --eval-every never reached the validation path"
    )

# The warm start is the candidate's defining ingredient; nothing else in the gate proves the
# averaged weights were the ones loaded.
warm = []
with open(log_path, encoding="utf-8") as handle:
    warm = [line.strip() for line in handle if line.startswith("[init] warm-started from")]
if not warm:
    raise SystemExit(f"{log_path}: no '[init] warm-started from' line: the warm start is unproven")
if "weights=ema" not in warm[-1]:
    raise SystemExit(f"{log_path}: the warm start is not the averaged weights: {warm[-1]}")

print(
    f"[gate] {rate:.2f} steps/s over steps {first['step']}-{last['step']}, "
    f"KL {min(losses):.3f}-{max(losses):.3f}, last {losses[-1]:.3f}"
)
print(f"[gate] {len(scored)} val evaluations, last {json.dumps(scored[-1], sort_keys=True)}")
print(f"[gate] {warm[-1]}")
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

train_arm() {  # <out-dir> <steps> <log-every> <eval-every> [resume]
  local out=$1 steps=$2 log_every=$3 eval_every=$4
  local -a resume_flag=()
  if [ -n "${5:-}" ]; then
    resume_flag=(--resume)
  fi
  python -m training.distill_train \
    --preset h448 --stream-shards "$(cat "$ROOT/logs/spec.txt")" \
    --audio-root "$VAL_CLIPS" --out-dir "$out" --steps "$steps" --batch-size 32 \
    --learning-rate 5e-5 --warmup-steps 500 --cooldown-steps 3000 \
    --temperature 2.0 --nonblank-weight 3.0 --confirm-weight 2.0 \
    --ema-decay 0.999 --seed "$SEED" --save-every 2000 --log-every "$log_every" \
    --eval-every "$eval_every" --num-workers 1 --exposure-use "$EXPOSURE_USE" \
    --teacher-revision "$TEACHER_REVISION" \
    --augment --noise-pool "$POOL" --init-from "$INIT" --init-weights ema \
    "${resume_flag[@]}"
}

manifest() {
  # The frame the round-2 candidate's all-Tadabur substream has to replay exactly: the shard
  # list `assert_stream` just verified, the two RNG seeds and the augmentation recipe record
  # (keyed by source row). Written into the run directory before the first step, so the input
  # the run actually reads is what the run note copies.
  python -m training.frame_manifest --shards "$(cat "$ROOT/logs/spec.txt")" \
    --shard-order-seed "$SEED" --exposure-use "$EXPOSURE_USE" --run-dir "$OUT"
}

chain() {
  assert_ready || return 1
  manifest || return 1
  # A resume keeps the smoke it already passed; the smoke directory itself is scratch (its
  # checkpoint is deleted after the gate), so a retried smoke starts from an empty one.
  if [ ! -e "$ROOT/logs/smoke.done" ]; then
    rm -rf "$SMOKE"
    printf '%s smoke start\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
    if ! train_arm "$SMOKE" "$SMOKE_STEPS" 25 "$SMOKE_EVAL_EVERY" > "$ROOT/logs/smoke.log" 2>&1
    then
      printf '%s smoke failed\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
      return 1
    fi
    gate_smoke "$SMOKE/metrics.jsonl" "$ROOT/logs/smoke.log" >> "$ROOT/logs/events" || return 1
    smoke_has_ema "$SMOKE/checkpoint.pt" >> "$ROOT/logs/events" || return 1
    rm -f "$SMOKE/checkpoint.pt"  # the metrics stay; the run's weights supersede them
    : > "$ROOT/logs/smoke.done"
  fi
  printf '%s full start\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
  # `assert_monitor` counts the shard-completion lines of the whole run, so a resume appends
  # to the previous segment's log instead of truncating the evidence it checks.
  if [ -n "${RESUME:-}" ]; then
    exec 3>>"$ROOT/logs/train.log"
  else
    exec 3>"$ROOT/logs/train.log"
  fi
  if ! train_arm "$OUT" "$STEPS" 50 "$EVAL_EVERY" "${RESUME:-}" >&3 2>&1; then
    printf '%s full failed\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
    return 1
  fi
  assert_monitor "$ROOT/logs/train.log" >> "$ROOT/logs/events" || return 1
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
  activate
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
  # The monitoring guard first: a run that streamed the shard its validation clips came from
  # has in-sample val rows, and must not produce a clean report.
  ssh "$HOST" "cd '$ROOT/repo/tools' && '$ROOT/repo/scripts/issue-223-control.sh' monitor" \
    || return 1
  ssh "$HOST" "python3 - '$ROOT' <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / 'runs' / 'consonant2_control'
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
config = json.loads((root / 'run_config.json').read_text())
print('step', step)
print('teacher_revision', config.get('teacher_revision') or 'unpinned')
print('initialisation', json.dumps(config.get('initialisation', {}), sort_keys=True))
print('sha256', digest.hexdigest())
print('throughput {:.2f} steps/s over {} logged steps'.format(rate, len(rows)))
print('kl min {:.4f} max {:.4f} last {:.4f}'.format(min(losses), max(losses), losses[-1]))
print('grad_norm_preclip max {:.2f}'.format(max(grads)))
if final is not None:
    print('final val', json.dumps(final, sort_keys=True))
PY"
}

# --- on the Mac -------------------------------------------------------------------------


launch() {  # [--resume]
  local repo_root head held resume=""
  if [ "${1:-}" = "--resume" ]; then
    resume=1
  fi
  repo_root=$(git rev-parse --show-toplevel)
  head=$(git -C "$repo_root" rev-parse HEAD)
  ssh "$HOST" "mkdir -p '$ROOT/logs'"
  # #221: the run directories live outside the archived repo, and an existing one is refused
  # unless this is a resume -- the checkpoints in it are the arm's only artifact.
  if ssh "$HOST" "test -e '$OUT' && echo exists" | grep -q exists; then
    if [ -z "$resume" ]; then
      echo "refusing: $OUT exists (a previous run's checkpoints)." \
        "Rerun with '$0 launch --resume' to continue it." >&2
      return 4
    fi
    echo "resuming: keeping $OUT"
  fi
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
  ssh "$HOST" "'$ROOT/repo/scripts/issue-223-control.sh' setup"
  ssh "$HOST" "BUDGET=$BUDGET RESUME=$resume setsid flock /root/scratch/gpu.lock \
    '$ROOT/repo/scripts/issue-223-control.sh' run \
    < /dev/null >> '$ROOT/logs/run.out' 2>&1 & echo 'started: '\$!"
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
  launch) launch "${2:-}" ;;
  check) check ;;
  report) report ;;
  setup) setup ;;
  run) run ;;
  # The warm start and noise pool's assertion alone, in the shape `chain` reaches it through
  # `assert_ready || return 1`. It exists so a test can prove a warm start that is not #178's
  # checkpoint refuses the launch. Run it from `tools/` with the env active.
  ready)
    status=0
    assert_ready || status=$?
    exit "$status"
    ;;
  # The frame manifest alone, from the spec `assert_stream` wrote: the run's input record,
  # written at launch (and rewritten identically by a resume, whose frame cannot change).
  # Run it from `tools/`; it exists so a test can read what the launch records.
  manifest) manifest ;;
  # The data assertion alone, in the shape `chain` reaches it: through `assert_ready || return
  # 1`, where bash disables errexit inside the function. Run it from `tools/` with the env
  # active; it exists so a test can prove the assertion can fail at all.
  verify)
    status=0
    assert_stream || status=$?
    exit "$status"
    ;;
  # The smoke gate alone, in the shape `chain` calls it, so a test can prove it rejects a
  # smoke that never evaluated the staged clips or never proved the warm start.
  gate) gate_smoke "${2:?metrics.jsonl}" "${3:?smoke.log}" ;;
  # The post-run monitoring guard alone (`chain` runs it after the full run, `report`
  # before printing). It activates the box's environment itself: `report` reaches it from a
  # fresh SSH shell, where neither conda nor a bare `python` is on PATH. Run it from `tools/`.
  monitor) activate; assert_monitor "${2:-$ROOT/logs/train.log}" ;;
  *)
    echo "usage: $0 {launch [--resume]|check|report} \
(on the box: setup|run|verify|gate|monitor|manifest)" >&2
    exit 2
    ;;
esac

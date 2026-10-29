#!/usr/bin/env bash
# issue-228-train.sh — round 2's candidate: the control's recipe and frame, 3 parts tlog to 1
# part Tadabur.
#
# The control (#223) trains #178's warm start under #178's schedule and seed on Tadabur alone.
# This arm replays its 211-shard frame exactly (the committed `consonant2.control.training`
# use is the contract) and makes every batch 24 tlog + 8 Tadabur windows, so candidate minus
# control is the tlog data ingredient rather than 10,000 more steps. Recipe, data and readings:
# docs/consonant-round2-run.md (the run note) and docs/consonant-round2-preregistration.md.
#
# Mac side, from the worktree root, once the code is committed:
#   scripts/issue-228-train.sh launch           # freeze-check, gate, archive HEAD, start detached
#   scripts/issue-228-train.sh launch --smoke   # the 700-step smoke only; no freeze needed
#   scripts/issue-228-train.sh check            # done-file check; 0 = the smoke and the run finished
#   scripts/issue-228-train.sh report           # checkpoint SHA-256, KL range, consumed prefix
#
# `launch` refuses (full run only) unless the round-2 pre-registration, the consonant2 freeze
# record, `tlog/training_schedule.json` and `tlog/training_prefix.json` are committed in HEAD;
# a smoke may run before the freeze. It also refuses while `gpu.lock` is held (exit 3), while a
# previous attempt's checkpoint exists (4), below MIN_FREE_GB on /root/scratch (5). It never touches
# the box before the freeze check passes. `--resume` is refused: the tlog consumed-keys log
# cannot be replayed without duplicating rows, so the round's one attempt restarts from scratch
# with the same seed and schedule (remove the run directory, then launch again).
#
# `run` asserts the data first (the candidate's committed shard use, the committed tlog rows'
# disjointness, the warm start's checksum), writes the frame manifest the run records, samples
# the tlog cache's shard count while it runs, smoke-tests 700 steps, gates the smoke on
# throughput, both sources' finite KL, every batch 24 + 8, both splits consumed, at most
# interleave-width tlog shards on disk and the averaged warm start, then runs 10,000 steps. It
# appends one line per stage to logs/events and writes, last, logs/train.exit and then
# logs/train.done; `check` reads that, never a log tail. A run stopped by the budget writes exit
# 2 and a `budget-exceeded` done-file. No evaluation set is decoded during training: validation
# reads clips staged from the frame's monitor shard, and `assert_monitor` refuses a finished log
# that entered it.
#
# Budget: the 700-step smoke is ~0.25 GPU-h and 10,000 steps ~3.5 GPU-h at #178's 0.83 steps/s, so BUDGET
# defaults to six hours from the lock's acquisition.
set -euo pipefail

HOST=${HOST:-root@cuda-dev}
ROOT=${ISSUE_ROOT:-/root/scratch/issue-228}
CONDA=${CONDA:-/root/miniconda3/etc/profile.d/conda.sh}
BUDGET=${BUDGET:-21600}
# GiB free on /root/scratch a launch needs. The mixed stream holds up to eight tlog shards
# (0.77 GiB each at the index's largest, ~6.2 GiB) plus one Tadabur shard (~2.5 GiB), the run
# keeps two checkpoints (~1.9 GiB each: the smoke's until the gate deletes it, and the run's),
# and the archived tree and staged validation clips add ~0.5 GiB: ~13 GiB, with headroom.
MIN_FREE_GB=${MIN_FREE_GB:-16}

# The warm start: #178's candidate checkpoint, the same one the control continues. Its SHA-256
# is the file the #178 run note records -- live and averaged weights in one file.
INIT=${INIT:-/root/scratch/issue-178/repo/tools/runs/consonant1_candidate/checkpoint.pt}
INIT_SHA=057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2
POOL=${POOL:-/root/scratch/issue-94/noise_pool.npz}
POOL_SHA=d2978d0ff3fcc14b08b21b9360d0933202e1a9f7b92fae5de9376d6ba2b3fd43
# In-training validation is monitoring only and must touch no evaluation set, so it reads clips
# staged from the frame's monitor shard -- the last of the seed-2026 order, which the step budget
# never reaches; `assert_monitor` asserts that from the finished log.
VAL_CLIPS=${VAL_CLIPS:-$ROOT/val_clips}
VAL_CLIPS_COUNT=${VAL_CLIPS_COUNT:-200}
# The shard index the launch assertion recomputes the candidate's shard use from.
INDEX=${INDEX:-/root/scratch/issue-89/stage/full_index.jsonl}

SEED=2026
STEPS=10000
# 700, not the 300 the issue text names (orchestrator amendment, 2026-10-29): the frozen
# schedule's first `unclean` row is its 5,292nd, which 24 tlog windows a step reaches around
# step 600, so a 300-step smoke could never satisfy the gate's both-splits criterion. The gate
# keeps that refusal; the smoke runs long enough to meet it.
SMOKE_STEPS=700
# The teacher commit #174 pre-registers (`obadx/muaalem-model-v3_2` @ 01a1ef9). Pinned
# through preflight and training: the hub's default branch can move between the two loads.
TEACHER_REVISION=01a1ef9fbe40d144ef845101e89ff924aed3fef5
# The pre-registered batch: three parts tlog to one part Tadabur, 32 windows per step.
MIX=tlog=24,tadabur=8
# The smoke evaluates early enough to exercise the staged validation clips before the gate; the
# full run keeps #94's cadence.
SMOKE_EVAL_EVERY=150
EVAL_EVERY=2000
OUT=$ROOT/runs/consonant2_candidate
SMOKE=$ROOT/runs/consonant2_candidate_smoke
TLOG_CACHE=$SMOKE/tlog_cache
EXPOSURE_USE=consonant2.training
# The control's committed use is the frame this run replays.
FRAME_USE=consonant2.control.training
TLOG_SCHEDULE=$ROOT/repo/tools/tadabur/tlog/training_schedule.json
# The freeze artefacts the full run refuses without, committed in HEAD. The freeze record is
# #227's; its path is named here until that issue lands the readout spec that resolves it.
FREEZE_RECORD=${FREEZE_RECORD:-docs/consonant2-freeze.json}
FROZEN=(docs/consonant-round2-preregistration.md "$FREEZE_RECORD"
        tools/tadabur/tlog/training_schedule.json tools/tadabur/tlog/training_prefix.json)

# --- the frame ----------------------------------------------------------------------------


derive_frame() {
  # The candidate's frame is the control's, read from its committed shard use: its 211 shards,
  # and the monitor shard this run's seed orders last (validation clips are staged from it).
  # Fills MONITOR, POSITION and SPEC, or returns 1 -- an unreadable registry must stop a launch
  # rather than leave it streaming a different frame.
  local derivation
  derivation=$(python - "$FRAME_USE" "$SEED" <<'PY'
import sys

from tadabur.exposure import load_registry
from tadabur.shard_reader import shard_spec
from training.consumed_prefix import monitoring_shard

use = load_registry().shard_uses.get(sys.argv[1])
if use is None:
    raise SystemExit(f"{sys.argv[1]} has no committed shard use")
monitor, position = monitoring_shard(use.shards, int(sys.argv[2]))
print(f"monitor={monitor}")
print(f"position={position}")
print(f"stream={shard_spec(use.shards)}")
PY
  ) || return 1
  MONITOR=""
  POSITION=""
  SPEC=""
  while IFS='=' read -r key value; do
    case $key in
      monitor) MONITOR=$value ;;
      position) POSITION=$value ;;
      stream) SPEC=$value ;;
    esac
  done <<<"$derivation"
  if [ -z "$MONITOR" ] || [ -z "$POSITION" ] || [ -z "$SPEC" ]; then
    echo "[assert] derivation printed no monitor shard, position and spec:" \
      "(${derivation})" >&2
    return 1
  fi
}

# --- on the box ---------------------------------------------------------------------------


activate() {
  # The one place the box's environment is entered. Every subcommand that runs Python on the
  # box goes through it (`report` reaches `monitor` from a fresh SSH shell, where the conda env
  # -- and a bare `python` -- are not on PATH).
  # shellcheck source=/dev/null
  source "$CONDA"
  conda activate hifzguide
}


setup() {
  mkdir -p "$ROOT/logs" "$VAL_CLIPS"
  cd "$ROOT/repo/tools"
  activate
  derive_frame || exit 1
  python - "$VAL_CLIPS" "$VAL_CLIPS_COUNT" "$SPEC" "$MONITOR" "$POSITION" <<'PY'
import pathlib
import sys

import soundfile as sf

from tadabur.audio import TARGET_SAMPLE_RATE
from tadabur.exposure import CONSONANT2_TRAINING, RowExclusion
from tadabur.shard_reader import iter_shard_rows, parse_shard_spec
from training.distill_data import clip_split
from training.distill_stream import decode_row_waveform

out = pathlib.Path(sys.argv[1])
wanted = int(sys.argv[2])
# The shard the run monitors on: the last of its seed permutation. The run's step budget never
# completes the shards before it; `assert_monitor` refuses a finished log that entered it, so
# the validation clips were never training audio.
remaining = parse_shard_spec(sys.argv[3])
monitor, position = int(sys.argv[4]), int(sys.argv[5])
for stale in out.glob("*.wav"):
    stale.unlink()

exclusion = RowExclusion.for_use(CONSONANT2_TRAINING)
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
  # The data: the candidate's committed shard use recomputed against the 385-shard index, the
  # committed tlog rows checked against every evaluation use, and only then the spec written.
  #
  # `check_row_set_disjoint` is called through `tlog-training-disjoint`: the shard/clip check
  # refuses tlog by design, so it is never used for these rows. Every failure is an explicit
  # `|| return 1`: called as `assert_ready || return 1`, bash disables errexit for the whole
  # body, so a bare failing command would be swallowed and a drift would launch anyway.
  mkdir -p "$ROOT/logs"
  derive_frame || return 1
  echo "[assert] monitor shard $MONITOR: validation clips are staged from it, and the run" \
    "enters it only after $POSITION completions -- asserted by assert_monitor"
  echo "[assert] stream shards: $SPEC"
  python -m tadabur.exposure training-stream --use "$EXPOSURE_USE" --index "$INDEX" \
    --shards "$SPEC" --membership exact --note "launch assertion" --check || return 1
  python -m tadabur.exposure tlog-training-disjoint || return 1
  printf '%s\n' "$SPEC" > "$ROOT/logs/spec.txt"
  printf '%s\n' "$MONITOR" "$POSITION" > "$ROOT/logs/val_shard.txt"
}

assert_monitor() {  # <log>
  # The monitor shard is inside the run's --stream-shards, so the only thing keeping validation
  # out of training is the step budget. This reads the finished log's shard-completion lines and
  # refuses if the run entered the shard the validation clips were decoded from.
  derive_frame || return 1
  python - "$1" "$SPEC" "$MONITOR" "$POSITION" "$SEED" <<'PY'
import pathlib
import sys

from tadabur.shard_reader import parse_shard_spec
from training.consumed_prefix import completion_order, monitor_reached

log = pathlib.Path(sys.argv[1])
remaining = parse_shard_spec(sys.argv[2])
monitor, position = int(sys.argv[3]), int(sys.argv[4])
# A run that restarts replays nothing; the guard counts the shards the run has left, once each.
completed = list(dict.fromkeys(completion_order(log.read_text(encoding="utf-8"))))
if monitor_reached(remaining, completed, int(sys.argv[5])):
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

gate_smoke() {  # <metrics.jsonl> <smoke.log> <shards.log> <consumed keys> <mix>
  python - "$1" "$2" "$3" "$4" "$MIX" <<'PY'
import gzip
import json
import math
import sys

metrics_path, log_path, shards_path, consumed_path, mix = sys.argv[1:6]
counts = dict(part.split("=") for part in mix.split(","))
wanted = {f"batch_{name}": int(count) for name, count in counts.items()}

rows = []
with open(metrics_path, encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        if "split" not in record:  # the val records are evaluations, not steps
            rows.append(record)
if len(rows) < 2:
    raise SystemExit(f"{metrics_path}: {len(rows)} training records, too few to gate")
first, last = rows[0], rows[-1]
rate = (last["step"] - first["step"]) / (last["elapsed_s"] - first["elapsed_s"])
if rate < 0.6:
    raise SystemExit(f"throughput {rate:.2f} steps/s is below the 0.6 threshold")

# Both sources trained: their KL must exist and be finite on every logged step, and every
# batch must be exactly the pre-registered mix.
for record in rows:
    for name in counts:
        value = record.get(f"kl_{name}")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(
                f"step {record['step']}: kl_{name} is {value!r}, not a finite number")
    seen = {key: record.get(key) for key in wanted}
    if seen != wanted:
        raise SystemExit(
            f"step {record['step']}: batch composition {seen}, not {wanted} (the mix {mix})")

# Both splits must have been consumed: one split alone is not the round-2 pool.
with gzip.open(consumed_path, "rt", encoding="utf-8") as handle:
    splits = {json.loads(line)["split"] for line in handle if line.strip()}
if missing := {"clean", "unclean"} - splits:
    raise SystemExit(f"{consumed_path}: no {sorted(missing)} rows were consumed ({sorted(splits)})")

# The disk bound: at most interleave_width (8) tlog shards on disk at once, sampled while the
# smoke ran. An empty sample log means the sampler never ran, so the bound is unproven.
with open(shards_path, encoding="utf-8") as handle:
    samples = [int(line.split()[-1]) for line in handle if line.strip()]
if not samples:
    raise SystemExit(f"{shards_path}: no shard-count samples: the disk bound is unproven")
if not 1 <= max(samples) <= 8:
    raise SystemExit(
        f"{shards_path}: {max(samples)} tlog shards on disk at once, outside 1..8")

# The warm start is the candidate's defining ingredient; nothing else in the gate proves the
# averaged weights were the ones loaded.
with open(log_path, encoding="utf-8") as handle:
    warm = [line.strip() for line in handle if line.startswith("[init] warm-started from")]
if not warm:
    raise SystemExit(f"{log_path}: no '[init] warm-started from' line: the warm start is unproven")
if "weights=ema" not in warm[-1]:
    raise SystemExit(f"{log_path}: the warm start is not the averaged weights: {warm[-1]}")

losses = [row["total"] for row in rows]
print(
    f"[gate] {rate:.2f} steps/s over steps {first['step']}-{last['step']}, "
    f"KL {min(losses):.3f}-{max(losses):.3f}, last {losses[-1]:.3f}"
)
print(f"[gate] every batch {wanted}; splits consumed: {sorted(splits)}; "
      f"max {max(samples)} tlog shards on disk")
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

train_arm() {  # <out-dir> <steps> <log-every> <eval-every>
  local out=$1 steps=$2 log_every=$3 eval_every=$4
  python -m training.distill_train \
    --preset h448 --stream-shards "$(cat "$ROOT/logs/spec.txt")" \
    --audio-root "$VAL_CLIPS" --out-dir "$out" --steps "$steps" --batch-size 32 \
    --learning-rate 5e-5 --warmup-steps 500 --cooldown-steps 3000 \
    --temperature 2.0 --nonblank-weight 3.0 --confirm-weight 2.0 \
    --ema-decay 0.999 --seed "$SEED" --save-every 2000 --log-every "$log_every" \
    --eval-every "$eval_every" --num-workers 1 --exposure-use "$EXPOSURE_USE" \
    --teacher-revision "$TEACHER_REVISION" \
    --augment --noise-pool "$POOL" --init-from "$INIT" --init-weights ema \
    --mix "$MIX" --tlog-schedule "$TLOG_SCHEDULE" \
    --frame-manifest "$OUT/frame_manifest.json"
}

manifest() {
  # The frame the run records: the control's shard list `assert_stream` verified, the two RNG
  # seeds and the augmentation recipe record (keyed by source row). Written before the first
  # step, so the input the run actually reads is what the run note copies.
  python -m training.frame_manifest --shards "$(cat "$ROOT/logs/spec.txt")" \
    --shard-order-seed "$SEED" --exposure-use "$EXPOSURE_USE" --run-dir "$OUT"
}

smoke_is_current() {  # <commit>
  # The smoke's gate result is evidence for the code that ran it, not for the arm: a launch of
  # another commit must smoke again. `launch` writes logs/launch.txt (`<commit> <time>`) before
  # the chain starts, and `chain` stores that commit in smoke.done when the smoke passes.
  [ -s "$ROOT/logs/smoke.done" ] && [ "$(cat "$ROOT/logs/smoke.done")" = "$1" ]
}

chain() {
  assert_ready || return 1
  manifest || return 1
  local commit
  commit=$(cut -d' ' -f1 "$ROOT/logs/launch.txt" 2>/dev/null || true)
  if [ -z "$commit" ]; then
    echo "[smoke] $ROOT/logs/launch.txt names no launch commit: the smoke's evidence is" \
      "tied to the code that ran it, so this run cannot decide whether to smoke again" >&2
    return 1
  fi
  if ! smoke_is_current "$commit"; then
    rm -f "$ROOT/logs/smoke.done"
    rm -rf "$SMOKE"
    printf '%s smoke start\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
    if ! train_arm "$SMOKE" "$SMOKE_STEPS" 25 "$SMOKE_EVAL_EVERY" > "$ROOT/logs/smoke.log" 2>&1
    then
      printf '%s smoke failed\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
      return 1
    fi
    gate_smoke "$SMOKE/metrics.jsonl" "$ROOT/logs/smoke.log" "$ROOT/logs/shards.log" \
      "$SMOKE/tlog_consumed.jsonl.gz" "$MIX" >> "$ROOT/logs/events" || return 1
    smoke_has_ema "$SMOKE/checkpoint.pt" >> "$ROOT/logs/events" || return 1
    rm -f "$SMOKE/checkpoint.pt"  # the metrics stay; the run's weights supersede them
    printf '%s\n' "$commit" > "$ROOT/logs/smoke.done"
  fi
  # A smoke-only launch exists for the window before the freeze: it must not spend the round's
  # one full attempt.
  if [ -n "${SMOKE_ONLY:-}" ]; then
    printf '%s smoke only; full run not started\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
    return 0
  fi
  printf '%s full start\n' "$(date -u +%FT%TZ)" >> "$ROOT/logs/events"
  # A mixed run cannot resume (its consumed-keys log would duplicate rows), so no checkpoint is
  # ever continued here: `run` starts the full arm from scratch and `launch` refuses --resume.
  exec 3>"$ROOT/logs/train.log"
  if ! train_arm "$OUT" "$STEPS" 50 "$EVAL_EVERY" >&3 2>&1; then
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
  : > "$ROOT/logs/shards.log"
  # Globals: the EXIT trap can fire after `run` returns, when its locals are gone.
  SAMPLER=""
  SHARD_SAMPLER=""
  WATCHDOG=""
  CHAIN=""
  finish() {
    local code=$?
    local group
    for group in "$CHAIN" "$SHARD_SAMPLER" "$SAMPLER" "$WATCHDOG"; do
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
  # Job control gives every background job its own process group, so the trap and the deadline
  # stop the chain and the trainer with every worker beneath them.
  set -m
  ( while true; do
      printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$(du -sb "$ROOT" | cut -f1)" \
        "$(df -B1 --output=avail /root/scratch | tail -1)" >> "$ROOT/logs/df.log"
      sleep 60
    done ) >/dev/null 2>&1 &
  SAMPLER=$!
  # The tlog disk bound, sampled: the reader deletes each shard as it is exhausted, so the
  # peak count in this log is what the smoke gate checks against the schedule's 8.
  ( while true; do
      printf '%s %s\n' "$(date -u +%FT%TZ)" \
        "$(find "$TLOG_CACHE" -name '*.parquet' 2>/dev/null | wc -l)" \
        >> "$ROOT/logs/shards.log"
      sleep 5
    done ) >/dev/null 2>&1 &
  SHARD_SAMPLER=$!
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
  ssh "$HOST" "cd '$ROOT/repo/tools' && '$ROOT/repo/scripts/issue-228-train.sh' monitor" \
    || return 1
  ssh "$HOST" "python3 - '$ROOT' <<'PY'
import gzip
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / 'runs' / 'consonant2_candidate'
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
consumed = {'clean': 0, 'unclean': 0}
with gzip.open(root / 'tlog_consumed.jsonl.gz', 'rt', encoding='utf-8') as handle:
    for line in handle:
        consumed[json.loads(line)['split']] += 1
print('step', step)
print('teacher_revision', config.get('teacher_revision') or 'unpinned')
print('mix', config.get('mix'))
print('tlog_schedule_sha256', config.get('tlog_schedule_sha256'))
print('frame_manifest_sha256', config.get('frame_manifest_sha256'))
print('initialisation', json.dumps(config.get('initialisation', {}), sort_keys=True))
print('sha256', digest.hexdigest())
print('throughput {:.2f} steps/s over {} logged steps'.format(rate, len(rows)))
print('kl min {:.4f} max {:.4f} last {:.4f}'.format(min(losses), max(losses), losses[-1]))
print('kl_tlog last {:.4f} kl_tadabur last {:.4f}'.format(rows[-1]['kl_tlog'],
                                                         rows[-1]['kl_tadabur']))
print('grad_norm_preclip max {:.2f}'.format(max(grads)))
print('consumed clips', json.dumps(consumed, sort_keys=True))
if final is not None:
    print('final val', json.dumps(final, sort_keys=True))
PY"
}

# --- on the Mac ---------------------------------------------------------------------------


assert_disk() {
  # The box's disk is the binding constraint, and a full disk kills a multi-hour run partway.
  # Refuse before anything is archived or staged, so nothing is half-copied onto a box that
  # cannot hold the run.
  local avail
  avail=$(ssh "$HOST" "df -B1 --output=avail /root/scratch | tail -1")
  case $avail in
    ''|*[!0-9]*)
      echo "refusing: could not read free space on $HOST:/root/scratch ($avail)" >&2
      return 1
      ;;
  esac
  if [ "$avail" -lt $((MIN_FREE_GB * 1024 * 1024 * 1024)) ]; then
    echo "refusing: $((avail / 1024 / 1024 / 1024)) GiB free on /root/scratch, below the" \
      "$MIN_FREE_GB GiB a mixed run needs (eight tlog shards, one Tadabur shard, two" \
      "checkpoints, the archived tree); free space, or set MIN_FREE_GB if the budget is" \
      "genuinely lower" >&2
    return 1
  fi
  echo "free space: $((avail / 1024 / 1024 / 1024)) GiB on /root/scratch"
}

freeze_check() {
  # The full run may start only after the round's pre-registration and freeze artefacts are
  # **committed**, not merely on disk: a draft is not a pre-registration, and a run launched
  # against one cannot be cited. Locally runnable so the refusal is tested without a box.
  local root missing="" name
  root=$(git rev-parse --show-toplevel) || return 1
  for name in "${FROZEN[@]}"; do
    if ! git -C "$root" cat-file -e "HEAD:$name" 2>/dev/null; then
      echo "freeze-check: $name is not committed in HEAD" >&2
      missing=1
    fi
  done
  if [ -n "$missing" ]; then
    echo "refusing the full run: commit the round-2 freeze artefacts (the smoke may run" \
      "before them with 'launch --smoke')" >&2
    return 1
  fi
  echo "freeze: ${#FROZEN[@]} artefacts committed in HEAD"
}

launch() {
  local repo_root head smoke=""
  case "${1:-}" in
    --smoke) smoke=1 ;;
    --resume)
      echo "refusing --resume: the mixed stream's tlog consumed-keys log cannot be replayed" \
        "without duplicating rows, so the round's one attempt restarts from scratch with the" \
        "same seed and schedule. Remove $OUT and launch again." >&2
      return 2
      ;;
    "") ;;
    *)
      echo "usage: $0 launch [--smoke]" >&2
      return 2
      ;;
  esac
  repo_root=$(git rev-parse --show-toplevel)
  head=$(git -C "$repo_root" rev-parse HEAD)
  # Nothing touches the box before the full run's freeze check passes.
  if [ -z "$smoke" ]; then
    freeze_check || return 1
  fi
  ssh "$HOST" "mkdir -p '$ROOT/logs'"
  # The lock first: nothing on the box is touched while another run holds the GPU.
  if ! ssh "$HOST" "flock -n /root/scratch/gpu.lock true" 2>/dev/null; then
    echo "gpu.lock is held; not launching. Holder:" >&2
    ssh "$HOST" "pgrep -af 'flock /root/scratch/gpu.lock' || true" >&2
    return 3
  fi
  # The run directories live outside the archived repo. A previous full attempt's checkpoint is
  # refused: a mixed run cannot resume, and a restart must not append to its metrics log. A
  # directory a smoke-only launch left behind (it holds the frame manifest, no checkpoint) is
  # scratch, so it is cleared and the full attempt starts empty.
  if ssh "$HOST" "test -s '$OUT/checkpoint.pt' && echo exists" | grep -q exists; then
    echo "refusing: $OUT holds a previous attempt's checkpoint. A mixed run cannot resume;" \
      "remove the directory to start the attempt again." >&2
    return 4
  fi
  ssh "$HOST" "rm -rf '$OUT'"
  assert_disk || return 5
  git -C "$repo_root" archive "$head" \
    | ssh "$HOST" "rm -rf '$ROOT/repo' && mkdir -p '$ROOT/repo' && tar -x -C '$ROOT/repo'"
  ssh "$HOST" "printf '%s %s\n' '$head' \"\$(date -u +%FT%TZ)\" > '$ROOT/logs/launch.txt'"
  echo "launch commit $head"
  ssh "$HOST" "'$ROOT/repo/scripts/issue-228-train.sh' setup"
  local env_prefix="BUDGET=$BUDGET"
  if [ -n "$smoke" ]; then
    env_prefix="$env_prefix SMOKE_ONLY=1"
  fi
  ssh "$HOST" "$env_prefix setsid flock /root/scratch/gpu.lock \
    '$ROOT/repo/scripts/issue-228-train.sh' run \
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
  # The frame alone: the control's 211 shards and the monitor shard this run's seed orders
  # last. A test reads it; `setup` and `assert_stream` derive it through the same function.
  frame) derive_frame && printf 'monitor=%s\nposition=%s\nstream=%s\n' \
    "$MONITOR" "$POSITION" "$SPEC" ;;
  # The freeze check alone: the full run's gate, locally runnable from a checkout.
  freeze-check) freeze_check ;;
  # The warm start's assertion alone, in the shape `chain` reaches it through
  # `assert_ready || return 1`. Run it from `tools/` with the env active.
  ready)
    status=0
    assert_ready || status=$?
    exit "$status"
    ;;
  # The frame manifest alone, from the spec `assert_stream` wrote. Run it from `tools/`.
  manifest) manifest ;;
  # The data assertion alone, in the shape `chain` reaches it. Run it from `tools/`.
  verify)
    status=0
    assert_stream || status=$?
    exit "$status"
    ;;
  # The smoke gate alone, so a test can prove it rejects a smoke that missed a criterion.
  gate) gate_smoke "${2:?metrics.jsonl}" "${3:?smoke.log}" "${4:?shards.log}" \
    "${5:?consumed keys}" "$MIX" ;;
  # Whether smoke.done belongs to this launch commit (`chain`'s decision alone). It exists so a
  # test can prove a code change between the smoke and the full launch re-runs the smoke.
  smoke-valid) smoke_is_current "${2:?commit}" ;;
  # The post-run monitoring guard alone. It activates the box's environment itself: `report`
  # reaches it from a fresh SSH shell, where neither conda nor a bare `python` is on PATH.
  monitor) activate; assert_monitor "${2:-$ROOT/logs/train.log}" ;;
  *)
    echo "usage: $0 {launch [--smoke]|check|report} \
(on the box: setup|run|verify|frame|ready|manifest|freeze-check|gate|smoke-valid|monitor)" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
# Drive the ADR-0016 re-read corpus run end to end (HifzGuide #70).
#
# Five stages, and the order is the point. Only stage 1 is resumable and only stage 1
# costs per shard; stages 2-5 reprocess their whole input, so they run **once** over the
# accumulated sink, not per shard:
#
#   1 filter    mine rejects + stage the clean re-read WAVs   per shard, resumable
#   2 stage     timed decode + VAD intervals over those WAVs  once
#   3 recut     clip neighbour-ayah bleed, re-gate, keep/drop  once
#   4 scenario  stage the corpus bundle + excision pairs       once
#   5 report    yield, bleed prevalence                        once, no GPU
#
# Two wiring details this exists to get right, because getting either wrong is silent:
#   * ``--delete-shards`` throws away the 2.4 GB parquet after each shard but keeps
#     $RUN/clips — the staged clean re-read WAVs every later stage reads.
#   * stage 4 takes ``--recuts``. Without it the bundle stages un-re-cut audio and
#     nothing complains.
#
# Usage:
#   bash tadabur/run_corpus.sh                 # all five, in order
#   bash tadabur/run_corpus.sh stage recut     # just those, in the order given
#   RUN=tadabur/topup SHARDS=31-40 bash tadabur/run_corpus.sh
#
# Run it detached (tmux) and read the artifacts back; stage 1 is hours on a full run.
set -euo pipefail

RUN=${RUN:-tadabur/corpus_run}
SHARDS=${SHARDS:-20-30}
BATCH=${BATCH:-4}
TARGET=${TARGET:-500}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

shards_run() {  # shards the spec names, for the yield extrapolation
  python -c "from tadabur.shard_reader import parse_shard_spec; print(len(parse_shard_spec('$SHARDS')))"
}

stage_filter() {
  python -m tadabur.filter \
    --manifest         "$RUN/passing.jsonl" \
    --rejects          "$RUN/rejects.jsonl" \
    --reject-audio-out "$RUN/clips" \
    --shards "$SHARDS" --batch-size "$BATCH" --delete-shards
}

stage_stage() {
  python -m tadabur.bleed_stage \
    --rejects "$RUN/rejects.jsonl" \
    --clips   "$RUN/clips" \
    --out     "$RUN/decodes.jsonl" \
    --batch-size "$BATCH"
}

stage_recut() {
  python -m tadabur.bleed_recut \
    --decodes     "$RUN/decodes.jsonl" \
    --clips       "$RUN/clips" \
    --out         "$RUN/recuts.jsonl" \
    --recut-clips "$RUN/recut_clips"
}

stage_scenario() {
  python -m tadabur.scenario \
    --rejects "$RUN/rejects.jsonl" \
    --clips   "$RUN/clips" \
    --recuts  "$RUN/recuts.jsonl" \
    --out     "$RUN/scenario" \
    --batch-size "$BATCH"
  python -m tadabur.scenario --verify "$RUN/scenario"
}

stage_report() {
  python -m tadabur.reject_yield \
    --passing "$RUN/passing.jsonl" --rejects "$RUN/rejects.jsonl" \
    --shards-run "$(shards_run)" --target "$TARGET" --json "$RUN/yield.json"
  python -m tadabur.bleed_detect --rejects "$RUN/rejects.jsonl" --json "$RUN/bleed.json"
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(filter stage recut scenario report)

for stage in "${stages[@]}"; do
  echo "=== $stage  $(date -Is) ==="
  case "$stage" in
    filter)   time stage_filter ;;
    stage)    time stage_stage ;;
    recut)    time stage_recut ;;
    scenario) time stage_scenario ;;
    report)   time stage_report ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done
echo "=== done  $(date -Is) ==="
df -h . | tail -1

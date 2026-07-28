#!/usr/bin/env bash
# Run the remaining ADR-0004 ablation rungs back to back once rung 2 finishes.
#
# Rung 2 (whole-clip phoneme) is already running; this waits on its PID, then trains
# rung 3 (joint phoneme + detached waqf) and rung 1 (the segment-scoped control) so the
# GPU never idles between them. All three consume the ADR-0003-correct labels
# (windowed_labels_v2 / segmented_labels_v2) and the pre-existing soft-label store, whose
# window geometry was verified to match those labels exactly.
set -euo pipefail

RUNG2_PID="${1:?usage: run_remaining_rungs.sh <rung2-pid>}"
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hifzguide

D=tadabur/audit_run/seg_v21
AUDIO=tadabur/audit_run/clips_v2
EVAL_ARGS=(--eval-segment-manifest tadabur/audit_run/segment_manifest_v2.jsonl
           --eval-audio-dir tadabur/audit_run/segment_audio_v2)
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "waiting for rung 2 (pid $RUNG2_PID)..."
while kill -0 "$RUNG2_PID" 2>/dev/null; do sleep 60; done
echo "rung 2 finished at $(date -Is)"

echo "=== rung 3: joint phoneme + detached waqf ==="
python -u -m training.joint_waqf train \
    --labels "$D/windowed_labels_v2.jsonl" \
    --soft-labels "$D/soft" \
    --audio-dir "$AUDIO" \
    --out-dir "$D/rung3_v2" \
    "${EVAL_ARGS[@]}" > "$D/rung3_v2.log" 2>&1
echo "rung 3 finished at $(date -Is)"

echo "=== rung 1: segment-scoped control ==="
python -u -m training.whole_clip_phoneme train \
    --labels "$D/segmented_labels_v2.jsonl" \
    --audio-dir "$AUDIO" \
    --out-dir "$D/rung1_v2" \
    "${EVAL_ARGS[@]}" > "$D/rung1_v2.log" 2>&1
echo "rung 1 finished at $(date -Is)"

echo "ALL RUNGS COMPLETE at $(date -Is)"

# Waqf event fixtures (P7.F0, #27)

Human-adjudicated candidate waqf boundaries — the event-level ground truth ADR-0004
(`docs/adr/0004-waqf-head-and-joint-whole-clip-fine-tune.md`) requires, because a
silence VAD detects *silence*, not whether a boundary is a real stop. Grades the
three things the F0 eval must measure: **false-waqf** at true-wasl boundaries,
**false-wasl** at genuine stops, and a **mid-word-closure** rejection set (qalqala on
ق/ط, the hamza in شَيء — silence the snap must not treat as waqf).

- `waqf_events.jsonl` — one `WaqfEventEntry` per line (see
  `tools/tadabur/waqf_event_fixtures.py` for the schema and validating loader).

This directory ships the schema and the (empty) canonical file — **not** the labelled
data. The adjudication UI (`python -m tadabur.waqf_audit_ui`) fills it in from a
candidate-boundary worklist (`python -m tadabur.waqf_event_sampler`), and F0's
event-level eval reads it back through `load_waqf_events`.

## Pipeline (F0, #30)

The whole flow is **torch-free** — every candidate is read off the segmentation pass's
existing artifacts (`segment_score` manifest + its VAD pause map), no model re-run:

```bash
# 1. Derive the candidate-boundary manifest (waqf / wasl / mid-word-closure) from the
#    segmentation artifacts. Emits one WaqfCandidate per boundary.
python -m tadabur.waqf_candidates \
  --segment-manifest audit_run/segment_manifest_v4.jsonl \
  --vad-pauses audit_run/vad_pauses_v4.json \
  --out audit_run/waqf_candidates.jsonl

# 2. Split into reciter-disjoint calibration + test partitions, both excluding the
#    D2/D3 fine-tune reciters (pass their ids and/or the training manifest). A reciter
#    lands wholly on one side, so calibration/test/training share no reciter.
python -m tadabur.waqf_partition \
  --candidates audit_run/waqf_candidates.jsonl \
  --train-reciters <D2/D3 reciter ids...> [--train-manifest <fine-tune manifest.jsonl>] \
  --calibration audit_run/waqf_cand.calibration.jsonl \
  --test        audit_run/waqf_cand.test.jsonl \
  --test-fraction 0.5 --seed 0 --report audit_run/waqf_partition.json

# 3. Sample a per-partition clip worklist (the clips the reviewer walks through).
python -m tadabur.waqf_event_sampler \
  --candidates audit_run/waqf_cand.calibration.jsonl \
  --clips audit_run/waqf_clip_worklist.calibration.jsonl --seed 0

# 4. Adjudicate each partition into its own frozen fixture file (repeat for --test).
python -m tadabur.waqf_audit_ui \
  --candidates audit_run/waqf_cand.calibration.jsonl \
  --clips      audit_run/waqf_clip_worklist.calibration.jsonl \
  --audio-dir  audit_run/clips_v2 \
  --fixtures   waqf_event_fixtures/waqf_events.calibration.jsonl
```

Freeze both fixture files (and `waqf_partition.json`) once adjudicated — they are the
versioned calibration/test sets F1/F2 read, both held out of training. Candidate class
tallies are approximate on purpose: `wasl` covers every interior word edge, while
`mid_word_closure` is rare (only VAD pauses that fell mid-word).

## Correction-based per-clip adjudication

The UI works on the **clip** as the review unit, not one card per boundary. The
candidate manifest (`--candidates`) is treated as the **assumed-correct baseline**: for
each clip the reviewer plays the whole recitation, and only marks the boundaries the
detector got wrong — a **false positive** (a predicted stop that is really `wasl`), a
**false negative** (a word edge the detector called `wasl` that is actually a stop), or
a **class fix** (`waqf` ↔ `mid_word_closure`). Every word edge is a candidate, so a
false negative can be marked on any edge, not just the sampled ones.

Only **overrides** (`verdict` ≠ `predicted`) are written to `waqf_events.jsonl`; the
ground truth for a clip is `predicted ⊕ overrides`. A clip must be explicitly marked
**reviewed** to enter the eval set — this distinguishes "reviewed, no errors found"
from "never seen". Reviewed clip ids persist in a sibling `waqf_reviewed_clips.json`
next to the fixtures file, so the per-line event schema stays unchanged.

The UI plays whole clips from `--audio-dir`, which is the staging directory
`tadabur.waqf_segments` already wrote (each clip under its `audio_filename` — the same
16 kHz clips the VAD/segmentation pass analysed to propose the candidates). No
separate audio-export step is needed: each clip is served by that filename.

Each line carries: `clip_id`, `audio_ref`, `surah_ayah`, `boundary_index`,
`word_index`, `start_s`, `end_s`, `predicted`, `verdict`, `note`. Both `predicted`
(the detector's class) and `verdict` (the human's) are one of `waqf` / `wasl` /
`mid_word_closure`; their disagreement is the metric.

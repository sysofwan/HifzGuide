# Waqf event fixtures (P7.F0, #27)

Human-adjudicated candidate waqf boundaries. They were collected as the event-level
ground truth for the waqf head (ADR-0004), which ADR-0011 has since dropped. The verdicts
remain **human truth** and are reused as truth sites (ADR-0011): `waqf` verdicts give
*sukun at a pause* sites and `wasl` verdicts give word-final haraka sites. Never edit or
regenerate these files.

- `waqf_events.calibration.jsonl`, `waqf_events.test.jsonl` — the frozen, self-contained
  per-boundary ground truth, one `WaqfEventEntry` per line. Read them with
  `tadabur.waqf_event_fixtures.load_waqf_events`, which validates the schema.
- `waqf_partition.json` — the reciter→partition assignment, counts,
  `must_exclude_reciters`, the calibration/test clip lists that
  `training.windowed_labels --held-out-clips` keeps out of training, and a `binary` block
  of waqf / not-waqf / `closure_tag` counts per partition.

## How they were produced

Everything ran torch-free on the segmentation pass's artifacts (`segment_score` manifest
plus its VAD pause map). `tadabur.waqf_candidates` derived one candidate per boundary
(`waqf` / `wasl` / `mid_word_closure`), and `tadabur.waqf_event_sampler` drew a
per-class clip worklist. A human then adjudicated each clip in a waqf audit UI. A freeze
step materialized the ground truth over the reviewed clips and split it
reciter-disjoint into the two partitions above. The audit UI, the partition/freeze
step and the event eval were removed with the waqf head. They remain in git history
before the commit that removed them.

## Correction-based per-clip adjudication

The clip was the review unit. The candidate manifest was the **assumed-correct
baseline**: the reviewer played the whole recitation and marked only the boundaries the
detector got wrong — a **false positive** (a predicted stop that is really `wasl`), a
**false negative** (a word edge called `wasl` that is actually a stop), or a **class
fix** (`waqf` ↔ `mid_word_closure`). Only clips explicitly marked **reviewed** entered
the frozen set, which separates "reviewed, no errors found" from "never seen".

## Schema

Each line carries: `clip_id`, `audio_ref`, `surah_ayah`, `boundary_index`,
`word_index`, `start_s`, `end_s`, `predicted`, `verdict`, `note`. Both `predicted`
(the detector's class) and `verdict` (the human's) are one of `waqf` / `wasl` /
`mid_word_closure`. **Scoring is binary** (see below): a boundary is a positive iff
`verdict == "waqf"`; `wasl` and `mid_word_closure` are both *not-waqf*.

## Freezing

The freeze carried every candidate boundary of each reviewed clip with its human
`verdict` (an override where one existed, else the detector's `predicted`). An override
whose `(clip_id, boundary_index)` no longer named a boundary in the candidate baseline
was **stale**: it was dropped from the frozen set and listed under `waqf_partition.json`'s
`stale_overrides`, so it never distorted the ground truth.

## Scoring is binary (waqf vs not-waqf)

The deployed system only ever decides **waqf** (a real stop) vs **not-waqf**, so the eval
is scored that way: `verdict == "waqf"` is the only positive; both `wasl` and
`mid_word_closure` are not-waqf. `mid_word_closure` is **not a third product outcome** — it
is a VAD silence the segmenter did *not* treat as a segment boundary, i.e. a within-word
articulation closure (qalqala on ق/ط, hamza in شَيء, madd elongation) — and it is kept only
as a **diagnostic tag** on the not-waqf class to report the hard-negative rejection rate.
That tag is best-effort: reviewers may collapse an interior silence straight to `wasl`, so
it under-counts, but every `mid_word_closure` is unambiguously not-waqf and so never moves
the binary ground truth. `waqf_partition.json`'s `binary` block records the
waqf / not-waqf / `closure_tag` counts per partition.


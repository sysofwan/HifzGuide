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

Each line carries: `clip_id`, `audio_ref`, `surah_ayah`, `boundary_index`,
`word_index`, `start_s`, `end_s`, `predicted`, `verdict`, `note`. Both `predicted`
(the detector's class) and `verdict` (the human's) are one of `waqf` / `wasl` /
`mid_word_closure`; their disagreement is the metric.

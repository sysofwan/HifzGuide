# Eval fixtures — should-accept / should-reject

Two curated, hand-labelled eval sets that the **P3.5 poison audit (#6)** produces
and the **P3.6 eval harness (#7)** consumes. Per ADR-0001 the fine-tune eval is
**two-sided and targeted**, not aggregate PER:

- **`should_accept.jsonl`** — acceptable-imperfect amateur clips the fine-tuned
  model *should admit*. Measures the **recall gain** vs the over-strict base model.
- **`should_reject.jsonl`** — genuinely-wrong substitutions the model *must still
  reject*. Measures that **discrimination is retained**, not collapsed (aggregate
  PER can improve while this distinction collapses, so it cannot be the metric).

These two files ship **empty**. #6 fills them in; do not commit labelled data
here without going through the audit. Load them via
`tadabur.eval_fixtures.load_should_accept()` / `load_should_reject()`, which
validate every line against the schema below.

## Schema

One JSON object per line (JSONL). Blank lines and lines starting with `#` are
ignored. Fields (see `EvalFixtureEntry` in `../eval_fixtures.py`):

| field         | type   | notes                                                        |
| ------------- | ------ | ------------------------------------------------------------ |
| `clip_id`     | string | stable id for the labelled clip                              |
| `audio_ref`   | string | Tadabur `audio_filename` (matches worklist + filter manifest)|
| `surah_ayah`  | string | `"surah:ayah"`                                               |
| `contrast`    | string | one of the audit buckets (see below)                         |
| `verdict`     | string | `"accept"` or `"reject"` — must match the file it lives in    |
| `note`        | string | optional free-text rationale from the labeller (default `""`)|

`contrast` must be one of the seven audit buckets — the six balanced soft pairs
`ذ↔ز, ت↔ط, ض↔ظ, ق↔ك, س↔ص, ح↔ه` (codepoint-ordered labels), `shadda`, or the
`marginal` `match_ratio` band — i.e. `tadabur.contrast_attribution.contrast_vocabulary()`.

## Example line

```json
{"clip_id": "acc-0001", "audio_ref": "reciter42/002/000123.wav", "surah_ayah": "2:255", "contrast": "س↔ص", "verdict": "accept", "note": "amateur س reads slightly emphatic; still acceptable"}
```

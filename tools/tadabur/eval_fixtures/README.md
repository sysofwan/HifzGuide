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

## Labelling workflow (the audit UI)

The audit is done in a small local web UI — no framework, stdlib only — that plays
each admitted clip and records one **B (accept)** / **C (reject)** verdict per
worklist row, writing straight into the two files above:

```bash
# 1. Sample the per-contrast worklist and export the clips' audio (see audit_sampler).
python -m tadabur.audit_sampler --manifest passing_subset.jsonl \
  --worklist audit_worklist.jsonl --seed 0 --audio-dir audit_audio/

# 2. Label in the browser (writes should_accept.jsonl / should_reject.jsonl here).
python -m tadabur.audit_ui --worklist audit_worklist.jsonl \
  --manifest passing_subset.jsonl --audio-dir audit_audio/
# → open http://127.0.0.1:8000  (A = accept, R = reject, ←/→ navigate)
```

The UI persists after every verdict and resumes from whatever these files already
hold, so the audit can be paused and continued. It shows the **poison rate**
(`reject / labelled`) per contrast live — the direct input to the #6 go/no-go gate.
`accept` labels become the should-accept set, `reject` labels the should-reject set.

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

---

# `reject_reread_verdicts.jsonl` — what the ear says about a mined re-read

A different fixture with a different job. The two files above label *pronunciation* on
clips the gate **admitted**; this one labels *transmission* on clips the gate
**rejected** and ADR-0016 mines anyway (`tadabur.rejects.is_clean_re_read`).

The question it answers is narrow. `normalize_phonemes` deletes short vowels, so a
wholly non-Hafs recitation can score a perfect `match_ratio` — no automated screen can
see it. ADR-0016 decision 10 originally answered that by requiring every corpus clip to
be heard; the amendment replaced that with a **sampled audit** (`tadabur.reread_audit`),
on the argument that a *vowel-only* divergence cannot move the alignment cursor and is
therefore inert to everything the corpus measures.

That argument is only as good as vowel-only dominance, which is why the schema carries
`divergence` and why a blank one on a `nonhafs` row is reported as `unclassified` rather
than folded into the inert bucket.

| field               | type   | notes                                                            |
| ------------------- | ------ | ---------------------------------------------------------------- |
| `clip_id`           | string | staged clip id; `audio/<clip_id>.wav` in the scenario bundle      |
| `audio_ref`         | string | Tadabur `audio_filename`                                          |
| `surah_ayah`        | string | `"surah:ayah"`                                                    |
| `reciter_id`        | int    | joins to the manifest                                             |
| `match_ratio`       | float  | the gate's ratio, for context — never the basis of the verdict    |
| `band`              | string | which ratio band the clip was drawn from                          |
| `verdict`           | string | `"clean"` or `"nonhafs"`                                          |
| `divergence`        | string | on `nonhafs`: `"vowel_only"` or `"consonantal"`; else `""`        |
| `note`              | string | optional free text (bleed, unfinished ayah, anything heard)       |

```bash
# Draw the sample and stage its audio off a bundle, then listen and append rows here.
python -m tadabur.reread_audit --bundle corpus_run/scenario --out corpus_run/audit --size 50
# What the verdicts so far say.
python -m tadabur.reread_audit --summary
```

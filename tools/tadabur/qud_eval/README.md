# QUD correct-recitation eval set (#144)

An **evaluation-only** set of truth sites from professional Hafs recitation that neither the
base teacher nor `h448` has heard, built from QUD
([`QUD-Technologies/quranic-universal-ayahs`](https://huggingface.co/datasets/QUD-Technologies/quranic-universal-ayahs)).
It is never a training signal. Phase 1 is the selection and the size estimate; the pilot
(owner approval 2026-10-14) checks two mushafs for overlap by audio, decodes them with both
models and measures them without listening ([`docs/qud-pilot.md`](../../../../docs/qud-pilot.md)).

| file | what it is | written by |
|---|---|---|
| `sources.json` | what the selection reads, at pinned revisions: QUD's `mushafs` catalog and every mushaf's parquet shards with sizes; Muaalem's reciter and moshaf metadata (`obadx/muaalem-annotated-v3`) and the model's revision; Tadabur's `sheikh_dict.json` | `python -m tadabur.qud_selection snapshot` |
| `selection.json` | every QUD Hafs mushaf with its provenance, Muaalem match, Tadabur name screen, decision and reason; totals per decision | `python -m tadabur.qud_selection select` |
| `coverage.json` | per selected mushaf and surah: the ayat present, rows, duration and source URLs (parquet column reads, no audio) | `python -m tadabur.qud_selection coverage` |
| `date_check.json` | titles and upload dates of six source videos per selected mushaf, the evidence for its recording year | `python -m tadabur.qud_selection date-check` |
| `size_estimate.json` | carriers per pair direction and in-word geminates, expected agreed / disagreed / abstained sites, listening minutes and disk, per mushaf and in total | `python -m tadabur.qud_sizing` |

| `overlap_controls.json` | the overlap check's controls: synthetic positives, hard negatives and cross-aligner pairs, committed before any pilot audio met Tadabur | `python -m tadabur.qud_overlap controls` |
| `overlap_rules.json` | every control's fingerprint BER and timing distance, and the two decision lines fixed from them | `python -m tadabur.qud_overlap calibrate` |
| `overlap_screen.json` | stage 1: every pilot ayah's word timing against every Tadabur row of the ayah, and the candidates | `python -m tadabur.qud_overlap screen` |
| `overlap_results.json` | stage 2: each candidate's fingerprint, and the pilot against Muaalem's opening segments | `python -m tadabur.qud_overlap confirm` |
| `pilot/decodes/<slug>.<model>.jsonl.gz` | every pilot item's `spans`, `stream_b0` and `checks` decodes, merged from the GPU box | `python -m training.qud_decode` (GPU), then `python -m training.qud_pilot merge` |
| `pilot/truth_sites/<slug>.jsonl.gz` | the pilot's truth sites (source `qud`): agreed sites with the mushaf's letter as `heard`, disagreements `pending` | `python -m training.qud_pilot build` |
| `pilot/worklist.jsonl` | one row per disagreed carrier with its excerpt; **not queued** for listening | the same |
| `pilot/report.json` | items, classes, per-direction counts and rates, the app outcome, the listening estimate, compute | `build`, `app` (Mac with the Muraja harness) |

Run from `tools/`, in that order. The pilot's ayat are registered in the exposure registry as
the QUD use `qud.correct_recitation` (`../exposure/qud.correct_recitation.qud.jsonl`). The rules and the estimate's method are in the module
docstrings of [`../qud_selection.py`](../qud_selection.py) and [`../qud_sizing.py`](../qud_sizing.py).

**Licence.** QUD is CC BY 4.0; its card adds that the recitation audio "is not relicensed,
and remains the property of upstream sources/reciters". The repo commits no QUD audio, only
provenance (source URL, offsets, checksums), as for Tadabur.

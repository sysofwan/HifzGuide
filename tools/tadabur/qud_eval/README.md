# QUD correct-recitation eval set (#144)

An **evaluation-only** set of truth sites from professional Hafs recitation that neither the
base teacher nor `h448` has heard, built from QUD
([`QUD-Technologies/quranic-universal-ayahs`](https://huggingface.co/datasets/QUD-Technologies/quranic-universal-ayahs)).
It is never a training signal. Phase 1 (this directory today) is the selection and the size
estimate; no audio has been downloaded or decoded.

| file | what it is | written by |
|---|---|---|
| `sources.json` | what the selection reads, at pinned revisions: QUD's `mushafs` catalog and every mushaf's parquet shards with sizes; Muaalem's reciter and moshaf metadata (`obadx/muaalem-annotated-v3`) and the model's revision; Tadabur's `sheikh_dict.json` | `python -m tadabur.qud_selection snapshot` |
| `selection.json` | every QUD Hafs mushaf with its provenance, Muaalem match, Tadabur name screen, decision and reason; totals per decision | `python -m tadabur.qud_selection select` |
| `coverage.json` | per selected mushaf and surah: the ayat present, rows, duration and source URLs (parquet column reads, no audio) | `python -m tadabur.qud_selection coverage` |
| `date_check.json` | titles and upload dates of six source videos per selected mushaf, the evidence for its recording year | `python -m tadabur.qud_selection date-check` |
| `size_estimate.json` | carriers per pair direction and in-word geminates, expected agreed / disagreed / abstained sites, listening minutes and disk, per mushaf and in total | `python -m tadabur.qud_sizing` |

Run from `tools/`, in that order. The rules and the estimate's method are in the module
docstrings of [`../qud_selection.py`](../qud_selection.py) and [`../qud_sizing.py`](../qud_sizing.py).

**Licence.** QUD is CC BY 4.0; its card adds that the recitation audio "is not relicensed,
and remains the property of upstream sources/reciters". The repo commits no QUD audio, only
provenance (source URL, offsets, checksums), as for Tadabur.

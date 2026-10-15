# Non-professional eval pilot (#156)

An **evaluation-only** reading of non-professional recitation that neither the base teacher nor
`h448` has heard, through the external-corpus path QUD's pilot uses (#144,
`training.external_corpus`). It is never a training signal and sets no threshold. The doc is
[`docs/nonpro-pilot.md`](../../../docs/nonpro-pilot.md).

| source | dataset | licence | what is committed here |
|---|---|---|---|
| Tarteel v1 | [`greentechapps/user_recited_verses_full`](https://huggingface.co/datasets/greentechapps/user_recited_verses_full) | CC BY 4.0 | decodes, truth sites, both worklists |
| RetaSy | [`RetaSy/quranic_audio_dataset`](https://huggingface.co/datasets/RetaSy/quranic_audio_dataset) | unknown | aggregates (in `report.json`) and clip ids (the exposure use) only |
| qdat | [`obadx/qdat`](https://huggingface.co/datasets/obadx/qdat) | MIT | decodes, truth sites, both worklists |
| qdat_bench | [`obadx/qdat_bench`](https://huggingface.co/datasets/obadx/qdat_bench) | unstated | aggregates and clip ids only (its clips are qdat's, byte for byte) |

Revisions are pinned in `tadabur.nonpro_sources`. No audio is committed.

| file | what it is | written by |
|---|---|---|
| `<source>/decodes.<model>.jsonl.gz` | every clip's provenance (id, SHA-256 of its audio as published, the canonical words it recites, reciter, dataset labels) and its `spans`, `stream_b0` and `checks` decodes | `python -m training.nonpro_decode` (GPU), then `python -m training.nonpro_eval merge` |
| `<source>/truth_sites.jsonl.gz` | truth sites (source = the dataset): agreed with the mushaf's letter as `heard`, disagreements `pending` | `python -m training.nonpro_eval build` |
| `<source>/worklist.jsonl` | one row per disagreed target-pair carrier or in-word geminate; **not queued** | the same |
| `<source>/consonant_worklist.jsonl` | one row per disagreed carrier of any consonant (option B); **not queued** | the same |
| `report.json` | selection, items, classes, per-direction and per-consonant counts, rates, the app outcome and its comparison, qdat's views, the bench's letter agreement, RetaSy's label contrast, the listening estimate, compute | `merge`, `build`, `app` (Mac with the Muraja harness) |
| `overlap.json` | the cheap overlap sample against EveryAyah | `python -m tadabur.nonpro_overlap` |
| `restricted/` | RetaSy's per-clip outputs, written locally by the same commands; ignored by git | the same |

Run from `tools/`, in that order, then `python -m training.nonpro_eval doc`. Every decoded clip
and every qdat_bench clip is registered in the exposure registry as the evaluation-only use
`nonpro.evaluation_only` (`../exposure/nonpro.evaluation_only.clips.jsonl`).

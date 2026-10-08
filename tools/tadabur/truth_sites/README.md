# Truth sites (#79)

A **truth site** is one human-labelled position in one piece of Tadabur audio: what the
mushaf prescribes there and what the reciter actually said. It holds no model output.
[ADR-0011](../../../docs/adr/0011-transcription-fidelity-and-tashkeel-abstention.md) judges
every model against these sites, never against the teacher or the mushaf.

The files in this directory are committed source-of-truth (AGENTS.md, *Data & assets*). Each
`*.jsonl` needs its own `!` entry in the root `.gitignore`. The schema and the validating
loader are in [`tools/tadabur/truth_sites.py`](../truth_sites.py).

| file | what it is |
|---|---|
| `waqf_boundaries.jsonl` | the adjudicated waqf boundaries, converted to tashkeel sites |
| `waqf_boundaries.summary.json` | where each of the 2,050 boundary rows went (a site or an exclusion) |

## Schema

One JSON object per line, keys sorted, UTF-8. Every field is always present. A field that
is not known yet is `null`. It is never guessed.

| field | type | when | meaning |
|---|---|---|---|
| `site_id` | str | now | stable, unique id. Waqf sites use `waqf_boundary:<clip>#<boundary_index>` |
| `source` | str | now | `p35_fixture`, `waqf_boundary`, `new_audit` or `synthetic_edit` |
| `assumes_competent_reciter` | bool | now | `true` when the human judged something else and the mark holds only if the recitation was correct (the weaker label) |
| `audio_filename` | str | now | Tadabur `audio_filename` of the whole clip |
| `shard` | int \| null | **#83** | full-config parquet shard (0–384) the clip is re-downloaded from |
| `start_sample` | int | now | first sample of the item in the clip (16 kHz). 0 for a whole clip |
| `end_sample` | int \| null | **#83** | end sample (exclusive) of the item. For a whole clip, its length once staged |
| `audio_sha256` | str \| null | **#83** | SHA-256 of the staged 16 kHz mono PCM_16 WAV's bytes (`truth_sites.audio_sha256`) |
| `surah_ayah` | str | now | `"surah:ayah"` |
| `reference` | str | now | the item's **realized reference**: the phonetizer's output for what was recited, with waqf/wasl as recited |
| `reference_index` | int | now | index in `reference` of the **carrier letter**, the consonant the mark under test sits on (for shaddah, the first of the doubled pair) |
| `mark` | str | now | what is under test: `fatha`, `damma`, `kasra`, `sukun`, `shaddah`, or a soft pair labelled as `tadabur.phoneme_sifat.soft_pair_contrasts()` spells it, in codepoint order (`ت↔ط`, `ح↔ه`, `ذ↔ز`, `س↔ص`, `ض↔ظ`, `ق↔ك`) |
| `prescribed` | str | now | what the mushaf prescribes there (table below) |
| `heard` | str | now | what the human heard (table below). `unclear` leaves the denominator |
| `stratum` | str | now | the sampling stratum the site was drawn from |
| `stratum_population` | int | now | how many positions that stratum holds, so sampled estimates can be weighted (weight = population / sites in the stratum) |

`shard`, `end_sample` and `audio_sha256` are the **staging fields**. #83 fills them when it
re-stages the audio. All three are `null` (not staged yet) or all three are set (staged).
The loader rejects a mix.

| `mark` | `prescribed` | `heard` |
|---|---|---|
| `fatha` / `damma` / `kasra` / `sukun` | the mark itself | `fatha`, `damma`, `kasra`, `sukun` or `unclear` |
| `shaddah` | `held` | `held`, `not_held` or `unclear` |
| a soft pair `a↔b` | `a` or `b` (the mushaf's letter) | `a`, `b` or `unclear` |

`load_truth_sites(path, audio_dir=None)` rejects:

- **malformed rows**: invalid JSON, a missing or unknown field, or a wrong JSON type (`true` is not an int, `1.0` is not an int);
- **unknown sources or marks**, and a `prescribed` / `heard` the mark cannot take;
- **missing provenance**: an empty `audio_filename`, a `null` required field, partial staging fields, a bad shard, span or checksum format;
- **a reference index that does not carry the mark**: the carrier is not a consonant, a haraka site is not followed by that haraka, a sukun site is followed by a haraka, a shaddah site is not doubled, or a pair site's carrier is not the prescribed letter;
- **inconsistent files**: duplicate `site_id`, sites on one clip that disagree on its `shard` or `audio_sha256`, sites on one item (`audio_filename`, `start_sample`) that disagree on its `end_sample`, `surah_ayah` or `reference`, or a stratum whose population is not one value at least as large as its site count;
- **a checksum mismatch** when `audio_dir` is given. Every item's `<audio_dir>/<audio_filename>` must hash to its recorded `audio_sha256`, and an item with no checksum fails, because it cannot be verified.

`write_truth_sites` runs the same checks before it writes anything, then writes atomically.

## `waqf_boundaries.jsonl`: the adjudicated waqf boundaries

Source: the 2,050 frozen rows in `../waqf_event_fixtures/waqf_events.{calibration,test}.jsonl`
(126 clips, human verdicts `waqf` 179 / `wasl` 1,857 / `mid_word_closure` 14). Regenerate it
deterministically, from `tools/` with `quran-transcript` installed:

```bash
python -m tadabur.waqf_truth_sites
```

**Item and reference.** Each clip is one item: `start_sample = 0`, and the staging fields
stay `null` until #83. The boundary times in the fixture are whole-clip times, so no
re-segmentation is needed. The realized reference splits the clip's words at every human
`waqf` and phonetizes each recited run on its own (Hafs, `generate_phonemes.HAFS_MOSHAF`):
the terminal word in waqf form, the rest in wasl. The runs are joined by a space. A re-read
shows up as a run that restarts at an earlier word.

**Site.** The site is the boundary word's final mark: the haraka after its last realized
consonant, or sukun when there is none. If the word ends in a madd letter, the mark sits on
the letter before it (`عَلَى` → `عَلَ` / `عَلَاا`).

- `waqf` and the word ends in a consonant → **sukun at a pause**,
  `assumes_competent_reciter: false`. The pause the human heard is the evidence.
- `waqf` and the word ends in a long vowel (a madd letter, or tanween fatha, which becomes
  `اا` at waqf) → the haraka before the madd, `assumes_competent_reciter: true`. A pause on
  a long vowel produces no sukun, so the verdict says nothing about that mark.
- `wasl` → the word's wasl ending: a haraka, or the sukun the mushaf writes there,
  `assumes_competent_reciter: true`. The human heard continuation, not the mark. This is
  the weaker label: it holds only if the reciter recited correctly.

`prescribed == heard == mark` on every site. These labels are all correct recitation; none
is a real mistake. Strata are `waqf_boundary:waqf` and `waqf_boundary:wasl`. The 126
reviewed clips were labelled in full, so the strata are a **census** of them: the
population equals the site count, every weight is 1, and the clips themselves were not
sampled in a way that can be weighted back to the corpus.

### Counts

Sites by stratum and mark (1,791 sites on 123 clips; 1,667 assume a competent reciter):

| stratum \ mark | `damma` | `fatha` | `kasra` | `sukun` | total |
|---|---|---|---|---|---|
| `waqf_boundary:waqf` | 11 | 31 | 0 | 124 | 166 |
| `waqf_boundary:wasl` | 365 | 778 | 300 | 182 | 1625 |

All 1,791 sites have `source: waqf_boundary`. Where each fixture row went:

| fixture verdict \ outcome | `closure_merged` | `closure_unplaced` | `final_letter_assimilated` | `inconsistent_clip` | `mid_word_closure` | `phonetizer_unsupported` | `waqf_boundary:waqf` | `waqf_boundary:wasl` | total |
|---|---|---|---|---|---|---|---|---|---|
| `mid_word_closure` | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 14 |
| `waqf` | 13 | 4 | 0 | 3 | 0 | 5 | 154 | 0 | 179 |
| `wasl` | 8 | 0 | 143 | 41 | 0 | 28 | 12 | 1625 | 1857 |

### Why this is not 179 sukun-at-pause and 1,857 wasl sites

ADR-0011 counted the fixture's **verdicts**. A verdict is a site only when it fixes one
tashkeel mark on one letter of a reference that can be built:

- **Closure candidates are not word edges.** The detector also proposed VAD silences inside
  words (`predicted: mid_word_closure`). When the human called one `waqf` or `wasl`, the
  verdict is folded into the regular edge at the same word next to it in time
  (`closure_merged`: 13 waqf, 8 wasl). A `waqf` there turns that edge into a pause, which is
  why 12 `wasl` rows land in the `waqf` stratum. Four `waqf` closures have no regular edge
  at their word beside them, so the word the stop followed is unknown (`closure_unplaced`).
- **42 pauses fall on a long vowel** and are haraka sites, not sukun (see above). So the
  `waqf` stratum is 124 sukun + 42 haraka.
- **143 wasl boundaries have no word-final mark of their own** (`final_letter_assimilated`).
  The last letter merges into the next word through idgham (`أَن نَّمُنَّ` → `ءَننننَمُ…`,
  `أَن يَ` → `ءَييَ`) or is nasalized into it through ikhfa / iqlab (`مِّن قَوْمِهِۦ` → `مِںںںقَومِه`).
  The haraka or sukun under test does not exist there.
- **3 clips yield no reference.** In `spk0192_S1_A178` the edges run 0–7 within 0.6 s and
  then restart at word 0 with no pause before it, so they are not one recitation
  (`inconsistent_clip`, 44 rows). `spk0177_S26_A88` and `spk0345_S10_A52` each have a recited
  run that ends in waqf on a leen letter followed by the final consonant (`شَىْءٍ`,
  `إِلَيْهِ`). quran-transcript cannot phonetize that ending, so these clips are
  `phonetizer_unsupported` (33 rows).
- **14 `mid_word_closure` verdicts** are not boundaries and are excluded, as the issue specifies.

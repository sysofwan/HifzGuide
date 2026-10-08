# Pausal taa marbuta: affected ayahs and the quran-transcript pin

Findings for #100. No asset was regenerated and no release was published. A `quran.db`
release is a separate owner step under the `hifzguide-asset-release` skill.

## The bug and the fix

quran-transcript 0.5.2's `MaddAlewad` turns every final tanween fatha into fatha + alif before
`CleanEnd` / `NormalizeTaa` run. A pausal `ةً` therefore comes out `تَاا` instead of `ه`:
`رَحْمَةًۭ` gives `رَحمَتَاا`, when it should give `رَحمَه`. The phonetizer treats the end of
its input as a waqf, so the bug hits every phonetized string that ends on `ةً`.

The workaround from #79 now lives in one place, `tools/hafs_phonetizer.py`. Its `phonetize(text)`
is the only entry point to `quran_phonetizer` in the repo. It applies the Hafs moshaf
(`HAFS_MOSHAF`, moved there from `generate_phonemes.py`) and rewrites only the input's last word
with `pausal_taa_marbuta`. That rewrite covers `ةً` with no mark after it, with the small low
meem (`ۭ`) and with the small high meem (`ۢ`). The last word is found the way the phonetizer finds it:
any whitespace separates words, and trailing whitespace is not a word. Its char mappings index the
text as given: a dropped mark maps to a deleted, empty span where the word's phonemes end, which
is exactly what quran-transcript 0.6.x returns. Every caller goes through it:

- `tools/generate_phonemes.py`: `generate_reference_phonemes`, the ayah phonemes, and through
  it `tadabur.reference_phonemes`, the scorer gate's reference cache.
- `tools/generate_quran_db.py`: phonetizes on its own, so it is routed through the wrapper too.
  It builds `ayahs.phonemes`, `phoneme_groups` and `phoneme_char_map`.
- `tools/tadabur/waqf_segments.py`: `hafs_phonetizer`, `hafs_segment_reference`,
  `hafs_word_reference` and `hafs_normalized_word_reference`, which feed the waqf-segment
  references and the windowed and segmented training labels.
- `tools/tadabur/waqf_truth_sites.py`: `hafs_realizer`. It still re-exports
  `pausal_taa_marbuta` for #83.

`tools/test_hafs_phonetizer.py` fails if any other module under `tools/` imports
`quran_phonetizer` or `quran_transcript.phonetics`. `tadabur.reference_phonemes.CACHE_VERSION`
now includes `hafs_phonetizer.REVISION`, so a reference cache built before the fix is rebuilt
instead of trusted.

`phonetize` takes only quran-transcript's own Uthmani text, which is what every caller passes
(`Aya(...).uthmani` or its words). It raises `ValueError` on any character outside that
alphabet. The phonetizer cannot handle such characters anywhere in a word, so this is not only
about the pausal word. Examples:

- **QPC spellings** (`data/qpc-hafs-word-by-word.json`) use 23 such characters, including
  U+0656, U+0657 and U+065E for tanween and U+06E1 for sukun. 88:4:3 is spelled `حَامِيَةٗ`
  there. The open-tanween block (U+08F0 and after) is just as unsupported.
  quran-transcript passes these marks through into the phonemes (`رَحْمَةࣰ كَذَٰلِكَ` gives
  `رَحمَتࣰ كَذَاالِك`) or raises `IndexError` (`عَلِيمࣰا`), on 0.5.2 and on 0.6.4 alike.
  Rewriting only a final one would hide that the rest of the text is unsupported. No caller
  phonetizes QPC text: `generate_quran_db.py` reads it only for the `words` table.
- **Ayah-end markers and numbers** (`۝`, `١`) are passed through as phonemes, and the waqf is
  not applied to the word before them.

## Affected ayahs: 9 of 6,236

All 6,236 ayahs were phonetized with and without the fix. Each ayah went through both
`generate_reference_phonemes` and `generate_quran_db.main` (written to a scratch path, then
compared table by table). Exactly 9 ayahs change. Each ends on `ةً`, and in each only the
final word changes:

| ayah | final word | before | after |
| --- | --- | --- | --- |
| 56:7 | ثَلَـٰثَةًۭ | ثَلَااثَتَاا | ثَلَااثَه |
| 69:10 | رَّابِيَةً | ررَاابِيَتَاا | ررَاابِيَه |
| 69:14 | وَٰحِدَةًۭ | وَااحِدَتَاا | وَااحِدَه |
| 74:52 | مُّنَشَّرَةًۭ | ممممُنَششَرَتَاا | ممممُنَششَرَه |
| 79:11 | نَّخِرَةًۭ | ننننَخِرَتَاا | ننننَخِرَه |
| 88:4 | حَامِيَةًۭ | حَاامِيَتَاا | حَاامِيَه |
| 88:11 | لَـٰغِيَةًۭ | لَااغِيَتَاا | لَااغِيَه |
| 89:28 | مَّرْضِيَّةًۭ | ممممَرضِييَتَاا | ممممَرضِييَه |
| 98:2 | مُّطَهَّرَةًۭ | ممممُطَههَرَتَاا | ممممُطَههَرَه |

The "before" and "after" columns show the word as it appears in `quran.db`'s `ayahs.phonemes`,
split at word boundaries. In `generate_reference_phonemes`, a wasl merge can join that word to
the one before it, as in 56:7 `ءَزوَااجَںںںثَلَااثَتَاا`. The fixed text there is the same.

**No non-final occurrence changed.** Besides the 9 ayah-final ones, the mushaf has 498 words
ending in `ةً`, and every one keeps its wasl form. 89:28 shows both in a single ayah:
`رَااضِيَتَممممَرضِييَه`.

The fixed output equals quran-transcript 0.6.4's own output for all 9 ayahs.

**Segment references.** No segment manifest is committed, so the count here is per word, not
per segment. Any waqf segment, realized run or window label that ends on one of those
507 `ةً` words (9 ayah-final, 498 not) carried the `تَاا` form. Run outputs built before this
change, such as segment manifests, windowed labels and scenario bundles on the GPU box, need
regenerating on the next run. The waqf-boundary truth sites from #79 already used the fix and
regenerate identically.

## Committed generated files: owner's release decision

- **`data/quran.db`** is tracked. It was committed in the initial commit, even though
  `.gitignore` lists `quran.db`. It carries the bug in all 9 ayahs above. Regenerating it from
  this branch with quran-transcript 0.5.2 would change:
  - `ayahs`: the 9 rows above.
  - `phoneme_groups`: 18 rows removed and 9 added (the `تَ` + `اا` groups become one `ه`).
  - `phoneme_char_map`: 17 rows in those 9 ayahs.
  - **Unrelated to this fix:** 457,872 of the 707,423 `phoneme_char_map` rows, in 6,219
    ayahs, already differ from what the generator produces today with or without the fix.
    The differences are in `ph_start` / `ph_end` around word boundaries, so the committed file
    was evidently built with a different quran-transcript version. All other tables match the
    current generator exactly.
- `data/ayah_phonemes.json` (the output of `generate_phonemes.py`) is not committed.
- `tools/tadabur/tashkeel_counterfactual_fixtures/counterfactual_items.jsonl` is unaffected.
  Its one segment ending on `ةً` (cf042, 2:245) tests a different word.

## Upstream: fixed in quran-transcript 0.6.0, not adopted

Upstream fixed the bug in commit `97d397d` (2026-08-16), "حل مشكلة الوقف بالهاء على هاء
التأنيث المنونة بالفتح". It shipped in **0.6.0** (2026-08-17). 0.6.0, 0.6.1 and 0.6.4, the
latest, all give `رَحمَه`. On 0.6.4 the wrapper gives the same output as upstream, mappings
included, with one exception: upstream's fix misses the small high meem spelling `ةًۢ`. 0.6.4
still gives `غُرفَتَاا` for 2:249's `غُرْفَةًۢ` at a waqf, and the wrapper gives `غُرفَه`.
None of the 10 `ةًۢ` words in the mushaf ends an ayah, so this matters only for a waqf inside
an ayah.

**Not upgraded**, because 0.6.4 also changes 52 other ayahs' reference phonemes (phoneme strings
compared; mappings and sifat not compared). Most of these look like upstream corrections, but
the Muaalem labels came from older phonetizer output:

| change in 0.6.4 | ayahs |
| --- | --- |
| Lafz al-jalala after lam: alif madd restored (`لِللَهِ` → `لِللَااهِ`) | 14: 2:98, 2:142, 2:165, 2:284, 4:172, 6:12, 9:114, 13:31, 16:48, 16:120, 22:56, 39:44, 42:49, 82:19 |
| Fatha restored before a doubled waw/yaa (`عَصووَ` → `عَصَووَ`, `بِءييِ` → `بِءَييِ`) | 22: 2:61, 2:137, 3:20, 3:112, 3:188, 5:78, 5:93, 7:95, 8:23, 8:72, 8:74, 9:50, 9:76, 9:92, 13:35, 16:128, 19:72, 23:60, 38:3, 64:6, 68:6, 83:3 |
| Word-final madd letter shortened before hamzat wasl (`ءِذَاا` → `ءِذَ`, `ذُۥۥ` → `ذُ`) | 14: 2:104, 2:256, 3:4, 5:95, 9:5, 9:95, 12:62, 14:47, 39:37, 43:55, 48:15, 55:37, 57:13, 83:31 |
| 27:1 `طسٓ تِلْكَ`: ikhfa ghunna on the noon (`سِۦۦۦۦۦۦن تِلكَ` → `…ںںںتِلكَ`) | 1 |
| 2:72 `فَٱدَّٰرَْٰٔتُمْ`: `فَددَاارَااءتُم` → `فَددَاارَءتُم` | 1 |

0.6.x also phonetizes the 8 ayahs that `FALLBACK_PHONEMES` covers instead of raising. Seven of
them match the hand-written fallback exactly. 106:4 differs, with a 4-count madd on `ٱلَّذِىٓ`
(`ءَللَذِۦۦۦۦ`) where the fallback has 2.

The repo now pins **`quran-transcript==0.5.2`** in `tools/requirements-train.txt`. The earlier
`>=0.5` would have installed 0.6.4 on a fresh environment and silently changed these 53 ayahs.
Moving to 0.6.x is an owner decision. It would also change Muraja's graded references through
`quran.db`.

# HifzGuide

Assets and data-generation tools for the HifzGuide / Muraja Quran recitation checker: the Quran
database, phoneme data, mushaf layout/fonts, and the ML model-conversion pipeline that produces
the on-device CoreML models.

This glossary is the canonical terminology for this repo. When naming a concept in an issue,
commit, ADR, or script, use the term as defined here and avoid the listed synonyms.

## Quran Structure

**Surah**
: One of the 114 chapters of the Quran.
: _Avoid_: Chapter

**Ayah**
: A unit of Quran text within a surah, delineated by an ayah marker (۝).
: _Avoid_: Verse

**Word**
: A single token in the mushaf text, identified by a `(surah, ayah, word_index)` triple.
: _Avoid_: Token (too generic), kalima

**Page**
: A mushaf page (1–604) following the Medina Mushaf standard numbering.
: _Avoid_: Screen

**Juz**
: One of 30 equal divisions of the Quran. Metadata only.

**Mushaf**
: The specific printed physical format of the Quran (Medina Mushaf, 604 pages). Implies visual
  layout, page boundaries, and line breaks. Distinct from the abstract Quran text content.
: _Avoid_: Quran (when referring to visual layout)

**Tashkeel**
: Diacritical vowel marks (حركات) on Arabic letters. Includes fatha, damma, kasra, sukun,
  shadda, tanween.
: _Avoid_: Diacritics (too generic), vowels

**Uthmani script**
: The canonical Uthmani orthography of the Quran text, source for phonetic transcription.

## Phonemes & Model

**Phoneme**
: A model output token — a discrete Arabic character class from the CTC vocabulary (43 classes).
  Represents a unit of pronunciation, not a linguistic phoneme.
: _Avoid_: Character, letter, token (too generic)

**Phoneme map**
: The expected phoneme sequence for a given Quran word or ayah, used as the alignment reference.
  Generated from Uthmani text via the phonetizer.

**Muaalem**
: The upstream Wav2Vec2-BERT phoneme-recognition model (`obadx/muaalem-model-v3_2`), a
  multi-level CTC model with 11 heads. This repo uses only the phoneme head.

**Wav2Vec2-BERT**
: The model backbone (Meta's `w2v-bert-2.0`) that Muaalem fine-tunes.

**Multi-level CTC**
: The Muaalem architecture — one CTC head per aspect (phoneme identity + 10 sifat attributes).

**Recitation VAD**
: `obadx/recitation-segmenter-v2` — a Wav2Vec2-BERT frame-level speech/silence classifier
  (20 ms frames) fine-tuned for waqf. The teacher for the waqf head and the source of waqf
  pauses in the Tadabur labelling pipeline (`tadabur.vad`).

**Waqf head**
: A per-frame speech/silence classification head on the Muaalem backbone, distilled from the
  Recitation VAD. It rides the adapter + CTC output at the phoneme head's **40 ms** lattice, so the
  20 ms VAD teacher is pooled 2:1; unlike the CTC phoneme/sifat heads it is per-frame (no blank
  collapse). The scorer consumes it to detect waqf and pick the realized (waqf vs wasl) reference form.

**Sifat**
: Articulatory attributes of Arabic letters in tajweed (e.g. hams/jahr, shidda/rakhawa,
  tafkheem/tarqeeq). Modeled by the non-phoneme CTC heads.

**Phonetizer**
: `quran_phonetizer` from the `quran-transcript` library — converts Uthmani text to the Quranic
  phonetic script (all tajweed rules), producing the training/reference labels.

**Moshaf attributes**
: The recitation configuration passed to the phonetizer (rewaya, madd lengths). Default is Hafs.

**Waqf**
: A stop/pause in recitation. At a waqf the final word drops its ending haraka and loses the
  cross-word gemination/idgham it would carry in continuation; `quran_phonetizer`'s CleanEnd op
  produces this realized form. _Avoid_: stop, pause (ambiguous).

**Wasl**
: Continuous recitation across a word boundary (no stop), carrying idgham/ikhfa/gemination.
  The opposite of waqf.

**Word alignment**
: Tadabur's per-word forced-alignment timestamps (`metadata.word_alignments`: `word`, `start`,
  `end`), in recitation order. The source for detecting waqf pauses.

**Waqf segment**
: A contiguous run of an ayah's words with no internal waqf, bounded by detected pauses (an
  inter-word gap above the threshold) or the clip ends. Each segment gets a realized reference
  (terminal word waqf, interior words wasl) and is a lightweight `(start_s, end_s)` view into the
  whole clip. Distinct from **Chunk** (a CoreML encoder-layer segment).
: _Avoid_: Segment (unqualified — collides with Chunk), split, clip

## Scoring & Evaluation

The distinction between the first two entries is load-bearing: conflating them is what produced
the wrong headline metric for the whole fine-tune (ADR-0008). Name which one you mean.

**Decode**
: The model's raw output phoneme string for a clip or window — what the model *heard*. The
  fine-tune is judged on this: does the decode carry the vowel and the consonant the reciter
  actually produced (ADR-0003, ADR-0008).
: _Avoid_: Transcription, prediction, output (all too generic)

**Scorer gate**
: The ported `.balanced` Smith-Waterman scorer (`tadabur.scorer`) used as ADR-0001's
  **training-data filter**: it decides whether a clip's decode matches its reference closely
  enough to keep as a training example. It is not a model of any decision Muraja ships, and it is
  **not** the fine-tune's success metric (ADR-0005, ADR-0008).
: _Avoid_: Eval, metric, the scorer (unqualified)

**match_ratio**
: The gate's score — Smith-Waterman alignment score over query phoneme count. Computed after
  `normalize_phonemes`, so it is **vowel-blind by construction**: a decode with every short vowel
  wrong scores 1.0 (ADR-0005). Never quote it, or anything derived from it, as tashkeel evidence.

**Scoring mode**
: Muraja's `ScoringParameters` presets — `.strict` / `.balanced` / `.lenient`. In Muraja these are
  **word-grading** parameters; in this repo only `correct_threshold` has any effect, because the
  port is score-only. `.strict` here therefore means a threshold, not the app's grading behaviour.

**Soft pair**
: One of the six confusable consonant pairs (`ذ↔ز`, `ت↔ط`, `ض↔ظ`, `ك↔ق`, `س↔ص`, `ح↔ه`) that
  `.balanced` forgives and `.strict` does not. The discrimination the fine-tune must sharpen
  rather than collapse (ADR-0001).

**should-accept set / should-reject set**
: The curated, human-labelled fixture clips from the P3.5 poison audit (#6): acceptable-imperfect
  recitation that the model should render faithfully, and genuinely-wrong recitation whose
  deviation the model should still emit. The two sides carry **opposite sign** — the same
  confusion cell is a mishearing on one and correct behaviour on the other — so they are always
  reported separately (ADR-0008).
: _Avoid_: Positive/negative set (loses the sign convention)

**Vowel outcome**
: How one reference short vowel fared in a decode, carrier-anchored
  (`training.tashkeel_eval`): `matched`, `omitted` (no vowel emitted), `swapped` (a different
  vowel on a carrier the decode got right), `unanchored` / `unanchored_wrong` (right or wrong
  vowel on a **misheard** carrier — a decode failure, not an i'raab claim), `spurious`.
: _Avoid_: Color swap (ADR-0003's informal term; say `swapped`)

**Poison**
: A training example whose label does not match what was recited — the mislabel risk ADR-0001's
  filter and the P3.5 audit exist to bound. A property of the *corpus*, never of a checkpoint.

## Re-read Corpus

The vocabulary of Muraja ADR-0016, which mines the Tadabur clips the ADR-0001 gate **rejects**
for the natural re-reads its follow-along tuning needs. Nothing here is a training-data concept.

**Reject pile**
: The clips `tadabur.filter` turned away, kept with the `GateResult` behind the verdict
  (`tadabur.rejects`). The population the corpus is mined from — the *passing* subset holds no
  re-reads at all, by construction, since a long repeat is itself a gate reject.

**Re-read**
: A reciter saying a span of words again before carrying on. In an alignment it is an
  **insertion run** — decoded phonemes the reference does not contain — which is why it barely
  dents `match_ratio` and is caught by `max_insertion_run` instead.
: _Avoid_: repetition, stumble (a stumble is shorter than the gate's bar and stays in the
  passing subset)

**Clean re-read**
: A reject meeting the mining predicate — `max_insertion_run >= 5 and not added_shadda` — i.e. a
  recitation that matches its reference everywhere except for a repeated span. The `match_ratio`
  floor was dropped (`docs/tadabur-ratio-floor.md`): it filtered repeat *length*, not correctness.

**Seam**
: Where a re-read joins the recitation around it — the two edges of the insertion run, in the
  clip's own time. Measured, not assumed, to carry a waqf: 0% of shard-20 seams have a usable
  VAD silence at *both* edges (`docs/tadabur-excision-yield.md`).
: _Avoid_: boundary (collides with waqf boundary), cut point

**Excision differential**
: The pair ADR-0016 decision 4 asserts over: a clip and the same clip with its repeat cut out.
  Both must drive the engine to the same terminal state. The control clip is re-decoded and
  re-gated, so a bad cut costs a **pair**, never a finding.

**Covered word range**
: The half-open Uthmani word range of a clip's own ayah that its decode actually reached, taken
  from the alignment's reference span and mapped through the phonetizer's per-word phoneme
  boundaries. The only range an oracle may grade over (ADR-0016 decision 1) — and it is word
  space, never time.

**Bleed**
: Audio of the neighbouring ayah a staged clip carries because it was cut from a continuous
  recitation. Detected by aligning the decode against `prev | this | next` and reading off the
  reference consumed outside this ayah, then clipped (`tadabur.bleed_detect`,
  `tadabur.bleed_recut`).
: _Avoid_: lead-in, spill, overlap (unqualified)

**Scenario record**
: One line of `scenario.jsonl` — the cross-repo interface HifzGuide hands Muraja for one mined
  re-read. See `tools/README.md` for the schema.

## ML Pipeline & Assets

**CoreML pipeline**
: The 6-chunk, 6-bit palettized CoreML model produced from Muaalem for on-device inference. See
  `ml-model-transformation.md`.

**Chunk**
: One of the 6 encoder-layer segments the model is split into to fit the on-device compute budget.

**Palettization**
: K-means weight quantization (6-bit) applied to model chunks to shrink size with minimal
  accuracy loss.

**Mel spectrogram**
: The 80-bin Kaldi-compatible audio feature fed to the model. Exported filter bank and window
  live in `mel_filters.bin` / `window.bin`.

**Feature extraction**
: Converting 16 kHz audio into normalized `(1, 250, 160)` model input features.

**QCF2**
: Quran Complex Font v2 — per-page glyph fonts (604 TTFs) rendering the mushaf with exact visual
  fidelity to the printed Medina edition.

**Ligature**
: A glyph-string mapping used to render decorative fonts (surah names, common symbols).

**Manifest**
: `manifest.json` in each GitHub release — version metadata with SHA256 checksums for every
  downloadable asset.

**Asset release**
: A versioned GitHub release bundling `quran.db`, `models.zip`, `fonts.zip`, `mel_filters.bin`,
  `window.bin`, and `manifest.json` for client apps to download.

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

# Waqf-aware Tadabur reference labels

The Tadabur filter (ADR-0001) admits each clip on a **single full-ayah reference** — one
canonical guess at the recitation. But a reciter who **stops** (makes waqf) partway through an
ayah realizes the pre-stop word differently from one reciting continuously (wasl): the word
before the stop drops its final haraka and loses the cross-word gemination/idgham of
continuation. Labelling every clip with the full-ayah wasl reference therefore injects
**phantom pre-waqf gemination mismatches** into the fine-tune data — the model emits the
(correct) waqf form while the label demands the wasl form. This mirrors the discipline that
gives upstream Muaalem its accuracy: every clip's label matches what was *actually* recited.

## Decision

- **Split each clip at intra-ayah waqf pauses, label each segment in its realized form.**
  Phonetize each **waqf segment**'s Uthmani text on its own so `quran_phonetizer`'s CleanEnd puts
  the terminal word in **waqf** form and the interior words in **wasl** — the realized reference.
  Output is a per-segment **offsets manifest** plus audit-only sliced WAVs (see below).

- **Detect pauses with a dedicated VAD model, not the shipped forced alignment.** Tadabur's
  `metadata.word_alignments` **absorbs waqf silence into the adjacent word's span**, so the
  inter-word gap is ~0 even at a clear stop (verified on 3:159, 40:20, 2:159, …). Gap-based
  detection therefore both *under*-splits on absorbed pauses and *over*-splits on madd elongations
  (voiced audio the aligner leaves *outside* word bounds, indistinguishable from a stop in the
  timestamps). Instead, run **`obadx/recitation-segmenter-v2`** — a Wav2Vec2-BERT VAD fine-tuned
  on recitation for waqf (20 ms frames, F1 ≈ 0.996) — over the whole clip to get clean **speech
  intervals**; the interior silence *between* two speech spans is a waqf pause (`tadabur.vad`).
  The VAD's own post-processing supplies the levers: silences shorter than **300 ms**
  (`--min-silence-ms`) are merged away and speech shorter than **700 ms** (`--min-speech-ms`) is
  dropped — both taken from the model's training labels — so brief inter-phoneme / stop-consonant
  closures never surface as pauses. This is immune to madd (a madd is voiced, so the VAD keeps it
  inside a speech span) and to the aligner's silence absorption.

- **Map each pause to a *word* boundary via Smith-Waterman; self-reject mid-word stops.** A VAD
  pause gives a *time*, but a segment must cut on a word so it can be phonetized. The whole clip is
  still decoded once with the Muaalem **phoneme head** to get timed phonemes; these are
  Smith-Waterman-aligned to the ayah's per-word reference, and each pause is placed at the
  reference position of the last phoneme before its start time. A pause that lands within **3
  phonemes** (`--boundary-tol`) of a word edge is a waqf; one that lands mid-word is a stop-consonant
  closure (qalqala on ق/ط, the hamza in شَيء) and is **not** split. This Δ-to-edge test is what
  cleanly separates a real waqf (Δ≈0) from a mid-word closure (Δ≫tol) — the discriminator the
  acoustic gate lacked.

- **Split a re-read at its seam; separate a genuine re-read from a phantom over-read by the
  earlier chunk's own decode support.** When a pause's two neighbouring chunks overlap in reference
  words (the post-pause chunk resumes *behind* where the pre-pause chunk ended) the seam is
  ambiguous: either the reciter genuinely stopped and re-read the overlap word (a real waqf seam,
  overlapping realized references), or the pre-pause decode merely **over-snapped** its end a word or
  two past where the reciter actually stopped (a phantom — an ordinary forward waqf that must stay a
  clean contiguous cut). Segment word ranges alone cannot tell these apart. The discriminator is the
  runtime-faithful signal the app also has: whether the *earlier chunk's own local decode*
  contiguously **recited** each overlap word (`_word_supported`: prefix-anchored, ≥1 exact match,
  ≥0.55 coverage of the word's phoneme span). `_supported_end` scans forward from the chunk's start
  over the words it keeps supporting — bridging a **one-word** interior gap (a routine single-word
  CTC dropout mid-recitation) but stopping at a ≥2-word gap (the phantom's far snap onto an isolated
  duplicate) — and returns the half-open word end the reciter actually reached. The seam is a genuine
  **re-read** when the later chunk resumes inside that real coverage (`later.start_word <
  supported_end`), which covers both an adjacent re-read and a gross whole-ayah restart; otherwise it
  is an ordinary forward waqf and the phantom over-read is **un-inflated**, the segment ending at
  `supported_end` instead of the over-snapped word. This changes the segment manifest for the
  phantom cases only (a segment's `word_end` shrinks to the truly-recited word; the interior wasl
  edges shift accordingly) and correctly keeps a re-read's overlap and a restart's whole-ayah span
  intact. Thresholds (`_MAX_SUPPORT_GAP = 1`, coverage `0.55`) are uncalibrated but empirically
  validated on the audit worklist; the "nothing supported" fallback (trust the snap) never fires
  there. **The display marker** (`waqf_candidates`) moves off the resume anchor **only for a genuine
  re-read** — to the decode-supported stop word — so ordinary forward-waqf markers are unchanged.

- **Bound the final segment by the last reliable chunk's decode support, not a blind snap to the
  whole ayah.** The final segment has no terminating split pause, so its `word_end` used to snap to
  `n_words` unconditionally — which invents word markers for words never recited when the reciter
  **stops early** (ends mid-ayah). It now reuses the same `_supported_end` frontier: `word_end` is
  where the last *reliable* chunk in the final group actually reached. The *last reliable* chunk is
  used (not simply the last chunk) so a trailing **unreliable** fragment — an elongated final-word
  tail or a post-ayah artifact chunk after a pause — cannot collapse a completed recitation (those
  keep `word_end = n_words`). This changes the segment manifest for genuine early stops only (the
  final segment's `word_end` shrinks to the truly-recited word; phantom wasl markers for the
  never-recited tail words are dropped and the surviving interior wasl edges re-interpolate over the
  true, shorter range). Empirically 3 worklist clips (17:88, 18:110, 58:20).

- **The ayah-aligned span is the `ClipStatus` recut bounds, not the raw staged clip.** A staged
  clip often carries a **neighbour-ayah lead-in or trailing bleed** — audio of the previous/next
  ayah the reciter ran into (e.g. 11:52/spk0345 carries `afalā taʿqilūn` from 11:51 for its first
  2.833 s). `segment_clip` already recuts to the ayah's true onset/offset and records them as
  `recitation_start_s` / `recitation_end_s` on `ClipStatus`; those bounds — not `[0, clip_duration]`
  — are the span every downstream reader must trust. This matters **most for training, not just the
  audit**: `windowed_labels` and `waqf_distill` enumerate the fixed 5 s windows over the *recitation
  span* (`recitation_start_sample` .. `+ recitation_num_samples`) on the shared grid, so a window's
  phoneme CTC target and its waqf soft target both exclude the bleed — otherwise the fine-tune would
  be trained to transcribe (and to place stops in) a neighbouring ayah's audio (the `--whole-clip`
  flag deliberately re-includes the bleed, and is exactly what the labels must *not* use). The audit
  UI keys off the same bounds: it starts playback at `recitation_start_s`, floors the marker/seek
  pre-roll there, shades the trimmed regions, and auto-pauses at `recitation_end_s`, so a reviewer
  grades the ayah itself and a neighbour-ayah stop is never mis-heard as an in-ayah waqf (native
  scrub stays unrestricted so the trimmed audio is still inspectable).

- **Per-word phoneme boundaries come from the phonetizer's char `mappings`, not a space-split.**
  `quran_phonetizer` **merges adjacent words at a wasl** (liaison), so the phonetized whole ayah has
  *fewer* space-parts than Uthmani words for ~62 % of ayat — splitting the output on spaces would
  mislabel most clips. Instead the whole ayah is phonetized once and each word's phoneme offset is
  read from `output.mappings[word_start_char].pos` (spaces then stripped, offsets remapped to the
  space-free string the decode aligns against). This yields correct per-word boundaries even across
  wasl-merges and hamzat-wasl elisions (`tadabur.waqf_segments.hafs_word_reference`).

- **This stage owns the model (GPU).** Segmentation + per-segment scoring live in
  `tadabur.segment_score`, which first runs one VAD pass over all clips to get pauses, then frees
  the VAD and loads Muaalem; each clip is decoded **one at a time** (a batched decode of full clips
  OOMs — attention is quadratic in sequence length). The torch-free half, `tadabur.waqf_segments`,
  only **stages** each passing clip as a whole 16 kHz mono WAV on local disk and owns the shared
  realized-reference vocabulary (`SegmentRecord`, `hafs_phonetizer`, `hafs_word_reference`,
  `_uthmani_words`). The pure detection logic (`tadabur.waqf_detect`) takes an injected frame-id
  sequence, the injected VAD **pauses**, plus the reference/boundaries, so it is unit-tested with
  synthetic `class_ids` and no model; the VAD interval→pause logic (`tadabur.vad`) is likewise
  unit-tested torch-free.

- **The manifest carries per-segment offsets — the P4 label source.** Each manifest row is a
  `SegmentRecord` view into its whole clip (`audio_filename`, `surah_ayah`, `reciter_id`,
  `segment_index`, word range, `start_s`/`end_s` offsets, `realized_reference_phonemes`); P4 (#8)
  slices audio from those offsets at collate time. The whole passing clips are kept locally as
  16 kHz mono WAV; the full 937 GB Tadabur source is streamed, never landed. `segment_score` also
  writes each segment as a **pre-sliced** WAV under `--audio-out` — an **audit-UI convenience only**
  (so the UI plays a segment without re-slicing), not the P4 source of truth.

- **Skip segmentation safely, keep the clip whole, never silently mislabel.** Two whole-clip cases
  cannot be segmented reliably and are **tallied and kept as a single whole-ayah segment** (still
  audited, just not sub-split): a decode far longer than the reference (`repeated_recitation` — the
  reciter repeated the ayah, breaking the one-pass word map, default `--max-decode-ratio 1.6`) and a
  decode that barely aligns to the ayah at all (`low_alignment`, `--min-align-ratio 0.45`). The 8
  ayat `quran_phonetizer` cannot handle at all (leen madd on a final sukoon,
  `generate_phonemes.FALLBACK_PHONEMES`) are skipped entirely (`phonetizer_unsupported`).

### Superseded: CTC blank-run pause detection

A first model-driven revision detected pauses from the **Muaalem phoneme head** itself: greedy CTC
emits the blank token on silence, so a blank-run ≥ 0.35 s (`--min-pause`) was taken as a waqf
candidate. It removed the aligner dependency but had a **single** lever and **no** minimum-speech
filter, so it over-split — mapping *every* blank-run that landed near a word edge to a cut (671
segments on the preview subset, e.g. spk0029 split into 6). The dedicated VAD above replaces the
blank-run source wholesale (`--min-pause` / `find_blank_runs` removed); the Smith-Waterman
word-boundary mapping is kept, now fed VAD pause *times* instead of blank-runs.

### Superseded: the alignment-gap + acoustic-silence gate

An earlier revision detected pauses from alignment gaps (`word[i+1].start − word[i].end ≥
--pause-threshold 0.25 s`) and confirmed each against the audio with an **acoustic silence gate**
(gap-window RMS below `--silence-ratio 0.15` of clip RMS). It shipped (#18) and cut multi-segment
preview clips from 78 to 15, but auditing revealed the aligner **absorbs** real pauses into word
spans, so gap-based detection misses them entirely — no threshold or gate can recover a gap the
alignment does not expose. The model-driven approach above replaces it wholesale;
`--pause-threshold` / `--silence-ratio` and the gap/gate code are removed.

## Consequences

- **Feeds P4 data-prep (#8).** The offsets manifest is the label source; the reciter split is
  computed over the **post-segmentation** units; the CTC collator slices audio by the offsets.

  > **Amended by ADR-0004.** Training no longer slices audio: the fine-tune runs on **whole clips**
  > (so the new waqf head sees interior waqf in context), and the phoneme CTC label is the
  > **concatenation** of these per-segment realized references. Segmentation is retained here as the
  > label-construction step and as the poison-audit unit (#6).

- **One VAD pass, then the clip decoded twice.** A dedicated VAD pass over all clips finds the
  pauses; the VAD is then freed and Muaalem loaded. The whole clip is decoded once for segmentation
  (frame ids, mapping pauses to word edges) and each resulting segment is decoded again for scoring
  against its realized reference — the same `.balanced` normalization / Smith-Waterman / contrast
  attribution the full-ayah filter uses. The second decode is what the audit samples on; it is
  observational (no gate-reject).

- **Out of scope: residual gemination under-detection at a *genuine* waqf** (the model emits one
  held-consonant token where the phonetizer's waqf form still doubles it). That is a fine-tune /
  filter-side scoring-tolerance concern, not a labelling one, and is left to a later stage.

- **First pass is preview-scale.** The ~291-clip preview passing subset (<100 MB) is enough to
  validate the pipeline end-to-end; the full-corpus thresholds are re-checked before scaling up.

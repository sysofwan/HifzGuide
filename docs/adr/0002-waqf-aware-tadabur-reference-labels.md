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

- **Split each clip at intra-ayah waqf pauses, label each segment in its realized form.** Detect
  pauses from Tadabur's shipped forced alignment (`metadata.word_alignments`) as an inter-word
  gap `word[i+1].start - word[i].end ≥ threshold`; continuous recitation shows overlapping/near-
  zero gaps. Phonetize each **waqf segment**'s Uthmani text on its own so `quran_phonetizer`'s
  CleanEnd puts the terminal word in **waqf** form and the interior words in **wasl** — the
  realized reference. **No model, no GPU** for this stage: it reads alignments the dataset
  already ships.

- **Pause threshold = 0.25 s (validated, tunable) — plus an acoustic silence gate.** Over 300
  preview clips, inter-word gaps are overwhelmingly negative (words overlap in continuous
  recitation; median ≈ −0.12 s, p95 ≈ 0.10 s), and genuine pauses form a clear tail beyond
  ~0.15 s. 0.25 s sits in the stable middle of the 0.15–0.30 s band proposed by #18, so it catches
  real stops without splitting on alignment slack. Exposed as `--pause-threshold` for re-validation
  on the full corpus.

- **A timestamp gap alone is not a waqf — confirm it against the audio.** A **madd** (elongation,
  especially madd munfasil before a hamza) is *sustained voicing* the forced aligner leaves outside
  its word bounds, so it looks identical to a stop in the timestamps. Auditing the preview run,
  ~84% of timestamp-only splits were such elongations (or plain aligner underestimation), not stops
  — which would re-inject the very phantom mismatches this ADR removes, inverted (a wasl utterance
  wrongly labelled waqf). We therefore gate every candidate split on **acoustic silence**: a true
  waqf is (near-)silent, a madd is voiced, so a boundary splits only when the gap window's RMS is
  below `--silence-ratio` (default **0.15**) of the clip's overall speech RMS. Gap-window RMS is
  bimodal — genuine stops near 0.03–0.10× the clip RMS, elongations near 0.5–1.0× — and 0.15 sits
  in the empty band between them. This stays **model-free and GPU-free** (a mean-square over the
  saved 16 kHz clip) and needs no tajweed-specific rules: it catches madd munfasil, madd muttasil,
  and aligner slack uniformly. On the preview subset it cut multi-segment clips from 78 to 15.

- **Output is an offsets manifest — no new audio, no derived dataset.** Each segment is a
  lightweight `(start_s, end_s)` view into the whole clip (`tadabur.waqf_segments.SegmentRecord`
  JSONL: `audio_filename`, `surah_ayah`, `reciter_id`, `segment_index`, word range, offsets,
  `realized_reference_phonemes`). Audio is sliced from these offsets at collate time by P4 (#8).
  Whole passing clips are kept locally as 16 kHz mono WAV; the full 937 GB Tadabur source is
  streamed, never landed. The manifest is written by a deterministic sort-then-rewrite, so
  re-running reproduces it byte-for-byte.

- **Skip, never silently mislabel, the two per-clip data-quality cases.** A clip whose alignment
  word count differs from its Uthmani word count (the vocative `يا` is a separate simple-text
  word but merged in Uthmani, e.g. `يَـٰٓأَيُّهَا` — ~7% of clips) has no safe positional word
  map, and the 8 ayat `quran_phonetizer` cannot handle (leen madd on a final sukoon,
  `generate_phonemes.FALLBACK_PHONEMES`) cannot be re-phonetized per segment. Both are **skipped
  and tallied**, not mapped by guesswork.

## Consequences

- **Feeds P4 data-prep (#8).** The offsets manifest is the label source; the reciter split is
  computed over the **post-segmentation** units; the CTC collator slices audio by the offsets.

- **Fewer phantom soft-negative labels.** The build prints a before/after report on the audit's
  shadda-contrast bucket (ADR-0001's poison audit vocabulary): for each clip that contained an
  intra-ayah waqf, it re-attributes the shadda contrast of the model's stored decode against the
  full-ayah reference vs the realized reference, and counts the phantom pre-waqf gemination
  mismatches removed — measured, not assumed, and with no model inference.

- **Out of scope: residual gemination under-detection at a *genuine* waqf** (the model emits one
  held-consonant token where the phonetizer's waqf form still doubles it). That is a fine-tune /
  filter-side scoring-tolerance concern, not a labelling one, and is left to a later stage.

- **First pass is preview-scale.** The ~291-clip preview passing subset (<100 MB) is enough to
  validate the pipeline end-to-end; the full-corpus threshold is re-checked before scaling up.

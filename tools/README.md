# Tools

Scripts for model conversion, data processing, and Quran database generation.

## Environments

Two separate environments, because the training and export paths target different platforms:

### Linux + CUDA — filtering & fine-tuning (`tools/tadabur/`, `tools/training/`)

Verified on an NVIDIA RTX 5060 Ti (16 GB, **Blackwell / sm_120**). Blackwell requires
CUDA 12.8 PyTorch wheels — a plain PyPI/conda torch will not run on this GPU.

```bash
conda env create -f tools/environment.yml
conda activate hifzguide
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r tools/requirements-train.txt
```

### macOS — CoreML export (`convert_to_coreml.py`, `palettize_chunks.py`, `verify_coreml.py`, `compile_models.sh`)

`compile_models.sh` uses Xcode's `coremlcompiler`, so this path runs on macOS / Apple Silicon.

```bash
pip install -r tools/requirements.txt
```

## Python Scripts

### `tadabur.smoke_decode` (Linux + CUDA)

Walking-skeleton smoke test for the Tadabur filter's PyTorch inference→decode path
(PRD #1, Phase 0). Streams one clip from `FaisaI/tadabur` (no full-corpus download),
resamples it to 16 kHz mono, loads Muaalem (`obadx/muaalem-model-v3_2`, vendored
`Wav2Vec2BertForMultilevelCTC`) in bf16 on the GPU, runs one variable-length forward
pass, and greedy-CTC-decodes the phoneme head to a sanity phoneme string (recording
VRAM footprint, ~1.2 GB backbone).

```bash
cd tools
python -m tadabur.smoke_decode --config-name preview   # small row groups → fast
python -m tadabur.smoke_decode                          # default config (2.4 GB shards)
```

The default config's shards are a single ~2.4 GB / 1000-row Parquet row group, so
streaming even one clip pulls that whole group; `--config-name preview` uses the
dataset's small-row-group preview config for a quick check. The 43-class phoneme
vocabulary (`tadabur.phoneme_vocab`) is asserted to match the live model in
`tadabur/test_phoneme_vocab.py`.

### `tadabur.filter` (Linux + CUDA)

The Phase 3 filtering pipeline (PRD #1, ADR-0001): the passing-subset generator.
Streams `FaisaI/tadabur` once, resamples each clip to 16 kHz mono, runs **batched**
bf16 GPU inference in one variable-length full-ayah pass (no 250-frame windowing),
greedy-CTC-decodes the phoneme head, and scores each decoded string against the
cached `quran-transcript` reference for its `surah:ayah` (`tadabur.reference_phonemes`)
with the ported `.balanced` gate (`tadabur.scorer`). Passers are appended to a JSONL
manifest — `audio_filename`, `surah:ayah`, `match_ratio`, `ayah_duration_s`, `reciter_id`.

```bash
cd tools
python -m tadabur.filter --manifest passing_subset.jsonl --batch-size 64
python -m tadabur.filter --manifest passing_subset.jsonl --config-name preview --limit 200
```

Filtering is light on VRAM (~1.5 GB), so use a large `--batch-size` for throughput
over the 365k+ clips. The run is **resumable and idempotent**: a sibling
`<manifest>.progress.json` checkpoints how many clips have been scored, so a restart
skips them (rejected clips leave no manifest line but are still skipped), and a
per-`audio_filename` seen-set keeps the manifest duplicate-free if the last in-flight
batch is replayed after a crash.

### `tadabur/run_corpus.sh` (Linux — GPU) — the whole re-read corpus run, in order

The five stages behind an ADR-0016 corpus, wired. The order is not cosmetic: **only stage 1 is
resumable and only stage 1 costs per shard**, so it runs across every shard first and the rest
run once over the accumulated sink. Chaining all five per shard would pay the downstream tail
eleven times.

| | stage | cost |
| --- | --- | --- |
| 1 | `tadabur.filter` — mine rejects, stage the clean re-read WAVs | per shard, resumable |
| 2 | `tadabur.bleed_stage` — timed decode + VAD intervals | once |
| 3 | `tadabur.bleed_recut` — clip bleed, re-gate, keep or drop | once |
| 4 | `tadabur.scenario` — stage the bundle and the excision pairs | once |
| 5 | `tadabur.reject_yield` + `tadabur.bleed_detect` — the numbers | once, no GPU |

```bash
cd tools
RUN=tadabur/corpus_run SHARDS=20-30 bash tadabur/run_corpus.sh          # all five
RUN=tadabur/corpus_run bash tadabur/run_corpus.sh scenario report       # or some of them
```

Run it detached (`tmux`) — stage 1 is ~3.6 minutes per shard and the sink is checkpointed, so
the right way to use it is to start it, disconnect, and read the artifacts back. Two wirings it
exists to get right, both silent when wrong: `--delete-shards` throws away the parquet but must
keep the staged clips, and stage 4 must be given `--recuts` or it stages un-re-cut audio.
`docs/tadabur-corpus-run.md` is the 11-shard run's report.

### `tadabur.bleed_stage` (Linux — GPU) — the timed decode the re-cut reads

The reject sink carries a decode *string* and the staged WAV carries audio; neither carries a
**time**. This pass produces what `tadabur.bleed_recut` needs from both: per-frame CTC class ids
(so `waqf_detect.collapse_with_times` can put an onset on every phoneme) and the recitation VAD's
clean speech intervals (so a cut can prefer a pause over a signal).

It decodes the **staged WAV**, not the source shard. The shards are 2.4 GB each and
`--delete-shards` discards them as the filter walks; re-reading parquet for a timed decode would
re-download the entire run. The staged WAV is the same 16 kHz mono waveform the gate scored, and
the run reports how often the restaged decode reproduces the stored one — reading the wrong audio
is then visible rather than silent. Only clean re-reads are staged; prevalence over the whole
reject pile is `tadabur.bleed_detect`, which needs no audio at all.

```bash
cd tools
python -m tadabur.bleed_stage --rejects corpus_run/rejects.jsonl \
    --clips corpus_run/clips --out corpus_run/decodes.jsonl [--limit N]
```

### `tadabur.waqf_segments` (Linux/macOS — no GPU) — clip staging

Waqf-aware reference labelling (PRD #1, ADR-0002) splits each admitted clip at its intra-ayah
**waqf pauses** and labels each segment in the form the reciter *actually* recited (terminal word
in **waqf** form, interior words in **wasl**), removing phantom pre-waqf gemination mismatches from
the fine-tune data. The work is split across two stages because pause detection needs the model
(the shipped forced alignment *absorbs* pauses into word spans — see ADR-0002).

`tadabur.waqf_segments` is the torch-free half: it **stages** each passing clip as a whole 16 kHz
mono WAV on local disk (the full Tadabur source is streamed, never landed) and owns the shared
realized-reference vocabulary (`SegmentRecord`, `hafs_phonetizer`, `hafs_word_reference`). No model,
no GPU.

```bash
cd tools
python -m tadabur.waqf_segments --passing passing_subset.jsonl \
    --audio-dir clips/ --config-name preview
```

A full build that cannot locate a passing clip in the stream **fails loudly** (a partial clip set
is a data-integrity failure); a `--limit` smoke run instead tallies the unreached clips as
`missing_due_to_limit`.

### `tadabur.segment_score` (Linux — GPU) — model waqf pass + scoring

Owns the model pass end to end. It first runs a dedicated VAD (`obadx/recitation-segmenter-v2`
via `tadabur.vad`) over all clips to find the **waqf pauses** — the interior silences between
speech spans, with silences < `--min-silence-ms` merged away and speech < `--min-speech-ms`
dropped (both from the VAD's training labels). Then, freeing the VAD, it decodes each staged clip's
**whole** waveform once to per-frame phoneme ids and hands them, with the VAD pauses, to
`tadabur.waqf_detect` — along with the ayah's per-word phoneme boundaries (`hafs_word_reference`
derives these from the phonetizer's char `mappings`, robust to wasl word-merges) — which maps each
pause to a word boundary via Smith-Waterman, splitting at a word edge (waqf) but not mid-word (a
stop-consonant closure). Each resulting segment is then decoded again and scored against its
realized reference with the `.balanced` gate (same normalization / Smith-Waterman / contrast
attribution as the full-ayah filter, per segment). Output is one scored segment manifest (carrying
per-segment offsets — the P4 label source) plus each segment's sliced audio for the audit UI,
feeding the audit sampler + UI.

```bash
python -m tadabur.segment_score --passing passing_subset.jsonl --clips-dir clips/ \
    --out-manifest segment_manifest.jsonl --audio-out segment_audio/ \
    [--min-silence-ms 300] [--min-speech-ms 700] [--boundary-tol 3] [--vad-dtype bfloat16]
```

A clip that cannot be segmented safely (`repeated_recitation` / `low_alignment`) is kept whole (one
whole-ayah segment) and tallied; the 8 phonetizer-gap ayat are skipped (`phonetizer_unsupported`).
Feeds P4 data-prep (#8): the manifest is the label source, the reciter split is computed over the
post-segmentation units, and the collator slices audio by these offsets.

### `tadabur.scenario` (Linux — GPU) — the re-read corpus Muraja consumes

Stages the **clean re-reads** the reject sink mined (`tadabur.rejects`) into the cross-repo
interface of Muraja ADR-0016: a `scenario.jsonl` plus 16 kHz mono WAVs, one record per clip.
It re-cuts each clip to the recitation span `tadabur.bleed_recut` found (#67/#68), decodes the
result once with timing, locates the re-read seam in phoneme space (`tadabur.seam`), maps the
alignment's reference span to a **word** range, and then cuts the repeat out and re-gates the
result (`tadabur.excision`) to produce the paired control clip decision 4's second oracle needs.

```bash
cd tools
python -m tadabur.scenario --rejects reject_run/rejects.jsonl \
    --clips bleed_run/clips/ --recuts bleed_run/recuts.jsonl \
    --out scenario_run/shard20 [--limit N] [--batch-size 4]

# On the Mac, after the transfer — no model, GPU or reference cache needed:
python -m tadabur.scenario --verify tadabur_corpus/
```

Clips are processed in `clip_id` order, so a `--limit` run is a prefix of the full one and two
runs of the same command produce byte-identical output. Measured on shard 20: 45 clips in 47 s,
64 MB (35 MB staged audio, 30 MB control clips, 68 KB manifest). See
`docs/tadabur-excision-yield.md` for the yield and its limits.

#### The `scenario.jsonl` schema

One JSON object per line, keys sorted, written in `clip_id` order. Paths are **relative to the
manifest**, so the bundle moves as one directory. `schema_version` is `v1+norm<N>`, tied to
`tadabur.normalization.ALGORITHM_VERSION` because every phoneme offset in the record indexes
into a normalized string.

| field | meaning |
| --- | --- |
| `schema_version` | `v1+norm2` — reject a bundle whose version you do not know |
| `clip_id` | stable identity; the reject sink's join key is `<clip_id>.wav` |
| `audio` | `audio/<clip_id>.wav` — the staged clip, 16 kHz mono PCM_16, bleed already cut |
| `surah_ayah`, `reciter_id`, `duration_s` | the clip's ayah, reciter, and staged length |
| `match_ratio`, `max_insertion_run`, `leading_trim`, `trailing_trim`, `added_shadda` | the `.balanced` gate over the **staged** audio (`tadabur.scorer.GateResult`) |
| `predicted_phonemes` | the staged clip's decode — what the seam offsets index into, once normalized |
| `word_start`, `word_end` | **half-open, 0-indexed Uthmani word range the oracle may assert over.** Words only fully covered by the alignment; ADR-0016 decision 1 |
| `ref_covered_start`, `ref_covered_end`, `ref_length` | the same span in reference-phoneme space, as provenance |
| `uncovered_head`, `uncovered_tail` | reference phonemes before/after the covered span — a clip begun late or left unfinished |
| `early_start` | the lead-in is long enough (`leading_trim >= 5`) that starting a session one ayah earlier may help. Rare since #68 re-cuts bleed; read it per clip, do not apply a blanket policy |
| `recut_applied`, `recitation_start_s`, `recitation_end_s` | what was kept of the source clip, shaped like `ClipStatus`'s fields |
| `seams` | one entry per re-read: `query_start`/`query_end` (normalized-decode indices), `ref_position`, `phonemes`, `start_s`/`end_s` (where a cut lands), `pause_anchored_start`/`pause_anchored_end` |
| `excised_audio` | `excised/<clip_id>.wav`, or **null** when the pair did not survive re-gating |
| `excised_duration_s`, `excised_phonemes` | the control clip's length and decode — kept for a refused pair too, so a discard can be explained |
| `excision` | always carries `reason`; carries the full re-gate numbers when a cut was made |

**No timestamp from Tadabur's `word_alignments` or from `ClipStatus.word_times` appears in this
file.** The seconds that do appear are the clip's own duration and the spans cut out of it —
neither is an assertion, and both describe audio the gate re-scored.

A sibling `staging.json` carries the run's tallies (selected, staged, seams, pairs attempted and
kept, refusal reasons, early starts, truncated clips).

### `tadabur.reread_audit` (Linux/macOS — no GPU) — the sampled listening audit

ADR-0016 decision 10 no longer requires every corpus clip to be heard. A **vowel-only** non-Hafs
divergence cannot move the alignment cursor — alignment runs on the normalized string, and
normalization deletes short vowels — so it is inert to cycles-to-resync and falsely-skipped
words alike. What replaces the full pass is a ~50-clip random audit, and that audit is not a
screen: it is the test of the inertness argument's one premise, that vowel-only divergence
dominates. So it records **how** a non-Hafs reading diverges, and a `nonhafs` verdict with no
mode is reported as `unclassified` rather than assumed inert.

```bash
cd tools
python -m tadabur.reread_audit --bundle corpus_run/scenario --out corpus_run/audit \
    [--size 50] [--seed 0]     # stage the worklist and its audio
python -m tadabur.reread_audit --summary   # what the verdicts so far say
```

Sampling is off the staged bundle, so what is heard is what Muraja replays: post-re-cut audio at
the boundaries the corpus asserts. The draw is seeded and sorted, so the same bundle and seed
reproduce it and a re-run tops the sample up rather than redrawing it. Verdicts are appended to
`tadabur/eval_fixtures/reject_reread_verdicts.jsonl`; its schema is in that directory's README.

### `training.waqf_distill` (Linux — GPU teacher, CPU pooling) — waqf soft labels

The teacher half of the waqf-head distillation (ADR-0004). Because the deployed / student
model sees **fixed 5 s windows** (250 feature → 125 student frames), the teacher must too:
a transformer frame classifier's window-local posteriors differ from a whole-clip pass
(attention context, window-edge padding), so the generator cuts each clip's waveform into
the same fixed windows the student uses and runs the same Recitation VAD
(`obadx/recitation-segmenter-v2` via `tadabur.vad`) over each **window waveform**. It keeps
the raw **per-20 ms silence posteriors** (`P(silence)`, not the cleaned intervals) and
**pools each window 2:1 to Muaalem's 40 ms CTC lattice** by a pinned rule: student frame
`i` owns teacher frames `2i`/`2i+1` and is silent iff *both* are (min-pool silence /
max-pool speech), left-anchored so a ±few-frame feature-extractor drift is absorbed at the
window tail, never by shifting an interior boundary. **A window's student-frame count comes
from its audio span, not from how many frames the VAD emitted** — `feature_frames_for_samples`
reproduces the `SeamlessM4TFeatureExtractor` length exactly (`(num_samples - 80) // 320`, *not*
the naive `num_samples // 320`, which over-counts by one on most spans), and
`training.windowed_labels` derives its `logit_frames` from the same expression, so the phoneme
CTC target and its silence teacher always land on the grid the model itself produces. Targets
are emitted **per training window**, keyed to the passing-subset manifest by `(audio_filename, window_index)`. The
window length is the deployed 5 s and the spacing defaults to a **provisional
non-overlapping tiling** (`--hop-feature-frames`) pending the #24 inference-window contract
(overlap/edge/stitch). Output goes into a deterministic, idempotent `SoftLabelStore`
(per-window `.npy` arrays + a `soft_labels.jsonl` index, one line per clip listing its
windows). The exact generation contract (window/hop, pooling rule, adapter + frame
geometry, VAD id) is stored as `contract.json` and **re-checked on resume**, so a run that
would append labels built under a different contract **fails fast** instead of silently
corrupting the artifact. Generation **streams one clip at a time** and fsyncs each clip
before the next, so a crash mid-run keeps every clip already written and a resumed run
skips them — the whole manifest is never held in memory. The pooling/windowing/alignment is
torch-free and covered by golden fixtures (`training/test_waqf_distill.py`); only the VAD
forward pass needs the GPU. The windowed collator (#8) consumes these per-window targets
against the phoneme lattice.

```bash
cd tools
python -m training.waqf_distill --manifest passing_subset.jsonl --clips-dir clips/ \
    --out-dir waqf_soft_labels/ [--window-feature-frames 250] [--hop-feature-frames 250] \
    [--device cuda] [--dtype bfloat16] [--batch-size 8]
```

### `training.distill_*` (Linux — GPU) — size distillation to a single ANE chunk

Distils the 578M teacher's **phoneme head** into a thin student so the deployed model can
drop from six 6-bit CoreML chunks (504 MB) to one (ADR-0010). The driver is per-window cost,
not size for its own sake: Muraja issues **~6 inferences per audio-second** — one full-window
pass plus ~5 throttled previews (`previewMinNewSamples` = 200 ms) — and each pays the full
static `(1, 250, 160)` cost, which at the 150–200 ms A15 figure saturates the ANE. The
5 s/1 s windowing contract is frozen, so per-window cost is the only remaining lever.

The objective is **behavioural cloning, not accuracy**: the student is correct insofar as it
reproduces the teacher, because the teacher's behaviour is what Muraja's scorer and fixtures
are tuned against. Two things follow — **no labels are needed** (the teacher generates every
target, so any recitation audio is training data) and **only the phoneme head is built**
(Muraja consumes 1 of the teacher's 11 heads).

- **`distill_student`** — the width ladder. Every preset keeps all **24 layers** and shrinks
  `hidden_size`, because `ml-model-transformation.md` §6 shows depth is the axis that
  destroys this backbone (24→12 gave 99.4% CER). `h512` 151.8M → 108.6 MB → 2 chunks;
  **`h384` 85.5M → 61.7 MB → 1 chunk** at 7.1x less compute; `h256` 38.2M → 27.3 MB.
  `--verify` instantiates each preset and asserts the `(1, 250, 160) → (1, 125, 43)` contract.

  Size is modelled over **graph constants, not parameters**, and students use **rotary**
  position embeddings rather than the teacher's `relative_key`. A traced `relative_key`
  attention bakes a `(250, 64, 250)` constant into the graph *per layer* — 96M values over
  24 layers, set by sequence length and head dim and **independent of width**. That is 16%
  of the teacher and would be 53% of `h384`: exporting `h384` with `relative_key` measured
  **130.3 MB** (over budget, 2 chunks) against the same student with rotary at **61.7 MB**.
  Since students train from random init, they need not inherit the teacher's scheme.
- **`distill_data`** — windows are cut from the **waveform before** feature extraction, never
  after: `SeamlessM4TFeatureExtractor` normalizes per utterance while the device normalizes
  per 5 s window, so slicing extracted features would train off-distribution. Short trailing
  windows are kept and zero-padded because the device pads too (5 of its ~6 inferences per
  second run on a partially filled buffer). The train/val split hashes the **clip** name, so
  overlapping windows cannot leak across it.
- **`distill_loss`** — frame-weighted logit KL + tapped feature matching + a CTC anchor.
  Non-blank frames are up-weighted because CTC output is blank-dominated (per Muraja's
  `CTCStats`, 30–60% blank during speech, 85–98% during silence), so a flat KL is mostly a
  lesson in predicting blank; this is the mirror image of what `training.waqf_head` solved
  with `pause_frame_weights`, and takes the same two answers — a weighting and a collapse
  diagnostic. The first **25** timesteps are up-weighted because `predictSplit` commits only
  those to the transcript (`seg.midpoint < 25`), a phoneme being confirmed when its window
  position is *oldest* and it has the full 4 s of right context.

  **Neither of those escapes the all-blank basin**, which is why `ctc_anchor_loss` exists.
  Both are *per-frame* objectives, and a per-frame objective cannot break alignment
  symmetry: until the student knows which frames carry which phoneme, blank is locally
  optimal at every frame, so all-blank is a stable fixed point — reweighting frames does not
  help, because the problem is not which frames are weighted. Measured on the KL-only recipe
  at step 2000, the teacher's class sat at rank **8.2** with P=**0.027** against blank's
  **0.822**, while top-5 agreement of 0.51 (chance 0.12) showed the encoder had genuinely
  learned. The CTC anchor is a *sequence* objective against the teacher's decoded tokens: its
  forward-backward sums over every valid alignment, and an all-blank output has probability
  zero under any alignment of a non-empty target. `--ctc-weight 0` reproduces the old recipe.
- **`distill_train`** — frozen bf16 teacher **online** rather than cached: caching its logits
  is cheap but forfeits feature matching, and caching hidden states costs ~3 MB/window
  (terabytes). Carries the `whole_clip_phoneme` VRAM preflight pattern — measured **10.97 GiB
  at batch 32** on the 16 GB card; batch 48 OOMs. Watch `blank_collapse_margin` and
  `nonblank_agreement`: the all-blank basin scores ~67% frame agreement for free and is the
  observed starting point, with `--nonblank-weight` as the knob.
- **`distill_eval`** — the release gate. Replays the deployed protocol (5 s window, 1 s hop,
  `scanCTC` collapse, `midpoint < 25` confirmation) and compares the **confirmed transcripts**,
  reporting exact-match and char-accuracy in the same shape as the quantization rows in
  `ml-model-transformation.md` §1.3. Frame agreement is used during training but is not the
  gate — it averages over 100 timesteps the user never sees.

  `--breakout` answers a different question, for use while the student is still
  blank-collapsed: *is this run converging or stuck?* `nonblank_agreement` reads a flat 0.0
  for thousands of steps either way, because argmax cannot distinguish "the teacher's class
  holds 40% and is about to overtake blank" from "it holds 0.1%". The diagnostic reports the
  continuous quantities over teacher-non-blank frames — `target_prob`, `target_rank`
  (1 = agrees), `prob_margin` (≤0 means escaped) — so the decision to keep waiting or
  intervene is measured rather than guessed.

```bash
cd tools
python -m training.distill_student --verify            # sizing ladder + shape contract
python -m training.distill_data --audio-root <wav-dir> # clips, hours, window counts

python -m training.distill_train --preset h384 --audio-root <wav-dir> \
    --out-dir runs/h384 --batch-size 32 --preflight-only   # check VRAM before committing
python -m training.distill_train --preset h384 --audio-root <wav-dir> \
    --out-dir runs/h384 --batch-size 32 --steps 60000 [--resume]

python -m training.distill_eval --checkpoint runs/h384/checkpoint.pt \
    --audio-root <wav-dir> --num-clips 200
```

### `convert_to_coreml.py`

Converts the Wav2Vec2-BERT TorchScript model (`obadx/muaalem-model-v3_2`) to CoreML format optimized for Apple Neural Engine. Traces the model with a fixed input shape `(1, 250, 160)`, exports to FP32 `.mlpackage`, and optionally creates INT8 and 4-bit compressed variants.

```bash
python convert_to_coreml.py [--output-dir ./coreml_models] [--skip-quantization] [--pruned-model path/]
```

### `generate_phonemes.py`

Downloads the `obadx/muaalem-annotated-v3` dataset from Hugging Face and reconstructs per-ayah phoneme reference strings. Merges overlapping segments, deduplicates, and selects the most complete variant per ayah.

```bash
python generate_phonemes.py
# Output: ayah_phonemes.json
```

### `palettize_chunks.py`

Applies 6-bit palettization to the six chunked FP32 CoreML model packages using `coremltools`. Skips chunks that already have palettized output.

```bash
python palettize_chunks.py [nbits]  # default: 6
# Input:  coreml_models_chunked/*_FP32.mlpackage
# Output: coreml_models_chunked/*_6BIT.mlpackage
```

### `verify_coreml.py`

Runs identical random inputs through both the PyTorch model and exported CoreML model(s), comparing outputs numerically (max/mean absolute error, cosine similarity, argmax agreement).

```bash
python verify_coreml.py [--model-dir ./coreml_models] [--variant FP32|INT8|4BIT]
```

### `compile_models.sh`

Compiles palettized `.mlpackage` models to `.mlmodelc` bundles using Xcode's `coremlcompiler`. The compiled models can be loaded directly on-device without runtime compilation. Creates a `models.zip` for uploading to a GitHub Release.

```bash
bash compile_models.sh [input_dir] [output_dir]
# Default input:  coreml_models_chunked/
# Default output: compiled_models/
# Output zip:     models.zip
```

### `generate_quran_db.py`

Builds the consolidated `quran.db` SQLite database from multiple Quran data sources. Creates tables for surahs, ayahs, words, word-phoneme mapping, mushaf page layout, and ligature mappings.

```bash
python generate_quran_db.py
# Input:  ../data/*.json, ../data/*.db
# Output: quran.db
```

### `requirements.txt`

Python dependencies: `torch`, `coremltools`, `transformers`, `huggingface_hub`, `numpy`, `soundfile`, `librosa`.

---

For a detailed deep dive on the model conversion and ANE deployment process, see [ML Model Transformation](../ml-model-transformation.md).

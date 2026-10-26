# tlog hold-out and cleaning: pre-registered rules (#173)

**Status:** frozen 2026-10-24, owner-approved 2026-10-24 (calls 1–5), before any tlog audio is decoded,
played or transcribed. Nothing below changes after S3a starts reading audio. A change needs a new, dated
pre-registration, never a re-run. Index commit: `683c3e8039838a3d589307a8458390693d4b8e7a`.

## Source
`tarteel-ai/tlog` @ `9fe4f7e6d1b372c2c11157d6d2f0f91ca344eb51`, splits `clean` and `unclean`. A row is
keyed by `(split, shard_file, row)` and identified by `audio_path` and, once read, by `audio_sha256` (the
SHA-256 of the stored audio bytes). No tlog audio is ever committed or published.

## Groups and roles
1. **Id group:** rows that share an `audio_path`, or a parsed recording id, across both splits.
2. **Speaker cluster** (S2 validates it, S3b applies it):
   - Embedding: L2-normalised speaker embeddings, one per clip, from the first 20 s at 16 kHz mono.
     Candidate models, tried in order: (A) `speechbrain/spkrec-ecapa-voxceleb`,
     (B) `microsoft/wavlm-base-plus-sv`. The revision is pinned when S2 starts. S2 picks the first model
     that passes the validation rule.
   - Graph: exact cosine k-nearest neighbours (k = 20), computed in blocks. There is an edge where
     cosine ≥ τ. Clusters are the connected components.
   - Cap: any component with more than 1% of rows is re-clustered by the same procedure at τ + 0.02,
     repeatedly, until no component exceeds 1%.
   - Embeddings are deleted after the split freeze. Only cluster ids are kept, and only internally.
3. **τ and validation, on Tadabur only** (training-complement shards, never tlog):
   - The sample: ≥ 200 reciters, ≤ 20 ayah clips each. Ayat are chosen so that ≥ 50 ayat are recited
     by ≥ 5 sampled reciters each. Clip durations match Tarteel v1 look's distribution by decile.
   - τ is the lowest value on the grid 0.30, 0.32, …, 0.90 at which both hold:
     - (i) **no chaining:** the largest cluster is at most 3× the largest single reciter's clip share;
     - (ii) **text-independence:** the co-cluster rate of same-ayah, different-reciter pairs is
       ≤ 2× that of different-ayah, different-reciter pairs, and ≤ 1% in absolute terms.
   - **Pass** if, at that τ, the pairwise recall of same-reciter pairs is ≥ 0.9.
   - If model A fails, try B. If both fail, use the **fallback**.
4. **Group:** connected components over two kinds of link: same id group, and same speaker cluster.
   Under the fallback, the group is the id group alone.
   - `group_key` = the lexicographically smallest `audio_path` in the group.
5. **Role:** `u` = the first 8 bytes (big-endian) of `sha256("tlog-holdout-2026-10:" + group_key)`,
   divided by 2^64. Then:

   | u | role |
   |---|---|
   | [0.00, 0.02) | `look` |
   | [0.02, 0.04) | `confirm.1` |
   | [0.04, 0.06) | `confirm.2` |
   | [0.06, 0.08) | `confirm.3` |
   | [0.08, 0.10) | `confirm.4` |
   | otherwise | `train` |

   Every row has a role. Exclusion is recorded separately.
6. **The freeze is a commit** (S3b) of:
   - every row's cluster id, group key and role;
   - the embedding model id and revision, the clustering code SHA, τ, and the S2 validation report.
7. **Tranche use:**
   - `look` may be inspected, decoded by any model and audited.
   - `confirm.k` is decoded by student models only for one round's verdict, opened in the order 1 → 4.
     It is never listened to, audited or inspected before that round.
   - Teacher-only identification of sealed tranches (S7) is allowed; it is not a student read.
   - Each tranche's surah and ayah distribution is reported against the training pool's.
8. **Fallback consequences:** every tlog verdict states "clip-level split, same users possible on both
   sides", and that round's pre-registration adds a qdat consistency bar.

## Exclusions (rows are never deleted, only marked `excluded` with a reason)
- `eval_overlap:tarteel_v1`: the recording id is among the 8,950 decoded Tarteel v1 clips (S1), or the
  audio matches any frozen eval set (S4).
- `eval_overlap:<set>` (S4): checked against Tarteel v1 (both halves), QUD, Tadabur's eval frame
  (`held_out_shards()`), gate_eval, qdat, qdat_bench and RetaSy. The checks are an equal `audio_sha256`,
  or a fingerprint match (`qud_fingerprint`, Haitsma–Kalker inverted-index lookup, verified by
  `best_match` at `qud_overlap`'s threshold rule).
- `duplicate` (S4): the same audio inside tlog, by checksum or fingerprint.
  - Across roles, exclude the `train` copy; never an eval copy.
  - Across two eval tranches, keep the copy in the earliest-opened tranche (look, then confirm.1 … 4) and
    exclude the others.
  - Within one role, keep the copy with the smallest `audio_path`.
- `codec`: the audio does not decode, or its duration is 0.
  - Amended 2026-10-24 (S3a, pre-outcome): "decodes" means **soundfile, or the pinned PyAV
    fallback** (`av`, pinned in `tools/requirements-train.txt` and recorded in
    `pass1/meta.json`; its wheel bundles ffmpeg), which S3a's pass 1 uses only when soundfile
    refuses the container. The unclean split carries MP4/AAC bytes behind `.wav` paths (48 of
    the tracer's 465 unclean-0 rows) next to 268 zero-byte rows; the zero-byte rows stay
    `codec_ok = false`, and only container support was added. No tlog outcome had been read
    when this was written, so it is not outcome-informed. The floors are unaffected: all 4,537
    Tarteel v1 look clips decoded with soundfile, so `silence_floors.json` needed no rerun.
- `silence`: speech ratio below f_s, or speech seconds below d_s.
  - Speech is measured by Silero VAD, pinned in S3a.
  - f_s and d_s are the 0.5th percentiles over Tarteel v1 look clips, computed and committed in S3a,
    before S3b reads tlog.

## Ayah identification (base teacher only; no student or candidate output enters any rule)
- **Decode:** the base teacher (`obadx/muaalem-model-v3_2` @ `01a1ef9`), a whole-clip greedy decode, the
  same mode as Tarteel v1's committed `decodes.base` field `spans`.
- **Reference database:**
  - each ayah is phonetized on its own in its waqf form (as `generate_phonemes.py` does), and the ayat
    are concatenated in order per surah: 114 sequences, with word offsets;
  - **text class:** ayat with identical references are one class. Agreement, relabel and margin compare
    classes, never ayah numbers;
  - **decoys:** each ayah's reference reversed, concatenated the same way.
- Amended 2026-10-25 (S5, pre-outcome; no Tarteel v1 look clip had been scored when this was
  written). These pins fill in procedures the text above leaves open, and correct one count. No
  threshold rule changes.
  - **Alphabet:**
    - decodes and references are compared after `normalize_phonemes` (algorithm version recorded);
    - each ayah is normalised separately, and word offsets are remapped through its `offset_map`;
    - `len(decode)` is the normalised length;
    - words are quran-transcript `uthmani.split()` words;
    - the 8 `FALLBACK_PHONEMES` ayat take boundaries 0…n−2 from phonetizing `words[:-1]`, and the
      last word's start from an infix alignment of that prefix into the fallback string;
    - text classes are equality of the normalised per-ayah references.
  - **Decoys:** each normalised ayah reference is reversed by grapheme cluster (a base plus its
    combining marks), and the reversed ayat are concatenated in ayah order per surah.
  - **Shortlist:**
    - consonant-skeleton 3-grams over windows of an ayah ± 1;
    - a window's rank is its number of distinct query 3-grams, ties by (surah, ayah) of the centre
      ayah;
    - shortlist recall for the guard means the shortlist's best score equals the best score over all
      windows;
    - the library is `edlib` 1.3.9.post1, mode `HW`.
  - **The best decoy score uses the same search as the real one:** the same shortlist construction
    over the decoy sequences, the same k, the same scoring. The best over all decoy windows is
    reported, never used.
  - **Real-mistake clips:** the bar on b uses the **52** clips that hold the 56 sites
    ("42" above was the reciter count). They are listed in
    `tools/tadabur/tlog/ident/real_mistake_clips.jsonl`
    (sha256 `6b8c4002b098a77218d68e3cfa730b088703b37dd03a9c3768df2ffbcbec59df`).
- **Search:**
  - a consonant-skeleton n-gram index shortlists the top k = 20 windows (an ayah ± 1 neighbour);
  - infix (semi-global) edit distance with a pinned C-backed library scores each window:
    `score = 1 − distance / len(decode)`;
  - the result is the best location as a **word span** `surah, ayah_from:word_from, ayah_to:word_to`.
    Its ayah is the one with the most aligned words. `straddles` is true when the span crosses an ayah
    boundary;
  - **Guard:** on 1,000 Tarteel v1 look clips, the shortlist recall against exhaustive search must be
    ≥ 99.9%. Otherwise k doubles until it is.
- **Floor (non-recitation):**
  - length bins of the decode: ≤ 20, 21–40, 41–80, 81–160, > 160 characters;
  - the floor of a bin is the 99th percentile of the best **decoy** score of Tarteel v1 look decodes in
    that bin;
  - `non_recitation` = speech present and the best real score ≤ floor;
  - **reported, never tuned on:**
    - the pass rate of Freesound noise (`nonpro_gap/noise_pool.json`), which must be ≤ 1%;
    - the retention of Tarteel v1 look;
    - the retention of the look tail with `cer_wav ≥ 0.2`;
  - **stop:** if look retention is < 90% or the noise pass rate is > 1%, stop and re-pre-register.
- **Relabel (rows with a filename label):**
  - Δ = score(best class) − score(the filename ayah's class, aligned within its window of ayah ± 1);
  - relabel to the best class when all of these hold: the floor is passed, Δ ≥ b, and the best span does
    not overlap the filename ayah;
  - b is the smallest value on the grid 0.00, 0.01, …, 0.50 at which both hold: ≤ 0.5% of Tarteel v1
    look clips would be relabelled, and 0 of the 52 clips holding the 56 owner-verified real-mistake
    sites (`docs/truth-baseline.md`) would be relabelled (52, not 42: amended 2026-10-25);
  - the relabel rate in the `cer_wav ≥ 0.2` tail is reported;
  - if no b on the grid meets both, stop and ask the owner.
- **Identify (rows without a filename label):**
  - m = the best score minus the best score of a different text class at a non-overlapping location;
  - assign when the floor is passed and m ≥ b_u. b_u is the smallest grid value at which ≥ 99% of
    Tarteel v1 look clips, with their labels hidden, are assigned their own class.
- **`label_source`** takes one of these values:

  | value | meaning |
  |---|---|
  | `filename` | below the bar, kept under the filename label |
  | `both` | the best class equals the filename class |
  | `identified` | relabelled, or assigned to an unlabelled row |
  | `none` | no label |

## Evaluation population
- **Gating items** meet all of these:
  - in a `look` or `confirm.k` role;
  - in the `clean` split, with a filename label;
  - not excluded, not `non_recitation`, not relabelled.

  Their reference is the filename ayah's full text. The span is stored, never used to trim the
  reference.
- **Reported strata, never gating:** `identified` rows, `unclean` rows, and `straddles` rows.
- **Bootstrap clusters:** the group (speaker cluster); under the fallback, the surah.

## Training pool
- `tlog.training` = rows with `role == train`, not excluded, codec ok and not silence, at a fixed
  overlay commit. It uses no teacher output and no label. Training is KL to the teacher, and filename
  labels never enter a loss.
- **5% rule:** S6 identifies a seeded simple random sample of 2,000 `tlog.training` rows. If the upper
  95% Wilson bound of their `non_recitation` rate is above 5%, the whole pool is decoded, and
  `not non_recitation` is added under a new overlay commit and a new exposure use. The S6 audit's
  `non_recitation` stratum reports how precise that flag is.

## Audit (S6)
- About 150 items from `look` and from the 2,000-row training sample only, never from `confirm.k`,
  stratified:

  | stratum | items |
  |---|---|
  | `both` | 40 |
  | relabelled | 30 |
  | `non_recitation` | 25 |
  | silence | 15 |
  | unclean identified | 25 |
  | below-bar disagreement | 15 |

- **Verdicts:** right ayah, other ayah (which), silence, not recitation, unclear.
- **Outputs:** precision per stratum with a Wilson 95% interval. The audit is never a filter.
- **Outside-ASR trigger:** if the residual wrong-ayah rate in `both` is above 2%, open an issue for an
  outside-family ASR cross-check.

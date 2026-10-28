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
  shadda, tanween. In the model's output, only fatha/damma/kasra (**haraka**) have their own
  classes. Sukun is today an *empty* slot, shaddah is a doubled consonant with a CTC blank
  between the two, and tanween is a haraka followed by ن or ں (ADR-0011).
: _Avoid_: Diacritics (too generic), vowels

**Haraka**
: One of the three short-vowel marks fatha, damma, kasra — the tashkeel the model emits as
  classes 32–34.
: _Avoid_: vowel

**Empty tashkeel slot**
: A letter on which the model emits no tashkeel. Today it means *either* sukun *or* the model
  being unsure, which is why Muraja cannot grade it reliably. Under ADR-0011 sukun gets its own
  output, so an empty slot means only **unsure**, and Muraja does not grade it.

**Commit rate**
: Of the tashkeel a reciter actually said, the share the model emitted a mark for (right or
  wrong) instead of leaving the slot empty. Reported with the **accuracy of committed marks**,
  never alone (ADR-0011).

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
  (20 ms frames) fine-tuned for waqf. The source of waqf pauses in the Tadabur labelling
  pipeline (`tadabur.vad`). It was also the teacher for the retired waqf head.

**Waqf head** _(retired, ADR-0011)_
: A per-frame speech/silence head on the Muaalem backbone, distilled from the Recitation VAD,
  meant to let the scorer tell waqf from wasl (ADR-0004). Dropped as a requirement on
  2026-10-08; Muraja keeps its end-word allowance instead.

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

**Wasl-dependent site**
: A site whose prescribed mark changes when a waqf is placed at a boundary of its word: a
  geminate made by joining two words (idgham, cross-word assimilation), a ta marbuta read `ت`, a
  word-final haraka. Decided through the phonetizer, never hand-listed; a sun-letter `لام` is
  not one (`tadabur.waqf_guard`, #134).

**Local pause check**
: The recitation VAD's frame-level silence probability read between two words' aligned times,
  more sensitive than the segmenter, at a wasl-dependent boundary (`tadabur.local_pause`).

**Waqf-realized**
: A wasl-dependent site at a boundary judged a pause (by the local pause check, or an owner
  `waqf` note): no site of its stratum, out of every evaluation denominator; its verdict stays a
  training target.

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

**Commit block**
: The one-second block of each 5 s streaming window (25 of its 125 CTC timesteps) that the
  streaming protocol commits to the transcript; the rest is re-decoded by later windows.
  Muraja commits **b=0**, the oldest second (no left context, 4 s of right). At `b > 0` the
  first window also commits the blocks before `b`, because no earlier window covers them, and
  the last window flushes the blocks after it (`training.decoding`).

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
  Muraja's `.balanced` exempts from the phoneme gate, so a swap costs only score and the word
  passes while it still scores ≥ 0.65; `.strict` caps the word at minor at best. `ذ↔ظ` is **not**
  a soft pair (gated in `.balanced`) but is a target pair all the same. The
  discrimination the model must sharpen rather than collapse (ADR-0001). Every soft pair is a
  **target pair**; not every target pair is a soft pair.

**Target pair**
: A consonant pair the model must tell apart: the six soft pairs plus `ذ↔ظ`, `س↔ش` and
  `ط↔ق`. Only the six are Muraja soft pairs; the app forgives none of the other three.

**Allowance**
: A Muraja grading rule that forgives a model error rather than a reciter's: soft-pair
  forgiveness, shaddah suppression, the dropped-haraka exemptions. Each exists because the model
  is not faithful there. ADR-0012 names the ones the model should make unnecessary (soft-pair
  forgiveness, shaddah suppression, the Tashkeel toggle); the dropped-haraka exemptions stay.
: _Avoid_: override, tolerance (unqualified)

**App outcome**
: The grade a reciter would see for a word, simulated through Muraja's real grading: every full
  window and preview regrades the word and only the best grade is kept. A candidate model is
  judged on the app outcome **with the allowance under test switched off** (decided 2026-10-09);
  the per-decode numbers explain why it moved.
: _Avoid_: decode accuracy (that is per decode, not per word)

**False flag / missed mistake**
: The two sides of an **app outcome**: a correctly recited word shown as wrong, and a real
  mistake not flagged, whether shown as correct or left ungraded. Always reported separately,
  never pooled (ADR-0008).

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

**Truth site**
: One human-labelled position, independent of any model: an item's audio provenance, the
  **carrier letter** in its realized reference, the mark under test, what the mushaf prescribes
  and what the reciter was heard to say (`tadabur.truth_sites`, `tadabur/truth_sites/README.md`).
  The unit ADR-0011 scores models against. A label that holds only if the reciter recited
  correctly is flagged `assumes_competent_reciter`.
: _Avoid_: fixture (the older per-clip label sets), ground truth (unqualified)

**External corpus**
: Recitation from outside Tadabur that neither the base teacher nor `h448` has heard, read for
  evaluation only (`training.external_corpus`): QUD's professional mushafs (#144) and the
  non-professional corpora (#156: Tarteel v1 app users, RetaSy, qdat). Its agreed sites
  `assume_competent_reciter`; its exposure uses are refused by `check_disjoint`.
: _Avoid_: held-out set (it was never part of any split), test set

**Look half / confirm half**
: Tarteel v1's two halves by clip hash (`tadabur.tarteel_split`, pre-registered in #128): a
  clip is in the look half when the first 8 bytes of `sha256("tarteel-v1-split-2026-10:" +
  clip_id)` are below 2**63. The look half may be read while a recipe is chosen; the confirm
  half is read once, for the final candidate.
: _Avoid_: dev/test (they are halves of one evaluation corpus, never training data)

**Excess flag**
: A word `h448` flags under Muraja's shipped default where the base teacher's checks grade it
  correct, on the same recitation (#160): the gap the augmentation experiment tries to close.

## tlog Hold-out

The vocabulary of `tarteel-ai/tlog` (#185, `tadabur.tlog_index`), whose clips are split by
**group** before any audio is decoded. The rules are frozen in
`docs/tlog-holdout-preregistration.md`; this section only names them.

**tlog**
: The gated Hub corpus of 719,853 amateur app clips in the `clean` and `unclean` source splits.
  Only its metadata (`audio.path`) is used and committed; its audio is never copied, and its own
  `label` column is never read.

**Metadata index**
: The committed per-shard metadata (`tools/tadabur/tlog/index/`, built by
  `python -m tadabur.tlog_index build`): one row per `(split, shard_file, row)` with the clip's
  path, filename label, recording id, **id group** and Tarteel v1 overlap flags. The committed
  copy is the source of truth; a rebuild only runs on a host with an accepted Hub token.

**Id group**
: Rows sharing an `audio_path`, or a parsed recording id, across both source splits — one
  component, named by its lexicographically smallest `audio_path`. It is the unit the S3b
  **group** and every later count are built on.
: _Avoid_: duplicate (that is one specific exclusion rule), cluster (that is the speaker cluster)

**Group**
: The connected components over two kinds of link: same **id group**, and same **speaker
  cluster**. Under the pre-registered fallback the group is the id group alone. Its
  `group_key` is its lexicographically smallest `audio_path`.

**Speaker cluster**
: Clips the S2 clustering would link as one voice: L2-normalised speaker embeddings, exact
  cosine k-nearest neighbours, connected components, capped at 1% of rows. S2 failed its
  pre-registered validation and the owner accepted the fallback on 2026-10-24, so **no
  embedding was ever computed for tlog and no cluster exists**: every freeze row's
  `speaker_cluster` is `null`, and the group is the id group alone. (Tadabur's own clusters
  are a different corpus and rule.)

**Role**
: Which use a group's rows are reserved for, from `u` — the first 8 bytes, big-endian, of
  `sha256("tlog-holdout-2026-10:" + group_key)` over 2**64 (`tadabur.tlog_index.role`):
  `look` in [0.00, 0.02), `confirm.1` … `confirm.4` in the next four 2% bands, else `train`.
  A role is not one of the Tarteel v1 halves: different corpus, different salt, six values.
  The committed freeze lives in `tools/tadabur/tlog/split/` — the issue's directory name, not
  a data split; each file carries `split` (clean/unclean) and `role` as separate fields.
: _Avoid_: split (that is `clean`/`unclean`, or Tarteel v1's look/confirm halves), fold

**Tranche**
: One role's rows as a frozen set: `look`, or `confirm.k` for k = 1…4. `look` may be inspected
  freely. `confirm.k` is opened by one round, in order, and is never listened to, audited or
  inspected before that round: **no student, C95 or candidate model decodes a confirm row before
  its opening round**, and only for that round's verdict (`docs/tlog-holdout-preregistration.md`
  §7). S7's identification of all four tranches used the base teacher only
  (`docs/tlog-gating-amendment.md`).

**Gating item**
: A clean row with a filename label, not excluded, non-straddling and `label_source == both`: the
  base teacher's best class is the filename ayah's text class. The recitation floor and the Δ
  relabel bar do not decide it (`docs/tlog-gating-amendment.md` §1). Only gating items carry a
  round's verdict; every other manifest row is reported.
: _Avoid_: eval set, test set (that is the whole role, exposure `tlog.<role>`)

**Straddle**
: A decoded span that crosses an ayah boundary rather than sitting inside one, so the row's ayah
  is assigned but not trusted (`straddles`, `docs/tlog-holdout-preregistration.md`). A straddling
  row is reported, never gating.

**Manifest stratum**
: A tranche manifest row's one value, in a fixed precedence: `reported:unclean` (the unclean
  split), `reported:undecided` (`label_source == none`), `reported:straddles`, `gating`
  (`both`), `reported:identified`, `reported:below_bar` (the old below-bar rows)
  (`tadabur.tlog_tranches.stratum_of`, architect r24 item 2). Total and single-valued, so a
  re-slice is a filter over committed rows.
: _Avoid_: audit stratum (S6's listening draw), bucket, bin

**Claim**
: A committed `tranches/confirm.k.claim.json` (`{role, issue, commit, date}`) written by the round
  that opens that tranche (`tadabur.tlog_tranches.claim_tranche`). `confirm.k` cannot be opened or
  read while `confirm.(k-1)`'s claim is not in HEAD with no staged or unstaged change; reading
  `tranches/confirm.k.jsonl` needs `confirm.k`'s own claim in HEAD unchanged too, and the manifest
  itself must be committed. One round may claim two consecutive tranches, in order. `commit` names
  the existing input commit the round opens from — it holds the frozen tranche manifest and the
  round's pre-registration, so the manifest at that commit is what the round read; the claim is
  committed after it is written and cannot name its own enclosing commit.

**Freeze**
: The committed `tools/tadabur/tlog/split/` tree: one row per tlog row with its `group_key`
  and `role`, plus `meta.json` (provenance, counts, distribution) and `pool.json` (the usable
  population per role and split). From the freeze commit on, every role is fixed. A row's
  identity is `(split, shard_file, row)`; an `audio_path` can appear in several rows and is not
  a key. The directory keeps the issue's path; the concept is the freeze.
: _Avoid_: split (the directory name only — `split` itself means clean/unclean), hold-out set
  (that is look ∪ confirm)

**Silence floor**
: The pre-registration's two Exclusions thresholds from
  `tools/tadabur/tlog/silence_floors.json`: `f_s` (0.392731) and `d_s` (1.756378 seconds), the
  type-7 0.5th percentiles of `speech_ratio` and `speech_s` over the Tarteel v1 look clips that
  decode. A pass-1 row is excluded as **silence** when its ratio is below `f_s` **or** its
  speech seconds below `d_s`; a row exactly on both is kept.

**Usable**
: A row that survives the full frozen exclusion union: the two floors (`codec`, `silence`),
  S1's `eval_overlap:tarteel_v1` id flag, and S4's `excluded` reason (`eval_overlap:<set>`
  or `duplicate`). `split/pool.json` stops after the floors and S1's id flag, so it is not a
  training pool; `overlap/meta.json`'s `usable_after_s4` (per role, train included) is the
  population after the whole union, and `tadabur.tlog_overlap.iter_pool_rows` is the reader
  that exposes it as one `exclusions` tuple.
: _Avoid_: valid (that is the app's own label column, never read)

**Exclusion union**
: The ordered, deduplicated tuple a row carries after S4: `codec`, `silence`,
  `eval_overlap:tarteel_v1` (S1's id overlap), S4's primary `excluded`
  (`eval_overlap:<set>` before `duplicate`), then usable. "Deferred" in the frozen
  `split/meta.json` and `split/pool.json` means "decided in `overlap/`"; those files cannot be
  rewritten.

**Overlap record**
: One row's S4 fact in `overlap/<split>/<shard>.jsonl.gz`: its `eval_overlap` list (best match
  per external set, `{set, clip, ber}` in the fixed set order), `duplicate_of`, `duplicate_ber`
  (`0` only for an equal `audio_sha256`) and `excluded` (S4's primary reason). Keyed by
  `(split, shard_file, row)`, never by `audio_path`.

**Survivor**
: A duplicate component's kept copy. Resolution is the pre-registered earliest-opened eval role
  (`look`, then `confirm.1`…`confirm.4`, then `train`), then the smallest
  `(audio_path, split, shard_file, row)`; every other member's `duplicate_of` names it directly.
  Usability does not enter the choice — `meta.json` records the rows lost to an unusable
  survivor instead.

**Stop-word**
: `0` and `0xFFFFFFFF`, the sub-fingerprint words of digital silence. They are neither indexed
  nor queried; each verified match reports the share of its overlap window that is a stop-word
  in either fingerprint, and `meta.json` splits the mostly-stop-word share into all candidates
  and accepted exclusions (external and near-duplicate).

**Ambiguous zone**
: Verified matches above the highest synthetic-positive BER (`0.0625`) and inside the `0.238`
  line. They are listed with their set, clip and BER, never silently passed.

**Text class**
: Ayat with an identical normalised reference. Agreement, Δ and the margin compare classes, never
  ayah numbers, so a repeated ayah (55:13 and its 30 repeats) is never a disagreement
  (`tadabur.tlog_ident`). A class is named by its earliest ayah.

**Decoy**
: An ayah's reference with its grapheme clusters reversed, concatenated in ayah order per surah.
  The best decoy score per decode-length bin is the **floor**'s population; it is a control for
  non-recitation, never a candidate answer.

**Floor** (non-recitation)
: The S5b per-decode rule (`docs/tlog-ident-floor-preregistration.md`, #204): a decode is
  `non_recitation` when its normalised length is under `L_min = 8` characters **or** its best
  real score is within `DELTA = 0.30` of its own best **decoy** score, both from the same
  shortlisted search. Every identification record carries `floor_rule {path, sha256}`. The
  retired per-bin floors — the type-7 99th percentile of the best decoy score per decode-length
  bin (`0-20`, `21-40`, `41-80`, `81-160`, `161-`) — stay in the report as history and decide
  nothing.
: _Avoid_: threshold (the bars are the thresholds), silence floor (that is the pass-1 `silence`
  exclusion, a different rule)

**Bar b**
: The smallest value on the grid 0.00, 0.01, …, 0.50 at which at most 0.5% of look clips would be
  **relabelled** and none of the 52 real-mistake clips would be. Δ, not the margin, is what `b`
  bars.

**Bar b_u**
: The smallest grid value at which at least 99% of the look clips the identifier **names** (floor
  passed, margin ≥ bar, labels hidden) are their own class — the owner's post-look erratum
  (`docs/tlog-ident-bu-amendment.md`). Both bars are **provisional** unless `floor_status` is
  exactly `confirmed`.

**Owner margin approval**
: The owner's clearance of the pinned **margin** definition at the S5b check-in (frozen floor
  document §7), a separate committed record `tlog/ident/margin_approval.json` written by
  `python -m tadabur.tlog_ident approve`. It cites the margin definition by SHA-256 and the `b_u`
  it clears; `load_calibration` and `load_identifier` refuse until one matches the report. The
  oracle's mechanical `final` is **not** this approval.

**Margin**
: `m`, for a row without a filename label: the best score minus the best score of a **different
  text class** at a non-overlapping location. The competitor is pinned by
  `docs/tlog-ident-margin-amendment.md` (word spans, edlib's first optimal alignment per span);
  the pin's known gap is one-directional — `m` can be overstated, never understated.

**Label source**
: `filename` (below the bar, kept under the filename label), `both` (the best class equals the
  filename's), `identified` (relabelled or assigned) or `none`. The floor rule gates relabelling
  and assignment only: a `non_recitation` row is never `identified` — though a matching filename
  class is still `both` — and an unlabelled row below the floor stays `none`.

**Decode mode**
: `whole_clip` or `not_decoded:over_d_max`, required on every decode and every identification row
  (`identification_record(..., decode_mode=)`), so no tree can mix modes silently.

**D_max**
: The whole-clip decode limit, measured on seeded synthetic audio on the GPU box before any tlog
  decode and committed in `tlog/ident/decode_limit.json`: the largest ladder rung keeping 1.5 GiB
  of headroom (90 s on the S6 box). A row longer than D_max is `not_decoded:over_d_max` with
  `label_source = none` — not decoded, not identified, in no audit stratum, assigned nothing.
: _Avoid_: max duration, clip cap (D_max bounds the decode, not the clip's role)

**Audit stratum**
: One pre-registered slice of the S6 listening audit — `both`, `relabelled`,
  `non_recitation`, `silence`, `unclean identified`, `below-bar disagreement` — each drawn by
  salted rank from its own eligible rows (`docs/tlog-s6-amendment.md` §4). A row satisfying two
  predicates is drawn in both, heard once and scored in each
  (`docs/tlog-s6-amendment-note-overlap.md`).
: _Avoid_: bucket, category, bin (that is the retired decode-length bins)

**§9 sample**
: The first 25 rows by the frozen salt's rank among every decoded row failing the floor (2,439),
  whatever other stratum they belong to; the worklist marks them `floor_sample`. Three or more
  heard as real recitation suspend the 5% rule's consequence pending a dated floor revision the
  owner decides.

**Verdict referent**
: The label a listening verdict is scored against, per stratum: `both` the shared label, the
  relabel strata the assigned label, `below-bar disagreement` the filename label (the best class
  is reported too), `non_recitation` and `silence` recitation itself. An `unclear` verdict is
  undecided, never a wrong ayah.

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

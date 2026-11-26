# Follow-along audit S5: the draw's run note (#311 PR 1 and PR 2)

The S5 listening round's draw, snips and arm traces were built before any listening and are
committed; this note records what produced them, what they cost, and the inspection the frozen
design asked for. It changes no number, stratum or rule.

Inputs: the pre-registration (`docs/followalong-audit-preregistration.md`, SHA-256
`33d3ba26dbe6df3a72b0a03637934274a1c2526ac9e0dc306afd1545ea28ed39`), draw amendment 1
(`docs/followalong-audit-draw-amendment.md`, SHA-256
`3c5cc33803a3497c80cba2f08b6de146afd955d5bd1262b70f3207cd40369d91`) and the §6 erratum
(`docs/followalong-audit-edge-erratum.md`). The amendment commit is M =
`03bc6d071a37a9bb264fcd6513381dbedee1c96e`, and the draw's seed is **9665042067186208708**
(`sha256("followalong-audit-draw:" + M)`'s first eight bytes, big-endian). `s5_draw/draw_seed.json`
pins all four.

## The draw

- Frame: the committed App arm's covered words, 49,000 (S1 5,156 in 14 `cause:base` cells, S3
  1,459, S4 38,598, S5 2,038 leading / 1,553 trailing / 196 interior).
- The F-ref gate compared a second full App run with `app_arm/word_grades.jsonl.gz` word by word:
  57,935 words both sides, **0 flips**, no word removed, so every cell's population is its frame
  count and each inclusion probability is the design's tabulated n/N.
- Drawn: 540 words over 498 clips (S1 200, S3 250, S4 60, S5 30), with 25 repeats (7 pilot, 18
  main) — 565 served items. The pilot sitting serves 157, the main 408.
- `worklist.jsonl` is blind: the opaque id, the clip's provenance and the word. It carries no
  stratum, cell, arm or weight; the key does.

## Compute

**0 GPU-hours.** Nothing in this round touched a GPU. The App arm's regeneration ran on the Mac's
CPU through the pinned harness at `028bc51` (the gate certified it and `reproduction.json` pins its
5,742 trace files by digest); the arm traces replay committed decodes and the oracle reference
through that same harness on the CPU. The machine is a laptop, not `cuda-dev`.

## Snips

One snip per served id, from #264's rule at its owner-approved `ll_per_frame` gate (-3.5): 565
rows, 471 aligned excerpts. 94 play the whole clip: **22** are the S5 leading/trailing words this
draw serves whole by design, and **72** are alignment fallbacks (49 below the likelihood gate, 16
unclipped alignments, 7 whose excerpt covers the clip).
`s5_draw/snips.report.json` gives both rates: the whole-clip share **0.1664** (94/565, the sidecar's
own convention) and the alignment fallback rate **0.1274** (72/565), which is the number a fallback
claim should quote.

## The arms

`s5_draw/arms.jsonl.gz`: one row per drawn word (540, 0.38 MB gz) holding the App trace (both audit
scorings, full check list), the App's flag causes, the Oracle, Teacher (`base`) and Student-PT
(`r3_candidate_ema`) kept grades, and F1-align's reference. The App cells are the gate's own traces;
the other three arms replayed the committed decodes and the oracle reference through the pinned
harness. Each arm's kept outcome was checked against its committed word table
(`app_arm/word_grades.jsonl.gz`, `oracle_sweep/word_grades.jsonl.gz`) word for word.

F1-align's reference is **the oracle's canonical text**, not Muraja's own spelling: R and its two
neighbours are the canonical phoneme words mapped to engine words w, w-1 and w+1, concatenated as
the oracle text joins them (preregistration §5, l.284-288: "the word's reference R joined with a
neighbour as the oracle text joins them"). The unit stays the engine word, so a wasl-merged word is
its canonical words with no separator. **0** of the 540 drawn words have no mapped canonical word,
so F1-align abstains nowhere in this draw.

A drawn word the gate's traces hold no row for — the engine reported it and gave it no kept entry —
keeps its checks instead. The clip's own saved checks (their hop, overlap and flush, the statuses
dropped) are replayed through that same pinned harness, and the replay must reproduce the clip's
saved kept grades to the word or its statuses are refused. **29** of the 540 drawn words have no
saved row: **20**, over 19 clips, were re-graded this way, their App cell a pending row
(`quality: "pending"`, no kept score or error set, every check's own status) so that a check which
graded the word keeps it off M-nograde ("the engine never saw it", A2). 19 of the 20 hold at least
one status; `35_5_1337214249.flac#0` word 9 holds none. The other **9** words, one per clip, sit in
clips whose trace file holds no row at all — no checks left to replay — so their cells stay empty
and nothing changes for them. Every kept outcome is unchanged, as the App word table requires. Read
as mistakes, 16 of the 19 changed cells now read M1u (then M3, or M2 where the Teacher flagged), 3
read M3 alone (`2_170_4222931689.flac#0` word 21 among them), and the ninth stays M-nograde/M3.

## The five S1-pending words

The pre-registration counts S1 ∪ S1-pending as the S1 pool (N = 5,156). These five words are App
strict flags the oracle left ungraded, so no oracle verdict explains them. Two are the clip's first
word and three its last, the clip-edge positions the pre-registration discloses. They stay in the
pool as frozen; this is the inspection, not a rule. All five: the App kept a **flagged** strict
grade, and the oracle's committed row is
`graded: false` — no flag, no cause, no disagreement. The oracle's session committed no final
grade for them: in four of the five clips it committed **none** of the clip's words (55_28 0/4,
56_34 0/2, 79_28 0/3, 84_5 0/3), and in 84_19 it committed words 1–3 but left word 4 provisional at the
closing flush. Replaying 84_19 on its own shows the shape: every check graded all four words
correct, and `final` still held only 1–3. So the ungraded row is where the sweep's session ended,
not a claim about the word.

| item | word | App strict causes | note |
|---|---:|---|---|
| `55_28_6162424172.flac#0` | 4 | tashkeel | The clip's **last** word (4/4). `tashkeelError` on one haraka mark (expected `ُ`); the engine graded 4 of 5 checks. A strict-display deviation on a final vowel: reads as a follow-along effect, not a consonant the decoder misheard. |
| `56_34_2147808602.flac#0` | 1 | tashkeel | The clip's **first** word (1/2). `tashkeelError` on one haraka mark (expected `ُ`); both checks graded. Same reading: the strict display flags a final vowel, at the clip's opening word. |
| `79_28_420573590.flac#0` | 1 | unrecorded | The clip's **first** word (1/3). `minor` with **no kept errors**: the grade came from a check's score, not from a marked letter, and the cause is `unrecorded` — the phonetizer did not record the word. Nothing localisable to hear; it reads as an engine/score effect. |
| `84_19_9013039918.flac#0` | 4 | soft_pair | The clip's **last** word (4/4, the provisional one). `minor` with a consonant mismatch (`ط` expected, `ت` heard) plus a tashkeel mark; the engine graded 4 of 6 checks. Reads as a real decode deviation, at the clip's end. |
| `84_5_2752243060.flac#0` | 3 | other_consonant, shaddah | The clip's **last** word (3/3). `minor` with three marks: a consonant mismatch (`و` expected, `ب` heard), a tashkeel and a **deletion** (`ق`); 4 of 5 checks graded. Reads as a decode error, not a display effect. |

## The scorer

`s5-score` reads the committed draw and the verdicts file and writes `summary.json` and
`attribution.jsonl`; it never writes into `s5_draw/`. There are **no verdicts yet** — the listening
page is PR3 — so the only run it can make today is over no verdicts: it attributes nothing, reports
every rate as `None` and rule 4 as "direction only", which is the intended behaviour rather than a
placeholder.

Beside the four rules the scorer reports prereg §5's **R-alone row** (`rules.r_alone`): how many
decisive words' setting checks hold R alone, how many of those the primary attribution put on the
model side, and rule 3's share with them read as F1-align. It is labelled a sensitivity and decides
nothing; the primary buckets, sides and rules are the neighbour-required ones.

The row matches R alone at **any reference length**. The four-code-point floor is the primary
rule's, and only for a clip with no neighbour at all; the prereg states none for this row. So the
row counts words the primary's neighbour requirement declines even where R is three characters:
`567a698f9ee97bc0` (`2_189_4280756139.flac#0` word 4, R = `قُل`, in setting checks 36–47 with
neither neighbour) is one — the primary stays F3 on the model side and the sensitivity moves it to
the follow-along side.

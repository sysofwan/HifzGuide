# The owner's listening at `h448`'s excess flags on Tarteel v1 (#160)

On Tarteel v1 app-user audio (#156, [`docs/nonpro-pilot.md`](../../../docs/nonpro-pilot.md))
`h448` flags 57.0 words per 1,000 against the base teacher's 21.3, and nobody has verified the
recitations. The augmentation experiment's main bar (#128) is a drop in `h448`'s Tarteel v1
flags, so this sitting asks whether `h448`'s **excess** flags are model errors or the users'
real mistakes. Code: [`../tarteel_excess.py`](../tarteel_excess.py) (frame, draw, audio, page)
and [`../tarteel_excess_report.py`](../tarteel_excess_report.py) (the analysis).

## The look / confirm split

Pre-registered in #128 (2026-10-15) and defined here, in
[`../tarteel_split.py`](../tarteel_split.py): a clip is in the **look** half when the first
8 bytes (big-endian) of `sha256("tarteel-v1-split-2026-10:" + clip_id)` are below 2**63, else
in the **confirm** half. `clip_id` is the clip's own id in the dataset, its `audio.path`
(`audio_filename` in #156's decodes, e.g. `100_10_4115049010.wav`), not the decode key.
[`../nonpro_eval/tarteel_v1/split.json`](../nonpro_eval/tarteel_v1/split.json) freezes it
over #156's 8,950 clips: **4,537 look, 4,413 confirm**. The experiment's readout imports the
same `half()`. This sitting samples the look half only; the confirm half is never read here.

## What is asked

**Frame.** Every graded word of a look-half item (the whole ayah each clip recites, as #156
read it) that `h448`'s recorded checks flag under Muraja's shipped default **and** the base
teacher's checks grade correct, both replayed through Muraja's engine (protocol 3, Muraja
`99c326f`) from #156's committed decodes. The look half holds 4,530 items;
`h448` grades 43,683 words there and flags 2,479, of which
**1,724 are excess flags**, in 1,281 clips. (Graded over the whole corpus,
the same harness reproduces #156's counts exactly: base 1,825 of 85,655, `h448` 4,856 of
85,145.)

**Strata.** A word's top substitution is the consonant error of its `h448` flag ranked highest
in #156's tally (`h448`, shipped default: `ء→∅` 210, `ا→∅` 149, `ج→ش` 123, …). Its stratum
is that substitution when it is one the issue names (`ء→∅`, `ا→∅`, `ج→ش`, `ل→ن`, `ن→∅`, `د→ت`,
`ن→ل`, `ز→س`, all `other_consonant` in the tally), else `other`, as is a flag with no
consonant error (tashkeel, skipped, unrecorded).

**Draw.** Seeded by salted SHA-256 ranks (`tarteel-v1-excess-listening-2026-10`), in two
stages: each clip keeps one excess word (its representative, chance `1 / m` for a clip with
`m`), then each stratum draws from its representatives (chance `n / N`). The inclusion
probability is the product, recorded per row. `other` holds about four in five of the frame,
so it draws 44 and each named stratum 7.

| stratum | frame words | representatives (`N`) | drawn (`n`) | `n / N` |
|---|---|---|---|---|
| `other` | 1374 | 1021 | 44 | 0.0431 |
| `ء→∅` | 63 | 42 | 7 | 0.1667 |
| `ا→∅` | 30 | 26 | 7 | 0.2692 |
| `ج→ش` | 58 | 43 | 7 | 0.1628 |
| `د→ت` | 35 | 24 | 7 | 0.2917 |
| `ز→س` | 34 | 26 | 7 | 0.2692 |
| `ل→ن` | 46 | 42 | 7 | 0.1667 |
| `ن→ل` | 37 | 23 | 7 | 0.3043 |
| `ن→∅` | 47 | 34 | 7 | 0.2059 |

**100 words in 100 clips; about 47.1 minutes** (the whole clip twice plus 6 s
to answer; the drawn clips average 11.1 s). Inclusion probabilities run from
0.0054 (an `other` word in a clip with eight excess words) to 0.30. `worklist.summary.json`
freezes `worklist.jsonl` by its SHA-256; every reader refuses another file.

**What is played: the whole clip, the word highlighted in the ayah's text.** No corpus here
times its words (#156), and #156's decodes keep strings, not the CTC frames a `TimedClip`-style
span needs. A span would mean re-decoding with the teacher and trusting its alignment exactly
where `h448` heard something else: most strata are a letter dropped at a word's edge (`ء→∅`,
`ا→∅`, `ن→∅`), where a cut goes wrong first. A Tarteel v1 clip is one ayah, so the whole clip
fits the sitting, and the owner hears the word in context (ل/ن, ج/ش, د/ت and ز/س are easy by
ear in context; س/ص/ث, the hard ones, are not a named stratum). **Whole** (`w`) plays the same
clip.

**Blind (ADR-0007).** The page is `tadabur.tashkeel_audit_ui`'s word mode, unchanged. It
receives the ayah's text, the highlighted word and the answer, as for any sampled word: never
a model's output, the flag, the stratum or the design, and the queue is one seeded order of
the ids. Every word on the page is one `h448` flagged; nothing says what it flagged.

## Files

| file | what it is | written by |
|---|---|---|
| `worklist.jsonl` | one drawn word per line: the clip's provenance (#156's `audio_sha256` of the published bytes, the whole-clip span), the ayah's realized reference and word offsets, the canonical words the Muraja word covers, `h448`'s flag (causes, substitutions, errors), the stratum and the design (`clip_words`, `stratum_words`, `stratum_population`, `representative_probability`, `draw_probability`, `inclusion_probability`) | `python -m tadabur.tarteel_excess draw` (a Mac with the Muraja harness) |
| `worklist.summary.json` | the look half's counts, the frame and the draw per stratum, the listening minutes, the Muraja build, the inputs' SHA-256 and the SHA-256 that freezes `worklist.jsonl` | the same |
| `staged.jsonl` | per clip: #156's dataset shard and row group, the published bytes' SHA-256, and the staged 16 kHz PCM_16 WAV's SHA-256 and length | `python -m tadabur.tarteel_excess stage` |
| `verdicts.jsonl` | the answers, `{site_id, heard, note}` per line (`heard`: `correct`, `mistake` or `unclear`) | the page, as they are given |
| `review.jsonl` | words queued for another listen | the page |
| `report.json` | the analysis, after the sitting | `python -m tadabur.tarteel_excess_report` |

No audio is committed.

## The sitting

The page runs on the Mac, from the checkout whose `verdicts.jsonl` should receive the answers.
The clips are re-read once from the Hub straight to the Mac (no GPU box): only the row groups
holding them, by HTTP range requests, with `HF_HOME` in a scratch directory so nothing lands
in `~/.cache`. Each clip's bytes are checked against #156's SHA-256 and its decoded length
against #156's decode, and the staged files against `staged.jsonl`; the server checks them
again every time it starts.

```bash
cd ~/repos/HifzGuide/tools
# Once: re-read the drawn clips from greentechapps/user_recited_verses_full at #156's revision
# (eae2dbb4) and verify each (35 row groups, ~3 GB streamed, 34 MB kept; ~15 min).
HF_HOME=$(mktemp -d) /Users/sysofwan/repos/HifzGuide/.venv-test/bin/python \
  -m tadabur.tarteel_excess stage --audio-dir ~/hifzguide-listening/issue-160
# Launch, reachable from a phone on the LAN.
/Users/sysofwan/repos/HifzGuide/.venv-test/bin/python -m tadabur.tarteel_excess serve \
  --audio-dir ~/hifzguide-listening/issue-160 --host 0.0.0.0 --port 8000
```

It prints `0 of 100 answered`. Open `http://<the Mac's LAN IP>:8000/`
(`ipconfig getifaddr en0`); `curl -s -o /dev/null -w "%{http_code}\n"
http://$(ipconfig getifaddr en0):8000/` must print `200` before the phone is handed over.
The keys, the review list and the firewall note are as in
[`../listening_session/README.md`](../listening_session/README.md). Ctrl-C stops the server;
the answers are already on disk. Afterwards:

```bash
git add tools/tadabur/tarteel_excess/verdicts.jsonl tools/tadabur/tarteel_excess/review.jsonl
python -m tadabur.tarteel_excess_report   # writes report.json, prints the table
```

Every word asks **"Was the highlighted word recited correctly?"**: Yes (`correct`: the flag
is a model error), No (`mistake`: a real mistake the teacher missed) or Unclear (`unclear`),
with an optional note. A note helps most on a No: what was wrong.

## The analysis

`tadabur.tarteel_excess_report` gives, per stratum and pooled, the weighted share of the
excess flags heard as correct (model error), as a mistake and as unclear, plus the model-error
share of the clear answers. Each word weighs `1 / inclusion_probability`, so the pooled share
estimates the share among **every** excess flag of the look half, not among the words drawn.
Intervals follow acceptance rules §1 (the bootstrap clustered by surah, #156's clusters; a
Wilson bound for a sparse, equally weighted cell; none otherwise). A stratum holds about ten
words: read the strata as where `h448` errs, the pooled share as how often.

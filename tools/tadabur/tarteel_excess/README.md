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

**Frame.** The graded words of the look-half items (the whole ayah each clip recites, as #156
read it), both models' recorded checks replayed through Muraja's engine (protocol 3, Muraja
`99c326f`) under the shipped default, from #156's committed decodes. The look half holds
4,530 items; `h448` grades 43,683 words there and flags 2,479.
(Graded over the whole corpus, the same harness reproduces #156's counts exactly: base 1,825
of 85,655, `h448` 4,856 of 85,145.)

- **Excess words:** `h448` flags the word, the base teacher grades it correct. **1,724 words
  in 1,281 clips.**
- **Control words:** both models grade it correct. 40,963 words in
  4,459 clips. They give the owner's baseline: how often a word nobody
  flagged is heard as a mistake.

**Strata.** An excess word's top substitution is the consonant error of its `h448` flag ranked
highest in #156's tally (`h448`, shipped default: `ء→∅` 210, `ا→∅` 149, `ج→ش` 123, …). Its
stratum is that substitution when it is one the issue names (`ء→∅`, `ا→∅`, `ج→ش`, `ل→ن`, `ن→∅`,
`د→ت`, `ن→ل`, `ز→س`, all `other_consonant` in the tally), else `other`, as is a flag with no
consonant error (tashkeel, skipped, unrecorded).

**Draw.** Seeded by salted SHA-256 ranks (`tarteel-v1-excess-listening-2026-10`), at most one
word per clip across the whole worklist, every probability recorded per row:

1. **Excess** (two stages): each clip keeps one excess word (its representative, chance `1 / m`
   for a clip with `m`), then each stratum draws from its representatives (chance `n / N`).
   `other` holds about four in five of the frame, so it draws 32 and each named stratum 6.
2. **Controls** (two stages): of the look-half clips with a control word, less the 80 clips the
   excess draw took, 20 clips by rank (chance `20 / C`), then one control word in each (chance
   `1 / k` for a clip with `k`). The probability is conditional on the excess draw, which the
   ids fix.

| stratum | frame words | population (`N`: representatives; controls: eligible clips) | drawn (`n`) | `n / N` |
|---|---|---|---|---|
| `control` | 40,963 | 4,379 | 20 | 0.0046 |
| `other` | 1,374 | 1,021 | 32 | 0.0313 |
| `ء→∅` | 63 | 42 | 6 | 0.1429 |
| `ا→∅` | 30 | 26 | 6 | 0.2308 |
| `ج→ش` | 58 | 43 | 6 | 0.1395 |
| `د→ت` | 35 | 24 | 6 | 0.2500 |
| `ز→س` | 34 | 26 | 6 | 0.2308 |
| `ل→ن` | 46 | 42 | 6 | 0.1429 |
| `ن→ل` | 37 | 23 | 6 | 0.2609 |
| `ن→∅` | 47 | 34 | 6 | 0.1765 |

**100 words in 100 clips (80 excess, 20 controls); about 48.1 minutes** (the
whole clip twice plus 6 s to answer; the drawn clips average 11.4 s). Excess inclusion
probabilities run from 0.0039 to 0.26, the controls' from 0.00022 to 0.00152.
`worklist.summary.json` freezes `worklist.jsonl` by its SHA-256; every reader refuses another
file. (The 80 excess words are a subset of the first, 100-word draw, which nobody listened
to: the ranks are the same, and only the sizes fell.)

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
a model's output, the flag, the stratum or the design. Excess and control words share one
seeded order of their ids and look the same, so nothing on the page says whether a word was
flagged, let alone what for.

## Files

| file | what it is | written by |
|---|---|---|
| `worklist.jsonl` | one drawn word per line: the clip's provenance (#156's `audio_sha256` of the published bytes, the whole-clip span), the ayah's realized reference and word offsets, the canonical words the Muraja word covers, `h448`'s flag (causes, substitutions, errors; empty for a control), the stratum (`control` for a control) and the design (`clip_words`, `stratum_words`, `stratum_population`, `representative_probability`, `draw_probability`, `inclusion_probability`) | `python -m tadabur.tarteel_excess draw` (a Mac with the Muraja harness) |
| `worklist.summary.json` | the look half's counts, the frame and the draw per stratum, the listening minutes, the Muraja build, the inputs' SHA-256 and the SHA-256 that freezes `worklist.jsonl` | the same |
| `staged.jsonl` | per clip: #156's dataset shard and row group, the published bytes' SHA-256, and the staged 16 kHz PCM_16 WAV's SHA-256 and length | `python -m tadabur.tarteel_excess stage` |
| `verdicts.jsonl` | the answers, `{site_id, heard, note}` per line (`heard`: `correct`, `mistake` or `unclear`) | the page, as they are given |
| `review.jsonl` | words queued for another listen | the page |
| `report.json` | the analysis, after the sitting | `python -m tadabur.tarteel_excess_report` |

No audio is committed.

## The sitting

The clips are re-read once from the Hub to the Mac: only the row groups holding them, by HTTP
range requests, with `HF_HOME` in a scratch directory so nothing lands in `~/.cache`. Each
clip's bytes are checked against #156's SHA-256 and its decoded length against #156's decode,
and the staged files against `staged.jsonl`; a clip already staged and matching is kept. The
server checks them again every time it starts. It prints `0 of 100 answered`.

```bash
cd ~/repos/HifzGuide/tools
# Once, on the Mac: re-read the drawn clips from greentechapps/user_recited_verses_full at
# #156's revision (eae2dbb4) and verify each (about 35 row groups, ~3 GB streamed, ~35 MB kept).
HF_HOME=$(mktemp -d) /Users/sysofwan/repos/HifzGuide/.venv-test/bin/python \
  -m tadabur.tarteel_excess stage --audio-dir ~/hifzguide-listening/issue-160
```

### On cuda-dev (the owner's preference)

The owner's sessions are hosted on cuda-dev (`root@cuda-dev`, http://10.0.1.195:8000/). Run
the UI from a scratch copy under `/root/scratch/issue-160/`, never from `/root/repos/HifzGuide`.
cuda-dev is scratch: pull `verdicts.jsonl` back and commit it after the sitting.

```bash
# On the Mac, from tools/: copy the 100 verified clips to /root/scratch/issue-160/clips/ (one
# rsync) and re-read every SHA-256 there.
/Users/sysofwan/repos/HifzGuide/.venv-test/bin/python -m tadabur.tarteel_excess push \
  --audio-dir ~/hifzguide-listening/issue-160
# The code: a scratch copy of the merged main (before the merge: agent/issue-160).
git fetch origin && git archive origin/main | ssh root@cuda-dev \
  'rm -rf /root/scratch/issue-160/HifzGuide && mkdir -p /root/scratch/issue-160/HifzGuide \
   && tar -x -C /root/scratch/issue-160/HifzGuide'
# Launch, detached (port 8000 must be free: ssh root@cuda-dev 'ss -ltnp | grep :8000').
ssh root@cuda-dev 'cd /root/scratch/issue-160/HifzGuide/tools \
  && source /root/miniconda3/etc/profile.d/conda.sh && conda activate hifzguide \
  && setsid nohup python -m tadabur.tarteel_excess serve \
       --audio-dir /root/scratch/issue-160/clips --host 0.0.0.0 --port 8000 \
       > /root/scratch/issue-160/ui.log 2>&1 < /dev/null &'
curl -s -o /dev/null -w "%{http_code}\n" http://10.0.1.195:8000/   # must print 200
```

The answers are written on the box, into
`/root/scratch/issue-160/HifzGuide/tools/tadabur/tarteel_excess/verdicts.jsonl`. Pull them
back after the sitting (and during it, to be safe), then commit:

```bash
cd ~/repos/HifzGuide
scp root@cuda-dev:/root/scratch/issue-160/HifzGuide/tools/tadabur/tarteel_excess/verdicts.jsonl \
    tools/tadabur/tarteel_excess/
# review.jsonl too, if the owner queued any word for another listen:
scp root@cuda-dev:/root/scratch/issue-160/HifzGuide/tools/tadabur/tarteel_excess/review.jsonl \
    tools/tadabur/tarteel_excess/ 2>/dev/null || true
git add tools/tadabur/tarteel_excess/verdicts.jsonl && git commit -m "..."
# Stop the server: find its PID (the [t] keeps pgrep from matching its own ssh command line),
# then kill that PID: ssh root@cuda-dev 'pgrep -af "[t]adabur.tarteel_excess serve"', then
# ssh root@cuda-dev 'kill <pid>'
```

### On the Mac (the alternative)

```bash
cd ~/repos/HifzGuide/tools
/Users/sysofwan/repos/HifzGuide/.venv-test/bin/python -m tadabur.tarteel_excess serve \
  --audio-dir ~/hifzguide-listening/issue-160 --host 0.0.0.0 --port 8000
```

Open `http://<the Mac's LAN IP>:8000/` (`ipconfig getifaddr en0`); `curl -s -o /dev/null -w
"%{http_code}\n" http://$(ipconfig getifaddr en0):8000/` must print `200` before the phone is
handed over. The answers go straight into this checkout's `verdicts.jsonl`; Ctrl-C stops the
server, and `git add tools/tadabur/tarteel_excess/verdicts.jsonl` follows.

The keys, the review list and the firewall note are as in
[`../listening_session/README.md`](../listening_session/README.md). After the sitting, either
way: `python -m tadabur.tarteel_excess_report` (writes `report.json`, prints the tables).

Every word asks **"Was the highlighted word recited correctly?"**: Yes (`correct`; on an
excess word, the flag is a model error), No (`mistake`: a real mistake) or Unclear
(`unclear`), with an optional note. A note helps most on a No: what was wrong.

## The analysis

`tadabur.tarteel_excess_report` gives, per excess stratum and pooled over the excess words, the
weighted share heard as correct (model error), as a mistake and as unclear, plus the
model-error share of the clear answers; the same shares on the controls (the owner's baseline
on unflagged words); and every excess share **net of the controls'** (the difference, with its
interval). Each word weighs `1 / inclusion_probability`, so the pooled share estimates the share
among **every** excess flag of the look half, not among the words drawn. Intervals follow
acceptance rules §1 (the bootstrap clustered by surah, #156's clusters; a Wilson bound for a
sparse, equally weighted share; none otherwise); a difference resamples both groups' surahs in
one draw and, its groups being disjoint, has no paired score fallback. A stratum holds about
six words: read the strata as where `h448` errs, the pooled share as how often.

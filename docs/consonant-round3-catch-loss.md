# Round 3: the catch-loss audit (#257)

The pre-registered blind audit of what round 3's candidate stops flagging that round 2's model
caught. It is a **pre-claim gate**: bar 2 measures fewer flags, and this measures whether the
further drop is in false flags or in real mistakes. A failure ends the round before confirm.1 or
confirm.2 is claimed.

The rules are the pre-registration's (`docs/consonant-round3-preregistration.md`, "Catch-loss
audit", FINAL TEXT 2026-11-01, committed at `b2470ff`), and the round-2 audit's precedent
is `docs/consonant-round2-catch-loss.md`. This note records the tooling, the committed draw,
the owner's instructions and the sitting's result.

## The populations

Look gating words graded, at the shipped default, by C95, the round-2 EMA (`r2`) and the chosen
round-3 weights (EMA; the live/EMA choice is commit `fd25c93f`).

* **New removals** (the gating stratum): C95 flags the word, `r2` flags it, and round 3 does
  not. Round 2's own removals — C95 flags, round 2 does not — are already audited (5 of 98
  decisive heard as real mistakes) and are in neither stratum.
* **New additions** (reported only): round 3 flags the word and neither C95 nor `r2` does.

A word any one of the three graded differently is in neither population. The draw reads the look
sitting's three-model export (`consonant3_readout/look_grades.jsonl.gz` plus
`look_grades_r2.jsonl.gz`) and refuses when the reference grades are absent, when they do not
repeat the export's baseline grades word for word, or when they do not reproduce the
pre-registration's disclosed counts.

## The tooling

`tools/tadabur/catch_loss_audit.py`, unchanged from round 2 except that `draw` and `score` take
`--spec` (the round's seed, reference model, root and bars):

* **`draw`** ranks each stratum by `sha256("round3-catch-loss-2026-11:<stratum>:<key>:<position>")`
  and takes the first 100 removed and 50 added (a smaller stratum whole): a seeded simple random
  sample without replacement that does not depend on row order. It writes, into
  `tools/tadabur/consonant3_readout/catch_loss/`:
  * `worklist.jsonl` — the page's whole view: an opaque id, the clip's provenance (split, shard,
    row, file name, `audio_sha256`, recorded span), the ayah, the 0-based Uthmani word index and
    the word's text, sorted by id;
  * `key.jsonl` — never served: stratum, sampling probability, and all three models' grades
    (outcome, causes, the consonant substitutions behind the flag);
  * `summary.json` — the seed, the choice's commit, each stratum's population and the sample
    size taken.
* **`score`** joins the UI's `verdicts.jsonl` (`correct` / `mistake` / `unclear`) to `key.jsonl`
  and writes `result.json`: among new removals with a decisive verdict, the real-mistake share is
  at most 20% (exactly 20% passes); fewer than 80 decisive removed verdicts cannot certify
  whatever the share. Additions, both strata by cause, unclear counts and the owner's harder
  pairs (س/ص/ث) are reported. It refuses an incomplete sitting and verdicts that are not the
  worklist's.

The UI is the #61 listening page unchanged: one seeded queue with the strata interleaved, the
whole clip from the box's look clip cache (`/root/scratch/tlog/r2/clips`, #212), each clip
verified against the worklist's `audio_sha256` before the server starts. The page receives the
ayah with the target word highlighted and nothing else: no stratum, model output, grade, weight
or tally. On cuda-dev, `scripts/issue-225-audit.sh` runs `draw` and `ui` from a bundle clone and
`backup` from the scratch root (its loop only copies files, so it never re-clones);
`SPEC=consonant3` selects round 3's readout, scratch root (`/root/scratch/catch_loss_r3`) and
port (`8001`), so round 2's listener on `8000` is left untouched. Verdicts and the review
list live beside the worklist in the scratch root, never inside the clone the launcher
rebuilds, so a relaunch keeps every answer and every "review later" marker. No audio is
committed and none leaves cuda-dev beyond playback.

## The committed draw, 2026-11-02

Drawn once from the round-3 look sitting's grades and committed at `df40628` (`#281`) before any
listening.

* **Seed:** `round3-catch-loss-2026-11`; choice commit `fd25c93f`.
* **Removed population:** 222 words, 100 sampled. The pre-registration disclosed 1,624 words
  flagged by both C95 and `r2`; the narrower three-model stratum (round 3 also not flagging) is
  the 222.
* **Added population:** 97 words, 50 sampled.
* **Worklist:** 150 items over 141 distinct clips; every clip's source bytes are already in the
  look clip cache, so nothing was prepared for playback.

## The hand-off, 2026-11-02

### The served page

`SPEC=consonant3 scripts/issue-225-audit.sh ui launch` on cuda-dev, with the backup loop
(`backup launch`) beside it. Round 2's listener is still bound on `:8000` and was left
untouched, but it answers nothing (its clone and verdicts are gone): round 2's page is closed,
not serving.

* **Page:** `http://10.0.1.195:8001/` — "0 of 150 answered".
* **Scratch root:** `/root/scratch/catch_loss_r3` (clone, `verdicts.jsonl`, `review.jsonl`,
  `backup/`); logs in its `logs/` (`ui.log`, `backup.log`), each job's exit status in
  `logs/<job>.exit`, absent while it runs.
* **Checked before hand-off:** the whole clip cache holds all 141 worklist hashes; over the LAN
  the page renders the ayah with the target word marked, `/api/audio?key=…&whole=0` returns mono
  16 kHz WAV and plays, and no verdict exists yet. `ui launch` itself now ends its footer with
  `audio: 150 items; <key> audio <n> bytes` — the same probe, run at every launch, so a queue
  that answers but cannot serve a clip fails the launch instead of surfacing mid-sitting.
* **Stopping:** the `ui launch` output ends with `stop: kill -- -<pid>`, the job's process group
  (the runner and its server). Use it only with the owner's OK.

Do not launch `ui` or `draw` again while the page is up: a launch re-clones the branch under the
scratch root, the page's html and js are read from disk, and the running process holds the port,
so the sitting goes down rather than restarts. A `ui launch` refuses an occupied port before it
stages anything, printing the listener it found, so the accidental relaunch fails loudly instead
of reporting the running sitting as a new one; a launch that comes up verifies the port's
listener is its own server before printing the footer, and that footer's `check with:` line
repeats every setting given explicitly (`SPEC`, `ISSUE_ROOT`, `PORT`, `CLIPS`, `HOST`). The open
page also starts a fresh session while the old answers stay in `verdicts.jsonl`;
`verdicts.jsonl` and `review.jsonl` survive either way.

### What the owner does

* Listen blind: the page shows the ayah with the target word highlighted, no scores and no model
  output.
* Answer one question per item — **was the highlighted word recited correctly?** Yes / No /
  Unclear. The strata are shuffled together, so item order says nothing.
* Notes are welcome; "review later" queues an item without answering it, and the queue is not
  finished until every item has an answer.
* 150 items, roughly an hour with breaks. Audio plays from the box; nothing is downloaded.

### After the sitting

```
SPEC=consonant3 scripts/issue-225-audit.sh ui fetch DIR
git add tools/tadabur/consonant3_readout/catch_loss/{verdicts,review}.jsonl   # then commit
cd tools && python -m tadabur.catch_loss_audit score --spec consonant3        # writes result.json
```

Commit `verdicts.jsonl`, `review.jsonl` (if present) and `result.json` in
`tools/tadabur/consonant3_readout/catch_loss/`. The verdicts file is irreplaceable human data
(AGENTS.md); the backup job keeps a per-minute copy in `/root/scratch/catch_loss_r3/backup`.

## The owner's sitting, 2026-11-17

The owner listened to all 150 items; the committed `verdicts.jsonl` is the sitting's file, SHA-256
`0a2d3c41d4a48ca7a3003fe0b400f75dff71f74623f5d2d17fca2fde9655976a`: 115 `correct`, 22 `mistake`,
13 `unclear`, 38 with a note. No `review.jsonl` was written (every item was answered).

**Result (the frozen rule, `result.json`).** Removed: **16 of 92 decisive** verdicts are real
mistakes, **17.39%** (Wilson 95% 11.00–26.40%), at most the pre-registered 20% with 92 decisive;
fewer than 80 decisive certifies nothing, whatever the share -> **pass**. Added, reported only:
6 of 45 decisive, 13.33% (6.26–26.18%).

**The owner's note rule (REPORTED, not in `result.json`).** The owner, verbatim, on what a note
means:

> Please count prioritize the correct/incorrect verdict that I gave. The note just says which part
> is incorrect. For the correct with note, please read the note for verdict as it may mean unclear

So the headline is the verdict as given: every `mistake` stays a mistake whatever its note, every
`unclear` stays unclear, and only a `correct` carrying a note is re-read, individually. The
removed stratum's eleven:

| site_id | note | reading |
|---|---|---|
| `1588286f` | unclear | `unclear` |
| `191c21a0` | Corrected with reread | correct |
| `1be3a83e` | fast | likely `unclear` |
| `1c38a407` | Previous word trimmed | correct |
| `2379e37f` | incorrect on the first pass but corrected | correct |
| `28232893` | Too fast | likely `unclear` |
| `382816d3` | first fathahah incorrect | correct |
| `52c94240` | Ghunnah missing | correct |
| `60416add` | very low volume | likely `unclear` |
| `a154b237` | possible ain -> alif substitution | `unclear` |
| `f1baad7b` | Incorrect Al at the end of word | correct |

The two `unclear` readings are certain, the three "likely" ones are a judgment (the word was
heard; the recording was trimmed, fast or quiet), and the last six stay correct: each note names a
defect the owner judged not to change the verdict (a reread corrected it, the previous word was
trimmed, or the slip is tajweed/tashkeel rather than the consonant). All readings are recomputed
from `verdicts.jsonl` and `key.jsonl` with the scorer's own count and Wilson machinery, as
`result.json` is.

| reading | removed real/decisive | removed real share (Wilson 95%) | removed-stratum verdict | added real/decisive (share) |
|---|---|---|---|---|
| k=2: the certain `unclear` readings | 16/90 | 17.78% (11.25–26.94%) | pass | 6/45 (13.33%) |
| k=5: plus the three likely ones | 16/87 | 18.39% (11.65–27.81%) | pass | 6/45 (13.33%) |
| k=11: bound, every noted `correct` `unclear` | 16/81 | 19.75% (12.54–29.70%) | pass | 6/44 (13.64%) |

All three readings keep at least 80 decisive, and the share rises towards 20% without crossing
it, so the pass is stable under the owner's rule. The added stratum has no bar; it carries one
`correct`-with-note ("start of ayah trimmed, baini clearly heard", read as correct) and 5 noted
`mistake`s, which stay mistakes.

**Superseded round-2-style readings (REPORTED).** The round-2 note's two blanket readings, kept
only because the owner's rule above supersedes them: (a) all 38 noted answers treated as
`unclear`: removed 3/68 (4.41%, Wilson 1.51–12.19%) -> cannot certify (68 decisive), added 1/39
(2.56%); (b) noted `correct` treated as a real mistake: removed 27/92 (29.35%, 21.02–39.33%) ->
fail, added 7/45 (15.56%). Neither is the owner's rule or the bar.

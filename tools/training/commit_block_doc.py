"""Render the commit-block replay (``docs/commit-block.md``) from its report (#148).

Presentation only: every number comes from :mod:`training.commit_block`'s report, and the doc
is regenerated with it, never edited by hand. The one hand-written part is
:data:`CODE_READING`: what Muraja's transcriber and engine do with the commit block, with a
``file:line`` citation for every claim, at the commit the harness is built from
(``training.muraja_policy.MURAJA_COMMIT``). Re-read it whenever that commit moves.
"""

from __future__ import annotations

from tadabur.exclusions import REASONS
from training.muraja_policy import APPROXIMATIONS, RECORDED_CHECKS
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE
from training.truth_baseline_doc import (
    SPARSE_MARK,
    allowance_rows,
    count_rate,
    format_delta,
    format_estimate,
    word_cause_table,
)

#: Muraja paths are under ``ios/HifzGuide/`` at ``99c326f``; a bare line number continues the
#: file last named in the same cell or bullet.
CODE_READING = """\
## The answer from the code

**Yes, the commit block reaches grading.** The block a window commits is the **hop** the engine
keeps. Every check's alignment query is built from the kept hops and the newest inference, and
every word the alignment reaches is regraded into the upward-only ratchet. Once a second of
audio is older than the newest window or preview, the app grades it from the decode of the
block that committed it. Committing block 1 changes which decode stands for that audio. It
does not change when anything runs.

Muraja has no transcriber that commits block 1. The "b = 1" column below is the hypothetical
transcriber #148 asks about. It runs the same inferences as today, sends each second of audio
once, and leaves the engine unchanged (`training.decoding`, "Another commit block"). Paths are
under `ios/HifzGuide/` of `sysofwan/Muraja` at `99c326f` (v1.0.27). A bare line number
continues the file last named in the same cell or bullet.

### What reaches the engine

| | Muraja today (b = 0) | b = 1, as replayed |
|---|---|---|
| **Inferences** | Nothing runs below 2 s pending (`Models/RealtimeTranscriber+iOS.swift:568-571`). With 5 s pending, the window runs, then the buffer advances 1 s (:575-581, :616, :459-468). Otherwise a preview of all pending audio runs per 200 ms of new audio (:238-241, :624-662). On stop, one inference covers all pending audio (:317-336). | The same inferences, on the same audio. |
| **A window's hop** (`confirmed`) | The segments whose midpoint is before timestep 25 (`Models/RealtimeTranscriber+iOS.swift:256, :580-583`; `Models/MuaalemInference+iOS.swift:128-149`). This is block 0, the window's first second, decoded with 0 s of left context and 4 s of right. | Block 1 (timesteps 25-50), with 1 s of left context and 3 s of right. The session's first window sends blocks 0-1. |
| **A window's overlap** (`unconfirmed`) | The rest: timesteps 25-125 (`Models/RealtimeTranscriber+iOS.swift:584, :612`). | Timesteps 50-125. Block 0 is not sent: an earlier window's block 1 already committed it. |
| **A preview** (`unconfirmed`, with an empty hop) | All pending audio, from the buffer's start (`Models/RealtimeTranscriber+iOS.swift:634-658, :441-449`). | The same, less its first second once the buffer has advanced, as that second is already committed. |
| **The stop** | All pending audio (`Models/RealtimeTranscriber+iOS.swift:317-336`), handed to the engine's `handleFinalFlush` (`FollowAlong/LifecycleCoordinator+iOS.swift:340-347`). It sends nothing if the CTC hallucination gate fires (`Models/RealtimeTranscriber+iOS.swift:326-332`). | The same, less its first second once the buffer has advanced. The gate still reads all its timesteps. |
| **A silence flush** (5 s with no meaningful output) | All pending audio, sent as a hop with `didFlush` (`Models/RealtimeTranscriber+iOS.swift:545-556, :676-700`). | It would drop its first second the same way. Not replayed: no speech detector is modelled. |

### What the engine does with them

| | Muraja today (b = 0) | b = 1, as replayed |
|---|---|---|
| **Route** | Each update goes to `FollowAlongEngine.handleUpdate` (`FollowAlong/FollowAlongManager+iOS.swift:719-746`). | Unchanged. |
| **Pending hops** | A hop is appended to the pending queue and to the heard stream (`FollowAlong/FollowAlongEngine.swift:625-628`; `FollowAlong/HeardStream.swift:41-53`). The overlap or preview replaces the previous one (`FollowAlong/FollowAlongEngine.swift:630`). | Unchanged. The hops are now block-1 text. |
| **Confirmation** | Pending hops move into the phoneme log when an alignment advances, or holds the same position for three checks, with a match ratio over a threshold that falls as the queue grows (`FollowAlong/FollowAlongEngine.swift:654-661, :696-704, :739-784`). The newest hop always stays pending (`trailingHopHoldback = 1`, :263). Past six pending hops, the oldest are dropped (:265, :868-877). | Unchanged; it now acts on block-1 text. |
| **Every check's query** | `prev5s` is the last 20 confirmed characters (`FollowAlong/FollowAlongEngine.swift:1566, :3656-3659`; `FollowAlong/PhonemeLog.swift:95-108`). `next5s` is every pending hop, then the current overlap or preview (`FollowAlong/FollowAlongEngine.swift:1567, :236-239`). | Assembled the same way; the next table shows what fills it. |
| **What is graded** | Every update with content runs one alignment (`FollowAlong/FollowAlongEngine.swift:636-639, :1628`) of `prev5s + next5s` against the reference (`FollowAlong/QuranFollowAlong.swift:289-326`). It grades every word that overlaps the aligned region, up to the matched word (`FollowAlong/QuranFollowAlong+WordScoring.swift:362-382`). The grades enter `GradeStore` (`FollowAlong/FollowAlongEngine.swift:3319-3342`), which keeps a new grade only if it ranks higher, or ranks the same and scores higher (`FollowAlong/GradeStore.swift:182-207, :335`). | Unchanged. |
| **The stop's check** | The stop's phonemes become one more pending hop. One last alignment then runs over the pending hops and the last overlap (`FollowAlong/FollowAlongEngine.swift:899-915`), and held grades are promoted (:937-939). | Unchanged. The stop's phonemes start 1 s further into its audio. |

### What each check grades from

Here `w` is the second the current window starts at, and "hops" are the pending hops and the
confirmed tail before them.

| check | b = 0: the query's audio, oldest to newest | b = 1 |
|---|---|---|
| the window starting at `w` | hops (block 0 of each earlier window, reaching `w`), then `[w, w+5)` from this window alone | hops (block 1 of each earlier window, reaching `w+1`), then `[w+1, w+5)` from this window |
| a preview after it (the buffer now starts at `w+1`) | hops through `w+1`, then `[w+1, now)` from the preview | hops through `w+2`, then `[w+2, now)` from the preview |
| the stop | hops, then the stop's whole audio, then the last overlap | hops, then the stop's audio after its first second, then the last overlap |

So a word is graded many times, and the ratchet keeps the best grade.

- While the word is inside the newest window or preview, it is graded from that one inference.
- After that, it is graded from the committed hops of the windows that held it, joined at the
  hop boundaries. This lasts while those hops are pending or within the last 20 confirmed
  characters.

At b = 1, a word moves from the first kind of check to the second 1 s **earlier**. Second `s`
is committed by the window starting at `s - 1` (which runs once `s + 4` s of audio have
arrived), not by the window starting at `s` (at `s + 5` s). Later previews leave it out 1 s
sooner. The exception is the session's first second: at both blocks it is committed by the
first window, at 5 s. The committed hops are each window's second second, not its first.

### What else b = 1 would change

- **Startup (the first seconds).** Nothing runs before 2 s are pending
  (`Models/RealtimeTranscriber+iOS.swift:568-571`). Until the first window at 5 s, every check
  is a preview of everything, at either block. The first window then sends `[0, 2)` as its hop
  instead of `[0, 1)`. Second 0 is the same decode at both blocks. Second 1 comes from window
  0's block 1 instead of window 1's block 0. A page change empties the buffer (:500-503), and
  the startup rule applies again.
- **Latency.** No inference is added or moved. Each second is committed by the window that
  starts 1 s before it: about 1 s sooner (at roughly `s + 4` s of audio, not `s + 5`), with 3 s
  of right context instead of 4. The checks that update what the reader sees are the same
  checks as today.
- **The final flush.** The stop decodes the same pending audio, and so would a silence flush.
  Their first second is already committed, so it is not sent again. A session whose pending
  audio never reaches 5 s, so no window runs, is unchanged.
- **Elsewhere in the engine.** Placement, confirmation, desync and stall handling, and the
  heard stream (ADR-0024) read the same hops and query. The replay runs all of them. The
  transcriber's "meaningful output" clock reads the text it sends
  (`Models/RealtimeTranscriber+iOS.swift:607-611`). At b = 1 a window sends 1 s less, which
  could change when a silence flush fires. That is not replayed.
"""

_ARM_ORDER = ("base/checks", "base/checks_b1", "h448/checks", "h448/checks_b1")
#: The scorings whose flips the doc lists (the report keeps every scoring's).
_FLIP_SCORINGS = ("shipped_default", "allowances_off")


def _scorings(report: dict) -> list[str]:
    return [s["name"] for s in report["muraja"]["scorings"]]


def _header(first: str, names: list[str]) -> list[str]:
    return [f"| {first} | " + " | ".join(f"`{n}`" for n in names) + " |",
            "|---|" + "---|" * len(names)]


def _difference(entry: dict) -> str:
    """``±point [interval]``, then the words flagged by the first arm only / the second only."""
    return f"{format_delta(entry)} (+{entry['only_first']} / −{entry['only_second']})"


def _side_rate(side: str) -> tuple[str, str]:
    """The app rate a side reads, and its title."""
    return ("word_flags", "false flags") if side == CORRECT_SIDE else ("muraja_missed", "missed mistakes")


def _replay_section(report: dict) -> list[str]:
    models = ", ".join(f"{name} `{ref}`" for name, ref in report["models"].items())
    build = report["muraja"]["build"]
    lines = [
        "## The replay",
        "",
        f"Models: {models}. Each model has two arms: `checks` is the committed b = 0 sequence of",
        "the truth baseline, and `checks_b1` is the b = 1 sequence. `Decoder.app_checks_at` cut the",
        "b = 1 sequence and a fresh b = 0 sequence from one set of decoded rows per item. Each b = 1",
        "record carries the digest of that fresh b = 0 sequence, and the run stops unless it matches",
        "the committed b = 0 sequence. The digest checks that the b = 0",
        "check payloads (the hop and overlap texts) are identical, not the timing of the historical",
        "argmax or its logits. Within this run, b = 0 and b = 1 differ only in what each inference",
        "sends. Graded by Muraja's own engine",
        f"at `{str(build.get('muraja_commit', ''))[:12]}`, compiled with {build.get('compiler', '?')}",
        "(`tools/muraja_harness`, `training/muraja_policy.py`).",
        "",
        f"**How the checks were made** (both arms). {APPROXIMATIONS[RECORDED_CHECKS]}",
        "",
        "| model | items | checks sent, b = 0 | checks sent, b = 1 | items whose committed transcript differs |",
        "|---|---|---|---|---|",
    ]
    for model, t in report["transcripts"].items():
        lines.append(f"| {model} | {t['items']} | {t['checks']['checks']} | {t['checks']['checks_b1']} "
                     f"| {t['different_transcripts']} |")
    lines += [
        "",
        "Scorings (balanced throughout; `None` keeps balanced's own flag):",
        "",
        "| scoring | `softPairsEnabled` | `shaddahSuppression` | tashkeel shown |",
        "|---|---|---|---|",
    ]
    for s in report["muraja"]["scorings"]:
        lines.append(f"| `{s['name']}` | {s['soft_pairs_enabled']} | {s['shaddah_suppression']} "
                     f"| {s['tashkeel_shown']} |")
    return lines + [""]


def _shown(view: dict) -> str:
    """One block's kept grade: outcome (quality; causes): errors as Muraja logs them."""
    causes = ", ".join(view["causes"])
    return (f"{view['outcome']} ({view['quality']}{'; ' + causes if causes else ''}): "
            f"{' '.join(view['errors']) or '–'}")


def _flip_rows(flipped: dict, extra: tuple[str, ...] = ()) -> list[str]:
    """Every flip under :data:`_FLIP_SCORINGS`, one row each, or ``None.``"""
    head = ["model", "scoring", "id", *extra, "b = 0", "b = 1"]
    rows = [
        "| " + " | ".join([model, f"`{scoring}`", f"`{entry['id']}`",
                           *(str(entry[k]) for k in extra), _shown(entry["b0"]),
                           _shown(entry["b1"])]) + " |"
        for model, by_scoring in flipped.items()
        for scoring in _FLIP_SCORINGS
        for entry in by_scoring.get(scoring, [])
    ]
    if not rows:
        return ["None."]
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *rows]


def _word_section(report: dict) -> list[str]:
    sample, names = report["word_sample"], _scorings(report)
    arms = [a for a in _ARM_ORDER if a in sample["arms"]]
    total = sample["arms"][arms[0]][names[0]]["correct_words"]
    verdicts = ", ".join(f"{n} {v}" for v, n in sample["verdicts"].items())
    taken = "; ".join(f"{len(sample[r])} {r}" for r in REASONS if sample.get(r))
    lines = [
        "## The word sample: correctly recited words the app flags",
        "",
        f"There are {sample['words']} sampled words, one per held-out reciter, drawn with no model.",
        f"The owner judged each word whole: {verdicts}. The shared exclusion takes out",
        f"{taken or 'none'}, which leaves {total} correctly recited words. A **false flag** is a",
        "correctly recited word the app shows as not correct. Its causes are read from the grade",
        "the engine kept (`training/word_flags.py`). A word can have several causes. Each cell",
        "gives the count, then the weighted rate (each word weighs the inverse of its selection",
        f"probability, from {sample['weight_range'][0]:g} to {sample['weight_range'][1]:g}) and its",
        f"§1 interval. **The sample is small: one word moves a count by one in {total}.**",
        "",
        "False flags per arm and scoring (count · weighted % [interval]):",
        "",
        *_header("arm", names),
        *(f"| {a} | " + " | ".join(count_rate(sample["arms"][a][n]["false_flags"], total)
                                   for n in names) + " |" for a in arms),
        "",
        "Paired differences in false flags, in percentage points with their interval, then the",
        "words flagged by the first arm only / by the second only:",
        "",
        *_header("difference", names),
        *(f"| {pair} | " + " | ".join(_difference(by_scoring[n]) for n in names) + " |"
          for pair, by_scoring in sample["differences"].items()),
    ]
    for a in arms:
        lines += ["", f"**`{a}`**, false flags by cause:", "", *word_cause_table(sample["arms"][a], names)]
    mistakes = sorted({w for a in arms for n in names for w in sample["arms"][a][n]["mistakes"]})
    for word in mistakes:
        lines += ["", f"The word heard as a mistake (`{word}`), its app outcome per arm:", "",
                  *_header("arm", names),
                  *(f"| {a} | " + " | ".join(sample["arms"][a][n]["mistakes"][word] for n in names)
                    + " |" for a in arms)]
    lines += ["", "Every word the app flags at one block only, under the shipped default and with",
              "the allowances off, with the grade the engine kept at each block (outcome, quality,",
              "causes, then the errors as Muraja logs them):", "", *_flip_rows(sample["flips"])]
    return lines + [""]


def _site_rows(cells: list[dict], names: list[str], arms: list[str]) -> list[str]:
    lines = ["| side | rate | sites / reciters | arm | " + " | ".join(f"`{n}`" for n in names) + " |",
             "|---|---|---|---|" + "---|" * len(names)]
    for cell in cells:
        rate, title = _side_rate(cell["side"])
        size = f"{cell['sites']} / {cell['reciters']}" + (f" ({SPARSE_MARK})" if cell["sparse"] else "")
        lines += [f"| {cell['side']} | {title} | {size} | {a} | "
                  + " | ".join(format_estimate(cell["arms"][a]["rates"].get(f"{rate}@{n}"))
                               for n in names) + " |" for a in arms]
    return lines


def _site_differences(cells: list[dict], names: list[str]) -> list[str]:
    lines = ["| side | rate | difference | " + " | ".join(f"`{n}`" for n in names) + " |",
             "|---|---|---|" + "---|" * len(names)]
    for cell in cells:
        rate, title = _side_rate(cell["side"])
        lines += [f"| {cell['side']} | {title} | {pair} | "
                  + " | ".join(format_delta(by_rate.get(f"{rate}@{n}")) for n in names) + " |"
                  for pair, by_rate in cell["differences"].items()]
    return lines


def _site_section(report: dict) -> list[str]:
    verified, names = report["verified"], _scorings(report)
    cells = sorted(verified["cells"], key=lambda c: c["side"] != CORRECT_SIDE)
    arms = [a for a in _ARM_ORDER if cells and a in cells[0]["arms"]]
    sides = {c["side"]: c for c in cells}
    sizes = ", ".join(f"{sides[s]['sites']} {s} ({sides[s]['reciters']} reciters)"
                      for s in (CORRECT_SIDE, MISTAKE_SIDE) if s in sides)
    return [
        "## Ship A's verified sites, held-out half",
        "",
        "These are the sites where `h448` swapped a target pair or collapsed an in-word geminate",
        "and the owner listened, in the held-out half of the reciter split. They are scored",
        "alone (`truth_scorer.score`), weighted by the design weights, less the shared",
        f"exclusion: {sizes or 'none'}. These sites were **selected where `h448` erred**, so the",
        "rates are conditional on that selection. They are not rates over correct recitation at",
        "large. On correct recitation a flagged word is a false flag. On a real mistake, an",
        "unflagged word is a missed mistake. % [95% interval].",
        "",
        *_site_rows(cells, names, arms),
        "",
        "Paired differences, in percentage points with their interval:",
        "",
        *_site_differences(cells, names),
        "",
        "Per allowance, on the sites it affects: flagged words with the allowance on",
        "(`allowances_on`) → switched off alone:",
        "",
        *allowance_rows(verified["allowances"], arms),
        "",
        "Every site whose word the app flags at one block only, under the shipped default and",
        "with the allowances off:",
        "",
        *_flip_rows(verified["flips"], ("side", "mark", "heard")),
        "",
    ]


GAPS = """\
## What the replay still differs from the device on

- **No speech detector.** No silence flush fires, and the VAD power gate and the window and
  preview hallucination gates are bypassed. The reciter is taken to speak from the item's
  first sample to its stop. At b = 1 the "meaningful output" clock reads 1 s less text per
  window, which this cannot show.
- **Ideal timing.** Audio arrives in exact 200 ms steps and every inference is
  instantaneous. There are no late windows and no skipped previews.
- **Numerics.** The decodes are PyTorch with bf16 weights under CUDA autocast, at batch size
  1. The device runs CoreML in fp16 with 6-bit palettized weights. For `h448`, the device would
  also need its own export.
- **Items, not sessions.** Each item is a segment of a clip, replayed as one session that starts
  on the item's first word with nothing before it: no ta'awwudh, no earlier context, no page
  change. The startup and the stop are exactly where b = 1 differs at the edges, and an item
  has one of each per few seconds of audio, which is far more often than a real session.
- **An engine tuned on b = 0.** The engine's confirmation thresholds, the hop holdback and the
  six-hop cap were set with block-0 hops. They are replayed unchanged at b = 1.
- **Display.** The Tashkeel toggle and `.lenient`'s minor mask (iOS-only
  `GradeFilter+iOS.swift`) are applied to the kept grade by `training.muraja_policy`, as in the
  truth baseline. Page-change discards (`UpdateCoalescer+iOS.swift`) are not modelled.

None of these is known to move grades one way only. The replay estimates the device's
outcome at each block; it does not bound it.
"""


def render(report: dict) -> str:
    lines = [
        "# Does committing block 1 change the app outcome? (#148)",
        "",
        "Generated by `python -m training.commit_block` (from `tools/`); do not edit by hand. The",
        "machine-readable report is `tools/tadabur/commit_block/report.json`. This measures only:",
        "it recommends no protocol and sets no threshold.",
        "",
        CODE_READING,
        *_replay_section(report),
        *_word_section(report),
        *_site_section(report),
        GAPS,
        "## Re-running",
        "",
        "```bash",
        "# from tools/, on the GPU box with the staged clips: decodes only what the b = 1 cache lacks",
        "python -m training.commit_block --audio-dir <clips> --decode-only",
        "# on a Mac with the Muraja harness built (bash tools/muraja_harness/build.sh), torch-free",
        "python -m training.commit_block",
        "```",
        "",
    ]
    return "\n".join(lines)

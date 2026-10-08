"""Where each synthetic edit and its decoy go, from the base teacher's frame times (#88).

Torch-free and audio-free: the input is each clip's realized reference and the base
teacher's whole-clip CTC segments (:class:`TimedClip`), the output is :class:`EditPair`
specs that :mod:`tadabur.synthetic_edits` renders. The operations, their decoys and their
labels are described there; this module owns the timing rules.

**Anchoring.** A CTC segment is one decoded character. A reference position is *anchored*
when it sits inside a block where the decode equals the reference (``difflib`` over the raw
strings, the reference's word spaces left out, since the decode has none) with at least
:data:`ANCHOR_CONTEXT` equal characters on each side, so its segment is the teacher's
emission of exactly that letter. Every position an edit or a decoy touches must be
anchored: nothing is cut where the teacher heard something else.

**Spans** come from the emissions at the 40 ms step resolution of the teacher
(:data:`SAMPLES_PER_STEP`):

* the **held span** of a doubled consonant runs from the centre of its first emission to
  the centre of its second (the hold; for a stop, its closure). Cropping it leaves one
  consonant. A crop of only the silent-to-the-model gap between the two emissions was
  tried first: the base teacher still heard the geminate in 16 of 25 such edits, against 7
  of 25 for the centre-to-centre crop (decoys: 25 of 25 unchanged either way);
* a **hold** of a single consonant is its own emission;
* a consonant's **cell** runs from the end of the previous emission to the end of the
  following haraka's, so a splice carries the consonant-to-haraka transition, which is
  where emphasis (``ص`` against ``س``) is heard. A splice ending at the haraka's start
  was heard as the donor's letter by the teacher in 6 of 68 edits, this one in 23 of 53;
  a splice aligns the donor's emission centre on the carrier's;
* a **madd span** runs from the first to the last emission of a long-vowel run.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from training.decoding import DEPLOYED_LOGIT_FRAMES
from training.distill_data import WINDOW_SAMPLES

from .phoneme_vocab import PHONEME_ID_TO_CHAR
from .truth_sites import CONSONANTS, HARAKA_CHARS, HELD, NOT_HELD, SHADDAH

#: Audio samples per CTC step of the teacher: 125 steps per 5 s window, 40 ms.
SAMPLES_PER_STEP = WINDOW_SAMPLES // DEPLOYED_LOGIT_FRAMES

SHADDAH_REMOVED = "shaddah_removed"
SHADDAH_ADDED = "shaddah_added"
CONSONANT_SWAP = "consonant_swap"
OPERATIONS = (SHADDAH_REMOVED, SHADDAH_ADDED, CONSONANT_SWAP)

EDIT = "edit"
DECOY = "decoy"

CROP = "crop"
STRETCH = "stretch"
SPLICE = "splice"
#: Where a change is made: at the carrier, or inside a long madd away from it.
AT_CARRIER = "carrier"
IN_MADD = "madd"

#: The fricative pairs a consonant swap may cross (the issue's three, plus ``ذ↔ظ`` per
#: acceptance rules §7). Each is labelled like a truth-site soft pair: codepoint order.
SWAP_PAIRS = tuple(sorted("↔".join(sorted(pair)) for pair in
                          ({"س", "ص"}, {"ذ", "ز"}, {"ض", "ظ"}, {"ذ", "ظ"})))
#: Single consonants whose hold is stretched: the voiceless fricatives, whose frication
#: extends as noise without a pitch to keep in phase.
STRETCHABLE = frozenset("ثحخسشصف")
#: Long-vowel letters; a run of :data:`MIN_MADD_RUN` of one of them is a decoy's place.
MADD_LETTERS = frozenset("اۥۦ")
MIN_MADD_RUN = 4

#: Equal characters required on each side of an anchored position.
ANCHOR_CONTEXT = 2
#: The shortest gap between a geminate's two emissions for it to be cropped: two steps,
#: so the hold is more than one step's jitter.
MIN_HELD_GAP = 2 * SAMPLES_PER_STEP
#: A madd span must exceed a decoy's change by this much on each side.
MADD_MARGIN = SAMPLES_PER_STEP
#: A decoy's change is at least this far (0.5 s) from where the edit's would be.
MIN_DECOY_DISTANCE = 8000
#: Context samples a splice's donor needs on each side, for the crossfade.
SPLICE_CONTEXT = 80

HARAKAT = frozenset(HARAKA_CHARS.values())


def rank(text: str, salt: str) -> str:
    """A salted SHA-256 hex digest: the order every draw in this work is made in."""
    return hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Token:
    """One emission of the teacher: its character and its ``[start, end)`` samples."""

    char: str
    start: int
    end: int

    @property
    def centre(self) -> int:
        return (self.start + self.end) // 2


@dataclass(frozen=True)
class TimedClip:
    """A clip's realized reference and the teacher's timed decode of it."""

    audio_filename: str
    reciter_id: int
    num_samples: int
    reference: str
    tokens: tuple[Token, ...]
    #: Reference index -> token index, for anchored positions only.
    anchors: dict[int, int] = field(compare=False, repr=False)

    @classmethod
    def from_steps(
        cls, audio_filename: str, reciter_id: int, num_samples: int, reference: str,
        steps: Iterable[tuple[int, int, int]],
    ) -> "TimedClip":
        """From CTC segments ``(token_id, first_step, last_step)`` of a whole-clip decode."""
        tokens = tuple(
            Token(PHONEME_ID_TO_CHAR[token], first * SAMPLES_PER_STEP,
                  min((last + 1) * SAMPLES_PER_STEP, num_samples))
            for token, first, last in steps
        )
        decode = "".join(t.char for t in tokens)
        return cls(audio_filename, reciter_id, num_samples, reference, tokens,
                   anchored_positions(reference, decode))

    def token(self, index: int) -> Token:
        return self.tokens[self.anchors[index]]


def anchored_positions(reference: str, decode: str) -> dict[int, int]:
    """Reference index -> decode index for every anchored position (module docstring)."""
    letters = [i for i, c in enumerate(reference) if c != " "]
    compact = "".join(reference[i] for i in letters)
    anchors: dict[int, int] = {}
    matcher = SequenceMatcher(None, compact, decode, autojunk=False)
    for tag, i1, i2, j1, _ in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i1 + ANCHOR_CONTEXT, i2 - ANCHOR_CONTEXT):
                anchors[letters[k]] = j1 + (k - i1)
    return anchors


# --- the specs -------------------------------------------------------------------------


@dataclass(frozen=True)
class Donor:
    """The span of a same-reciter clip a splice takes its material from."""

    audio_filename: str
    reference_index: int
    letter: str
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class Change:
    """One timing or splice change to a source clip, in its sample coordinates.

    ``[start_sample, end_sample)`` is replaced by ``inserted_samples`` new samples: none
    for a crop, a stretch inserts at a point, a splice replaces the span with as many
    samples of ``donor``. A stretch's samples are drawn from ``fill_region``.
    """

    place: str
    kind: str
    start_sample: int
    end_sample: int
    inserted_samples: int
    fill_region: tuple[int, int] | None
    donor: Donor | None

    @property
    def length_change(self) -> int:
        return self.inserted_samples - (self.end_sample - self.start_sample)


@dataclass(frozen=True)
class EditPair:
    """An edit and its decoy, on one carrier of one source clip.

    The edit changes the carrier from ``prescribed`` to ``edited``; the decoy makes the
    same kind of change (equal length change) somewhere that leaves ``prescribed`` there.
    """

    operation: str
    audio_filename: str
    reciter_id: int
    reference: str
    reference_index: int
    mark: str
    prescribed: str
    edited: str
    edit: Change
    decoy: Change

    @property
    def pair_id(self) -> str:
        return f"{self.operation}:{self.mark}:{self.audio_filename}:{self.reference_index}"


def labelled_reference(pair: EditPair, role: str) -> str:
    """The reference an item of ``pair`` carries: the decoy keeps the source's, the edit's
    follows the edit at the carrier."""
    ref, i = pair.reference, pair.reference_index
    if role == DECOY:
        return ref
    if pair.operation == SHADDAH_REMOVED:
        return ref[: i + 1] + ref[i + 2:]
    if pair.operation == SHADDAH_ADDED:
        return ref[: i + 1] + ref[i] + ref[i + 1:]
    return ref[:i] + pair.edited + ref[i + 1:]


def carrier_positions(reference_index: int, label: str) -> tuple[int, ...]:
    """The positions of an item's labelled reference that spell its label at the carrier:
    both halves of a held consonant, else the carrier alone."""
    return (reference_index, reference_index + 1) if label == HELD else (reference_index,)


# --- finding carriers --------------------------------------------------------------------


def _neighbour(ref: str, i: int, step: int) -> str:
    """The nearest character before (``step = -1``) or after ``ref[i]`` that is not a
    word space, or ``""`` at an end."""
    i += step
    while 0 <= i < len(ref) and ref[i] == " ":
        i += step
    return ref[i] if 0 <= i < len(ref) else ""


def _single(ref: str, i: int) -> bool:
    """``ref[i]`` is not half of a doubled consonant, across a word space included."""
    return ref[i] not in (_neighbour(ref, i, -1), _neighbour(ref, i, 1))


def _anchored(clip: TimedClip, *indices: int) -> bool:
    return all(i in clip.anchors for i in indices)


def held_spans(clip: TimedClip) -> dict[int, tuple[int, int]]:
    """Each anchored doubled consonant's held span, by the index of its first half.

    Only a pair exactly two long (a ghunna geminate is spelled with four) whose emissions
    are at least :data:`MIN_HELD_GAP` apart."""
    ref, gaps = clip.reference, {}
    for i in range(len(ref) - 1):
        c = ref[i]
        if (c in CONSONANTS and ref[i + 1] == c and _neighbour(ref, i, -1) != c
                and _neighbour(ref, i + 1, 1) != c and _anchored(clip, i, i + 1)):
            first, second = clip.token(i), clip.token(i + 1)
            if second.start - first.end >= MIN_HELD_GAP:
                gaps[i] = (first.centre, second.centre)
    return gaps


def stretchable_holds(clip: TimedClip) -> dict[int, tuple[int, int]]:
    """Each anchored single voiceless fricative between two harakat, and its hold."""
    ref, holds = clip.reference, {}
    for i in range(1, len(ref) - 1):
        if (ref[i] in STRETCHABLE and _single(ref, i) and ref[i - 1] in HARAKAT
                and ref[i + 1] in HARAKAT and _anchored(clip, i)):
            token = clip.token(i)
            holds[i] = (token.start, token.end)
    return holds


def splice_carriers(clip: TimedClip, letters: Iterable[str]) -> dict[int, tuple[int, int]]:
    """Each anchored single carrier of one of ``letters`` followed by a haraka, and its cell:
    from the end of the emission before it (anchoring makes that the previous letter's) to
    the end of the haraka's."""
    ref, letters, cells = clip.reference, frozenset(letters), {}
    for i in range(1, len(ref) - 1):
        if (ref[i] in letters and _single(ref, i) and ref[i + 1] in HARAKAT
                and _anchored(clip, i, i + 1)):
            cells[i] = (clip.tokens[clip.anchors[i] - 1].end, clip.token(i + 1).end)
    return cells


def madd_spans(clip: TimedClip) -> list[tuple[int, int]]:
    """The sample span of every anchored long-vowel run of :data:`MIN_MADD_RUN` or more."""
    ref, spans, i = clip.reference, [], 0
    while i < len(ref):
        j = i
        while j < len(ref) and ref[j] == ref[i]:
            j += 1
        if ref[i] in MADD_LETTERS and j - i >= MIN_MADD_RUN and _anchored(clip, *range(i, j)):
            spans.append((clip.token(i).start, clip.token(j - 1).end))
        i = j
    return spans


# --- decoys ------------------------------------------------------------------------------


def madd_change(
    clip: TimedClip, kind: str, length: int, avoid: int, salt: str
) -> Change | None:
    """A crop of ``length`` samples from, or a stretch of ``length`` samples into, the
    middle of a long madd of ``clip`` at least :data:`MIN_DECOY_DISTANCE` from ``avoid``
    and at least ``length`` longer than its margins; the hash-first such madd, or
    ``None``."""
    options = []
    for start, end in madd_spans(clip):
        middle = (start + end) // 2
        room = end - start - 2 * MADD_MARGIN
        if abs(middle - avoid) < MIN_DECOY_DISTANCE:
            continue
        if room < length:
            continue
        if kind == CROP:
            change = Change(IN_MADD, CROP, middle - length // 2, middle - length // 2 + length,
                            0, None, None)
        else:
            change = Change(IN_MADD, STRETCH, middle, middle, length,
                            (start + MADD_MARGIN, end - MADD_MARGIN), None)
        options.append((rank(f"{clip.audio_filename}:{start}", salt), change))
    return min(options)[1] if options else None


# --- the operations ----------------------------------------------------------------------


def shaddah_removed(clip: TimedClip, salt: str) -> list[EditPair]:
    """Every doubled consonant whose held span can be cropped, with a madd-crop decoy."""
    pairs = []
    for i, (start, end) in sorted(held_spans(clip).items()):
        edit = Change(AT_CARRIER, CROP, start, end, 0, None, None)
        decoy = madd_change(clip, CROP, end - start, start, salt)
        if decoy is not None:
            pairs.append(EditPair(SHADDAH_REMOVED, clip.audio_filename, clip.reciter_id,
                                  clip.reference, i, SHADDAH, HELD, NOT_HELD, edit, decoy))
    return pairs


def shaddah_added(clip: TimedClip, length: int, salt: str) -> list[EditPair]:
    """Every stretchable single fricative, stretched by ``length`` samples in the middle
    of its hold, with a madd-stretch decoy of the same length."""
    pairs = []
    for i, (start, end) in sorted(stretchable_holds(clip).items()):
        middle = (start + end) // 2
        edit = Change(AT_CARRIER, STRETCH, middle, middle, length, (start, end), None)
        decoy = madd_change(clip, STRETCH, length, middle, salt)
        if decoy is not None:
            pairs.append(EditPair(SHADDAH_ADDED, clip.audio_filename, clip.reciter_id,
                                  clip.reference, i, SHADDAH, NOT_HELD, HELD, edit, decoy))
    return pairs


@dataclass(frozen=True)
class _Occurrence:
    clip: TimedClip
    index: int

    @property
    def letter(self) -> str:
        return self.clip.reference[self.index]

    @property
    def haraka(self) -> str:
        return self.clip.reference[self.index + 1]

    @property
    def before(self) -> str:
        return _neighbour(self.clip.reference, self.index, -1)


def _donor_span(target: _Occurrence, cell: tuple[int, int], donor: _Occurrence) -> Donor | None:
    """The donor window aligned on emission centres, if it stays within the donor's
    neighbouring emissions and leaves room for the crossfade context."""
    centre = target.clip.token(target.index).centre
    donor_centre = donor.clip.token(donor.index).centre
    start = donor_centre - (centre - cell[0])
    end = donor_centre + (cell[1] - centre)
    low = donor.clip.tokens[donor.clip.anchors[donor.index] - 1].start
    high = donor.clip.token(donor.index + 1).end
    if start < max(low, SPLICE_CONTEXT) or end > min(high, donor.clip.num_samples - SPLICE_CONTEXT):
        return None
    return Donor(donor.clip.audio_filename, donor.index, donor.letter, start, end)


def _best_donor(
    target: _Occurrence, cell: tuple[int, int], letter: str,
    occurrences: list[_Occurrence], salt: str,
) -> Donor | None:
    """The best same-reciter donor of ``letter`` with the target's following haraka: one
    with the same preceding character first, then by hash; never the target itself."""
    options = []
    for donor in occurrences:
        if (donor.letter != letter or donor.haraka != target.haraka
                or (donor.clip.audio_filename, donor.index)
                == (target.clip.audio_filename, target.index)):
            continue
        span = _donor_span(target, cell, donor)
        if span is not None:
            key = f"{target.clip.audio_filename}:{target.index}:{donor.clip.audio_filename}:{donor.index}"
            options.append((donor.before != target.before, rank(key, salt), span))
    return min(options)[2] if options else None


def consonant_swaps(reciter_clips: list[TimedClip], salt: str) -> list[EditPair]:
    """Every splice-able carrier of a swap pair in one reciter's clips, spliced from a
    same-reciter donor of the other letter, with a same-letter splice as its decoy."""
    letters = {c for pair in SWAP_PAIRS for c in pair.split("↔")}
    cells = {clip.audio_filename: splice_carriers(clip, letters) for clip in reciter_clips}
    occurrences = [_Occurrence(clip, i) for clip in reciter_clips
                   for i in sorted(cells[clip.audio_filename])]
    pairs = []
    for target in occurrences:
        cell = cells[target.clip.audio_filename][target.index]
        for mark in SWAP_PAIRS:
            if target.letter not in mark.split("↔"):
                continue
            (other,) = set(mark.split("↔")) - {target.letter}
            edit_donor = _best_donor(target, cell, other, occurrences, salt)
            decoy_donor = _best_donor(target, cell, target.letter, occurrences, salt)
            if edit_donor is None or decoy_donor is None:
                continue
            width = cell[1] - cell[0]
            pairs.append(EditPair(
                CONSONANT_SWAP, target.clip.audio_filename, target.clip.reciter_id,
                target.clip.reference, target.index, mark, target.letter, other,
                Change(AT_CARRIER, SPLICE, *cell, width, None, edit_donor),
                Change(AT_CARRIER, SPLICE, *cell, width, None, decoy_donor),
            ))
    return pairs


def stretch_length(clips: Iterable[TimedClip]) -> int:
    """How far ``shaddah_added`` stretches a hold: the median held span of the frame's
    doubled consonants, the length ``shaddah_removed`` crops, so an added hold is as long
    as a typical real one."""
    gaps = [end - start for clip in clips for start, end in held_spans(clip).values()]
    return int(statistics.median(gaps))


def plan_pairs(clips: list[TimedClip], salt: str) -> tuple[list[EditPair], int]:
    """Every candidate pair of every operation, and the stretch length used."""
    length = stretch_length(clips)
    pairs: list[EditPair] = []
    by_reciter: dict[int, list[TimedClip]] = {}
    for clip in sorted(clips, key=lambda c: c.audio_filename):
        pairs += shaddah_removed(clip, salt) + shaddah_added(clip, length, salt)
        by_reciter.setdefault(clip.reciter_id, []).append(clip)
    for reciter in sorted(by_reciter):
        pairs += consonant_swaps(by_reciter[reciter], salt)
    return pairs, length


def select_pairs(
    pairs: list[EditPair], quota: int, per_reciter: int, salt: str
) -> list[EditPair]:
    """Per operation and mark, at most ``quota`` pairs in hash order, one per source clip
    and at most ``per_reciter`` per reciter, so no clip or voice dominates."""
    chosen: list[EditPair] = []
    groups: dict[tuple[str, str], list[EditPair]] = {}
    for pair in pairs:
        groups.setdefault((pair.operation, pair.mark), []).append(pair)
    for key in sorted(groups):
        clips: set[str] = set()
        reciters: dict[int, int] = {}
        taken = 0
        for pair in sorted(groups[key], key=lambda p: rank(p.pair_id, salt)):
            if taken == quota:
                break
            if pair.audio_filename in clips or reciters.get(pair.reciter_id, 0) >= per_reciter:
                continue
            clips.add(pair.audio_filename)
            reciters[pair.reciter_id] = reciters.get(pair.reciter_id, 0) + 1
            chosen.append(pair)
            taken += 1
    return sorted(chosen, key=lambda p: p.pair_id)

"""Generate the frozen P7.H integration fixtures from the phonetizer (#35).

Resolves each curated boundary into its two realized reference forms and the reciter-realization
decodes, then freezes them (:mod:`tadabur.waqf_integration_fixtures`) so the product-gate eval
(:mod:`tadabur.waqf_integration_eval`) is a pure function of the scorer — no ``quran_transcript``
or model at eval/test time. This is the phonetizer-dependent authoring step, run once (and re-run
to regenerate); it mirrors the realized-reference vocabulary of :mod:`tadabur.waqf_segments`.

For a boundary after Uthmani word *w* (next word *n*) in an ayah, the two realized forms are:

* **waqf form** — ``quran_phonetizer(w)``: the terminal word in its **pausal** CleanEnd realization
  (a tanwin becomes a madd, the final haraka drops), exactly as :mod:`tadabur.waqf_segments` labels
  a stop-terminated segment's terminal word.
* **wasl form** — ``quran_phonetizer(w n)`` sliced to drop *n* via the phonetizer's per-word char
  offsets (:func:`tadabur.waqf_segments._spaceless_word_offsets`): the terminal word in
  **continuation** into the next word — the pausal madd absent, the tanwin's noon/ghunna carried
  onto *n* (so it leaves the run) — the seam realization the cross-word idgham / i'raab turns on.

Both are normalized with the scorer's :func:`tadabur.normalization.normalize_phonemes` so they are
the exact strings ``STRICT_SCORER.gate`` compares against.

Only boundaries whose two forms **differ after normalization** and whose difference the `.strict`
gate actually **resolves** (the pausal decode is rejected against the wasl reference yet accepted
against the waqf reference, and the continuation decode is accepted against the wasl reference) are
admitted — the crisp-discrimination criterion. A curated boundary that fails it raises, so the set
never silently contains a boundary the coarse gate cannot grade. Note (ADR-0004 / this repo's
``normalization``) that a *pure short-vowel* final haraka is stripped by normalization and so is
**not** gradeable; the admitted i'raab cases are the consonantal desinences (tanwin → madd) and the
idgham cases the tanwin's ghunna, which survive normalization.

Each admitted boundary emits three cases: a legitimate **waqf/correct** (pausal decode — the pause
the hack must not punish), a **wasl/correct** (continuation decode), and a **wasl/dropped** (the
pausal decode said mid-continuation — the i'raab/idgham error the hack forgives). Boundaries also
emit a **wasl/interior** control (a genuine mistake *inside* the word) when one can be synthesised
that both the strict and the end-word-forgiven gate reject, proving the baseline still catches
non-boundary errors.

Every case also carries a synthesised **silence-posterior lattice + word-frame spans** encoding the
reciter's actual stop/continue behaviour: a true-waqf case gets a lattice with a ≥ 300 ms silence
run in the gap after the boundary word (a detectable stop), a true-wasl case a contiguous-speech
lattice (no stop). The generator self-validates that
:func:`tadabur.waqf_postprocess.waqf_events` snaps each lattice back to the labelled class, so the
eval's *predicted* full path (not an oracle label) drives the gate.

Usage:
  python -m tadabur.waqf_integration_gen [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .normalization import normalize_phonemes
from .scorer import STRICT, STRICT_SCORER
from .waqf_integration_fixtures import (
    ACCEPT,
    CORRECT,
    CROSS_WORD_IDGHAM,
    DROPPED,
    INTERIOR,
    IRAAB,
    REJECT,
    WAQF,
    WASL,
    IntegrationCase,
    write_integration_cases,
)

# Curated discriminating boundaries: (surah:ayah, word_index) — the boundary falls after the word
# at ``word_index`` (0-based) in the ayah's Uthmani text. Chosen (by scanning, see the module's
# validation) as clean tanwin-desinence / cross-word-idgham stops whose waqf vs wasl realized forms
# the `.strict` gate crisply resolves. The generator re-validates each; a stale one raises.
CURATED_BOUNDARIES: tuple[tuple[str, int], ...] = (
    ("2:38", 7),    # هُدًۭى → فَمَن   (tanwin fath, izhar/ikhfa — i'raab)
    ("3:186", 14),  # أَذًۭى → كَثِيرًۭا (tanwin fath — i'raab)
    ("2:259", 18),  # مِا۟ئَةَ → عَامٍۢ (ta-marbuta desinence — i'raab)
    ("2:249", 50),  # فِئَةٍۢ → قَلِيلَةٍ (tanwin kasra — i'raab)
    ("3:13", 7),    # فِئَةٌۭ → تُقَـٰتِلُ (tanwin damma — i'raab)
    ("2:71", 14),   # شِيَةَ → فِيهَا  (ta-marbuta desinence — i'raab)
    ("2:96", 13),   # سَنَةٍۢ → وَمَا  (tanwin + و idgham ghunna — cross-word idgham)
    ("2:255", 9),   # سِنَةٌۭ → وَلَا  (tanwin + و idgham ghunna — cross-word idgham)
    ("2:263", 7),   # أَذًۭى → وَٱللَّهُ (tanwin + و idgham ghunna — cross-word idgham)
    ("3:111", 3),   # أَذًۭى → وَإِن  (tanwin + و idgham ghunna — cross-word idgham)
)

# Idgham-with-ghunna receiving consonants; a boundary whose next word opens with one (and whose
# terminal word carries a tanwin/noon) exercises cross-word idgham rather than a bare desinence.
_GHUNNA_IDGHAM_HEADS = frozenset("ينمو")
_TASHKEEL = "ًٌٍَُِّْـٰ"

# A hard, clearly-audible consonant swap for the interior-error control (a ص↔ك class distance the
# scorer never treats as a soft pair), applied to the first consonant of the continuation decode so
# the mistake sits *inside* the word, not at its forgiven edge.
_INTERIOR_SWAPS = {"\u0635": "\u0643", "\u0643": "\u0635"}  # ص↔ك
_INTERIOR_FALLBACK = ("\u0635", "\u0643")  # (from, to) when neither swap key is present

# Synthesised silence lattice geometry (40 ms frames, matching tadabur.waqf_postprocess). Each word
# gets a speech span comfortably above the 700 ms min-speech (so it survives the VAD cleaning) and a
# stop gets a silence gap comfortably above the 300 ms min-silence (so it is detected). These are the
# *reciter behaviour* encoded per case: waqf_events snaps a stopped lattice back to `waqf`, a
# contiguous lattice to `wasl` — the predicted class the eval's conditional path selects from.
_SPEECH_FRAMES = 20  # 800 ms >= DEFAULT_MIN_SPEECH_MS (700 ms)
_GAP_FRAMES = 10  # 400 ms >= DEFAULT_MIN_SILENCE_MS (300 ms)


def _stopped_lattice(word_index: int) -> tuple[list[float], list[list[int]]]:
    """A lattice with a detectable stop after ``word_index`` (the true-waqf behaviour)."""
    total = _SPEECH_FRAMES + _GAP_FRAMES + _SPEECH_FRAMES
    silence = [0.0] * total
    for frame in range(_SPEECH_FRAMES, _SPEECH_FRAMES + _GAP_FRAMES):
        silence[frame] = 1.0
    word_spans = [
        [word_index, 0, _SPEECH_FRAMES],
        [word_index + 1, _SPEECH_FRAMES + _GAP_FRAMES, total],
    ]
    return silence, word_spans


def _continued_lattice(word_index: int) -> tuple[list[float], list[list[int]]]:
    """A contiguous-speech lattice with no stop (the true-wasl behaviour)."""
    total = _SPEECH_FRAMES + _SPEECH_FRAMES
    silence = [0.0] * total
    word_spans = [
        [word_index, 0, _SPEECH_FRAMES],
        [word_index + 1, _SPEECH_FRAMES, total],
    ]
    return silence, word_spans


def _lattice_for(true_class: str, word_index: int) -> tuple[list[float], list[list[int]]]:
    """The silence lattice a reciter of ``true_class`` produced at ``word_index``."""
    return (
        _stopped_lattice(word_index)
        if true_class == WAQF
        else _continued_lattice(word_index)
    )


def _phonetizers():
    """The Hafs phonetizer + per-word offset helper (imported lazily; needs quran_transcript)."""
    from quran_transcript import Aya, quran_phonetizer
    from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes

    import generate_phonemes  # tools/ sibling

    from .waqf_segments import _spaceless_word_offsets

    moshaf = MoshafAttributes(**generate_phonemes.HAFS_MOSHAF)

    def phon(text: str):
        out = quran_phonetizer(text, moshaf)
        return out.phonemes, out.mappings

    return Aya, phon, _spaceless_word_offsets


def _norm(s: str) -> str:
    return normalize_phonemes(s).normalized


def _strict_accepts(decode: str, reference: str) -> bool:
    return STRICT_SCORER.gate(decode, reference).passed


def _baseline_accepts(decode: str, reference: str) -> bool:
    """The end-word-forgiven `.strict` verdict (today's ignore-end-word-tashkeel behaviour).

    Scores against the continuation (wasl) reference but discounts the terminal edge the local
    aligner trims (``trailing_trim``) from the match denominator, so the boundary word's pausal
    ending never counts — a legitimate pause is not punished, but a dropped-i'raab / missed-idgham
    error there is equally forgiven (the discrimination ADR-0004 is regaining). Kept identical to
    :func:`tadabur.waqf_integration_eval.baseline_accepts`.
    """
    from .waqf_integration_eval import baseline_accepts

    return baseline_accepts(decode, reference)


_TANWIN = "ًٌٍ"  # fathatan / dammatan / kasratan


def _phenomenon(word: str, next_word: str) -> str:
    head = next((c for c in next_word if c not in _TASHKEEL), "")
    nunated = any(c in _TANWIN for c in word) or word.rstrip(_TASHKEEL).endswith("ن")
    return CROSS_WORD_IDGHAM if (nunated and head in _GHUNNA_IDGHAM_HEADS) else IRAAB


def _interior_decode(wasl_raw: str) -> str | None:
    """A continuation decode with its first consonant swapped — a genuine *interior* mistake."""
    chars = list(wasl_raw)
    for i, c in enumerate(chars):
        if c in _INTERIOR_SWAPS:
            chars[i] = _INTERIOR_SWAPS[c]
            return "".join(chars)
    frm, to = _INTERIOR_FALLBACK
    for i, c in enumerate(chars):
        if c not in " " and c not in _TASHKEEL and c != frm:
            chars[i] = to if c != to else frm
            return "".join(chars)
    return None


def _resolve_boundary(surah_ayah: str, word_index: int, Aya, phon, offsets):
    """One boundary's (word, next_word, waqf_raw, wasl_raw, waqf_ref, wasl_ref).

    ``*_raw`` are the un-normalized phonetizer forms (the model-decode stand-ins the scorer
    normalizes); ``*_ref`` are their normalized realized references (the scorer's cache form).
    """
    surah, ayah = (int(x) for x in surah_ayah.split(":"))
    words = Aya(surah, ayah).get().uthmani.split()
    if not 0 <= word_index < len(words) - 1:
        raise ValueError(
            f"{surah_ayah}#{word_index}: boundary must be interior (0..{len(words) - 2})"
        )
    word, next_word = words[word_index], words[word_index + 1]

    waqf_raw, _ = phon(word)
    run_next = [word, next_word]
    ph2, map2 = phon(" ".join(run_next))
    spaceless, offs = offsets(ph2, map2, run_next)
    wasl_raw = spaceless[: offs[1]]  # drop the appended next word
    return word, next_word, waqf_raw, wasl_raw, _norm(waqf_raw), _norm(wasl_raw)


def _validate_discriminating(
    waqf_raw: str, wasl_raw: str, waqf_ref: str, wasl_ref: str, source: str
) -> None:
    """Assert the boundary's two forms differ and the `.strict` gate resolves the difference.

    Uses the *raw* decodes against the *normalized* references (the scorer's contract): the pausal
    decode must be rejected against the wasl reference yet accepted against the waqf reference, and
    the continuation decode accepted against the wasl reference — the crisp-discrimination bar.
    """
    if waqf_ref == wasl_ref:
        raise ValueError(
            f"{source}: waqf and wasl references normalize identically ({waqf_ref!r}) — not "
            "gradeable (a pure short-vowel desinence is stripped by normalization)"
        )
    if _strict_accepts(waqf_raw, wasl_ref):
        raise ValueError(f"{source}: pausal decode is not rejected against the wasl reference")
    if not _strict_accepts(waqf_raw, waqf_ref):
        raise ValueError(f"{source}: pausal decode is not accepted against the waqf reference")
    if not _strict_accepts(wasl_raw, wasl_ref):
        raise ValueError(f"{source}: continuation decode is not accepted against the wasl reference")


def _validate_predicted_path(case: IntegrationCase, source: str) -> None:
    """Assert the synthesised lattice snaps back to the case's labelled ``true_class``.

    The eval selects the conditional path's reference from the class
    :func:`tadabur.waqf_integration_eval.predicted_class` snaps from the case's silence lattice, so
    a lattice that mis-predicts would silently break the product gate. Re-run the real
    post-processing here and fail loudly if the snapped class disagrees with the label the case was
    built for.
    """
    from .waqf_integration_eval import predicted_class

    got = predicted_class(case)
    if got != case.true_class:
        raise ValueError(
            f"{source}: synthesised lattice snaps to {got!r} but case is labelled "
            f"true_class {case.true_class!r} — the lattice does not encode its behaviour"
        )


def build_cases(
    boundaries: tuple[tuple[str, int], ...] = CURATED_BOUNDARIES,
) -> list[IntegrationCase]:
    """Resolve and validate every curated boundary into its frozen integration cases.

    Each boundary yields a waqf/correct, a wasl/correct and a wasl/dropped case (and a
    wasl/interior control when one can be synthesised that both the strict and end-word-forgiven
    gate reject). Raises on any boundary the coarse gate cannot crisply grade, so the frozen set is
    self-validating.
    """
    Aya, phon, offsets = _phonetizers()
    cases: list[IntegrationCase] = []
    for surah_ayah, word_index in boundaries:
        source = f"{surah_ayah}#{word_index}"
        word, next_word, waqf_raw, wasl_raw, waqf_ref, wasl_ref = _resolve_boundary(
            surah_ayah, word_index, Aya, phon, offsets
        )
        _validate_discriminating(waqf_raw, wasl_raw, waqf_ref, wasl_ref, source)
        phen = _phenomenon(word, next_word)

        def case(recitation: str, true_class: str, decode: str) -> IntegrationCase:
            silence, word_spans = _lattice_for(true_class, word_index)
            built = IntegrationCase(
                case_id=f"{source}/{recitation}",
                surah_ayah=surah_ayah,
                boundary_word_index=word_index,
                word=word,
                next_word=next_word,
                phenomenon=phen,
                waqf_reference=waqf_ref,
                wasl_reference=wasl_ref,
                silence=silence,
                word_spans=word_spans,
                true_class=true_class,
                recitation=recitation,
                decode=decode,
                expected_strict=ACCEPT if recitation == CORRECT else REJECT,
            )
            _validate_predicted_path(built, source)
            return built

        cases.append(case(CORRECT, WAQF, waqf_raw))  # legitimate pause: pausal decode
        cases.append(case(CORRECT, WASL, wasl_raw))  # continuation: continuation decode
        cases.append(case(DROPPED, WASL, waqf_raw))  # error: pausal decode said mid-continuation

        interior = _interior_decode(wasl_raw)
        if (
            interior is not None
            and not _strict_accepts(interior, wasl_ref)
            and not _baseline_accepts(interior, wasl_ref)
        ):
            cases.append(case(INTERIOR, WASL, interior))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=None, help="Output path (JSONL).")
    args = parser.parse_args()

    cases = build_cases()
    if args.out is None:
        write_integration_cases(cases)
        out = None
    else:
        write_integration_cases(cases, args.out)
        out = args.out
    from .waqf_integration_fixtures import WAQF_INTEGRATION_PATH

    print(
        f"Wrote {len(cases)} integration cases from {len(CURATED_BOUNDARIES)} boundaries "
        f"to {out or WAQF_INTEGRATION_PATH}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

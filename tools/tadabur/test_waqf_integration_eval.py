"""Unit tests for the P7.H conditional-reference integration eval (``tadabur.waqf_integration_eval``).

Covers the pure scorer harness: the end-word-forgiven baseline forgiving a boundary error the
conditional realized-reference path rejects, per-scenario `.strict` verdicts, the false-waqf /
false-wasl failure directions, the ``run_eval`` product-gate logic (regain must be real *and*
complete, legitimate-pause acceptance preserved), the fixtures loader's leak/invariant guards, and
a run over the real frozen fixtures asserting the gate passes. All torch-free and
``quran_transcript``-free: cases are hand-written normalized realized references + raw decodes.
"""

from __future__ import annotations

import pytest

from .waqf_integration_eval import (
    BASELINE,
    CONDITIONAL,
    FALSE_WAQF,
    FALSE_WASL,
    ORACLE,
    baseline_accepts,
    predicted_class,
    run_eval,
    score_scenario,
    strict_accepts,
    verdict,
)
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
    WAQF_INTEGRATION_PATH,
    _parse_entry,
    load_integration_cases,
)

# A single boundary's frozen forms (2:38#7 هُدًۭى→فَمَن): pausal madd vs continuation noon. The
# pausal decode is accepted against the waqf reference but rejected against the wasl reference —
# the crisp-discrimination the fixtures guarantee.
WAQF_REF = "هدا"  # pausal: tanwin fath → madd
WASL_REF = "هدن"  # continuation: tanwin's noon realized
PAUSAL_DECODE = "هُدَاا"  # raw pausal realization (normalizes to هدا)
CONT_DECODE = "هُدَںںں"  # raw continuation realization (normalizes to هدن)
INTERIOR_DECODE = "كُدَںںں"  # continuation with its opening consonant swapped ه→ك

# Synthesised silence lattice geometry (40 ms frames, mirroring tadabur.waqf_postprocess). A stopped
# lattice carries a ≥ 300 ms silence run in the gap after the boundary word (a detectable waqf); a
# continued lattice is contiguous speech (no stop). Each word span clears the 700 ms min-speech.
_SPEECH_FRAMES = 20  # 800 ms
_GAP_FRAMES = 10  # 400 ms
_BOUNDARY_INDEX = 7


def _stopped_lattice(word_index=_BOUNDARY_INDEX):
    total = _SPEECH_FRAMES + _GAP_FRAMES + _SPEECH_FRAMES
    silence = [0.0] * total
    for frame in range(_SPEECH_FRAMES, _SPEECH_FRAMES + _GAP_FRAMES):
        silence[frame] = 1.0
    spans = [
        [word_index, 0, _SPEECH_FRAMES],
        [word_index + 1, _SPEECH_FRAMES + _GAP_FRAMES, total],
    ]
    return silence, spans


def _continued_lattice(word_index=_BOUNDARY_INDEX):
    total = _SPEECH_FRAMES + _SPEECH_FRAMES
    silence = [0.0] * total
    spans = [[word_index, 0, _SPEECH_FRAMES], [word_index + 1, _SPEECH_FRAMES, total]]
    return silence, spans


def _case(recitation, true_class, decode, *, phenomenon=IRAAB, cid=None, lattice=None):
    """An :class:`IntegrationCase` over the 2:38#7 forms; verdict implied by ``recitation``.

    The silence lattice defaults to the one a reciter of ``true_class`` produced (a stop for waqf,
    contiguous speech for wasl) so the predicted full path recovers ``true_class``; pass ``lattice``
    to exercise a detection mismatch.
    """
    silence, spans = lattice or (
        _stopped_lattice() if true_class == WAQF else _continued_lattice()
    )
    return IntegrationCase(
        case_id=cid or f"2:38#7/{recitation}/{true_class}",
        surah_ayah="2:38",
        boundary_word_index=_BOUNDARY_INDEX,
        word="هُدًۭى",
        next_word="فَمَن",
        phenomenon=phenomenon,
        waqf_reference=WAQF_REF,
        wasl_reference=WASL_REF,
        silence=silence,
        word_spans=spans,
        true_class=true_class,
        recitation=recitation,
        decode=decode,
        expected_strict=ACCEPT if recitation == CORRECT else REJECT,
    )


def _discriminating_set():
    """The three-case set for one boundary: legit pause, legit continuation, dropped-i'raab error."""
    return [
        _case(CORRECT, WAQF, PAUSAL_DECODE),  # legitimate pause
        _case(CORRECT, WASL, CONT_DECODE),  # legitimate continuation
        _case(DROPPED, WASL, PAUSAL_DECODE),  # error: pausal ending said mid-continuation
    ]


# --- the scorer primitives --------------------------------------------------------------------


def test_strict_accepts_pausal_against_its_own_reference():
    assert strict_accepts(PAUSAL_DECODE, WAQF_REF)
    assert strict_accepts(CONT_DECODE, WASL_REF)


def test_strict_rejects_pausal_against_the_wasl_reference():
    # The dropped-i'raab error: a pausal ending scored under full strict against the continuation.
    assert not strict_accepts(PAUSAL_DECODE, WASL_REF)


def test_baseline_forgives_the_boundary_error_strict_rejects():
    # The whole point of the hack: forgiving the terminal edge accepts the dropped-i'raab error
    # that full strict rejects — no discrimination.
    assert baseline_accepts(PAUSAL_DECODE, WASL_REF)
    assert not strict_accepts(PAUSAL_DECODE, WASL_REF)


def test_baseline_still_rejects_interior_errors():
    # The forgiveness is only of the trimmed terminal edge; a mistake inside the word is not
    # recoverable by the trailing-trim discount.
    assert not baseline_accepts(INTERIOR_DECODE, WASL_REF)
    assert not strict_accepts(INTERIOR_DECODE, WASL_REF)


def test_baseline_rejects_too_short_a_decode():
    # Below MIN_QUERY_PHONEMES the gate is not meaningful — the baseline must not accept it.
    assert not baseline_accepts("ه", WASL_REF)


# --- per-scenario verdicts --------------------------------------------------------------------


# --- the predicted full path ------------------------------------------------------------------


def test_predicted_class_snaps_waqf_from_a_stopped_lattice():
    # A stopped lattice (≥300 ms gap after the boundary word) snaps to a waqf event.
    assert predicted_class(_case(CORRECT, WAQF, PAUSAL_DECODE)) == WAQF


def test_predicted_class_snaps_wasl_from_a_continued_lattice():
    # A contiguous-speech lattice has no stop → wasl.
    assert predicted_class(_case(DROPPED, WASL, PAUSAL_DECODE)) == WASL


def test_conditional_reference_follows_the_predicted_class_not_true_class():
    # The gate must be driven by the snapped prediction, not the oracle label: give a *true wasl*
    # case a *stopped* lattice and the conditional path selects the waqf reference (predicted waqf),
    # accepting a pausal decode that the oracle (true_class=wasl) would reject.
    mislabelled = _case(DROPPED, WASL, PAUSAL_DECODE, lattice=_stopped_lattice())
    assert predicted_class(mislabelled) == WAQF
    assert verdict(mislabelled, CONDITIONAL) == ACCEPT  # predicted waqf → waqf reference
    assert verdict(mislabelled, ORACLE) == REJECT  # oracle wasl → wasl reference


def test_conditional_selects_reference_by_predicted_class():
    # A legitimate pause: stopped lattice → predicted waqf → waqf reference → accept.
    assert verdict(_case(CORRECT, WAQF, PAUSAL_DECODE), CONDITIONAL) == ACCEPT
    # The dropped error: continued lattice → predicted wasl → wasl reference → reject.
    assert verdict(_case(DROPPED, WASL, PAUSAL_DECODE), CONDITIONAL) == REJECT


def test_oracle_selects_reference_by_true_class():
    assert verdict(_case(CORRECT, WAQF, PAUSAL_DECODE), ORACLE) == ACCEPT
    assert verdict(_case(DROPPED, WASL, PAUSAL_DECODE), ORACLE) == REJECT


def test_baseline_scenario_forgives_the_dropped_error():
    assert verdict(_case(DROPPED, WASL, PAUSAL_DECODE), BASELINE) == ACCEPT


def test_false_waqf_forgives_the_dropped_error_but_rejects_a_continuation():
    # A spurious stop: always the waqf reference. Forgives the pausal error…
    assert verdict(_case(DROPPED, WASL, PAUSAL_DECODE), FALSE_WAQF) == ACCEPT
    # …but rejects a legitimate continuation (its noon does not match the pausal madd).
    assert verdict(_case(CORRECT, WASL, CONT_DECODE), FALSE_WAQF) == REJECT


def test_false_wasl_rejects_a_legitimate_pause():
    # A missed stop: always the wasl reference under full strict — punishes the legitimate pause.
    assert verdict(_case(CORRECT, WAQF, PAUSAL_DECODE), FALSE_WASL) == REJECT


# --- scenario aggregation ---------------------------------------------------------------------


def test_score_scenario_counts_the_two_failure_directions():
    cases = _discriminating_set()
    baseline = score_scenario(cases, BASELINE)
    assert baseline.forgave_errors == 1  # the dropped error, wrongly accepted
    assert baseline.rejected_valid == 0

    false_wasl = score_scenario(cases, FALSE_WASL)
    assert false_wasl.rejected_valid == 1  # the legitimate pause, wrongly rejected
    assert false_wasl.forgave_errors == 0

    conditional = score_scenario(cases, CONDITIONAL)
    assert conditional.correct == conditional.total
    assert conditional.forgave_errors == 0
    assert conditional.rejected_valid == 0


# --- the product gate -------------------------------------------------------------------------


def test_run_eval_passes_when_conditional_regains_all_baseline_forgiven_errors():
    report = run_eval(_discriminating_set())
    assert report.passed
    assert report.baseline_forgave == 1
    assert report.regained_discrimination == 1
    assert report.scenarios[CONDITIONAL].accuracy == 1.0
    assert report.scenarios[BASELINE].forgave_errors == 1
    assert "passed" in report.to_json_dict() and "summary" in report.to_json_dict()


def test_run_eval_fails_when_the_regain_is_vacuous():
    # No error case the baseline forgives → the regain is not demonstrated, so the gate must fail.
    cases = [
        _case(CORRECT, WAQF, PAUSAL_DECODE),
        _case(CORRECT, WASL, CONT_DECODE),
    ]
    report = run_eval(cases)
    assert not report.passed
    assert report.baseline_forgave == 0


def test_run_eval_fails_when_conditional_mis_scores_a_case():
    # A corrupt case the conditional path cannot score correctly fails the gate even though a
    # separate boundary supplies a genuine regain.
    good = _discriminating_set()
    # A "correct" waqf case whose decode is actually the continuation form → conditional (waqf
    # reference) rejects it, so conditional accuracy < 1.
    broken = _case(CORRECT, WAQF, CONT_DECODE, cid="broken/waqf-with-cont-decode")
    report = run_eval(good + [broken])
    assert not report.passed
    assert report.scenarios[CONDITIONAL].accuracy < 1.0


def test_run_eval_rejects_empty_cases():
    with pytest.raises(ValueError):
        run_eval([])


# --- fixtures loader guards -------------------------------------------------------------------


def test_parse_entry_rejects_recitation_verdict_mismatch():
    data = _case_dict(CORRECT, WAQF, PAUSAL_DECODE)
    data["expected_strict"] = REJECT  # a correct realization cannot be a reject
    with pytest.raises(ValueError, match="expected_strict"):
        _parse_entry(data, "test")


def test_parse_entry_rejects_dropped_at_a_true_waqf():
    data = _case_dict(DROPPED, WAQF, PAUSAL_DECODE)  # dropped is only defined at a true wasl
    with pytest.raises(ValueError, match="requires true_class 'wasl'"):
        _parse_entry(data, "test")


def test_parse_entry_rejects_unknown_enum_value():
    data = _case_dict(CORRECT, WAQF, PAUSAL_DECODE)
    data["phenomenon"] = "tajweed"
    with pytest.raises(ValueError, match="phenomenon"):
        _parse_entry(data, "test")


def test_parse_entry_rejects_unknown_field():
    data = _case_dict(CORRECT, WAQF, PAUSAL_DECODE)
    data["extra"] = 1
    with pytest.raises(ValueError, match="unknown fixture field"):
        _parse_entry(data, "test")


def _case_dict(recitation, true_class, decode):
    from dataclasses import asdict

    return asdict(_case(recitation, true_class, decode))


# --- the real frozen fixtures -----------------------------------------------------------------


def test_frozen_fixtures_pass_the_product_gate():
    cases = load_integration_cases(WAQF_INTEGRATION_PATH)
    # The frozen fixtures ARE the product gate's source-of-truth (committed, un-ignored in
    # .gitignore). If they are missing the gate cannot be verified from the repo, so fail loudly —
    # never skip, which would let the whole gate silently vanish from CI.
    assert cases, (
        f"frozen integration fixtures missing at {WAQF_INTEGRATION_PATH} — regenerate with "
        "`python -m tadabur.waqf_integration_gen` (needs quran_transcript) and commit them"
    )
    report = run_eval(cases)
    assert report.passed, report.summary
    assert report.scenarios[CONDITIONAL].accuracy == 1.0
    assert report.scenarios[BASELINE].forgave_errors > 0
    assert report.regained_discrimination == report.baseline_forgave
    # The predicted full path recovers each case's true class on the honest frozen lattices.
    for case in cases:
        assert predicted_class(case) == case.true_class, case.case_id
    # Both ADR-named discriminations are represented in the frozen set.
    phenomena = {c.phenomenon for c in cases}
    assert IRAAB in phenomena and CROSS_WORD_IDGHAM in phenomena
